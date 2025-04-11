# main.py - Fixed (Speed Optimized)
import os
import re
import pytesseract
from PIL import Image
import numpy as np
import torch
from nltk.stem import PorterStemmer
import nltk
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import traceback
import time

# --- NLTK Download ---
try:
    print("Checking/downloading NLTK punkt...")
    nltk.download('punkt', quiet=True)
    print("NLTK punkt check complete.")
except Exception as e:
    print(f"Warning: Could not download NLTK punkt resource automatically. Error: {e}")

# --- Configuration & Model Loading ---
print("Starting CLI App Setup...")
# Tesseract Path
tesseract_cmd_path = None
tesseract_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract', 'tesseract']
for path in tesseract_paths:
    if os.path.exists(path): pytesseract.pytesseract.tesseract_cmd = path; tesseract_cmd_path = path; print(f"Using Tesseract at: {path}"); break
if not tesseract_cmd_path: print("Warning: Tesseract executable not found. OCR disabled unless path valid.")
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# --- Constants ---
# <<< NEW: Default toggles for CLI >>>
ENABLE_SENTENCE_COMPLETION_DEFAULT = True
ENABLE_SEMANTIC_DEDUPLICATION_DEFAULT = True
MAX_COMPLETION_CALLS = 10 # Limit sentence completions

# Load Models
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
EMBEDDER_MODEL_NAME = 'all-MiniLM-L6-v2'
tokenizer, model, embedder = None, None, None
stemmer = PorterStemmer()
try:
    print(f"Loading LLM model: {LLM_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map="auto")
    print("LLM Model loaded successfully!")
except Exception as e: print(f"FATAL ERROR loading LLM: {e}"); traceback.print_exc(); exit(1)
try:
    print(f"Loading Sentence Embedder: {EMBEDDER_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device=device)
    print("Embedder loaded successfully!")
except Exception as e: print(f"ERROR loading Embedder: {e}"); embedder = None; print("Warning: Semantic deduplication disabled.")
if not tokenizer or not model: print("Essential models failed. Exiting."); exit(1)
print("Model loading complete.")

MATH_SYMBOLS = { # Keep as is
    '∫': '\\int', '∑': '\\sum', '∏': '\\prod', '√': '\\sqrt', '∞': '\\infty', '≠': '\\neq', '≤': '\\leq',
    '≥': '\\geq', '±': '\\pm', '→': '\\to', '∂': '\\partial', '∇': '\\nabla', 'π': '\\pi', 'θ': '\\theta',
    'λ': '\\lambda', 'μ': '\\mu', 'σ': '\\sigma', 'ω': '\\omega', 'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma',
    'δ': '\\delta', 'ε': '\\epsilon'
}

# --- Utility Functions (Identical to speed-optimized app.py) ---
# (Includes modified complete_sentence, unchanged pdf/text processing)

def get_stemmed_key(sentence, num_words=5): # Keep as is
    words = re.findall(r'\w+', sentence.lower())[:num_words]; return ' '.join([stemmer.stem(word) for word in words])

# <<< MODIFIED: Sentence completion respects global toggle >>>
def complete_sentence(fragment, force_completion=False, enable_global_toggle=True):
    """Complete sentence fragments using the loaded LLM. More cautious & toggleable."""
    if not enable_global_toggle and not force_completion: return fragment + "." # Check specific toggle passed
    if not model or not tokenizer: return fragment + "."
    if re.search(r'[.!?]$', fragment.strip()): return fragment
    if len(fragment.split()) < 4 or len(fragment) < 20: return fragment + "."
    prompt = f"Complete this sentence fragment concisely:\nFragment: '{fragment}'\nCompleted sentence:"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=25, temperature=0.15, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True)
        completed_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        match = re.search(r"Completed sentence:\s*(.*)", completed_full, re.I | re.S)
        if match:
            completed = match.group(1).strip()
            if completed.lower().startswith(fragment.lower()): final = completed if len(completed)>len(fragment)+3 else fragment+"."
            else: final = completed
            final = re.sub(r'<\|eot_id\|>','',final).strip()
            if not final: return fragment+"."
            if final[-1].isalnum(): final += '.'
            return final
        else: return fragment + "."
    except Exception as e: print(f"Completion Error: {e}"); return fragment + "."

def extract_text_from_pdf(pdf_path, detect_math=True, ocr_enabled=False): # Use previous fixed version
    if not os.path.exists(pdf_path): raise FileNotFoundError(f"PDF not found: {pdf_path}")
    print(f"Extracting text from {pdf_path} (OCR: {ocr_enabled})...")
    extracted_text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            text = page.get_text("text", sort=True)
            if text: extracted_text += text + "\n"
            if ocr_enabled and tesseract_cmd_path and page.get_images(full=True):
                # print(f"  - OCR on page {page_num+1}...") # Reduce logging noise
                import io
                for img_index, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]; base_image = doc.extract_image(xref)
                        fmt = base_image.get("ext", "png").lower()
                        if fmt not in ["png", "jpeg", "jpg", "bmp", "gif", "tiff"]: continue
                        pil_image = Image.open(io.BytesIO(base_image["image"]))
                        processed_img = preprocess_image_for_math_ocr(pil_image) if detect_math else pil_image
                        ocr_text = pytesseract.image_to_string(processed_img, config='--psm 6 --oem 3')
                        if ocr_text.strip():
                            if detect_math:
                                for symbol, latex in MATH_SYMBOLS.items(): ocr_text = ocr_text.replace(symbol, f" {latex} ")
                            extracted_text += f"\n[OCR_IMG {img_index+1}] {ocr_text.strip()} [/OCR_IMG]\n"
                    except pytesseract.TesseractNotFoundError: print("Error: Tesseract not found. Disabling OCR."); ocr_enabled = False; break
                    except Exception as e: print(f"Warn: OCR img {img_index} pg {page_num+1} failed: {e}") # Use Warning
        doc.close()
        lines = extracted_text.split('\n')
        if len(lines) > 2:
            if len(lines[0].strip()) < 25 and re.match(r'^[\s\d\W]*?(\d{1,3})?[\s\d\W]*$', lines[0].strip()): lines = lines[1:]
            if len(lines) > 1 and len(lines[-1].strip()) < 25 and re.match(r'^[\s\d\W]*?(\d{1,3})?[\s\d\W]*$', lines[-1].strip()): lines = lines[:-1]
        extracted_text = '\n'.join(lines)
        # print(f"Extracted ~{len(extracted_text)} chars.") # Reduce logging noise
        return extracted_text
    except Exception as e: print(f"Extraction failed: {e}"); traceback.print_exc(); return ""

def preprocess_image_for_math_ocr(image): # Keep previous fixed
    if image.mode != 'L': image = image.convert('L')
    image_array = np.array(image); threshold = np.mean(image_array) * 0.85
    binary_image = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_image)

def detect_math_content(text): # Keep previous fixed
    math_keywords = [r'\b(equation|formula|theorem|lemma|proof|calculus|algebra|derivative|function|integral|vector|matrix|variable|constant|graph|plot|solve|calculate|measure|angle|degree)\b']
    math_symbols = r'[=><≤≥≠\+\-\*\/\^∫∑∏√∞≠≤≥±→∂∇πθλμσωαβγδε%]'
    function_notation = r'\b[a-zA-Z]\s?\([a-zA-Z0-9,\s\+\-\*\/]+\)'
    latex_delimiters = r'\$.*?\$|\\\(.*?\\\)|\\[a-zA-Z]+(\{.*?\})*'
    math_list_items = r'^\s*(\d+\.|\*|\-)\s*[=><≤≥≠\+\-\*\/\^∫∑∏√∞≠≤≥±→∂∇πθλμσωαβγδε\$\\]'
    if re.search('|'.join(math_keywords), text, re.IGNORECASE): return True
    text_sample = text[:30000]
    for pattern in [math_symbols, function_notation, latex_delimiters]:
        if re.search(pattern, text_sample): return True
    if re.search(math_list_items, text_sample, re.MULTILINE): return True
    return False

def clean_text(text): # Keep previous fixed
    text = re.sub(r'\f', ' ', text)
    text = re.sub(r'\[OCR_IMG.*?\[\/OCR_IMG\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])(\w)', r'\1 \2', text)
    lines = text.split('\n'); cleaned_lines = [l for l in lines if len(l.strip())>2 or l.strip() in ['.','!','?']]; text = '\n'.join(cleaned_lines)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_text_into_chunks(text, chunk_size=800, overlap=50): # Keep previous fixed
    try: sentences = nltk.sent_tokenize(text); assert sentences
    except Exception: sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]; sentences = sentences or [text]
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        sentence = sentence.strip(); sent_length = len(sentence.split());
        if not sentence: continue
        if current_length > 0 and current_length + sent_length > chunk_size:
            chunks.append(' '.join(current_chunk)); overlap_wc, overlap_idx = 0, []
            for i in range(len(current_chunk)-1, -1, -1):
                overlap_wc += len(current_chunk[i].split()); overlap_idx.insert(0, i)
                if overlap_wc >= overlap: break
            current_chunk = [current_chunk[i] for i in overlap_idx] + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
        else: current_chunk.append(sentence); current_length += sent_length
    if current_chunk: chunks.append(' '.join(current_chunk))
    refined_chunks, i = [], 0; min_chunk_w = max(50, chunk_size*0.2)
    while i < len(chunks):
        c_w = len(chunks[i].split()); n_w = len(chunks[i+1].split()) if i+1<len(chunks) else 0
        if c_w < min_chunk_w and i+1<len(chunks) and c_w+n_w <= chunk_size*1.5: refined_chunks.append(chunks[i]+" "+chunks[i+1]); i+=2
        else: refined_chunks.append(chunks[i]); i+=1
    chunks = refined_chunks; MAX_CHUNKS = 20
    if len(chunks) > MAX_CHUNKS: print(f"Warn: Reducing chunks {len(chunks)}->{MAX_CHUNKS}"); step=max(1,len(chunks)//MAX_CHUNKS); chunks=[chunks[i] for i in range(0,len(chunks),step)][:MAX_CHUNKS]
    print(f"Split into {len(chunks)} chunks (~{chunk_size}w, ~{overlap}w overlap).")
    return chunks

def determine_reading_level(grade): # Keep previous fixed
    if not isinstance(grade, int) or not (1 <= grade <= 12): grade = 6
    age = grade + 5
    if 1 <= grade <= 3: level, desc = "lower", f"early elementary (grades {grade}, ~age {age}-{age+1})"
    elif 4 <= grade <= 6: level, desc = "middle", f"late elem./middle school (grades {grade}, ~age {age}-{age+1})"
    elif 7 <= grade <= 9: level, desc = "higher", f"junior high/early high (grades {grade}, ~age {age}-{age+1})"
    else: level, desc = "higher", f"high school (grades {grade}, ~age {age}-{age+1})"
    return level, desc

# --- Prompts Dictionary (Keep Revised Structure) ---
# (Using the same strong, structured prompts from the previous fixed version)
prompts = { # Copy full prompts dict from app.py here
    "lower": { "standard": ("You are summarizing text for a young child (grades 1-3, ages 6-8).\nInstructions:\n1. Use VERY simple words and short sentences.\n2. Explain the absolute main idea first in one sentence under '# Simple Summary'.\n3. Then, list 3-5 key points using bullet points '- '. Each point should be a full, simple sentence.\n4. Do NOT include complex details or jargon.\n5. Finish with ONE fun, simple activity related to the text under '## Fun Activity'.\nText to summarize:\n{text}"), "math": ("You are explaining a math topic to a young child (grades 1-3, ages 6-8).\nInstructions:\n1. Use very simple words, short sentences, and analogies (like counting toys).\n2. Start with '# Math Fun'.\n3. Explain the main math idea very simply under '## What We Learned'.\n4. If there are steps, list them simply under '## Steps' using numbers (1., 2.).\n5. Give one simple example with small numbers under '## Example'.\n6. Finish with ONE easy practice question or drawing task under '## Practice Time'.\nText to explain:\n{text}") },
    "middle": { "standard": ("You are summarizing text for a student in grades 4-6 (ages 9-11).\nInstructions:\n1. Start with the main heading '# Summary'.\n2. Identify 2-4 main topics or sections from the text.\n3. For each main topic, create a subheading using '## Topic Name'.\n4. Under each subheading, list the key information using bullet points '- '. Use clear, complete sentences.\n5. Explain any important terms simply.\n6. Ensure the summary flows logically. Avoid just listing facts.\n7. Conclude with ONE practical activity or thought question related to the text under '## Try This'.\nText to summarize:\n{text}"), "math": ("You are explaining a math concept to a student in grades 4-6 (ages 9-11).\nInstructions:\n1. Start with the heading '# Math Explained'.\n2. Explain the core math concept clearly under '## The Concept'.\n3. Provide a step-by-step example of a typical problem under '## Step-by-Step Example'. Use numbered steps (1., 2.).\n4. Briefly explain why this math is useful or where it's used under '## Why It Matters'.\n5. Conclude with ONE practice problem (include the answer separately if possible) under '## Practice Problem'.\n6. Use clear language and formatting (headings, bullets, numbered steps).\nText to explain:\n{text}") },
    "higher": { "standard": ("You are creating a comprehensive, well-structured summary for a high school student (grades 7-12, ages 12-18).\nInstructions:\n1. Start with the main heading '# Comprehensive Summary'.\n2. Identify key themes, arguments, sections, or concepts. Create logical subheadings ('## Theme/Section Name') for each.\n3. Under each subheading, synthesize the key information. Use paragraphs for explanation and bullet points '- ' for specific details, evidence, or examples.\n4. Use appropriate academic vocabulary but ensure clarity. Define key terms if necessary.\n5. Connect ideas logically. Analyze or evaluate points where appropriate, don't just list them.\n6. Mention real-world implications, applications, or connections if relevant under a '## Connections' subheading.\n7. Conclude with ONE thought-provoking question, research idea, or analysis task under '## Further Thinking'.\n8. Structure the entire output clearly using Markdown.\nText to summarize:\n{text}"), "math": ("You are explaining an advanced math topic for a high school student (grades 7-12, ages 12-18).\nInstructions:\n1. Start with the heading '# Advanced Math Concepts'.\n2. Provide concise definitions of key terms/concepts under '## Definitions'.\n3. Explain the core theory, logic, or theorem under '## Core Theory'. Use paragraphs and potentially bullet points.\n4. Include a non-trivial worked example demonstrating the concept or technique under '## Worked Example'. Show steps clearly.\n5. Discuss applications or connections to other fields (science, engineering, etc.) under '## Applications'.\n6. Conclude with ONE challenging problem or extension idea under '## Challenge'.\n7. Use appropriate mathematical notation (like LaTeX placeholders if present in source) and structure clearly using Markdown subheadings (##) and formatting.\nText to explain:\n{text}") }
}


# <<< MODIFIED: model_generate - slightly lower temp >>>
def model_generate(prompt_text, max_new_tokens=1024, temperature=0.5): # Default temp lowered
    if not model or not tokenizer: return "Error: LLM not available."
    current_model_device = next(model.parameters()).device
    model_context_limit = getattr(tokenizer, 'model_max_length', 4096); buffer=100
    max_prompt_len = model_context_limit - max_new_tokens - buffer
    # print(f"Generating response (max_new={max_new_tokens}, temp={temperature})...") # Reduce logging
    try:
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(current_model_device)
        input_token_count = inputs['input_ids'].shape[1]
        if input_token_count >= max_prompt_len: print(f"Warning: Prompt potentially truncated ({input_token_count}/{max_prompt_len} tokens).")
        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True if temperature > 0.01 else False)
        gen_time = time.time() - start_time
        print(f"...Generation took {gen_time:.2f}s.") # Print timing here
        generated_text = tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True)
        generated_text = re.sub(r'<\|eot_id\|>', '', generated_text).strip()
        return generated_text
    except Exception as e: print(f"Generation Error: {e}"); traceback.print_exc(); return f"Error: Model generation failed - {e}"

# <<< MODIFIED: generate_summary - uses CLI toggles, optimized params >>>
def generate_summary(text_chunks, grade_level_category, grade_level_desc, duration_minutes, has_math=False,
                     enable_completion_cli=True, enable_deduplication_cli=True): # Accept CLI toggles
    if duration_minutes == 10: min_words, max_words = 1200, 1600
    elif duration_minutes == 20: min_words, max_words = 2400, 3200
    elif duration_minutes == 30: min_words, max_words = 3600, 4500
    else: min_words, max_words = 1200, 1600
    print(f"Targeting: {grade_level_desc}, {duration_minutes} mins ({min_words}-{max_words} words).")
    full_text = ' '.join(text_chunks); full_text_tokens = len(tokenizer.encode(full_text))
    model_context_limit = getattr(tokenizer, 'model_max_length', 4096)
    target_max_tokens=int(max_words*1.3)+200; safe_gen_limit=model_context_limit//2
    max_new_tokens=max(min(target_max_tokens, safe_gen_limit), 512)
    print(f"Max new tokens: {max_new_tokens}")
    prompt_buffer=700; req_tokens=full_text_tokens + max_new_tokens + prompt_buffer
    single_pass = req_tokens < (model_context_limit*0.9) and full_text_tokens < (model_context_limit*0.6) and max_new_tokens <= 2048

    initial_summary = ""
    if single_pass and text_chunks:
        print("Attempting single pass summary.")
        prompt = prompts[grade_level_category]["math" if has_math else "standard"].format(text=full_text)
        prompt += f"\n\nIMPORTANT: Ensure structure and length ({min_words}-{max_words} words)."
        initial_summary = model_generate(prompt, max_new_tokens=max_new_tokens, temperature=0.55) # Lower temp
    elif text_chunks:
        print(f"Iterative summary ({len(text_chunks)} chunks).")
        chunk_summaries = []; max_tokens_chunk = max(min((model_context_limit//(len(text_chunks)+1))-100, 200), 80) # Smaller target
        print(f"Max new tokens per chunk: {max_tokens_chunk}")
        for i, chunk in enumerate(text_chunks):
            chunk_prompt = f"Key points from chunk {i+1}/{len(text_chunks)}:\n{chunk}\n\nKey Points (CONCISE bullet list):"
            chunk_summary = model_generate(chunk_prompt, max_new_tokens=max_tokens_chunk, temperature=0.2) # Low temp
            if not chunk_summary.startswith("Error:") and len(chunk_summary.split()) >= 3:
                chunk_summary = re.sub(r'^.*?Key Points.*?\n','',chunk_summary,flags=re.IGNORECASE).strip()
                if chunk_summary: chunk_summaries.append(chunk_summary)
        if not chunk_summaries: return "Error: No valid chunk summaries."
        print("Consolidating chunks...")
        base_instr = prompts[grade_level_category]["math" if has_math else "standard"].split("Text to summarize:")[0]
        consol_prompt = f"{base_instr}\nSYNTHESIZE points below into ONE COHERENT, STRUCTURED summary for {grade_level_desc}.\nFollow original instructions (headings, bullets, activity). Organize logically. Aim for {min_words}-{max_words} words.\n\nChunk Summaries:\n\n"+"\n\n---\n\n".join(chunk_summaries)+"\n\nFinal Consolidated Summary:"
        initial_summary = model_generate(consol_prompt, max_new_tokens=max_new_tokens, temperature=0.6) # Med temp
    else: return "Error: Zero chunks."

    if initial_summary.startswith("Error:"): return initial_summary
    current_summary = initial_summary; current_words = len(current_summary.split())
    print(f"Initial summary: {current_words} words. Checking length.")

    attempts, max_attempts = 0, 2
    while current_words < min_words and attempts < max_attempts:
        print(f"Summary short. Elaborating (Attempt {attempts+1}/{max_attempts})...")
        prompt = f"Elaborate on points in the summary below within the existing structure...\nCurrent Summary:\n{current_summary}\n\nContinue summary with more detail:" # Shorter prompt
        needed=min_words-current_words; tokens_add=max(min(int(needed*1.5),max_new_tokens//2,700),150)
        new_part = model_generate(prompt, max_new_tokens=tokens_add, temperature=0.65) # Med temp
        if new_part.startswith("Error:") or len(new_part.split()) < 10: print("Stopping lengthening."); break
        current_summary += "\n\n" + new_part.strip(); current_words = len(current_summary.split()); attempts += 1
    if attempts == max_attempts and current_words < min_words: print(f"Warning: Max lengthening attempts reached ({current_words}/{min_words} words).")

    # Trimming (keep as is)
    words = current_summary.split()
    if len(words) > max_words:
        print(f"Trimming from {len(words)} to ~{max_words} words.")
        act_match=re.search(r'(##\s+(Activity|Practice|Thinking|Challenge|Try This|Fun Activity))',current_summary, re.I|re.M)
        if act_match:
            main_cont=current_summary[:act_match.start()]; act_cont=current_summary[act_match.start()]
            main_w=main_cont.split();
            if len(main_w) > max_words:
                limit_idx=len(' '.join(main_w[:max_words])); end_idx=main_cont.rfind('.',0,limit_idx)
                main_cont = main_cont[:end_idx+1] if end_idx != -1 else ' '.join(main_w[:max_words])+"..."
            current_summary = main_cont.strip() + "\n\n" + act_cont.strip()
        else: current_summary = ' '.join(words[:max_words]); current_summary += "..." if current_summary[-1].isalnum() else ""

    print("Post-processing summary...")
    # <<< MODIFIED: Pass CLI toggles >>>
    processed_summary = enhanced_post_process(current_summary, grade_level_category,
                                              enable_completion=enable_completion_cli,
                                              enable_deduplication=enable_deduplication_cli)

    # Fallback activity (keep as is)
    activity_pattern = r'^##\s+(Fun Activity|Practice Time|Try This|Practice Problem|Further Thinking|Challenge|Activity)\s*$'
    if not re.search(activity_pattern, processed_summary, re.I | re.M):
        print("Warning: Activity section missing. Generating fallback...")
        activity = generate_activity(processed_summary, grade_level_category, grade_level_desc)
        h_map={"lower":"## Fun Activity","middle":"## Try This","higher":"## Further Thinking"}; def_h="## Activity"
        if has_math: head = {"lower":"## Practice Time","middle":"## Practice Problem","higher":"## Challenge"}.get(grade_level_category, def_h)
        else: head = h_map.get(grade_level_category, def_h)
        processed_summary += f"\n\n{head}\n{activity}"

    final_word_count = len(processed_summary.split())
    print(f"Final summary: {final_word_count} words.")
    return processed_summary

# <<< MODIFIED: enhanced_post_process accepts toggles >>>
def enhanced_post_process(summary, grade_level_category, enable_completion=True, enable_deduplication=True):
    """Advanced post-processing with toggles for completion and deduplication."""
    if summary.startswith("Error:"): return summary
    print(f"Running Markdown-aware post-processing (Completion:{enable_completion}, Dedup:{enable_deduplication})...")
    completion_calls_made = 0 # Counter

    # --- 1. Cleanup & Heading (Keep as is) ---
    try: prompt_lines=prompts[grade_level_category]["standard"].splitlines(); head_line=next((l for l in prompt_lines if l.strip().startswith('#')), None); exp_head = head_line.strip().lstrip('# ').strip() if head_line else "Summary"
    except: exp_head = "Summary"
    summary = re.sub(r'^\s*#+.*?(\n|$)', f'# {exp_head}\n\n', summary.strip(), count=1, flags=re.I);
    if not summary.startswith("# "): summary = f'# {exp_head}\n\n' + summary

    # --- 2. Process Lines & Structure (Keep as is) ---
    lines = summary.split('\n'); processed_data = []; seen_frags = set()
    for line in lines:
        s_line = line.strip()
        if not s_line:
            if processed_data and processed_data[-1]["type"] != "blank": processed_data.append({"text":"", "type":"blank"})
            continue
        l_type, content, is_head, is_bullet = "paragraph", s_line, False, False
        if s_line.startswith('## '): l_type, content, is_head = "subheading", s_line[3:].strip(), True
        elif s_line.startswith('# '): l_type, content, is_head = "heading", s_line[2:].strip(), True
        elif s_line.startswith('- '): l_type, content, is_bullet = "bullet", s_line[2:].strip(), True
        elif re.match(r'^\d+\.\s+', s_line): l_type, content, is_bullet = "numbered", re.sub(r'^\d+\.\s+', '', s_line), True
        if not content: continue
        cont_key = ' '.join(content.lower().split()[:10])
        if not is_head and cont_key in seen_frags and len(content.split())<15: continue
        if not is_head: seen_frags.add(cont_key)

        # --- 3. Sentence Completion (Conditional & Limited) ---
        # <<< MODIFIED: Use enable_global_toggle in call >>>
        if enable_completion and l_type in ["paragraph", "bullet", "numbered"] and len(content.split()) > 4:
            if not re.search(r'[.!?:]$', content) and content[0].isupper() and completion_calls_made < MAX_COMPLETION_CALLS:
                original_content = content
                content = complete_sentence(content, enable_global_toggle=enable_completion) # Pass toggle
                if content != original_content: completion_calls_made += 1
        if l_type in ["paragraph", "bullet", "numbered"]:
             if content and content[0].islower() and not re.match(r'^[a-z]\s*\(', content): content = content[0].upper()+content[1:]
             if content and content[-1].isalnum(): content += '.'

        if l_type == "blank" and processed_data and processed_data[-1]["type"] == "blank": continue
        processed_data.append({"text":content, "type":l_type})

    # --- 4. Semantic Deduplication (Conditional) ---
    # (Identical logic as speed-optimized app.py - checks enable_deduplication)
    points_for_dedup = []; indices_map = {}
    if enable_deduplication and embedder:
        for i, data in enumerate(processed_data):
            if data["type"] in ["paragraph","bullet","numbered"] and len(data["text"].split()) > 6:
                cont = data["text"]; points_for_dedup.append(cont)
                if cont not in indices_map: indices_map[cont] = []
                indices_map[cont].append(i)
    else:
        if not enable_deduplication: print("Skipping semantic dedup (disabled by request).")
        elif not embedder: print("Skipping semantic dedup (embedder not loaded).")

    kept_indices = set(range(len(processed_data)))
    if points_for_dedup and enable_deduplication and embedder:
        print(f"Running semantic dedup on {len(points_for_dedup)} points...")
        try:
            unique_pts = remove_duplicates_semantic(points_for_dedup); unique_set = set(unique_pts)
            print(f"Reduced to {len(unique_pts)} unique points.")
            indices_to_remove = set(); processed_removal = set()
            for cont, orig_indices in indices_map.items():
                if cont in processed_removal: continue
                if cont not in unique_set:
                    for index in orig_indices:
                        is_only = False # Basic check
                        if index > 0 and processed_data[index-1]["type"] in ["heading","subheading"]:
                            if index==len(processed_data)-1 or processed_data[index+1]["type"] in ["heading","subheading","blank"]: is_only = True
                        if not is_only: indices_to_remove.add(index)
                processed_removal.add(cont)
            kept_indices -= indices_to_remove
            print(f"Marked {len(indices_to_remove)} lines for removal.")
        except Exception as e: print(f"Warning: Dedup failed: {e}")

    # --- 5. Final Assembly (Keep as is) ---
    final_text = ""; last_type = None
    kept_data = [processed_data[i] for i in sorted(list(kept_indices))]
    for i, data in enumerate(kept_data):
        curr_type, content = data["type"], data["text"]
        if i > 0:
            if curr_type in ["heading","subheading"]: final_text += "\n\n"
            elif curr_type == "paragraph" and last_type not in ["heading","subheading","blank"]: final_text += "\n\n"
            elif curr_type != "blank" and last_type != "blank": final_text += "\n"
            elif curr_type == "blank" and last_type == "blank": continue
        if curr_type == "heading": final_text += f"# {content}"
        elif curr_type == "subheading": final_text += f"## {content}"
        elif curr_type == "bullet": final_text += f"- {content}"
        elif curr_type == "numbered": final_text += f"1. {content}" # Basic numbering
        elif curr_type == "paragraph": final_text += content
        last_type = curr_type
    print("Post-processing finished.")
    return final_text.strip()

# <<< MODIFIED: remove_duplicates_semantic - added timing >>>
def remove_duplicates_semantic(points, similarity_threshold=0.90, batch_size=128):
    # (Identical logic as speed-optimized app.py)
    if not points or not embedder or len(points)<2: return points
    start_dedup = time.time()
    try:
        valid_pts = [p for p in points if len(p.split()) > 4]
        if not valid_pts:
            return points
        embeddings = embedder.encode(valid_pts, convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size, device=embedder.device)
        cos_sim = util.cos_sim(embeddings, embeddings)
        to_remove = set()
        for i in range(len(valid_pts)):
            if i in to_remove: continue
            for j in range(i+1, len(valid_pts)):
                if j in to_remove: continue
                if cos_sim[i][j] > similarity_threshold: to_remove.add(j)
        unique_pts = [valid_pts[i] for i in range(len(valid_pts)) if i not in to_remove]
        short_pts = [p for p in points if len(p.split()) <= 4]
        print(f"Semantic deduplication took {time.time() - start_dedup:.2f}s.")
        return unique_pts + short_pts
    except torch.cuda.OutOfMemoryError: print("OOM Error during dedup. Skipping."); return points
    except Exception as e: print(f"Dedup Error: {e}"); traceback.print_exc(); return points

# --- generate_activity (Keep previous fixed version) ---
def generate_activity(summary_text, grade_level_category, grade_level_desc):
    # (Identical logic as speed-optimized app.py)
    if not model or not tokenizer: return "- Review key points."
    print("Generating fallback activity...")
    act_type = "activity"
    if grade_level_category=="lower": act_type="fun activity/question"
    elif grade_level_category=="middle": act_type="practical activity/thought question"
    elif grade_level_category=="higher": act_type="provoking question/research idea/analysis task"
    prompt = f"Suggest ONE simple, engaging {act_type} for {grade_level_desc} based on this summary context:\n...{' '.join(re.sub(r'^#.*?\n','',summary_text).strip().split()[-200:])}\n\nActivity Suggestion:"
    activity = model_generate(prompt, max_new_tokens=80, temperature=0.7) # Slightly higher temp
    if activity.startswith("Error:"): print(f"Fallback activity failed: {activity}"); activity = ""
    else: activity = re.sub(r'^[\-\*\s]+','',activity.strip().replace("Activity Suggestion:","").strip()).strip(); activity = re.sub(r'\.$','',activity).strip()
    if activity:
        activity = f"- {activity[0].upper() + activity[1:]}"
        if activity[-1].isalnum(): activity += '.'
        return activity
    else: print("Warning: Failed fallback activity."); fallbacks={"lower":"- Draw!", "middle":"- Explain!", "higher":"- Find example."}; return fallbacks.get(grade_level_category, "- Review.")

######################################
# Main Execution Logic               #
######################################

def main():
    parser = argparse.ArgumentParser(description="Generate grade-level PDF summary.")
    parser.add_argument("pdf_path", help="Input PDF file path.")
    parser.add_argument("-g", "--grade", type=int, required=True, help="Target grade (1-12).")
    parser.add_argument("-d", "--duration", type=int, choices=[10, 20, 30], required=True, help="Target duration (10, 20, 30 mins).")
    parser.add_argument("-o", "--output", help="Output summary file path (optional).")
    parser.add_argument("--ocr", action="store_true", help="Enable image OCR (slow).")
    parser.add_argument("--chunk-size", type=int, default=500, help="Words/chunk (default: 500).")
    parser.add_argument("--overlap", type=int, default=50, help="Word overlap (default: 50).")
    # <<< NEW: CLI toggles for refinement >>>
    parser.add_argument("--no-completion", action="store_false", dest="completion", default=ENABLE_SENTENCE_COMPLETION_DEFAULT, help="Disable sentence completion in post-processing.")
    parser.add_argument("--no-dedup", action="store_false", dest="deduplication", default=ENABLE_SEMANTIC_DEDUPLICATION_DEFAULT, help="Disable semantic deduplication in post-processing.")

    args = parser.parse_args()
    main_start_time = time.time()

    # Input Validation (keep as before)
    if not os.path.exists(args.pdf_path): print(f"Error: PDF not found: '{args.pdf_path}'"); return
    if not (1 <= args.grade <= 12): print(f"Error: Grade must be 1-12."); return
    if not (100 <= args.chunk_size <= 2000): print(f"Warning: Chunk size invalid. Using 500."); args.chunk_size = 500
    if not (0 <= args.overlap <= args.chunk_size // 2): print(f"Warning: Overlap invalid. Using 50."); args.overlap = 50

    ocr_enabled = args.ocr
    if args.ocr and not tesseract_cmd_path: print("Warning: --ocr used, but Tesseract not found. Disabling OCR."); ocr_enabled = False

    if args.output: output_file = args.output
    else: name = os.path.splitext(os.path.basename(args.pdf_path))[0]; output_file = f"{name}_summary_grade{args.grade}_duration{args.duration}.txt"
    output_dir = os.path.dirname(output_file);
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)

    print(f"\n--- Configuration ---"); print(f"Input: {args.pdf_path}"); print(f"Grade: {args.grade}, Duration: {args.duration}m")
    print(f"Output: {output_file}"); print(f"Chunk: {args.chunk_size}, Overlap: {args.overlap}"); print(f"OCR: {'Enabled' if ocr_enabled else 'Disabled'}")
    # <<< NEW: Print refinement settings >>>
    print(f"Sentence Completion: {'Enabled' if args.completion else 'Disabled'}")
    print(f"Semantic Deduplication: {'Enabled' if args.deduplication and embedder else 'Disabled'}{'' if embedder else ' (Embedder unavailable)'}")
    print(f"Device: {device.upper()}"); print(f"LLM: {LLM_MODEL_NAME}"); print(f"Embedder: {EMBEDDER_MODEL_NAME if embedder else 'Off'}")
    print("-" * 21)

    try:
        grade_cat, grade_desc = determine_reading_level(args.grade)
        print(f"Reading level: {grade_cat} ({grade_desc})")
        print("\n--- Processing PDF ---")
        extract_start = time.time()
        raw_text = extract_text_from_pdf(args.pdf_path, detect_math=True, ocr_enabled=ocr_enabled)
        print(f"Extraction took {time.time()-extract_start:.2f}s")
        if not raw_text or len(raw_text.strip()) < 50: print("\nError: No significant text extracted."); return

        proc_start = time.time()
        has_math = detect_math_content(raw_text); # print(f"Math detected: {has_math}") # Reduce noise
        print("Cleaning & Chunking text...")
        cleaned = clean_text(raw_text); chunks = split_text_into_chunks(cleaned, chunk_size=args.chunk_size, overlap=args.overlap)
        if not chunks: print("\nError: Failed to split text."); return
        print(f"Processing took {time.time()-proc_start:.2f}s")

        print("\n--- Generating Summary ---")
        gen_start = time.time()
        # <<< MODIFIED: Pass CLI toggles >>>
        summary = generate_summary(chunks, grade_cat, grade_desc, args.duration, has_math,
                                   enable_completion_cli=args.completion,
                                   enable_deduplication_cli=args.deduplication)
        print(f"Generation took {time.time()-gen_start:.2f}s")

        if summary.startswith("Error:"): print(f"\n--- Summarization Failed ---\n{summary}"); return

        word_count = len(summary.split())
        print("\n--- Summary Generation Complete ---"); print(f"Final Word Count: {word_count}")

        try:
            with open(output_file, 'w', encoding='utf-8') as f: f.write(summary)
            print(f"\n✅ Summary saved to: {output_file}")
        except IOError as e:
            print(f"\nError writing to file '{output_file}': {e}")
            print("\n--- Generated Summary (Console) ---"); print(summary); print("--- End Summary ---")

    # (Keep existing error handling)
    except FileNotFoundError as e: print(f"Error: {e}")
    except pytesseract.TesseractNotFoundError: print("\nError: Tesseract issue during processing.")
    except torch.cuda.OutOfMemoryError: print("\nError: CUDA Out of Memory! Try smaller settings/doc."); traceback.print_exc()
    except Exception as e: print(f"\n--- Unexpected Error ---"); print(f"{type(e).__name__}: {e}"); traceback.print_exc()
    finally: print(f"\n--- Total Execution Time: {time.time() - main_start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()