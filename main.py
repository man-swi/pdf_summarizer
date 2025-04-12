# main.py (Modified for Llama-4-Scout-17B and 4-bit Quantization)

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
# <<< Import quantization config and updated model loader >>>
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import traceback
import time

# --- NLTK Download ---
try:
    print("Checking/downloading NLTK punkt...")
    # Use a simplified download for CLI, assuming standard paths might work
    # Or replicate the more robust path logic from app.py if needed
    nltk.download('punkt', quiet=True)
    print("NLTK punkt check complete.")
except Exception as e:
    print(f"Warning: Could not download NLTK punkt resource automatically. Error: {e}")

# --- Configuration & Model Loading ---
print("Starting CLI App Setup...")

# Tesseract Path (Keep existing logic)
tesseract_cmd_path = None
tesseract_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract', 'tesseract']
for path in tesseract_paths:
    if os.path.exists(path) and os.access(path, os.X_OK):
        pytesseract.pytesseract.tesseract_cmd = path
        tesseract_cmd_path = path
        print(f"Using Tesseract at: {path}")
        break
if not tesseract_cmd_path: print("Warning: Tesseract executable not found. OCR disabled unless path valid.")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Constants (Keep existing refinement toggles)
ENABLE_SENTENCE_COMPLETION_DEFAULT = True
ENABLE_SEMANTIC_DEDUPLICATION_DEFAULT = True
MAX_COMPLETION_CALLS = 10

# --- MODIFIED: Model Name and Quantization Setup ---
LLM_MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct" # <<< CHANGED Model Name
EMBEDDER_MODEL_NAME = 'all-MiniLM-L6-v2'

# Configure 4-bit quantization (same as in app.py)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
print(f"Using 4-bit quantization config: {quantization_config}")
# --- END MODIFIED ---

tokenizer, model, embedder = None, None, None
stemmer = PorterStemmer()

try:
    print(f"Loading LLM model: {LLM_MODEL_NAME} (with 4-bit quantization)...")
    # <<< NOTE: Ensure you provide authentication (e.g., HF_TOKEN env var or login) if needed >>>
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    print("Tokenizer loaded.")

    # --- MODIFIED: Load Model with Quantization ---
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=quantization_config, # <<< ADDED Quantization Config
        device_map="auto",                  # Accelerate handles device placement
    )
    print("LLM Model loaded successfully!")
    # <<< Optional Debugging >>>
    actual_context_limit = getattr(model.config, 'max_position_embeddings', None)
    if actual_context_limit is None: actual_context_limit = getattr(tokenizer, 'model_max_length', None)
    print(f"DEBUG: Model Config/Tokenizer reported context limit: {actual_context_limit}")

except Exception as e:
    print(f"FATAL ERROR loading LLM model {LLM_MODEL_NAME}: {e}")
    traceback.print_exc()
    if "out of memory" in str(e).lower(): print("Attempting to clear CUDA cache..."); torch.cuda.empty_cache()
    exit(1)

try:
    print(f"Loading Sentence Embedder: {EMBEDDER_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device=device)
    print("Embedder loaded successfully!")
except Exception as e:
    print(f"ERROR loading Embedder: {e}")
    embedder = None
    print("Warning: Semantic deduplication disabled.")

if not tokenizer or not model:
    print("Essential models (tokenizer/LLM) failed. Exiting.")
    exit(1)
print("Model loading complete.")

MATH_SYMBOLS = { # Keep as is
    '∫': '\\int', '∑': '\\sum', '∏': '\\prod', '√': '\\sqrt', '∞': '\\infty', '≠': '\\neq', '≤': '\\leq',
    '≥': '\\geq', '±': '\\pm', '→': '\\to', '∂': '\\partial', '∇': '\\nabla', 'π': '\\pi', 'θ': '\\theta',
    'λ': '\\lambda', 'μ': '\\mu', 'σ': '\\sigma', 'ω': '\\omega', 'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma',
    'δ': '\\delta', 'ε': '\\epsilon'
}

# --- Utility Functions ---
# (Keep functions like get_stemmed_key, complete_sentence, extract_text_from_pdf, etc.)
# Make sure model_generate uses the corrected context length logic from the modified app.py

def get_stemmed_key(sentence, num_words=5): # Keep as is
    words = re.findall(r'\w+', sentence.lower())[:num_words]; return ' '.join([stemmer.stem(word) for word in words])

def complete_sentence(fragment, force_completion=False, enable_global_toggle=True):
    # (Keep the version from the modified app.py - respects toggle)
    if not enable_global_toggle and not force_completion: return fragment + "."
    if not model or not tokenizer: return fragment + "."
    if re.search(r'[.!?]$', fragment.strip()): return fragment
    if len(fragment.split()) < 4 or len(fragment) < 20: return fragment + "."
    prompt = f"Complete this sentence fragment concisely:\nFragment: '{fragment}'\nCompleted sentence:"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=25, temperature=0.15, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True)
        completed_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        match = re.search(r"Completed sentence:\s*(.*)", completed_full, re.I | re.S)
        if match:
            completed = match.group(1).strip()
            final = completed if completed.lower().startswith(fragment.lower()) and len(completed)>len(fragment)+3 else (fragment+"." if completed.lower().startswith(fragment.lower()) else completed)
            final = re.sub(r'<\|eot_id\|>','',final).strip()
            if not final: return fragment+"."
            if final[-1].isalnum(): final += '.'
            return final
        else: return fragment + "."
    except Exception as e: print(f"Completion Error: {e}"); return fragment + "."

def extract_text_from_pdf(pdf_path, detect_math=True, ocr_enabled=False):
    # (Keep the robust version from modified app.py)
    if not os.path.exists(pdf_path): raise FileNotFoundError(f"PDF not found: {pdf_path}")
    print(f"Extracting text from {pdf_path} (OCR: {ocr_enabled})...")
    extracted_text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            text = page.get_text("text", sort=True)
            if text: extracted_text += text + "\n"
            if ocr_enabled and tesseract_cmd_path and page.get_images(full=True):
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
                    except Exception as e: print(f"Warn: OCR img {img_index} pg {page_num+1} failed: {e}")
        doc.close()
        lines = extracted_text.split('\n')
        if len(lines) > 2:
            if len(lines[0].strip()) < 25 and re.match(r'^[\s\d\W]*?(\d{1,3})?[\s\d\W]*$', lines[0].strip()): lines = lines[1:]
            if len(lines) > 1 and len(lines[-1].strip()) < 25 and re.match(r'^[\s\d\W]*?(\d{1,3})?[\s\d\W]*$', lines[-1].strip()): lines = lines[:-1]
        extracted_text = '\n'.join(lines)
        return extracted_text
    except Exception as e: print(f"Extraction failed: {e}"); traceback.print_exc(); return ""


def preprocess_image_for_math_ocr(image): # Keep previous fixed
    if image.mode != 'L': image = image.convert('L')
    image_array = np.array(image); threshold = np.mean(image_array) * 0.85
    binary_image = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_image)

def detect_math_content(text): # Keep previous fixed
    math_keywords=[r'\b(equation|formula|theorem|lemma|proof|calculus|algebra|derivative|function|integral|vector|matrix|variable|constant|graph|plot|solve|calculate|measure|angle|degree)\b']
    math_symbols=r'[=><≤≥≠\+\-\*\/\^∫∑∏√∞≠≤≥±→∂∇πθλμσωαβγδε%]'
    function_notation=r'\b[a-zA-Z]\s?\([a-zA-Z0-9,\s\+\-\*\/]+\)'
    latex_delimiters=r'\$.*?\$|\\\(.*?\\\)|\\[a-zA-Z]+(\{.*?\})*'
    math_list_items=r'^\s*(\d+\.|\*|\-)\s*[=><≤≥≠\+\-\*\/\^∫∑∏√∞≠≤≥±→∂∇πθλμσωαβγδε\$\\]'
    if re.search('|'.join(math_keywords), text, re.I): return True
    sample=text[:30000];
    for p in [math_symbols,function_notation,latex_delimiters]:
        if re.search(p,sample): return True
    if re.search(math_list_items,sample,re.M): return True;
    return False

def clean_text(text): # Keep previous fixed
    text=re.sub(r'\f',' ',text); text=re.sub(r'\[OCR_IMG.*?\[\/OCR_IMG\]','',text,flags=re.S); text=re.sub(r'\(cid:\d+\)','',text); text=re.sub(r'\s+',' ',text); text=re.sub(r'([.!?])(\w)',r'\1 \2',text);
    lines=text.split('\n'); cleaned=[l for l in lines if len(l.strip())>2 or l.strip() in ['.','!','?']]; text='\n'.join(cleaned);
    text=re.sub(r'\s+([.,;:!?])',r'\1',text); text=re.sub(r'(\w)-\s*\n\s*(\w)',r'\1\2',text); text=re.sub(r'\n',' ',text); text=re.sub(r'\s+',' ',text).strip(); return text

def split_text_into_chunks(text, chunk_size=800, overlap=50): # Keep previous fixed
    try: sentences = nltk.sent_tokenize(text); assert sentences
    except Exception: sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()] or [text]
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        sentence=sentence.strip(); sent_length=len(sentence.split());
        if not sentence: continue
        if current_length>0 and current_length+sent_length>chunk_size:
            chunks.append(' '.join(current_chunk)); overlap_wc,overlap_idx=0,[]
            for i in range(len(current_chunk)-1,-1,-1): overlap_wc+=len(current_chunk[i].split()); overlap_idx.insert(0,i);
            if overlap_wc>=overlap: break
            current_chunk=[current_chunk[i] for i in overlap_idx]+[sentence]; current_length=sum(len(s.split()) for s in current_chunk)
        else: current_chunk.append(sentence); current_length+=sent_length
    if current_chunk: chunks.append(' '.join(current_chunk))
    refined,i,min_w=[],0,max(50,chunk_size*0.2)
    while i<len(chunks):
        c_w=len(chunks[i].split()); n_w=len(chunks[i+1].split()) if i+1<len(chunks) else 0
        if c_w<min_w and i+1<len(chunks) and c_w+n_w<=chunk_size*1.5: refined.append(chunks[i]+" "+chunks[i+1]); i+=2
        else: refined.append(chunks[i]); i+=1
    chunks=refined; MAX_CHUNKS=20
    if len(chunks)>MAX_CHUNKS: print(f"Warn: Reducing chunks {len(chunks)}->{MAX_CHUNKS}"); step=max(1,len(chunks)//MAX_CHUNKS); chunks=[chunks[i] for i in range(0,len(chunks),step)][:MAX_CHUNKS]
    print(f"Split into {len(chunks)} chunks (~{chunk_size}w, ~{overlap}w overlap).")
    return chunks

def determine_reading_level(grade): # Keep previous fixed
    if not isinstance(grade,int) or not(1<=grade<=12): grade=6;
    age=grade+5;
    if 1<=grade<=3: level,desc="lower", f"early Elem ({grade}, ~age {age}-{age+1})"
    elif 4<=grade<=6: level,desc="middle",f"late Elem/Mid ({grade}, ~age {age}-{age+1})"
    elif 7<=grade<=9: level,desc="higher",f"Jr High/early High ({grade}, ~age {age}-{age+1})"
    else: level,desc="higher",f"High School ({grade}, ~age {age}-{age+1})"
    return level, desc

# --- Prompts Dictionary (Keep Revised Structure) ---
prompts = { # Keep full prompts dict from app.py here
    "lower": { "standard": ("..."), "math": ("...") },
    "middle": { "standard": ("..."), "math": ("...") },
    "higher": { "standard": ("..."), "math": ("...") }
    # <<< PASTE THE FULL PROMPTS DICTIONARY HERE FROM YOUR PREVIOUS VERSION >>>
}


# <<< MODIFIED: model_generate uses robust length calculation >>>
def model_generate(prompt_text, max_new_tokens=1024, temperature=0.5): # Default temp lowered
    if not model or not tokenizer: return "Error: LLM not available."
    current_model_device = next(model.parameters()).device

    # --- Get model context limit (handle potential inconsistencies) ---
    model_context_limit = getattr(model.config, 'max_position_embeddings', None)
    if model_context_limit is None: model_context_limit = getattr(tokenizer, 'model_max_length', None)
    if not isinstance(model_context_limit, int) or model_context_limit <= 512: model_context_limit = 8192

    # --- Robust Length Calculation ---
    if max_new_tokens >= model_context_limit: max_new_tokens = model_context_limit // 2
    buffer_tokens = 150
    max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens
    if max_prompt_len <= 0:
        needed = abs(max_prompt_len) + 10; max_new_tokens -= needed
        max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens
        if max_prompt_len <= 0 or max_new_tokens <= 50: return f"Error: Generation request too large for limit ({model_context_limit})."
        print(f"Warn: Reduced max_new_tokens to {max_new_tokens} to fit context.")
    max_prompt_len = min(max_prompt_len, model_context_limit)
    # print(f"DEBUG: Context={model_context_limit}, New={max_new_tokens}, PromptMax={max_prompt_len}") # Reduce noise

    try:
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(current_model_device)
        input_token_count = inputs['input_ids'].shape[1]
        if input_token_count >= max_prompt_len: print(f"Warn: Prompt potentially truncated.")

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                do_sample=True if temperature > 0.01 else False
            )
        gen_time = time.time() - start_time; print(f"...Generation took {gen_time:.2f}s.")

        generated_text = tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True)
        generated_text = re.sub(r'<\|eot_id\|>', '', generated_text).strip()
        return generated_text
    except torch.cuda.OutOfMemoryError as e: print(f"OOM during generation: {e}"); traceback.print_exc(); torch.cuda.empty_cache(); return f"Error: GPU OOM during generation."
    except Exception as e: print(f"Generation Error: {e}"); traceback.print_exc(); return f"Error: Model generation failed - {e}"

# <<< MODIFIED: generate_summary - uses CLI toggles, robust context limit >>>
def generate_summary(text_chunks, grade_level_category, grade_level_desc, duration_minutes, has_math=False,
                     enable_completion_cli=True, enable_deduplication_cli=True): # Accept CLI toggles
    # (Word count logic remains same)
    if duration_minutes == 10: min_words, max_words = 1200, 1600
    elif duration_minutes == 20: min_words, max_words = 2400, 3200
    elif duration_minutes == 30: min_words, max_words = 3600, 4500
    else: min_words, max_words = 1200, 1600
    print(f"Targeting: {grade_level_desc}, {duration_minutes} mins ({min_words}-{max_words} words).")

    # --- Get model context limit (handle potential inconsistencies) ---
    model_context_limit = getattr(model.config, 'max_position_embeddings', None)
    if model_context_limit is None: model_context_limit = getattr(tokenizer, 'model_max_length', None)
    if not isinstance(model_context_limit, int) or model_context_limit <= 512: model_context_limit = 8192

    full_text = ' '.join(text_chunks); full_text_tokens = len(tokenizer.encode(full_text))
    target_max_tokens=int(max_words*1.3)+200; safe_gen_limit=max((model_context_limit//2)-150, 512)
    max_new_tokens=max(min(target_max_tokens, safe_gen_limit), 512)
    print(f"Max new tokens: {max_new_tokens}")

    prompt_buffer=700; req_tokens=full_text_tokens + max_new_tokens + prompt_buffer
    single_pass = (req_tokens < (model_context_limit*0.9) and
                   full_text_tokens < (model_context_limit*0.6) and
                   max_new_tokens <= 2048) # Adjust limit if needed

    initial_summary = ""
    # (Single pass / iterative logic remains the same - uses corrected model_generate)
    if single_pass and text_chunks:
        print("Attempting single pass summary.")
        prompt = prompts[grade_level_category]["math" if has_math else "standard"].format(text=full_text)
        prompt += f"\n\nIMPORTANT: Structure summary ({min_words}-{max_words} words)."
        initial_summary = model_generate(prompt, max_new_tokens=max_new_tokens, temperature=0.55) # Lower temp
    elif text_chunks:
        print(f"Iterative summary ({len(text_chunks)} chunks).")
        chunk_summaries = []; max_tokens_chunk = max(min((model_context_limit//(len(text_chunks)+1))-150, 200), 80) # Smaller target
        print(f"Max new tokens per chunk: {max_tokens_chunk}")
        for i, chunk in enumerate(text_chunks):
            chunk_prompt = f"Key points from chunk {i+1}/{len(text_chunks)}:\n{chunk}\n\nKey Points (CONCISE bullet list):"
            chunk_summary = model_generate(chunk_prompt, max_new_tokens=max_tokens_chunk, temperature=0.2) # Low temp
            if not chunk_summary.startswith("Error:") and len(chunk_summary.split()) >= 3:
                chunk_summary = re.sub(r'^.*?Key Points.*?\n','',chunk_summary,flags=re.I).strip()
                if chunk_summary: chunk_summaries.append(chunk_summary)
        if not chunk_summaries: return "Error: No valid chunk summaries."
        print("Consolidating chunks...")
        base_instr = prompts[grade_level_category]["math" if has_math else "standard"].split("Text to summarize:")[0]
        consol_prompt = f"{base_instr}\nSYNTHESIZE points below into ONE summary for {grade_level_desc}.\nFollow original instructions. Aim for {min_words}-{max_words} words.\n\nChunk Summaries:\n\n"+"\n\n---\n\n".join(chunk_summaries)+"\n\nFinal Consolidated Summary:"
        initial_summary = model_generate(consol_prompt, max_new_tokens=max_new_tokens, temperature=0.6) # Med temp
    else: return "Error: Zero chunks."

    if initial_summary.startswith("Error:"): return initial_summary
    current_summary = initial_summary; current_words = len(current_summary.split())
    print(f"Initial summary: {current_words} words. Checking length.")

    # (Lengthening logic remains same)
    attempts, max_attempts = 0, 2
    while current_words < min_words and attempts < max_attempts:
        print(f"Summary short. Elaborating (Attempt {attempts+1}/{max_attempts})...")
        prompt = f"Elaborate on points...\nCurrent Summary:\n{current_summary}\n\nContinue summary:"
        needed=min_words-current_words; tokens_add=max(min(int(needed*1.5),max_new_tokens//2,700),150)
        new_part = model_generate(prompt, max_new_tokens=tokens_add, temperature=0.65)
        if new_part.startswith("Error:") or len(new_part.split()) < 10: print("Stopping lengthening."); break
        current_summary += "\n\n" + new_part.strip(); current_words = len(current_summary.split()); attempts += 1
    if attempts == max_attempts and current_words < min_words: print(f"Warning: Max lengthening reached ({current_words}/{min_words} words).")

    # (Trimming logic remains same)
    words = current_summary.split()
    if len(words) > max_words:
        print(f"Trimming from {len(words)} to ~{max_words} words.")
        act_match=re.search(r'(##\s+(Activity|Practice|Thinking|Challenge|Try This|Fun Activity))',current_summary, re.I|re.M)
        if act_match:
            idx=act_match.start(); main_cont,act_cont=current_summary[:idx],current_summary[idx:]
            main_w=main_cont.split();
            if len(main_w) > max_words:
                limit_idx=len(' '.join(main_w[:max_words])); end_idx=main_cont.rfind('.',0,limit_idx)
                main_cont = main_cont[:end_idx+1] if end_idx != -1 else ' '.join(main_w[:max_words])+"..."
            current_summary = main_cont.strip() + "\n\n" + act_cont.strip()
        else: current_summary = ' '.join(words[:max_words]); current_summary += "..." if current_summary[-1].isalnum() else ""

    print("Post-processing summary...")
    processed_summary = enhanced_post_process(current_summary, grade_level_category,
                                              enable_completion=enable_completion_cli,
                                              enable_deduplication=enable_deduplication_cli)

    # (Fallback activity logic remains same)
    activity_pattern = r'^##\s+(Fun Activity|Practice.*|Try This|Further Thinking|Challenge|Activity)\s*$'
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
    # (Keep the robust version from modified app.py)
    if summary.startswith("Error:"): return summary
    print(f"Running Markdown post-processing (Comp:{enable_completion}, Dedup:{enable_deduplication})...")
    comp_calls = 0
    try: head_line=next((l for l in prompts[grade_level_category]["standard"].splitlines() if l.strip().startswith('#')),None); exp_head=head_line.strip().lstrip('# ').strip() if head_line else "Summary"
    except: exp_head="Summary"
    summary=re.sub(r'^\s*#+.*?(\n|$)','',summary.strip()); summary=f'# {exp_head}\n\n'+summary # Ensure heading

    lines=summary.split('\n'); p_data,seen_frags=[],set()
    for line in lines:
        s_line=line.strip();
        if not s_line:
            if p_data and p_data[-1]["type"]!="blank": p_data.append({"text":"","type":"blank"}); continue
        l_type,cont,is_head,is_bullet="paragraph",s_line,False,False
        if s_line.startswith('## '): l_type,cont,is_head="subheading",s_line[3:].strip(),True
        elif s_line.startswith('# '): l_type,cont,is_head="heading",s_line[2:].strip(),True
        elif s_line.startswith('- '): l_type,cont,is_bullet="bullet",s_line[2:].strip(),True
        elif re.match(r'^\d+\.\s+',s_line): l_type,cont,is_bullet="numbered",re.sub(r'^\d+\.\s+','',s_line),True
        if not cont: continue
        key=' '.join(cont.lower().split()[:10])
        if not is_head and key in seen_frags and len(cont.split())<15: continue
        if not is_head: seen_frags.add(key)
        # Sentence Completion
        if enable_completion and l_type in ["paragraph","bullet","numbered"] and len(cont.split())>4:
            if not re.search(r'[.!?:]$',cont) and cont[0].isupper() and comp_calls<MAX_COMPLETION_CALLS:
                orig=cont; cont=complete_sentence(cont,enable_global_toggle=enable_completion);
                if cont!=orig and not cont.endswith(orig+"."): comp_calls+=1
        if l_type in ["paragraph","bullet","numbered"]:
            if cont and cont[0].islower() and not re.match(r'^[a-z]\s*\(',cont): cont=cont[0].upper()+cont[1:]
            if cont and cont[-1].isalnum(): cont+='.'
        if l_type=="blank" and p_data and p_data[-1]["type"]=="blank": continue
        p_data.append({"text":cont,"type":l_type})

    # Semantic Deduplication
    dedup_pts,idx_map=[],{}
    if enable_deduplication and embedder:
        for i,d in enumerate(p_data):
            if d["type"] in ["paragraph","bullet","numbered"] and len(d["text"].split())>6:
                c=d["text"]; dedup_pts.append(c);
                if c not in idx_map: idx_map[c]=[]
                idx_map[c].append(i)
    elif not enable_deduplication: print("Skip dedup (disabled).")
    elif not embedder: print("Skip dedup (no embedder).")

    kept_idx=set(range(len(p_data)))
    if dedup_pts and enable_deduplication and embedder:
        print(f"Running semantic dedup on {len(dedup_pts)} points...")
        try:
            unique=remove_duplicates_semantic(dedup_pts,batch_size=128); unique_set=set(unique)
            print(f"Reduced to {len(unique)} points.")
            to_remove,processed=set(),set()
            for c, orig_idx in idx_map.items():
                if c in processed: continue
                if c not in unique_set:
                    for idx in orig_idx:
                        is_only=(idx>0 and p_data[idx-1]["type"] in ["h","subh"] and (idx==len(p_data)-1 or p_data[idx+1]["type"] in ["h","subh","blank"]))
                        if not is_only: to_remove.add(idx)
                processed.add(c)
            kept_idx-=to_remove; print(f"Marked {len(to_remove)} lines for removal.")
        except Exception as e: print(f"Warn: Dedup failed: {e}")

    # Final Assembly
    final_text,last_type="",None
    kept_data=[p_data[i] for i in sorted(list(kept_idx))]
    for i,d in enumerate(kept_data):
        curr_t,cont=d["type"],d["text"]
        if i>0: # Spacing
            if curr_t in ["heading","subheading"]: final_text+="\n\n"
            elif curr_t=="paragraph" and last_type not in ["heading","subheading","blank"]: final_text+="\n\n"
            elif curr_t!="blank" and last_type!="blank": final_text+="\n"
            elif curr_t=="blank" and last_type=="blank": continue
        # Content
        if curr_t=="heading": final_text+=f"# {cont}"
        elif curr_t=="subheading": final_text+=f"## {cont}"
        elif curr_t=="bullet": final_text+=f"- {cont}"
        elif curr_t=="numbered": final_text+=f"1. {cont}" # Basic numbering
        elif curr_t=="paragraph": final_text+=cont
        last_type=curr_t
    print("Post-processing finished.")
    return final_text.strip()


def remove_duplicates_semantic(points, similarity_threshold=0.90, batch_size=128):
    # (Keep the robust version from modified app.py)
    if not points or not embedder or len(points)<2: return points
    start_dedup = time.time()
    try:
        valid_pts = [p for p in points if len(p.split()) > 4]
        if not valid_pts: return points
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

def generate_activity(summary_text, grade_level_category, grade_level_desc):
    # (Keep the robust version from modified app.py)
    if not model or not tokenizer: return "- Review key points."
    print("Generating fallback activity...")
    act_type = {"lower":"fun activity/question","middle":"practical activity/thought question","higher":"provoking question/research idea/analysis task"}.get(grade_level_category,"activity")
    prompt = f"Suggest ONE simple, engaging {act_type} for {grade_level_desc} based on this summary context:\n...{' '.join(re.sub(r'^#.*?\n','',summary_text).strip().split()[-200:])}\n\nActivity Suggestion:"
    activity = model_generate(prompt, max_new_tokens=80, temperature=0.7)
    if activity.startswith("Error:"): print(f"Fallback activity failed: {activity}"); activity = ""
    else: activity = re.sub(r'^[\-\*\s]+','',activity.strip().replace("Activity Suggestion:","").strip()).strip(); activity = re.sub(r'\.$','',activity).strip()
    if activity:
        activity = f"- {activity[0].upper() + activity[1:]}"
        if activity[-1].isalnum(): activity += '.'
        return activity
    else: print("Warn: Failed fallback activity."); fallbacks={"lower":"- Draw!", "middle":"- Explain!", "higher":"- Find example."}; return fallbacks.get(grade_level_category, "- Review.")

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