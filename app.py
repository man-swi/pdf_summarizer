# app.py (Modified for Unsloth Llama-4-Scout-17B 4-bit)

import os
import re
import pytesseract
from PIL import Image
import numpy as np
import torch
from nltk.stem import PorterStemmer
import nltk
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer, util # Keep for embedder

# --- MODIFIED: Import Unsloth ---
from unsloth import FastLanguageModel
# Keep AutoTokenizer for the non-Unsloth embedder model if needed, or for general use
# Keep AutoProcessor etc from transformers if the chosen Unsloth model specifically requires it (less likely for basic loading)
from transformers import AutoTokenizer # Still needed for embedder or if Unsloth tokenizer is compatible

import traceback
import tempfile
import io
from flask import Flask, request, render_template, jsonify, send_from_directory
import time

# --- NLTK Download (Keep as before) ---
try:
    print("Checking/downloading NLTK punkt...")
    nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
    if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
        os.makedirs(nltk_data_path, exist_ok=True)
        nltk.download('punkt', quiet=True, download_dir=nltk_data_path)
        if os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
            nltk.data.path.append(nltk_data_path)
            print(f"NLTK punkt downloaded to {nltk_data_path}")
        else: nltk.download('punkt', quiet=True)
    else: nltk.data.path.append(nltk_data_path)
    print("NLTK punkt check complete.")
except Exception as e: print(f"Warning: Could not download NLTK punkt: {e}")

# --- Configuration & Model Loading ---
print("Starting Flask App Setup...")

# Tesseract Path (Keep as before)
tesseract_cmd_path = None
tesseract_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract', 'tesseract']
for path in tesseract_paths:
    if os.path.exists(path) and os.access(path, os.X_OK):
        pytesseract.pytesseract.tesseract_cmd = path; tesseract_cmd_path = path
        print(f"Using Tesseract at: {path}"); break
if not tesseract_cmd_path: print("Warning: Tesseract not found. OCR unavailable.")

# Device (Unsloth generally handles device placement, but keep for embedder)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} (Note: Unsloth manages LLM placement)")

OCR_RESOLUTION = 300
MAX_SEQ_LENGTH = 8192 # Define max sequence length for Unsloth loading

# --- MODIFIED: Model Names & Unsloth Setup ---
LLM_MODEL_NAME = "unsloth/Llama-4-Scout-17B-16E-unsloth-dynamic-bnb-4bit" # <<< CHANGED Model Name
EMBEDDER_MODEL_NAME = 'all-MiniLM-L6-v2'

# Quantization is handled by Unsloth's FastLanguageModel when load_in_4bit=True
print(f"Will load {LLM_MODEL_NAME} with Unsloth (4-bit implied by name/load_in_4bit)")
# --- END MODIFIED ---

tokenizer = None # Will be loaded by FastLanguageModel
model = None     # Will be loaded by FastLanguageModel
embedder = None
stemmer = PorterStemmer()
MAX_COMPLETION_CALLS = 10

try:
    print(f"Loading LLM model & tokenizer: {LLM_MODEL_NAME} using Unsloth...")
    # --- MODIFIED: Use FastLanguageModel.from_pretrained ---
    # This loads the model and tokenizer together, handling quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LLM_MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,              # Unsloth handles dtype optimization with 4bit
        load_in_4bit = True,       # Explicitly load in 4bit
        device_map = "auto",       # Let Unsloth / Accelerate handle device placement
        # token = "hf_...", # Add if this specific Unsloth model requires auth (unlikely for community models)
    )
    # --- END MODIFIED ---

    print("Unsloth LLM Model and Tokenizer loaded successfully!")
    # <<< Get context limit (might be less reliable with Unsloth wrapper) >>>
    # Try model.config first, then tokenizer if needed
    actual_context_limit = getattr(model.config, 'max_position_embeddings', None)
    if actual_context_limit is None:
         actual_context_limit = getattr(tokenizer, 'model_max_length', None)
    # If still None, use the max_seq_length we passed
    if actual_context_limit is None:
         actual_context_limit = MAX_SEQ_LENGTH
    print(f"DEBUG: Using context limit: {actual_context_limit}")


except Exception as e:
    print(f"FATAL ERROR: Could not load Unsloth model {LLM_MODEL_NAME}: {e}")
    traceback.print_exc()
    if "out of memory" in str(e).lower(): print("Attempting to clear CUDA cache..."); torch.cuda.empty_cache()
    exit(1)

try:
    # Embedder loading remains the same (uses standard SentenceTransformer)
    print(f"Loading Sentence Embedder: {EMBEDDER_MODEL_NAME}...")
    # Use standard AutoTokenizer for the embedder model if needed,
    # but SentenceTransformer usually handles its own tokenizer.
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device=device)
    print("Embedder loaded successfully!")
except Exception as e:
    print(f"ERROR: Could not load Sentence Transformer {EMBEDDER_MODEL_NAME}: {e}")
    embedder = None
    print("Warning: Embedder failed to load. Semantic deduplication will be disabled.")

# Check if essential models loaded
if not tokenizer or not model:
     print("Essential models (tokenizer/LLM) failed to load. Exiting.")
     exit(1)

print("Model loading complete.")

# Math Symbols (Keep as is)
MATH_SYMBOLS = { # ... keep dictionary ...
    '∫': '\\int', '∑': '\\sum', '∏': '\\prod', '√': '\\sqrt', '∞': '\\infty',
    '≠': '\\neq', '≤': '\\leq', '≥': '\\geq', '±': '\\pm', '→': '\\to',
    '∂': '\\partial', '∇': '\\nabla', 'π': '\\pi', 'θ': '\\theta',
    'λ': '\\lambda', 'μ': '\\mu', 'σ': '\\sigma', 'ω': '\\omega',
    'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta', 'ε': '\\epsilon'
}


# --- Utility Functions ---
# NOTE: Most utility functions remain unchanged, but model_generate and
# generate_summary need to use the potentially updated context limit logic.

def get_stemmed_key(sentence, num_words=5): # Keep as is
    words = re.findall(r'\w+', sentence.lower())[:num_words]
    return ' '.join([stemmer.stem(word) for word in words])

def complete_sentence(fragment, enable_global_toggle=True): # Keep as is
    # ... (same logic as previous corrected version) ...
    if not enable_global_toggle: return fragment + "."
    if not model or not tokenizer: return fragment + "."
    if re.search(r'[.!?]$', fragment.strip()): return fragment
    if len(fragment.split()) < 3 or len(fragment) < 15: return fragment + "."
    prompt = f"Complete this sentence fragment naturally and concisely:\nFragment: '{fragment}'\nCompleted sentence:"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, temperature=0.2, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True)
        completed_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        match = re.search(r"Completed sentence:\s*(.*)", completed_full, re.I | re.S)
        if match:
            completed = match.group(1).strip()
            final = completed if completed.lower().startswith(fragment.lower()) and len(completed)>len(fragment)+3 else (fragment+"." if completed.lower().startswith(fragment.lower()) else completed)
            final = re.sub(r'<\|eot_id\|>','',final).strip();
            if not final: return fragment+"."
            if final[-1].isalnum(): final+='.'
            return final
        else: return fragment + "."
    except Exception as e: print(f"Completion Error: {e}"); return fragment + "."

def preprocess_image_for_math_ocr(image): # Keep as is
    if image.mode != 'L': image = image.convert('L')
    image_array = np.array(image); threshold = np.mean(image_array) * 0.85
    binary_image = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_image)

def extract_text_from_pdf(pdf_path, detect_math=True, ocr_enabled=False): # Keep as is
    # ... (same logic as previous corrected version) ...
    if not os.path.exists(pdf_path): raise FileNotFoundError(f"PDF not found: {pdf_path}")
    print(f"Extracting text from {pdf_path} (OCR: {ocr_enabled})...")
    extracted_text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            text = page.get_text("text", sort=True);
            if text: extracted_text += text + "\n"
            if ocr_enabled and tesseract_cmd_path and page.get_images(full=True):
                import io
                for img_index, img in enumerate(page.get_images(full=True)):
                    try:
                        xref=img[0]; base_image=doc.extract_image(xref); image_bytes=base_image["image"]
                        fmt=base_image.get("ext","png").lower();
                        if fmt not in ["png","jpeg","jpg","bmp","gif","tiff"]: continue
                        pil_image=Image.open(io.BytesIO(image_bytes))
                        processed_img = preprocess_image_for_math_ocr(pil_image) if detect_math else pil_image
                        ocr_text=pytesseract.image_to_string(processed_img,config='--psm 6 --oem 3')
                        if ocr_text.strip():
                            if detect_math:
                                for s,l in MATH_SYMBOLS.items(): ocr_text = ocr_text.replace(s, f" {l} ")
                            extracted_text += f"\n[OCR_IMG {img_index+1}] {ocr_text.strip()} [/OCR_IMG]\n"
                    except pytesseract.TesseractNotFoundError: print("Tesseract not found"); ocr_enabled=False; break
                    except Exception as e: print(f"OCR Img Err {img_index} pg {page_num+1}: {e}")
        doc.close()
        lines=extracted_text.split('\n'); # Basic header/footer removal
        if len(lines)>2:
            if len(lines[0].strip())<25 and re.match(r'^[\s\d\W]*?(\d{1,3})?[\s\d\W]*$',lines[0].strip()): lines=lines[1:]
            if len(lines)>1 and len(lines[-1].strip())<25 and re.match(r'^[\s\d\W]*?(\d{1,3})?[\s\d\W]*$',lines[-1].strip()): lines=lines[:-1]
        extracted_text = '\n'.join(lines)
        return extracted_text
    except Exception as e: print(f"Extraction failed: {e}"); traceback.print_exc(); raise

def detect_math_content(text): # Keep as is
    # ... (same logic as previous corrected version) ...
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

def clean_text(text): # Keep as is
    # ... (same logic as previous corrected version) ...
    text=re.sub(r'\f',' ',text); text=re.sub(r'\[OCR_IMG.*?\[\/OCR_IMG\]','',text,flags=re.S); text=re.sub(r'\(cid:\d+\)','',text); text=re.sub(r'\s+',' ',text); text=re.sub(r'([.!?])(\w)',r'\1 \2',text);
    lines=text.split('\n'); cleaned=[l for l in lines if len(l.strip())>2 or l.strip() in ['.','!','?']]; text='\n'.join(cleaned);
    text=re.sub(r'\s+([.,;:!?])',r'\1',text); text=re.sub(r'(\w)-\s*\n\s*(\w)',r'\1\2',text); text=re.sub(r'\n',' ',text); text=re.sub(r'\s+',' ',text).strip(); return text


def split_text_into_chunks(text, chunk_size=800, overlap=50): # Keep as is
     # ... (same logic as previous corrected version) ...
    try:
        if not any('tokenizers/punkt' in p for p in nltk.data.path): print("Warning: NLTK punkt path potentially not configured.")
        sentences = nltk.sent_tokenize(text)
        if not sentences: raise ValueError("NLTK tokenization resulted in empty list")
    except Exception as e: print(f"Warning: NLTK splitting failed ({e}), using regex fallback."); sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()] or [text]
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

def determine_reading_level(grade): # Keep as is
    # ... (same logic as previous corrected version) ...
    if not isinstance(grade,int) or not(1<=grade<=12): grade=6;
    age=grade+5;
    if 1<=grade<=3: level,desc="lower",f"early Elem ({grade}, ~age {age}-{age+1})"
    elif 4<=grade<=6: level,desc="middle",f"late Elem/Mid ({grade}, ~age {age}-{age+1})"
    elif 7<=grade<=9: level,desc="higher",f"Jr High/early High ({grade}, ~age {age}-{age+1})"
    else: level,desc="higher",f"High School ({grade}, ~age {age}-{age+1})"
    return level, desc

# --- Prompts Dictionary (Use the refined version for Scout) ---
prompts = {
    # <<< PASTE THE FULL REFINED PROMPTS DICTIONARY HERE >>>
    "lower": { "standard": ("..."), "math": ("...") },
    "middle": { "standard": ("..."), "math": ("...") },
    "higher": { "standard": ("..."), "math": ("...") }
}

# <<< MODIFIED: model_generate uses context limit from loaded model/tokenizer >>>
def model_generate(prompt_text, max_new_tokens=1024, temperature=0.6):
    if not model or not tokenizer: return "Error: LLM not available."
    current_model_device = model.device # Get device directly from Unsloth model

    # --- Get model context limit (handle potential inconsistencies) ---
    model_context_limit = getattr(model.config, 'max_position_embeddings', None)
    if model_context_limit is None: model_context_limit = getattr(tokenizer, 'model_max_length', None)
    if model_context_limit is None: model_context_limit = MAX_SEQ_LENGTH # Fallback to loading param
    if not isinstance(model_context_limit, int) or model_context_limit <= 512:
        print(f"Warning: Could not reliably determine context limit, using {MAX_SEQ_LENGTH}.")
        model_context_limit = MAX_SEQ_LENGTH

    # --- Robust Length Calculation ---
    if max_new_tokens >= model_context_limit:
         print(f"Warning: max_new_tokens ({max_new_tokens}) >= context limit ({model_context_limit}). Reducing.")
         max_new_tokens = model_context_limit // 2
    print(f"DEBUG: Using model_context_limit = {model_context_limit}, requested max_new_tokens = {max_new_tokens}")
    buffer_tokens = 150
    max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens
    if max_prompt_len <= 0:
        print(f"ERROR: Calculated max_prompt_len ({max_prompt_len}) is non-positive.")
        needed = abs(max_prompt_len) + 10; max_new_tokens -= needed
        max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens
        if max_prompt_len <= 0 or max_new_tokens <= 50: return f"Error: Generation request too large for limit ({model_context_limit})."
        print(f"Warning: Reduced max_new_tokens to {max_new_tokens} to fit context.")
    max_prompt_len = min(max_prompt_len, model_context_limit)
    print(f"DEBUG: Calculated max_prompt_len = {max_prompt_len} for tokenizer.")

    try:
        # Unsloth tokenizer usage is typically identical to transformers
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(current_model_device)
        input_token_count = inputs['input_ids'].shape[1]
        if input_token_count >= max_prompt_len: print(f"Warning: Prompt potentially truncated.")

        start_time = time.time()
        with torch.no_grad():
             # Unsloth model usage is typically identical to transformers
             outputs = model.generate(
                 **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                 pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                 do_sample=True if temperature > 0.01 else False
             )
        end_time = time.time(); print(f"...Generation took {end_time - start_time:.2f} seconds.")

        # Unsloth tokenizer usage is typically identical to transformers
        generated_text = tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True)
        generated_text = re.sub(r'<\|eot_id\|>', '', generated_text).strip() # Llama 3 specific token might not be relevant

        # (Keep checks for structure and length)
        if not re.search(r'(^#|^- )', generated_text, re.MULTILINE): print("Warning: Generated text seems to lack structure.")
        if len(generated_text) < 20: print("Warning: Generation resulted in very short text.")
        return generated_text
    except torch.cuda.OutOfMemoryError as e: print(f"OOM during generation: {e}"); traceback.print_exc(); torch.cuda.empty_cache(); return f"Error: GPU OOM during generation."
    except Exception as e: print(f"Generation Error: {e}"); traceback.print_exc(); return f"Error: Model generation failed - {str(e)}"
# --- End model_generate ---


# <<< MODIFIED: generate_summary uses context limit from loaded model/tokenizer >>>
def generate_summary(text_chunks, grade_level_category, grade_level_desc, duration_minutes, has_math=False,
                     enable_completion_web=True, enable_deduplication_web=True):
    # (Word count logic remains the same)
    if duration_minutes == 10: min_words, max_words = 1200, 1600
    elif duration_minutes == 20: min_words, max_words = 2400, 3200
    elif duration_minutes == 30: min_words, max_words = 3600, 4500
    else: min_words, max_words = 1200, 1600
    print(f"Targeting summary: {grade_level_desc}, {duration_minutes} mins ({min_words}-{max_words} words approx).")
    print(f"Refinement Options - Completion: {enable_completion_web}, Deduplication: {enable_deduplication_web}")

    # --- Get model context limit (handle potential inconsistencies) ---
    model_context_limit = getattr(model.config, 'max_position_embeddings', None)
    if model_context_limit is None: model_context_limit = getattr(tokenizer, 'model_max_length', None)
    if model_context_limit is None: model_context_limit = MAX_SEQ_LENGTH # Fallback
    if not isinstance(model_context_limit, int) or model_context_limit <= 512: model_context_limit = MAX_SEQ_LENGTH
    print(f"DEBUG (generate_summary): Using model_context_limit = {model_context_limit}")

    # (Rest of the logic uses the corrected context limit and corrected model_generate)
    full_text = ' '.join(text_chunks)
    try: full_text_tokens_estimate = len(tokenizer.encode(full_text))
    except: full_text_tokens_estimate = len(full_text) // 3

    estimated_target_max_tokens = int(max_words * 1.3) + 200
    safe_generation_limit = max((model_context_limit // 2) - 150, 512)
    max_new_tokens_summary = max(min(estimated_target_max_tokens, safe_generation_limit), 512)
    print(f"Calculated max_new_tokens for summary: {max_new_tokens_summary}")

    prompt_instruction_buffer = 700
    required_tokens_for_single_pass = full_text_tokens_estimate + max_new_tokens_summary + prompt_instruction_buffer
    can_summarize_all_at_once = (required_tokens_for_single_pass < (model_context_limit*0.9) and full_text_tokens_estimate < (model_context_limit*0.6) and max_new_tokens_summary <= 4096) # Increased allowance slightly

    initial_summary = ""
    # (Single pass / iterative logic remains the same)
    if can_summarize_all_at_once and text_chunks:
        print("Attempting summary generation in a single pass.")
        prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
        try: prompt = prompt_template.format(text=full_text) + f"\n\nIMPORTANT: Structure summary ({min_words}-{max_words} words)."
        except KeyError: prompt = f"Create summary for {grade_level_desc} ({min_words}-{max_words} words):\n{full_text}"
        initial_summary = model_generate(prompt, max_new_tokens=max_new_tokens_summary, temperature=0.6)
    elif text_chunks:
        print(f"Iterative summary ({len(text_chunks)} chunks).")
        chunk_summaries = []; max_new_tokens_chunk = max(min((model_context_limit//(len(text_chunks)+1))-150, 300), 100)
        print(f"Max new tokens per chunk: {max_new_tokens_chunk}")
        for i, chunk in enumerate(text_chunks):
            chunk_prompt = f"Key points from chunk {i+1}/{len(text_chunks)}:\n{chunk}\n\nKey Points (bullet list):"
            chunk_summary = model_generate(chunk_prompt, max_new_tokens=max_new_tokens_chunk, temperature=0.4)
            if chunk_summary.startswith("Error:") or len(chunk_summary.split())<5: continue
            chunk_summary = re.sub(r'^.*?Key Points.*?\n','',chunk_summary, flags=re.I).strip()
            if chunk_summary: chunk_summaries.append(chunk_summary)
        if not chunk_summaries: return "Error: No valid chunk summaries."
        print("Consolidating chunk summaries...")
        template = prompts[grade_level_category]["math" if has_math else "standard"]
        base_instr = template.split("Text to summarize:")[0]
        consol_prompt = f"{base_instr}\nSYNTHESIZE points below into ONE summary for {grade_level_desc}.\nFollow instructions. Aim for {min_words}-{max_words} words.\n\nChunk Summaries:\n\n"+"\n\n---\n\n".join(chunk_summaries)+"\n\nFinal Consolidated Summary:"
        initial_summary = model_generate(consol_prompt, max_new_tokens=max_new_tokens_summary, temperature=0.65)
    else: return "Error: Zero chunks."

    if initial_summary.startswith("Error:"): return initial_summary
    current_summary = initial_summary; current_word_count = len(current_summary.split())
    print(f"Initial summary: {current_word_count} words.")

    # (Lengthening / Trimming / Post-processing / Fallback Activity logic remains unchanged)
    # ... (Keep the rest of the generate_summary function as it was in the previous corrected version) ...
    # ... it will use the updated model_generate and enhanced_post_process ...
    attempts, max_attempts = 0, 2
    while current_word_count < min_words and attempts < max_attempts:
        print(f"Summary short. Elaborating (Attempt {attempts + 1}/{max_attempts})...")
        prompt = f"Elaborate on points...Current Summary:\n{current_summary}\n\nContinue summary:"
        needed = min_words - current_word_count
        tokens_add = max(min(int(needed * 1.5), max_new_tokens_summary // 2, 700), 150)
        new_part = model_generate(prompt, max_new_tokens=tokens_add, temperature=0.7)
        if new_part.startswith("Error:") or len(new_part.split()) < 10: print("Stopping lengthening."); break
        current_summary += "\n\n" + new_part.strip(); current_word_count = len(current_summary.split()); attempts += 1
    if attempts == max_attempts and current_word_count < min_words: print(f"Warning: Max lengthening attempts.")

    words = current_summary.split()
    if len(words) > max_words:
        print(f"Trimming summary from {len(words)} to ~{max_words} words.")
        activity_match = re.search(r'(##\s+(Activity|Practice|Thinking|Challenge|Try This|Fun Activity))', current_summary, re.I | re.M)
        if activity_match:
            idx = activity_match.start(); main_cont, act_cont = current_summary[:idx], current_summary[idx:]
            main_words = main_cont.split()
            if len(main_words) > max_words:
                 limit_idx = len(' '.join(main_words[:max_words])); end_idx = main_cont.rfind('.', 0, limit_idx)
                 main_cont = main_cont[:end_idx + 1] if end_idx != -1 else ' '.join(main_words[:max_words]) + "..."
            current_summary = main_cont.strip() + "\n\n" + act_cont.strip()
        else:
            current_summary = ' '.join(words[:max_words]); current_summary += "..." if current_summary[-1].isalnum() else ""
    summary = current_summary

    print("Post-processing summary...")
    processed_summary = enhanced_post_process(summary, grade_level_category,
                                              enable_completion=enable_completion_web,
                                              enable_deduplication=enable_deduplication_web)

    activity_pattern = r'^##\s+(Fun Activity|Practice.*|Try This|Further Thinking|Challenge|Activity)\s*$'
    if not re.search(activity_pattern, processed_summary, re.I | re.M):
        print("Warning: Activity section missing. Generating fallback...")
        activity = generate_activity(processed_summary, grade_level_category, grade_level_desc)
        h_map={"lower":"## Fun Activity","middle":"## Try This","higher":"## Further Thinking"}; def_h="## Activity"
        if has_math: head = {"lower":"## Practice Time","middle":"## Practice Problem","higher":"## Challenge"}.get(grade_level_category, def_h)
        else: head = h_map.get(grade_level_category, def_h)
        processed_summary += f"\n\n{head}\n{activity}"
    else: print("Activity section found in generated summary.")

    final_word_count = len(processed_summary.split())
    print(f"Final summary generated ({final_word_count} words).")
    return processed_summary
# --- End generate_summary ---


# --- Other Utility Functions (Keep as corrected before) ---
def enhanced_post_process(summary, grade_level_category, enable_completion=True, enable_deduplication=True):
    # ... (Keep the full corrected version from previous steps) ...
    if summary.startswith("Error:"): return summary
    print(f"Running enhanced post-processing (Comp:{enable_completion}, Dedup:{enable_deduplication})...")
    comp_calls = 0; # ... rest of function
    try: head_line=next((l for l in prompts[grade_level_category]["standard"].splitlines() if l.strip().startswith('#')),None); exp_head=head_line.strip().lstrip('# ').strip() if head_line else "Summary"
    except: exp_head="Summary"
    summary=re.sub(r'^\s*#+.*?(\n|$)','',summary.strip()); summary=f'# {exp_head}\n\n'+summary
    lines=summary.split('\n'); p_data,seen_frags=[],set()
    # ... (Keep loop processing lines, completion, casing logic) ...
    for line in lines: # Simplified loop representation
        s_line=line.strip(); #... type detection ... content ...
        # ... basic dup check ...
        # ... conditional completion call ...
        # ... casing/punctuation ...
        p_data.append({"text":cont,"type":l_type})
    # ... (Keep deduplication logic) ...
    # ... (Keep final assembly logic) ...
    final_text,last_type="",None #... loop kept_data ... add spacing ... add content
    return final_text.strip()


def remove_duplicates_semantic(points, similarity_threshold=0.90, batch_size=64):
    # ... (Keep the full corrected version from previous steps) ...
    if not points or not embedder or len(points)<2: return points # ... rest of function
    start_time=time.time(); # ... try block ... encode ... cos_sim ... remove indices ...
    print(f"Semantic deduplication took {time.time() - start_time:.2f}s.")
    return final_unique_points


def generate_activity(summary_text, grade_level_category, grade_level_desc):
     # ... (Keep the full corrected version from previous steps) ...
    if not model or not tokenizer: return "- Review points." # ... rest of function
    act_type = {...}.get(grade_level_category,"activity") # ... prompt ... model_generate ... clean ... format
    return activity_or_fallback


# --- Flask App Initialization & Routes (Keep as corrected before) ---
app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename): return send_from_directory(app.static_folder, filename)

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    # ... (Keep the full corrected version from previous steps) ...
    # ... get form data ... save temp file ... call extract/clean/chunk ...
    # ... call generate_summary (passes flags) ... handle errors ... return jsonify ...
    # ... finally block for cleanup ...
    start_time=time.time(); # ... check file ... get form data ... save temp ...
    try: # ... process ...
        # ... grade_cat, grade_desc = ...
        # ... raw_text = extract_text_from_pdf(...) ... check raw_text ...
        # ... cleaned = clean_text(...) ... chunks = split_text_into_chunks(...) ...
        # ... summary = generate_summary(...) ... check summary error ...
        # ... return jsonify(...) ...
        pass # Replace with actual logic block
    except Exception as e: # ... error handling ...
        pass
    finally: # ... cleanup ...
        pass
    return jsonify({"error": "Processing failed"}), 500 # Fallback


# --- Run Flask App ---
if __name__ == '__main__':
    print("Starting Flask development server (for testing only)...")
    # Port 8501 matches Dockerfile EXPOSE and Nginx proxy_pass
    app.run(host='0.0.0.0', port=8501, debug=False, threaded=True)