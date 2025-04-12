# app.py (Modified for Llama-4-Scout-17B and 4-bit Quantization)

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
# <<< Import quantization config >>>
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import traceback
import tempfile
import io
from flask import Flask, request, render_template, jsonify, send_from_directory
import time

# --- NLTK Download ---
try:
    print("Checking/downloading NLTK punkt...")
    nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
    if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
        os.makedirs(nltk_data_path, exist_ok=True)
        nltk.download('punkt', quiet=True, download_dir=nltk_data_path)
        if os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
            nltk.data.path.append(nltk_data_path)
            print(f"NLTK punkt downloaded to {nltk_data_path}")
        else:
            nltk.download('punkt', quiet=True)
    else:
        nltk.data.path.append(nltk_data_path)
    print("NLTK punkt check complete.")
except Exception as e:
    print(f"Warning: Could not download NLTK punkt resource automatically. Error: {e}")

# --- Configuration & Model Loading ---
print("Starting Flask App Setup...")

# Tesseract Path
tesseract_cmd_path = None
tesseract_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract', 'tesseract']
for path in tesseract_paths:
    if os.path.exists(path):
        try:
             if os.access(path, os.X_OK):
                 pytesseract.pytesseract.tesseract_cmd = path
                 tesseract_cmd_path = path
                 print(f"Using Tesseract at: {path}")
                 break
             else:
                 print(f"Found Tesseract at {path}, but it's not executable.")
        except Exception as e:
            print(f"Error checking Tesseract at {path}: {e}")

if not tesseract_cmd_path:
    print("Warning: Tesseract executable not found or not configured. OCR will be unavailable.")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

OCR_RESOLUTION = 300

# --- MODIFIED: Model Name and Quantization Setup ---
LLM_MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct" # <<< CHANGED Model Name
EMBEDDER_MODEL_NAME = 'all-MiniLM-L6-v2' # Keep the same embedder

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                # Use NF4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16,    # Use bfloat16 for compute speed
    bnb_4bit_use_double_quant=False,          # Optional: can save a bit more memory
)
print(f"Using 4-bit quantization config: {quantization_config}")
# --- END MODIFIED ---

tokenizer = None
model = None
embedder = None
stemmer = PorterStemmer()
MAX_COMPLETION_CALLS = 10

try:
    print(f"Loading LLM model: {LLM_MODEL_NAME} (with 4-bit quantization)...")
    # --- MODIFIED: Load Tokenizer (no quantization needed here) ---
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    print("Tokenizer loaded.")

    # --- MODIFIED: Load Model with Quantization ---
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=quantization_config, # <<< ADDED Quantization Config
        device_map="auto",                  # Accelerate handles device placement
        # torch_dtype should be implicitly handled by compute_dtype in BitsAndBytesConfig
        # torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, # Removed/Commented out
    )
    # --- END MODIFIED ---

    print("LLM Model loaded successfully!")
    actual_context_limit = getattr(model.config, 'max_position_embeddings', None) # <<< Get limit from model config
    if actual_context_limit is None:
        # Fallback to tokenizer if needed, though model config is usually better
        actual_context_limit = getattr(tokenizer, 'model_max_length', None)
    print(f"DEBUG: Model Config/Tokenizer reported context limit: {actual_context_limit}")


except Exception as e:
    print(f"FATAL ERROR: Could not load LLM model {LLM_MODEL_NAME}: {e}")
    traceback.print_exc()
    # Attempt to clear cache if OOM during loading
    if "out of memory" in str(e).lower():
        print("Attempting to clear CUDA cache...")
        torch.cuda.empty_cache()
    exit(1)

try:
    print(f"Loading Sentence Embedder: {EMBEDDER_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device=device) # Keep embedder on default device
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
MATH_SYMBOLS = {
    '∫': '\\int', '∑': '\\sum', '∏': '\\prod', '√': '\\sqrt', '∞': '\\infty',
    '≠': '\\neq', '≤': '\\leq', '≥': '\\geq', '±': '\\pm', '→': '\\to',
    '∂': '\\partial', '∇': '\\nabla', 'π': '\\pi', 'θ': '\\theta',
    'λ': '\\lambda', 'μ': '\\mu', 'σ': '\\sigma', 'ω': '\\omega',
    'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta', 'ε': '\\epsilon'
}

# --- Utility Functions ---

def get_stemmed_key(sentence, num_words=5):
    words = re.findall(r'\w+', sentence.lower())[:num_words]
    return ' '.join([stemmer.stem(word) for word in words])

def complete_sentence(fragment, enable_global_toggle=True):
    """Complete sentence fragments using the loaded LLM. Respects toggle."""
    # (Keep the previous version of this function - no changes needed here)
    if not enable_global_toggle:
        return fragment + "."
    if not model or not tokenizer:
        print("Warning: LLM not loaded, cannot complete sentence.")
        return fragment + "."
    if re.search(r'[.!?]$', fragment.strip()):
        return fragment
    if len(fragment.split()) < 3 or len(fragment) < 15:
        return fragment + "."
    prompt = f"Complete this sentence fragment naturally and concisely to make it grammatically correct and meaningful:\nFragment: '{fragment}'\nCompleted sentence:"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=30, temperature=0.2,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        completed_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completed_part_match = re.search(r"Completed sentence:\s*(.*)", completed_full, re.IGNORECASE | re.DOTALL)
        if completed_part_match:
            completed_part = completed_part_match.group(1).strip()
            final_sentence = completed_part if completed_part.lower().startswith(fragment.lower()) and len(completed_part) > len(fragment) + 3 else (fragment + "." if completed_part.lower().startswith(fragment.lower()) else completed_part)
            final_sentence = re.sub(r'<\|eot_id\|>', '', final_sentence).strip()
            if not final_sentence: return fragment + "."
            if final_sentence and final_sentence[-1].isalnum(): final_sentence += '.'
            return final_sentence
        else:
            return fragment + "."
    except Exception as e:
        print(f"Error during sentence completion for '{fragment}': {e}")
        return fragment + "."


def preprocess_image_for_math_ocr(image):
    # (Keep the previous version of this function - no changes needed here)
    if image.mode != 'L': image = image.convert('L')
    image_array = np.array(image)
    threshold = np.mean(image_array) * 0.85
    binary_image = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_image)

def extract_text_from_pdf(pdf_path, detect_math=True, ocr_enabled=False):
    # (Keep the previous version of this function - no changes needed here)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    print(f"Extracting text from {pdf_path} (OCR Enabled: {ocr_enabled})...")
    extracted_text = ""
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        for page_num, page in enumerate(doc):
            text = page.get_text("text", sort=True)
            if text: extracted_text += text + "\n"
            if ocr_enabled and tesseract_cmd_path and page.get_images(full=True):
                for img_index, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]; base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        fmt = base_image.get("ext", "png").lower()
                        if fmt not in ["png", "jpeg", "jpg", "bmp", "gif", "tiff"]: continue
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        processed_pil_image = preprocess_image_for_math_ocr(pil_image) if detect_math else pil_image
                        ocr_text = pytesseract.image_to_string(processed_pil_image, config='--psm 6 --oem 3')
                        if ocr_text.strip():
                            if detect_math:
                                for symbol, latex in MATH_SYMBOLS.items(): ocr_text = ocr_text.replace(symbol, f" {latex} ")
                            extracted_text += f"\n[OCR_IMG {img_index+1}] " + ocr_text.strip() + " [/OCR_IMG]\n"
                    except pytesseract.TesseractNotFoundError:
                         print("Error: Tesseract not found during OCR. Disabling OCR for this request."); ocr_enabled = False; break
                    except Exception as e: print(f"Error processing image {img_index} on page {page_num+1}: {str(e)}")
        doc.close()
        lines = extracted_text.split('\n')
        if len(lines) > 2:
            if len(lines[0].strip()) < 25 and re.match(r'^[\s\d\W]*?(\d{1,3})?[\s\d\W]*$', lines[0].strip()): lines = lines[1:]
            if len(lines) > 1 and len(lines[-1].strip()) < 25 and re.match(r'^[\s\d\W]*?(\d{1,3})?[\s\d\W]*$', lines[-1].strip()): lines = lines[:-1]
        extracted_text = '\n'.join(lines)
        return extracted_text
    except Exception as e: print(f"Text extraction failed: {str(e)}"); traceback.print_exc(); raise

def detect_math_content(text):
    # (Keep the previous version of this function - no changes needed here)
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

def clean_text(text):
    # (Keep the previous version of this function - no changes needed here)
    text = re.sub(r'\f', ' ', text)
    text = re.sub(r'\[OCR_IMG.*?\[\/OCR_IMG\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])(\w)', r'\1 \2', text)
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip()) > 2 or line.strip() in ['.','!','?']]
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_text_into_chunks(text, chunk_size=800, overlap=50):
    # (Keep the previous version of this function - no changes needed here)
    try:
        if not any('tokenizers/punkt' in p for p in nltk.data.path): print("Warning: NLTK punkt path potentially not configured.")
        sentences = nltk.sent_tokenize(text)
        if not sentences: raise ValueError("NLTK tokenization resulted in empty list")
    except Exception as e:
        print(f"Warning: NLTK splitting failed ({e}), using regex fallback.")
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()] or [text]
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        sent_length = len(sentence.split())
        if current_length > 0 and current_length + sent_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            overlap_word_count, overlap_sentence_indices = 0, []
            for i in range(len(current_chunk) - 1, -1, -1):
                 overlap_word_count += len(current_chunk[i].split())
                 overlap_sentence_indices.insert(0, i)
                 if overlap_word_count >= overlap: break
            current_chunk = [current_chunk[i] for i in overlap_sentence_indices] + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence); current_length += sent_length
    if current_chunk: chunks.append(' '.join(current_chunk))
    refined_chunks, i, min_chunk_words = [], 0, max(50, chunk_size * 0.2)
    while i < len(chunks):
        current_words = len(chunks[i].split())
        next_words = len(chunks[i+1].split()) if i+1 < len(chunks) else 0
        if current_words < min_chunk_words and i + 1 < len(chunks) and current_words + next_words <= chunk_size * 1.5:
            refined_chunks.append(chunks[i] + " " + chunks[i+1]); i += 2
        else:
            refined_chunks.append(chunks[i]); i += 1
    chunks = refined_chunks
    MAX_CHUNKS = 20
    if len(chunks) > MAX_CHUNKS:
        print(f"Warning: Reducing chunks from {len(chunks)} to {MAX_CHUNKS}")
        step = max(1, len(chunks) // MAX_CHUNKS); chunks = [chunks[i] for i in range(0, len(chunks), step)][:MAX_CHUNKS]
    print(f"Split into {len(chunks)} chunks (~{chunk_size}w, ~{overlap}w overlap).")
    return chunks

def determine_reading_level(grade):
    # (Keep the previous version of this function - no changes needed here)
    if not isinstance(grade, int) or not (1 <= grade <= 12): grade = 6
    age = grade + 5
    if 1 <= grade <= 3: level, desc = "lower", f"early elementary (grades {grade}, ~age {age}-{age+1})"
    elif 4 <= grade <= 6: level, desc = "middle", f"late elem./middle school (grades {grade}, ~age {age}-{age+1})"
    elif 7 <= grade <= 9: level, desc = "higher", f"junior high/early high (grades {grade}, ~age {age}-{age+1})"
    else: level, desc = "higher", f"high school (grades {grade}, ~age {age}-{age+1})"
    return level, desc

# Prompts dictionary (Keep the enhanced versions)
prompts = {
    "lower": { "standard": ("..."), "math": ("...") }, # Keep full prompts
    "middle": { "standard": ("..."), "math": ("...") }, # Keep full prompts
    "higher": { "standard": ("..."), "math": ("...") } # Keep full prompts
    # <<< PASTE THE FULL PROMPTS DICTIONARY HERE FROM YOUR PREVIOUS VERSION >>>
}


# --- MODIFIED: model_generate with robust length calculation ---
def model_generate(prompt_text, max_new_tokens=1024, temperature=0.6):
    if not model or not tokenizer: return "Error: LLM not available."
    current_model_device = next(model.parameters()).device

    # --- Get model context limit (handle potential inconsistencies) ---
    model_context_limit = getattr(model.config, 'max_position_embeddings', None)
    if model_context_limit is None:
        model_context_limit = getattr(tokenizer, 'model_max_length', None)

    # Default carefully if still not found or invalid
    if not isinstance(model_context_limit, int) or model_context_limit <= 512:
        print(f"Warning: Could not reliably determine context limit, using default 8192.")
        model_context_limit = 8192 # A common safe default

    # --- Robust Length Calculation ---
    if max_new_tokens >= model_context_limit:
         print(f"Warning: max_new_tokens ({max_new_tokens}) >= context limit ({model_context_limit}). Reducing.")
         max_new_tokens = model_context_limit // 2

    print(f"DEBUG: Using model_context_limit = {model_context_limit}, requested max_new_tokens = {max_new_tokens}")

    buffer_tokens = 150
    max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens

    if max_prompt_len <= 0:
        print(f"ERROR: Calculated max_prompt_len ({max_prompt_len}) is non-positive.")
        needed_reduction = abs(max_prompt_len) + 10
        max_new_tokens -= needed_reduction
        max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens
        if max_prompt_len <= 0 or max_new_tokens <= 50 :
             return f"Error: Generation request too large for model context limit ({model_context_limit}). Try a shorter duration."
        print(f"Warning: Reduced max_new_tokens to {max_new_tokens} to fit context.")

    max_prompt_len = min(max_prompt_len, model_context_limit) # Final cap
    print(f"DEBUG: Calculated max_prompt_len = {max_prompt_len} for tokenizer.")

    try:
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(current_model_device)
        input_token_count = inputs['input_ids'].shape[1]
        if input_token_count >= max_prompt_len: print(f"Warning: Prompt potentially truncated.")

        start_time = time.time()
        with torch.no_grad():
             outputs = model.generate(
                 **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                 pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                 do_sample=True if temperature > 0.01 else False
             )
        end_time = time.time(); print(f"...Generation took {end_time - start_time:.2f} seconds.")

        generated_text = tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True)
        generated_text = re.sub(r'<\|eot_id\|>', '', generated_text).strip()

        if not re.search(r'(^#|^- )', generated_text, re.MULTILINE): print("Warning: Generated text seems to lack structure.")
        if len(generated_text) < 20: print("Warning: Generation resulted in very short text.")
        return generated_text
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OutOfMemoryError during generation: {e}"); traceback.print_exc()
        torch.cuda.empty_cache()
        return f"Error: Model generation failed - CUDA Out of Memory."
    except Exception as e:
        print(f"Error during model generation: {e}"); traceback.print_exc()
        return f"Error: Model generation failed - {str(e)}"
# --- End model_generate ---


# --- MODIFIED: generate_summary uses robust context limit ---
def generate_summary(text_chunks, grade_level_category, grade_level_desc, duration_minutes, has_math=False,
                     enable_completion_web=True, enable_deduplication_web=True):
    # (Word count logic remains the same)
    if duration_minutes == 10: min_words, max_words = 1200, 1600
    elif duration_minutes == 20: min_words, max_words = 2400, 3200
    elif duration_minutes == 30: min_words, max_words = 3600, 4500
    else: min_words, max_words = 1200, 1600
    print(f"Targeting summary: {grade_level_desc}, {duration_minutes} mins ({min_words}-{max_words} words approx).")
    print(f"Refinement Options - Completion: {enable_completion_web}, Deduplication: {enable_deduplication_web}")

    full_text = ' '.join(text_chunks)
    try: full_text_tokens_estimate = len(tokenizer.encode(full_text))
    except: full_text_tokens_estimate = len(full_text) // 3

    # --- Get model context limit (handle potential inconsistencies) ---
    model_context_limit = getattr(model.config, 'max_position_embeddings', None)
    if model_context_limit is None: model_context_limit = getattr(tokenizer, 'model_max_length', None)
    if not isinstance(model_context_limit, int) or model_context_limit <= 512: model_context_limit = 8192
    print(f"DEBUG (generate_summary): Using model_context_limit = {model_context_limit}")

    estimated_target_max_tokens = int(max_words * 1.3) + 200
    safe_generation_limit = max((model_context_limit // 2) - 150, 512)
    max_new_tokens_summary = max(min(estimated_target_max_tokens, safe_generation_limit), 512)
    print(f"Calculated max_new_tokens for summary: {max_new_tokens_summary} (safe_gen_limit: {safe_generation_limit})")

    prompt_instruction_buffer = 700
    required_tokens_for_single_pass = full_text_tokens_estimate + max_new_tokens_summary + prompt_instruction_buffer
    can_summarize_all_at_once = (required_tokens_for_single_pass < (model_context_limit * 0.9) and
                                 full_text_tokens_estimate < (model_context_limit * 0.6) and
                                 max_new_tokens_summary <= 2048) # Adjust limit if needed

    initial_summary = ""
    # (Single pass / iterative logic remains the same - uses model_generate which now has fixes)
    if can_summarize_all_at_once and text_chunks:
        print("Attempting summary generation in a single pass.")
        prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
        try:
            prompt = prompt_template.format(text=full_text)
            prompt += f"\n\nIMPORTANT: Structure summary ({min_words}-{max_words} words)."
        except KeyError: prompt = f"Create summary for {grade_level_desc} ({min_words}-{max_words} words):\n{full_text}"
        initial_summary = model_generate(prompt, max_new_tokens=max_new_tokens_summary, temperature=0.6)
    elif text_chunks:
        print(f"Iterative summary ({len(text_chunks)} chunks).")
        chunk_summaries = []
        max_new_tokens_chunk = max(min( (model_context_limit // (len(text_chunks) + 1)) - 150, 300), 100)
        print(f"Max new tokens per chunk summary: {max_new_tokens_chunk}")
        for i, chunk in enumerate(text_chunks):
            # print(f"  Summarizing chunk {i + 1}/{len(text_chunks)}...") # Reduce noise
            chunk_prompt = f"Key points from chunk {i+1}/{len(text_chunks)}:\n{chunk}\n\nKey Points (bullet list):"
            chunk_summary = model_generate(chunk_prompt, max_new_tokens=max_new_tokens_chunk, temperature=0.4)
            if chunk_summary.startswith("Error:") or len(chunk_summary.split()) < 5: continue
            chunk_summary = re.sub(r'^.*?Key Points.*?\n', '', chunk_summary, flags=re.I).strip()
            if chunk_summary: chunk_summaries.append(chunk_summary)
        if not chunk_summaries: return "Error: Failed to generate summaries for any text chunks."
        print("Consolidating chunk summaries...")
        template = prompts[grade_level_category]["math" if has_math else "standard"]
        base_instr = template.split("Text to summarize:")[0]
        consol_prompt = f"{base_instr}\nSYNTHESIZE points below into ONE summary for {grade_level_desc}.\nFollow instructions. Aim for {min_words}-{max_words} words.\n\nChunk Summaries:\n\n" + "\n\n---\n\n".join(chunk_summaries) + "\n\nFinal Consolidated Summary:"
        initial_summary = model_generate(consol_prompt, max_new_tokens=max_new_tokens_summary, temperature=0.65)
    else: return "Error: Text processing resulted in zero chunks."

    if initial_summary.startswith("Error:"): return initial_summary
    current_summary = initial_summary
    current_word_count = len(current_summary.split())
    print(f"Initial summary: {current_word_count} words. Checking length.")

    # (Lengthening logic remains the same)
    attempts, max_attempts = 0, 2
    while current_word_count < min_words and attempts < max_attempts:
        print(f"Summary short. Elaborating (Attempt {attempts + 1}/{max_attempts})...")
        prompt = f"Elaborate on the points...Current Summary:\n{current_summary}\n\nContinue summary:"
        needed = min_words - current_word_count
        tokens_add = max(min(int(needed * 1.5), max_new_tokens_summary // 2, 700), 150)
        new_part = model_generate(prompt, max_new_tokens=tokens_add, temperature=0.7)
        if new_part.startswith("Error:") or len(new_part.split()) < 10: print("Stopping lengthening."); break
        current_summary += "\n\n" + new_part.strip(); current_word_count = len(current_summary.split()); attempts += 1
    if attempts == max_attempts and current_word_count < min_words: print(f"Warning: Max lengthening attempts.")

    # (Trimming logic remains the same)
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

    # (Fallback activity logic remains the same)
    activity_pattern = r'^##\s+(Fun Activity|Practice.*|Try This|Further Thinking|Challenge|Activity)\s*$' # Simplified pattern
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


# <<< MODIFIED: enhanced_post_process accepts refinement flags >>>
def enhanced_post_process(summary, grade_level_category, enable_completion=True, enable_deduplication=True):
    # (Keep the previous version of this function - no changes needed here)
    if summary.startswith("Error:"): return summary
    print(f"Running enhanced post-processing (Completion:{enable_completion}, Dedup:{enable_deduplication})...")
    completion_calls_made = 0
    try: # Get expected heading
        prompt_lines = prompts[grade_level_category]["standard"].splitlines()
        heading_line = next((l for l in prompt_lines if l.strip().startswith('#')), None)
        exp_head = heading_line.strip().lstrip('# ').strip() if heading_line else "Summary"
    except: exp_head = "Summary"
    summary = re.sub(r'^\s*#+.*?(\n|$)', f'# {exp_head}\n\n', summary.strip(), count=1, flags=re.I)
    if not summary.startswith("# "): summary = f'# {exp_head}\n\n' + summary

    lines = summary.split('\n'); processed_data, seen_frags = [], set()
    for line in lines:
        s_line = line.strip()
        if not s_line: # Blank line handling
            if processed_data and processed_data[-1]["type"] != "blank": processed_data.append({"text":"", "type":"blank"})
            continue
        l_type, content, is_head, is_bullet = "paragraph", s_line, False, False
        if s_line.startswith('## '): l_type, content, is_head = "subheading", s_line[3:].strip(), True
        elif s_line.startswith('# '): l_type, content, is_head = "heading", s_line[2:].strip(), True
        elif s_line.startswith('- '): l_type, content, is_bullet = "bullet", s_line[2:].strip(), True
        elif re.match(r'^\d+\.\s+', s_line): l_type, content, is_bullet = "numbered", re.sub(r'^\d+\.\s+', '', s_line), True
        if not content: continue
        cont_key = ' '.join(content.lower().split()[:10]) # Basic duplicate check
        if not is_head and cont_key in seen_frags and len(content.split()) < 15: continue
        if not is_head: seen_frags.add(cont_key)

        # Sentence Completion (Conditional & Limited)
        if l_type in ["paragraph", "bullet", "numbered"] and len(content.split()) > 4:
            if enable_completion and not re.search(r'[.!?:]$', content) and content[0].isupper() and completion_calls_made < MAX_COMPLETION_CALLS:
                original_content = content
                content = complete_sentence(content, enable_global_toggle=enable_completion)
                if content != original_content and not content.endswith(original_content + "."): completion_calls_made += 1
            if content and content[0].islower() and not re.match(r'^[a-z]\s*\(', content): content = content[0].upper() + content[1:]
            if content and content[-1].isalnum(): content += '.'

        if l_type == "blank" and processed_data and processed_data[-1]["type"] == "blank": continue
        processed_data.append({"text":content, "type":l_type})

    # Semantic Deduplication (Conditional)
    points_for_dedup, indices_map = [], {}
    if enable_deduplication and embedder:
        for i, data in enumerate(processed_data):
            if data["type"] in ["paragraph","bullet","numbered"] and len(data["text"].split()) > 6:
                cont = data["text"]; points_for_dedup.append(cont)
                if cont not in indices_map: indices_map[cont] = []
                indices_map[cont].append(i)
    elif not enable_deduplication: print("Skipping dedup (disabled).")
    elif not embedder: print("Skipping dedup (no embedder).")

    kept_indices = set(range(len(processed_data)))
    if points_for_dedup and enable_deduplication and embedder:
        print(f"Running semantic dedup on {len(points_for_dedup)} points...")
        try:
            unique_pts = remove_duplicates_semantic(points_for_dedup, batch_size=128); unique_set = set(unique_pts)
            print(f"Reduced to {len(unique_pts)} unique points.")
            indices_to_remove, processed_removal = set(), set()
            for cont, orig_indices in indices_map.items():
                if cont in processed_removal: continue
                if cont not in unique_set:
                    for index in orig_indices:
                        is_only = False # Basic check to avoid removing only content under heading
                        if index>0 and processed_data[index-1]["type"] in ["heading","subheading"] and (index==len(processed_data)-1 or processed_data[index+1]["type"] in ["heading","subheading","blank"]): is_only = True
                        if not is_only: indices_to_remove.add(index)
                processed_removal.add(cont)
            kept_indices -= indices_to_remove; print(f"Marked {len(indices_to_remove)} lines for removal.")
        except Exception as e: print(f"Warning: Dedup failed: {e}")

    # Final Assembly
    final_text, last_type = "", None
    kept_data = [processed_data[i] for i in sorted(list(kept_indices))]
    for i, data in enumerate(kept_data):
        curr_type, content = data["type"], data["text"]
        if i > 0: # Add spacing
            if curr_type in ["heading","subheading"]: final_text += "\n\n"
            elif curr_type == "paragraph" and last_type not in ["heading","subheading","blank"]: final_text += "\n\n"
            elif curr_type != "blank" and last_type != "blank": final_text += "\n"
            elif curr_type == "blank" and last_type == "blank": continue
        # Add content with markdown
        if curr_type == "heading": final_text += f"# {content}"
        elif curr_type == "subheading": final_text += f"## {content}"
        elif curr_type == "bullet": final_text += f"- {content}"
        elif curr_type == "numbered": final_text += f"1. {content}"
        elif curr_type == "paragraph": final_text += content
        last_type = curr_type
    print("Post-processing finished.")
    return final_text.strip()
# --- End enhanced_post_process ---


# <<< MODIFIED: remove_duplicates_semantic accepts batch_size >>>
def remove_duplicates_semantic(points, similarity_threshold=0.90, batch_size=64):
    # (Keep the previous version of this function - no changes needed here)
    if not points or not embedder or len(points) < 2: return points
    unique_pts, start_time = [], time.time()
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
        print(f"Semantic deduplication took {time.time() - start_time:.2f}s.")
        return unique_pts + short_pts
    except torch.cuda.OutOfMemoryError: print("OOM Error during dedup. Skipping."); return points
    except Exception as e: print(f"Dedup Error: {e}"); traceback.print_exc(); return points
# --- End remove_duplicates_semantic ---


def generate_activity(summary_text, grade_level_category, grade_level_desc):
    # (Keep the previous version of this function - no changes needed here)
    if not model or not tokenizer: return "- Review key points."
    print("Generating fallback activity suggestion...")
    act_type = {"lower": "fun activity/question", "middle": "practical activity/thought question", "higher": "provoking question/research idea/analysis task"}.get(grade_level_category, "activity")
    prompt_template = "Based on summary context:\n...{summary_snippet}\n\nSuggest ONE simple {activity_type} for {grade_desc}:"
    snippet = ' '.join(re.sub(r'^#.*?\n', '', summary_text).strip().split()[-200:])
    prompt = prompt_template.format(activity_type=act_type, grade_desc=grade_level_desc, summary_snippet=snippet)
    activity = model_generate(prompt, max_new_tokens=80, temperature=0.75)
    if activity.startswith("Error:"): print(f"Fallback activity failed: {activity}"); activity = ""
    else: activity = re.sub(r'^[\-\*\s]+','', activity.strip().replace("Activity Suggestion:","").strip()); activity = re.sub(r'\.$','',activity).strip()
    if activity:
        activity = f"- {activity[0].upper() + activity[1:]}"; activity += '.' if activity[-1].isalnum() else ''
        return activity
    else: print("Warning: Failed fallback activity."); fallbacks={"lower":"- Draw!", "middle":"- Explain!", "higher":"- Find example."}; return fallbacks.get(grade_level_category, "- Review.")
# --- End generate_activity ---


# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    # (Keep the previous version of this function - no changes needed here)
    start_time = time.time()
    if 'pdfFile' not in request.files: return jsonify({"error": "No PDF file provided."}), 400
    file = request.files['pdfFile'];
    if file.filename == '': return jsonify({"error": "No selected file."}), 400
    if not file.filename.lower().endswith('.pdf'): return jsonify({"error": "Invalid file type."}), 400
    try: # Get form data
        grade = int(request.form.get('grade', 6))
        duration = int(request.form.get('duration', 20))
        ocr_enabled = request.form.get('ocr', 'false').lower() == 'true'
        chunk_size = int(request.form.get('chunkSize', 500))
        overlap = int(request.form.get('overlap', 50))
        sentence_completion_enabled = request.form.get('sentenceCompletion', 'false').lower() == 'true'
        deduplication_enabled = request.form.get('deduplication', 'false').lower() == 'true'
        if not (100 <= chunk_size <= 2000): chunk_size = 500
        if not (0 <= overlap <= chunk_size // 2): overlap = 50
    except ValueError: return jsonify({"error": "Invalid form data."}), 400

    pdf_path = None
    try: # Process PDF
        fd, pdf_path = tempfile.mkstemp(suffix=".pdf"); os.close(fd); file.save(pdf_path)
        print(f"PDF saved temporarily to: {pdf_path}")
        grade_cat, grade_desc = determine_reading_level(grade)
        print(f"Processing for: {grade_desc}, OCR:{ocr_enabled}, Chunk:{chunk_size}, Overlap:{overlap}, Comp:{sentence_completion_enabled}, Dedup:{deduplication_enabled}")
        extract_start = time.time()
        raw_text = extract_text_from_pdf(pdf_path, detect_math=True, ocr_enabled=ocr_enabled)
        print(f"Extraction took {time.time() - extract_start:.2f}s")
        if not raw_text or len(raw_text.strip()) < 50: return jsonify({"error": "No significant text extracted."}), 400
        process_start = time.time()
        has_math = detect_math_content(raw_text); print(f"Math detected: {has_math}")
        print("Cleaning text..."); cleaned = clean_text(raw_text)
        print("Splitting text..."); chunks = split_text_into_chunks(cleaned, chunk_size=chunk_size, overlap=overlap)
        if not chunks: return jsonify({"error": "Failed to split text."}), 500
        print(f"Processing took {time.time() - process_start:.2f}s")
        print("Generating summary..."); gen_start = time.time()
        summary = generate_summary(chunks, grade_cat, grade_desc, duration, has_math,
                                   enable_completion_web=sentence_completion_enabled,
                                   enable_deduplication_web=deduplication_enabled)
        print(f"Core summary generation took {time.time() - gen_start:.2f}s")
        if summary.startswith("Error:"): error_msg = summary.split("Error:", 1)[1].strip(); print(f"Summarization failed: {error_msg}"); return jsonify({"error": f"Summarization failed: {error_msg}"}), 500
        word_count = len(summary.split()); total_time = time.time() - start_time
        print(f"Summary generated. Words: {word_count}. Total time: {total_time:.2f}s")
        return jsonify({"summary": summary, "word_count": word_count, "processing_time": round(total_time, 2)})
    except FileNotFoundError as e: print(f"Error: {e}"); return jsonify({"error": str(e)}), 404
    except pytesseract.TesseractNotFoundError: err = "Tesseract not found."; print(f"Error: {err}"); return jsonify({"error": err}), 500
    except torch.cuda.OutOfMemoryError: err = "GPU OOM."; print(f"Error: {err}"); traceback.print_exc(); return jsonify({"error": err}), 500
    except Exception as e: print(f"Unexpected Error: {e}"); traceback.print_exc(); return jsonify({"error": "Unexpected server error."}), 500
    finally: # Cleanup
        if pdf_path and os.path.exists(pdf_path):
            try: os.remove(pdf_path); print(f"Temp file removed: {pdf_path}")
            except Exception as e: print(f"Error removing temp file {pdf_path}: {e}")

# --- Run Flask App ---
if __name__ == '__main__':
    print("Starting Flask development server (for testing only)...")
    app.run(host='0.0.0.0', port=8501, debug=False, threaded=True)