# app.py (Complete - Modified for Unsloth Llama-4-Scout-17B 4-bit)

import os
import re
# --- MOVE UNSLOTH IMPORT HERE ---
from unsloth import FastLanguageModel
# --- END MOVE ---
import pytesseract
from PIL import Image
import numpy as np
import torch # Keep torch import after unsloth
from nltk.stem import PorterStemmer
import nltk
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer, util
# Keep other transformers imports if needed (like AutoTokenizer for embedder)
from transformers import AutoTokenizer #, BitsAndBytesConfig # Removed BNB config import
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
        else: # Fallback to system location
            nltk.download('punkt', quiet=True)
    else:
        nltk.data.path.append(nltk_data_path) # Ensure it's in the path if already exists
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
        try: # Check if executable
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

# Device (Unsloth generally handles device placement, but keep for embedder)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} (Note: Unsloth manages LLM placement)")

OCR_RESOLUTION = 300
MAX_SEQ_LENGTH = 8192 # Define max sequence length for Unsloth loading

# --- MODIFIED: Model Names & Unsloth Setup ---
LLM_MODEL_NAME = "unsloth/Llama-4-Scout-17B-16E-unsloth-dynamic-bnb-4bit" # <<< CHANGED Model Name
EMBEDDER_MODEL_NAME = 'all-MiniLM-L6-v2' # Keep the same embedder

# Quantization is handled by Unsloth's FastLanguageModel when load_in_4bit=True
print(f"Will load {LLM_MODEL_NAME} with Unsloth (4-bit implied by name/load_in_4bit)")
# --- END MODIFIED ---

tokenizer = None # Will be loaded by FastLanguageModel
model = None     # Will be loaded by FastLanguageModel
embedder = None
stemmer = PorterStemmer()
MAX_COMPLETION_CALLS = 10 # Limit sentence completions per request in post-processing

try:
    print(f"Loading LLM model & tokenizer: {LLM_MODEL_NAME} using Unsloth...")
    # <<< NOTE: Ensure you provide authentication (e.g., HF_TOKEN env var or login) if needed >>>
    # Use FastLanguageModel.from_pretrained
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LLM_MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,              # Unsloth handles dtype optimization with 4bit
        load_in_4bit = True,       # Explicitly load in 4bit
        device_map = {'': 0},       # Let Unsloth / Accelerate handle device placement
        # llm_int8_enable_fp32_cpu_offload = True, # Enable CPU offload for large models
        # token = "hf_...", # Add token if needed
    )
    print("Unsloth LLM Model and Tokenizer loaded successfully!")
    # Get context limit
    actual_context_limit = getattr(model.config, 'max_position_embeddings', None)
    if actual_context_limit is None: actual_context_limit = getattr(tokenizer, 'model_max_length', None)
    if actual_context_limit is None: actual_context_limit = MAX_SEQ_LENGTH # Fallback
    print(f"DEBUG: Using context limit: {actual_context_limit}")


except Exception as e:
    print(f"FATAL ERROR: Could not load Unsloth model {LLM_MODEL_NAME}: {e}")
    traceback.print_exc()
    # Attempt to clear cache if OOM during loading
    if "out of memory" in str(e).lower():
        print("Attempting to clear CUDA cache...")
        torch.cuda.empty_cache()
    exit(1)

try:
    # Embedder loading remains the same (uses standard SentenceTransformer)
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
    if not enable_global_toggle:
        # print("Sentence completion skipped (disabled by flag).") # Reduce noise
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
        # Use fewer tokens for simple completion
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        completed_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completed_part_match = re.search(r"Completed sentence:\s*(.*)", completed_full, re.IGNORECASE | re.DOTALL)
        if completed_part_match:
            completed_part = completed_part_match.group(1).strip()
            if completed_part.lower().startswith(fragment.lower()):
                 final_sentence = completed_part if len(completed_part) > len(fragment) + 3 else fragment + "."
            else:
                 final_sentence = completed_part
            final_sentence = re.sub(r'<\|eot_id\|>', '', final_sentence).strip()
            if not final_sentence: return fragment + "."
            if final_sentence and final_sentence[-1].isalnum(): final_sentence += '.'
            return final_sentence
        else:
            # print(f"Warning: Could not parse completion for fragment: '{fragment}'. Returning fragment.") # Reduce noise
            return fragment + "."
    except Exception as e:
        print(f"Error during sentence completion for '{fragment}': {e}")
        return fragment + "."

def preprocess_image_for_math_ocr(image):
    """Basic image preprocessing for OCR."""
    if image.mode != 'L': image = image.convert('L')
    image_array = np.array(image)
    threshold = np.mean(image_array) * 0.85 # Adjust threshold factor if needed
    binary_image = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_image)

def extract_text_from_pdf(pdf_path, detect_math=True, ocr_enabled=False):
    """Extracts text using PyMuPDF (fitz) and optionally Tesseract OCR."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    print(f"Extracting text from {pdf_path} (OCR Enabled: {ocr_enabled})...")
    extracted_text = ""
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        for page_num, page in enumerate(doc):
            # Extract standard text
            text = page.get_text("text", sort=True)
            if text: extracted_text += text + "\n"

            # Extract text from images via OCR if enabled
            if ocr_enabled and tesseract_cmd_path and page.get_images(full=True):
                # print(f"  - Page {page_num+1}: Found images, performing OCR...") # Reduce noise
                for img_index, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        fmt = base_image.get("ext", "png").lower() # Get extension safely
                        if fmt.lower() not in ["png", "jpeg", "jpg", "bmp", "gif", "tiff"]:
                             # print(f"    - Skipping image {img_index} with unsupported format: {fmt}") # Reduce noise
                             continue

                        pil_image = Image.open(io.BytesIO(image_bytes))
                        processed_pil_image = preprocess_image_for_math_ocr(pil_image) if detect_math else pil_image
                        # Use resolution setting if needed, default PSM/OEM often okay
                        custom_config = f'--psm 6 --oem 3 -c tessedit_do_invert=0' # Example config
                        ocr_text = pytesseract.image_to_string(processed_pil_image, config=custom_config) # Use config var

                        if ocr_text.strip():
                            if detect_math:
                                for symbol, latex in MATH_SYMBOLS.items():
                                    ocr_text = ocr_text.replace(symbol, f" {latex} ")
                            extracted_text += f"\n[OCR_IMG {img_index+1}] " + ocr_text.strip() + " [/OCR_IMG]\n"
                    except pytesseract.TesseractNotFoundError:
                         print("Error: Tesseract not found during OCR. Disabling OCR for this request.")
                         ocr_enabled = False
                         break
                    except Exception as e:
                        print(f"Error processing image {img_index} on page {page_num+1}: {str(e)}")
        doc.close()

        # Basic header/footer removal
        lines = extracted_text.split('\n')
        if len(lines) > 2:
            if len(lines[0].strip()) < 25 and re.match(r'^[\s\d\W]*?(\d{1,3})?[\s\d\W]*$', lines[0].strip()): lines = lines[1:]
            if len(lines) > 1 and len(lines[-1].strip()) < 25 and re.match(r'^[\s\d\W]*?(\d{1,3})?[\s\d\W]*$', lines[-1].strip()): lines = lines[:-1]
        extracted_text = '\n'.join(lines)
        # print(f"Extracted ~{len(extracted_text)} characters.") # Reduce noise
        return extracted_text
    except Exception as e:
        print(f"Text extraction failed: {str(e)}")
        traceback.print_exc()
        raise

def detect_math_content(text):
    """Simple heuristic to detect potential math content."""
    math_keywords = [r'\b(equation|formula|theorem|lemma|proof|calculus|algebra|derivative|function|integral|vector|matrix|variable|constant|graph|plot|solve|calculate|measure|angle|degree)\b']
    math_symbols = r'[=><≤≥≠\+\-\*\/\^∫∑∏√∞≠≤≥±→∂∇πθλμσωαβγδε%]'
    function_notation = r'\b[a-zA-Z]\s?\([a-zA-Z0-9,\s\+\-\*\/]+\)'
    latex_delimiters = r'\$.*?\$|\\\(.*?\\\)|\\[a-zA-Z]+(\{.*?\})*'
    math_list_items = r'^\s*(\d+\.|\*|\-)\s*[=><≤≥≠\+\-\*\/\^∫∑∏√∞≠≤≥±→∂∇πθλμσωαβγδε\$\\]'

    if re.search('|'.join(math_keywords), text, re.IGNORECASE): return True
    # Limit search scope for performance
    text_sample = text[:30000]
    for pattern in [math_symbols, function_notation, latex_delimiters]:
        if re.search(pattern, text_sample): return True
    if re.search(math_list_items, text_sample, re.MULTILINE): return True
    return False

def clean_text(text):
    """Cleans extracted text."""
    text = re.sub(r'\f', ' ', text) # Form feed
    text = re.sub(r'\[OCR_IMG.*?\[\/OCR_IMG\]', '', text, flags=re.DOTALL) # Remove OCR blocks
    text = re.sub(r'\(cid:\d+\)', '', text) # PDF artifacts
    text = re.sub(r'\s+', ' ', text) # Normalize whitespace
    text = re.sub(r'([.!?])(\w)', r'\1 \2', text) # Space after punctuation
    # Remove very short lines (likely artifacts)
    lines = text.split('\n')
    cleaned_lines = [l for l in lines if len(l.strip()) > 2 or l.strip() in ['.','!','?']]
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text) # Remove space before punctuation
    # Rejoin hyphenated words (use cautiously)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    text = re.sub(r'\n', ' ', text) # Convert remaining newlines to spaces
    text = re.sub(r'\s+', ' ', text).strip() # Final whitespace cleanup
    return text

def split_text_into_chunks(text, chunk_size=800, overlap=50):
    """Splits text into chunks using NLTK or regex fallback."""
    try:
        # Ensure NLTK can find its data
        if not any('tokenizers/punkt' in p for p in nltk.data.path):
             print("Warning: NLTK punkt path potentially not configured. Trying default.")
        sentences = nltk.sent_tokenize(text)
        if not sentences: raise ValueError("NLTK tokenization resulted in empty list")
    except Exception as e:
        print(f"Warning: NLTK splitting failed ({e}), using regex fallback.")
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()] or [text] # Handle case where no split occurs

    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        sent_length = len(sentence.split())
        # Simplified chunking logic
        if current_length > 0 and current_length + sent_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            # Determine overlap based on sentences, aiming for word count
            overlap_word_count = 0
            overlap_sentence_indices = []
            for i in range(len(current_chunk) - 1, -1, -1):
                 overlap_word_count += len(current_chunk[i].split())
                 overlap_sentence_indices.insert(0, i)
                 if overlap_word_count >= overlap:
                     break
            # Start new chunk with overlap sentences + current sentence
            current_chunk = [current_chunk[i] for i in overlap_sentence_indices] + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)

        else:
            current_chunk.append(sentence)
            current_length += sent_length

    if current_chunk: chunks.append(' '.join(current_chunk))

    # Merge small chunks
    refined_chunks = []
    i = 0
    min_chunk_words = max(50, chunk_size * 0.2) # Ensure min chunk size isn't too small
    while i < len(chunks):
        current_chunk_words = len(chunks[i].split())
        next_chunk_words = len(chunks[i+1].split()) if i+1 < len(chunks) else 0
        # If merging exceeds max size * 1.5, don't merge
        if current_chunk_words < min_chunk_words and i + 1 < len(chunks):
             if current_chunk_words + next_chunk_words <= chunk_size * 1.5:
                refined_chunks.append(chunks[i] + " " + chunks[i+1])
                i += 2
             else: # Don't merge if it makes the chunk too big
                 refined_chunks.append(chunks[i])
                 i += 1
        else:
            refined_chunks.append(chunks[i])
            i += 1
    chunks = refined_chunks

    # Limit chunks
    MAX_CHUNKS = 20
    if len(chunks) > MAX_CHUNKS:
        print(f"Warning: Reducing chunks from {len(chunks)} to {MAX_CHUNKS}")
        step = max(1, len(chunks) // MAX_CHUNKS)
        chunks = [chunks[i] for i in range(0, len(chunks), step)][:MAX_CHUNKS]

    print(f"Split into {len(chunks)} chunks (~{chunk_size} words each, overlap ~{overlap} words).")
    return chunks

def determine_reading_level(grade):
    """Determines reading level category and description from grade."""
    if not isinstance(grade, int) or not (1 <= grade <= 12): grade = 6
    age = grade + 5
    if 1 <= grade <= 3: level, desc = "lower", f"early elementary (grades {grade}, ~age {age}-{age+1})"
    elif 4 <= grade <= 6: level, desc = "middle", f"late elem./middle school (grades {grade}, ~age {age}-{age+1})"
    elif 7 <= grade <= 9: level, desc = "higher", f"junior high/early high (grades {grade}, ~age {age}-{age+1})"
    else: level, desc = "higher", f"high school (grades {grade}, ~age {age}-{age+1})"
    return level, desc

# --- Prompts Dictionary (Refined for Llama-4-Scout-17B-Instruct) ---
prompts = {
    "lower": { # Grades 1-3 (Ages 6-8) - Focus: Extreme Simplicity, Core Idea
        "standard": (
            "You are summarizing text for a young child (grades 1-3, ages 6-8).\n"
            "Instructions:\n"
            "1. Use VERY simple words and extremely short sentences.\n"
            "2. Start with the heading '# Simple Summary'.\n"
            "3. Under the heading, explain the absolute main idea in one simple sentence.\n"
            "4. Then, under a '## Key Points' heading, list 3-5 key points using bullet points '- '. Each point must be a full, simple sentence.\n"
            "5. Do NOT include complex details, numbers, or jargon.\n"
            "6. Finish with ONE fun, simple activity (like drawing or a simple question) under the heading '## Fun Activity'.\n"
            "7. Use clear Markdown formatting for headings and bullets.\n\n"
            "Text to summarize:\n{text}"
        ),
        "math": (
            "You are explaining a math topic to a young child (grades 1-3, ages 6-8).\n"
            "Instructions:\n"
            "1. Use very simple words, short sentences, and analogies (like counting toys or sharing cookies).\n"
            "2. Use very small, simple numbers in examples.\n"
            "3. Start with the main heading '# Math Fun'.\n"
            "4. Explain the main math idea very simply under the heading '## What We Learned'.\n"
            "5. If there are steps, list them very simply under '## Steps' using numbers (1., 2.).\n"
            "6. Give one clear, simple example with small numbers under '## Example'.\n"
            "7. Finish with ONE easy practice question or drawing task under '## Practice Time'.\n"
            "8. Use clear Markdown formatting for headings and numbered lists.\n\n"
            "Text to explain:\n{text}"
        )
    },
    "middle": { # Grades 4-6 (Ages 9-11) - Focus: Main Topics, Clear Explanation
        "standard": (
            "You are summarizing text for a student in grades 4-6 (ages 9-11).\n"
            "Instructions:\n"
            "1. Start with the main heading '# Summary'.\n"
            "2. Identify the 2-4 most important main topics or sections from the text.\n"
            "3. For each main topic, create a clear subheading using '## Topic Name'.\n"
            "4. Under each subheading, use bullet points '- ' to list the key information and supporting details. Use clear, complete sentences.\n"
            "5. Explain any important terms simply.\n"
            "6. Ensure the summary flows logically and synthesizes information, don't just list disconnected facts.\n"
            "7. Conclude with ONE practical activity or thought-provoking question related to the text under the heading '## Try This'.\n"
            "8. Use clear Markdown formatting for headings and bullets.\n\n"
            "Text to summarize:\n{text}"
        ),
        "math": (
            "You are explaining a math concept to a student in grades 4-6 (ages 9-11).\n"
            "Instructions:\n"
            "1. Start with the heading '# Math Explained'.\n"
            "2. Explain the core math concept clearly and concisely under the heading '## The Concept'.\n"
            "3. Provide a clear, step-by-step example of a typical problem under '## Step-by-Step Example'. Use numbered steps (1., 2.). Show the work clearly.\n"
            "4. Briefly explain why this math is useful or where it might be applied under '## Why It Matters'.\n"
            "5. Conclude with ONE relevant practice problem under '## Practice Problem'. If possible, provide the answer separately or indicate how to check it.\n"
            "6. Use clear language and structure the output using Markdown headings and numbered lists.\n\n"
            "Text to explain:\n{text}"
        )
    },
    "higher": { # Grades 7-12 (Ages 12-18) - Focus: Synthesis, Analysis, Structure
        "standard": (
            "You are creating a comprehensive, well-structured summary for a high school student (grades 7-12, ages 12-18).\n"
            "Instructions:\n"
            "1. Start with the main heading '# Comprehensive Summary'.\n"
            "2. Identify the key themes, arguments, sections, or concepts presented in the text.\n"
            "3. Create logical subheadings ('## Theme/Section Name') for each key area.\n"
            "4. Under each subheading, **synthesize** the essential information. Use clear paragraphs for explanation and bullet points '- ' for specific details, evidence, or examples where appropriate.\n"
            "5. Analyze or evaluate points where relevant, rather than just listing information.\n"
            "6. Use appropriate academic vocabulary but ensure clarity. Define key technical terms if necessary.\n"
            "7. If relevant, include a section '## Connections' discussing real-world implications, applications, or connections to other subjects.\n"
            "8. Conclude with ONE thought-provoking question, potential research idea, or analysis task under the heading '## Further Thinking'.\n"
            "9. Structure the entire output logically and clearly using Markdown formatting (headings, subheadings, paragraphs, lists).\n\n"
            "Text to summarize:\n{text}"
        ),
        "math": (
            "You are explaining an advanced math topic for a high school student (grades 7-12, ages 12-18).\n"
            "Instructions:\n"
            "1. Start with the main heading '# Advanced Math Concepts'.\n"
            "2. Provide concise definitions of key terms and concepts under the heading '## Definitions'.\n"
            "3. Explain the core theory, theorem, or method rigorously under '## Core Theory'. Use clear paragraphs and potentially bullet points for key steps or properties.\n"
            "4. Include a non-trivial worked example demonstrating the concept or technique under '## Worked Example'. Show all steps clearly and explain the reasoning.\n"
            "5. Discuss applications or connections to other fields (e.g., science, computer science, engineering, economics) under '## Applications'.\n"
            "6. Conclude with ONE challenging problem or an extension idea for further exploration under '## Challenge'.\n"
            "7. Use appropriate mathematical notation consistently (preserve LaTeX if present in source, otherwise use standard math symbols). Structure the output clearly using Markdown headings and formatting.\n\n"
            "Text to explain:\n{text}"
        )
    }
}
# --- End Prompts ---


# --- MODIFIED: model_generate uses context limit from loaded model/tokenizer ---
def model_generate(prompt_text, max_new_tokens=1024, temperature=0.6):
    """Generates text using the loaded Unsloth model."""
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
         max_new_tokens = model_context_limit // 2 # Reduce significantly
    # print(f"DEBUG: Using model_context_limit = {model_context_limit}, requested max_new_tokens = {max_new_tokens}") # Reduce noise
    buffer_tokens = 150
    max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens
    if max_prompt_len <= 0:
        print(f"ERROR: Calculated max_prompt_len ({max_prompt_len}) is non-positive.")
        needed = abs(max_prompt_len) + 10; max_new_tokens -= needed
        max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens
        if max_prompt_len <= 0 or max_new_tokens <= 50 : return f"Error: Generation request too large for limit ({model_context_limit})."
        print(f"Warning: Reduced max_new_tokens to {max_new_tokens} to fit context.")
    max_prompt_len = min(max_prompt_len, model_context_limit) # Final cap
    # print(f"DEBUG: Calculated max_prompt_len = {max_prompt_len} for tokenizer.") # Reduce noise

    try:
        # Unsloth tokenizer usage is typically identical to transformers
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(current_model_device)
        input_token_count = inputs['input_ids'].shape[1]
        if input_token_count >= max_prompt_len: print(f"Warning: Prompt potentially truncated.")

        start_time = time.time()
        with torch.no_grad():
             # Unsloth model usage is typically identical to transformers
             outputs = model.generate(
                 **inputs,
                 max_new_tokens=max_new_tokens,
                 temperature=temperature,
                 pad_token_id=tokenizer.eos_token_id,
                 eos_token_id=tokenizer.eos_token_id, # Ensure EOS token is used for stopping
                 do_sample=True if temperature > 0.01 else False, # Adjust sampling threshold
             )
        end_time = time.time(); print(f"...Generation took {end_time - start_time:.2f} seconds.")

        # Unsloth tokenizer usage is typically identical to transformers
        generated_text = tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True)
        generated_text = re.sub(r'<\|eot_id\|>', '', generated_text).strip() # Adjust if needed for new model tokens

        # (Keep checks for structure and length)
        if not re.search(r'(^#|^- )', generated_text, re.MULTILINE): print("Warning: Generated text seems to lack structure.")
        if len(generated_text) < 20: print("Warning: Generation resulted in very short text.")
        return generated_text
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM during generation: {e}"); traceback.print_exc(); torch.cuda.empty_cache()
        return f"Error: GPU OOM during generation."
    except Exception as e:
        print(f"Generation Error: {e}"); traceback.print_exc()
        return f"Error: Model generation failed - {str(e)}"
# --- End model_generate ---


# <<< MODIFIED: generate_summary uses context limit from loaded model/tokenizer >>>
def generate_summary(text_chunks, grade_level_category, grade_level_desc, duration_minutes, has_math=False,
                     enable_completion_web=True, enable_deduplication_web=True):
    """Generates the final summary, potentially calling model_generate multiple times."""
    # Target word counts based on duration
    if duration_minutes == 10: min_words, max_words = 1200, 1600
    elif duration_minutes == 20: min_words, max_words = 2400, 3200
    elif duration_minutes == 30: min_words, max_words = 3600, 4500
    else: min_words, max_words = 1200, 1600 # Default to 10 min
    print(f"Targeting summary: {grade_level_desc}, {duration_minutes} mins ({min_words}-{max_words} words approx).")
    print(f"Refinement Options - Completion: {enable_completion_web}, Deduplication: {enable_deduplication_web}")

    # --- Get model context limit (handle potential inconsistencies) ---
    model_context_limit = getattr(model.config, 'max_position_embeddings', None)
    if model_context_limit is None: model_context_limit = getattr(tokenizer, 'model_max_length', None)
    if model_context_limit is None: model_context_limit = MAX_SEQ_LENGTH # Fallback
    if not isinstance(model_context_limit, int) or model_context_limit <= 512:
        model_context_limit = MAX_SEQ_LENGTH
    print(f"DEBUG (generate_summary): Using model_context_limit = {model_context_limit}")

    full_text = ' '.join(text_chunks)
    # Estimate tokens more accurately using the tokenizer
    try:
        full_text_tokens_estimate = len(tokenizer.encode(full_text))
    except Exception as e:
        print(f"Warning: Tokenizer encoding failed for length estimation: {e}. Using character count approximation.")
        full_text_tokens_estimate = len(full_text) // 3 # Rough guess

    # Define model limits and target generation length
    estimated_target_max_tokens = int(max_words * 1.3) + 200
    # Ensure generation doesn't exceed roughly half the context limit for safety, leaving buffer
    safe_generation_limit = max((model_context_limit // 2) - 150, 512) # Ensure at least 512 buffer
    max_new_tokens_summary = max(min(estimated_target_max_tokens, safe_generation_limit), 512) # Ensure min of 512 generation
    print(f"Calculated max_new_tokens for summary: {max_new_tokens_summary} (safe_gen_limit: {safe_generation_limit})")

    # Determine if single pass is feasible (leaving ample room for prompt instructions)
    prompt_instruction_buffer = 700 # Increased buffer for more detailed prompts
    required_tokens_for_single_pass = full_text_tokens_estimate + max_new_tokens_summary + prompt_instruction_buffer
    can_summarize_all_at_once = (required_tokens_for_single_pass < (model_context_limit * 0.9) and
                                 full_text_tokens_estimate < (model_context_limit * 0.6) and
                                 max_new_tokens_summary <= 4096) # Allow larger single pass generation if context fits

    initial_summary = ""
    if can_summarize_all_at_once and text_chunks:
        print("Attempting summary generation in a single pass.")
        prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
        try:
            prompt = prompt_template.format(text=full_text)
            prompt += f"\n\nIMPORTANT: Ensure the final summary is well-structured with headings/bullets and between {min_words} and {max_words} words long."
        except KeyError:
            print("Warning: Prompt template key error. Using basic prompt.")
            prompt = f"Create a detailed, well-structured summary for {grade_level_desc} ({min_words}-{max_words} words):\n{full_text}"
        initial_summary = model_generate(prompt, max_new_tokens=max_new_tokens_summary, temperature=0.6)

    elif text_chunks:
        print(f"Text too long or requires large output ({full_text_tokens_estimate} input tokens est.). Summarizing {len(text_chunks)} chunks iteratively.")
        chunk_summaries = []
        # Calculate max tokens per chunk summary - aim small to capture key points
        max_new_tokens_chunk = max(min( (model_context_limit // (len(text_chunks) + 1)) - 150, 300), 100) # Reduce size per chunk, ensure > 0 buffer
        print(f"Max new tokens per chunk summary: {max_new_tokens_chunk}")

        for i, chunk in enumerate(text_chunks):
            # print(f"  Summarizing chunk {i + 1}/{len(text_chunks)}...") # Reduce noise
            # Use a simpler prompt for chunk summarization - focus on key points
            chunk_prompt = (f"Identify and list the key points or main ideas from this text chunk ({i+1}/{len(text_chunks)}). Be concise.\n\n"
                            f"Text Chunk:\n{chunk}\n\nKey Points (bullet list):")
            chunk_summary = model_generate(chunk_prompt, max_new_tokens=max_new_tokens_chunk, temperature=0.4) # Lower temp for factual extraction

            if chunk_summary.startswith("Error:") or len(chunk_summary.split()) < 5:
                print(f"  Skipping chunk {i+1} due to generation error or short output.")
                continue
            # Basic cleaning of chunk summary (remove potential headers)
            chunk_summary = re.sub(r'^.*?Key Points.*?\n', '', chunk_summary, flags=re.IGNORECASE).strip()
            if chunk_summary: chunk_summaries.append(chunk_summary)

        if not chunk_summaries: return "Error: Failed to generate summaries for any text chunks."

        print("Consolidating chunk summaries into a structured final summary...")
        consolidation_prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
        consolidation_base_instruction = consolidation_prompt_template.split("Text to summarize:")[0]
        consolidation_prompt = (
            f"{consolidation_base_instruction}\n"
            f"You are given several summaries from text chunks. Your task is to SYNTHESIZE these points into ONE COHERENT, WELL-STRUCTURED summary for {grade_level_desc}.\n"
            f"Follow the original instructions regarding headings (##), bullet points (-), paragraphs, and the final activity/question section.\n"
            f"Organize the information logically. Do NOT just list the chunk summaries.\n"
            f"Aim for a final word count between {min_words} and {max_words} words.\n\n"
            f"Chunk Summaries (Key Points from different parts):\n\n" + "\n\n---\n\n".join(chunk_summaries) +
            f"\n\nNow, generate the final, structured, consolidated summary based on ALL the points above:"
        )
        initial_summary = model_generate(consolidation_prompt, max_new_tokens=max_new_tokens_summary, temperature=0.65)
    else: return "Error: Text processing resulted in zero chunks."

    if initial_summary.startswith("Error:"): return initial_summary
    current_summary = initial_summary
    current_word_count = len(current_summary.split())
    print(f"Initial summary generated ({current_word_count} words). Checking length.")

    # Lengthening logic
    attempts, max_lengthening_attempts = 0, 2
    while current_word_count < min_words and attempts < max_lengthening_attempts:
        print(f"Summary too short ({current_word_count} words). Attempting to elaborate (Attempt {attempts + 1}/{max_lengthening_attempts})...")
        lengthen_prompt = (f"This summary is too short. Elaborate on the existing points within the current structure...\nCurrent Summary:\n{current_summary}\n\nContinue the summary by adding more detail:")
        words_needed = min_words - current_word_count
        tokens_to_add = max(min(int(words_needed * 1.5), max_new_tokens_summary // 2, 700), 150)
        new_part = model_generate(lengthen_prompt, max_new_tokens=tokens_to_add, temperature=0.7)
        if new_part.startswith("Error:") or len(new_part.split()) < 10: print("Stopping lengthening."); break
        current_summary += "\n\n" + new_part.strip()
        current_word_count = len(current_summary.split()); attempts += 1
    if attempts == max_lengthening_attempts and current_word_count < min_words: print(f"Warning: Reached max lengthening attempts.")

    # Trimming logic
    words = current_summary.split()
    if len(words) > max_words:
        print(f"Trimming summary from {len(words)} to approximately {max_words} words.")
        activity_pattern = r'(##\s+(Activity|Practice|Thinking|Challenge|Try This|Fun Activity))'
        activity_match = re.search(activity_pattern, current_summary, re.IGNORECASE | re.MULTILINE)
        if activity_match:
            activity_start_index = activity_match.start()
            main_content = current_summary[:activity_start_index]; activity_content = current_summary[activity_start_index:]
            main_words = main_content.split()
            if len(main_words) > max_words:
                 limit_char_index = len(' '.join(main_words[:max_words])); last_sentence_end = main_content.rfind('.', 0, limit_char_index)
                 main_content = main_content[:last_sentence_end + 1] if last_sentence_end != -1 else ' '.join(main_words[:max_words]) + "..."
            current_summary = main_content.strip() + "\n\n" + activity_content.strip()
        else:
            current_summary = ' '.join(words[:max_words]); current_summary += "..." if not re.search(r'[.!?]$', current_summary) else ""
    summary = current_summary

    print("Post-processing summary...")
    processed_summary = enhanced_post_process(summary, grade_level_category,
                                              enable_completion=enable_completion_web,
                                              enable_deduplication=enable_deduplication_web)

    # Fallback activity logic
    activity_headings_pattern = r'^##\s+(Fun Activity|Practice.*|Try This|Further Thinking|Challenge|Activity)\s*$' # Simplified
    if not re.search(activity_headings_pattern, processed_summary, re.IGNORECASE | re.MULTILINE):
        print("Warning: Activity section missing. Generating fallback...")
        activity = generate_activity(processed_summary, grade_level_category, grade_level_desc)
        activity_heading_map = {"lower": "## Fun Activity", "middle": "## Try This", "higher": "## Further Thinking"}; default_heading = "## Activity Suggestion"
        if has_math: head_map_math = {"lower": "## Practice Time", "middle": "## Practice Problem", "higher": "## Challenge"}; activity_heading = head_map_math.get(grade_level_category, default_heading)
        else: activity_heading = activity_heading_map.get(grade_level_category, default_heading)
        processed_summary += f"\n\n{activity_heading}\n{activity}"
    else: print("Activity section found in generated summary.")

    final_word_count = len(processed_summary.split())
    print(f"Final summary generated ({final_word_count} words).")
    return processed_summary
# --- End generate_summary ---


# --- MODIFIED: enhanced_post_process accepts refinement flags ---
def enhanced_post_process(summary, grade_level_category, enable_completion=True, enable_deduplication=True):
    """Advanced post-processing with toggles for completion and deduplication."""
    if summary.startswith("Error:"): return summary
    print(f"Running enhanced post-processing (Completion:{enable_completion}, Dedup:{enable_deduplication})...")
    completion_calls_made = 0 # Counter

    # --- 1. Basic Cleanup & Heading Standardization ---
    expected_heading_text = "Summary" # Default
    try:
        prompt_lines=prompts[grade_level_category]["standard"].splitlines()
        head_line=next((l for l in prompt_lines if l.strip().startswith('#')), None)
        exp_head = head_line.strip().lstrip('# ').strip() if head_line else "Summary"
    except: exp_head = "Summary" # Keep default on error
    summary = re.sub(r'^\s*#+.*?(\n|$)', f'# {exp_head}\n\n', summary.strip(), count=1, flags=re.I)
    if not summary.startswith("# "): summary = f'# {exp_head}\n\n' + summary

    # --- 2. Process Lines & Structure ---
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
        if enable_completion and l_type in ["paragraph", "bullet", "numbered"] and len(content.split()) > 4:
            if not re.search(r'[.!?:]$', content) and content[0].isupper() and completion_calls_made < MAX_COMPLETION_CALLS:
                original_content = content
                content = complete_sentence(content, enable_global_toggle=enable_completion) # Pass toggle
                if content != original_content and not content.endswith(original_content + "."): completion_calls_made += 1
        if l_type in ["paragraph", "bullet", "numbered"]: # Apply casing/punctuation regardless of completion
             if content and content[0].islower() and not re.match(r'^[a-z]\s*\(', content): content = content[0].upper()+content[1:]
             if content and content[-1].isalnum(): content += '.'

        if l_type == "blank" and processed_data and processed_data[-1]["type"] == "blank": continue
        processed_data.append({"text":content, "type":l_type})

    # --- 4. Semantic Deduplication (Conditional) ---
    points_for_dedup = []; indices_map = {}
    if enable_deduplication and embedder:
        for i, data in enumerate(processed_data):
            if data["type"] in ["paragraph","bullet","numbered"] and len(data["text"].split()) > 6:
                cont = data["text"]; points_for_dedup.append(cont)
                if cont not in indices_map: indices_map[cont] = []
                indices_map[cont].append(i)
    elif not enable_deduplication: print("Skipping semantic dedup (disabled).")
    elif not embedder: print("Skipping semantic dedup (no embedder).")

    kept_indices = set(range(len(processed_data)))
    if points_for_dedup and enable_deduplication and embedder:
        print(f"Running semantic dedup on {len(points_for_dedup)} points...")
        try:
            unique_pts = remove_duplicates_semantic(points_for_dedup, batch_size=128); unique_set = set(unique_pts)
            print(f"Reduced to {len(unique_pts)} unique points.")
            indices_to_remove = set(); processed_removal = set()
            for cont, orig_indices in indices_map.items():
                if cont in processed_removal: continue
                if cont not in unique_set:
                    for index in orig_indices:
                        is_only = (index > 0 and processed_data[index-1]["type"] in ["heading","subheading"] and
                                   (index == len(processed_data)-1 or processed_data[index+1]["type"] in ["heading","subheading","blank"]))
                        if not is_only: indices_to_remove.add(index)
                processed_removal.add(cont)
            kept_indices -= indices_to_remove; print(f"Marked {len(indices_to_remove)} lines for removal.")
        except Exception as e: print(f"Warn: Dedup failed: {e}")

    # --- 5. Final Assembly ---
    final_text, last_type = "", None
    kept_data = [processed_data[i] for i in sorted(list(kept_indices))]
    for i, data in enumerate(kept_data):
        curr_type, content = data["type"], data["text"]
        # Add spacing
        if i > 0:
            if curr_type in ["heading","subheading"]: final_text += "\n\n"
            elif curr_type == "paragraph" and last_type not in ["heading","subheading","blank"]: final_text += "\n\n"
            elif curr_type != "blank" and last_type != "blank": final_text += "\n"
            elif curr_type == "blank" and last_type == "blank": continue
        # Add content with markdown
        if curr_type == "heading": final_text += f"# {content}"
        elif curr_type == "subheading": final_text += f"## {content}"
        elif curr_type == "bullet": final_text += f"- {content}"
        elif curr_type == "numbered": final_text += f"1. {content}" # Basic numbering
        elif curr_type == "paragraph": final_text += content
        last_type = curr_type
    print("Post-processing finished.")
    return final_text.strip()
# --- End enhanced_post_process ---


# --- MODIFIED: remove_duplicates_semantic accepts batch_size ---
def remove_duplicates_semantic(points, similarity_threshold=0.90, batch_size=64):
    """Removes semantically similar points using Sentence Transformers."""
    if not points or not embedder or len(points)<2: return points
    start_dedup = time.time()
    try:
        valid_pts = [p for p in points if len(p.split()) > 4]
        if not valid_pts: return points
        # print(f"Encoding {len(valid_pts)} points for dedup (batch: {batch_size})...") # Noise
        embeddings = embedder.encode(valid_pts, convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size, device=embedder.device)
        cos_sim = util.cos_sim(embeddings, embeddings)
        to_remove = set()
        for i in range(len(valid_pts)):
            if i in to_remove: continue
            for j in range(i+1, len(valid_pts)):
                if j in to_remove: continue
                if cos_sim[i][j] > similarity_threshold: to_remove.add(j)
        unique_pts = [valid_pts[i] for i in range(len(valid_pts)) if i not in to_remove]
        short_pts = [p for p in points if len(p.split()) <= 4] # Add back short points
        print(f"Semantic deduplication took {time.time() - start_dedup:.2f}s.")
        return unique_pts + short_pts
    except torch.cuda.OutOfMemoryError: print("OOM Error during dedup. Skipping."); return points
    except Exception as e: print(f"Dedup Error: {e}"); traceback.print_exc(); return points
# --- End remove_duplicates_semantic ---


# --- generate_activity (Keep as corrected before) ---
def generate_activity(summary_text, grade_level_category, grade_level_desc):
    """Fallback function to generate an activity if missing from main summary."""
    if not model or not tokenizer: return "- Review key points."
    print("Generating fallback activity suggestion...")
    act_type = {"lower":"fun activity/question","middle":"practical activity/thought question","higher":"provoking question/research idea/analysis task"}.get(grade_level_category,"activity")
    prompt = f"Suggest ONE simple, engaging {act_type} for {grade_level_desc} based on this summary context:\n...{' '.join(re.sub(r'^#.*?\n','',summary_text).strip().split()[-200:])}\n\nActivity Suggestion:"
    activity = model_generate(prompt, max_new_tokens=80, temperature=0.7)
    if activity.startswith("Error:"): print(f"Fallback activity failed: {activity}"); activity = ""
    else: activity = re.sub(r'^[\-\*\s]+','',activity.strip().replace("Activity Suggestion:","").strip()).strip(); activity = re.sub(r'\.$','',activity).strip()
    if activity:
        activity = f"- {activity[0].upper() + activity[1:]}"
        if activity[-1].isalnum(): activity += '.'
        return activity
    else: print("Warn: Failed fallback activity."); fallbacks={"lower":"- Draw!", "middle":"- Explain!", "higher":"- Find example."}; return fallbacks.get(grade_level_category, "- Review points.")
# --- End generate_activity ---


# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serves static files (CSS, JS)."""
    return send_from_directory(app.static_folder, filename)

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    """API endpoint to handle PDF summarization."""
    start_time = time.time() # Track total request time

    if 'pdfFile' not in request.files: return jsonify({"error": "No PDF file provided."}), 400
    file = request.files['pdfFile']
    if file.filename == '': return jsonify({"error": "No selected file."}), 400
    if not file.filename.lower().endswith('.pdf'): return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

    # Get form data
    try:
        grade = int(request.form.get('grade', 6))
        duration = int(request.form.get('duration', 20))
        ocr_enabled = request.form.get('ocr', 'false').lower() == 'true'
        chunk_size = int(request.form.get('chunkSize', 500))
        overlap = int(request.form.get('overlap', 50))
        sentence_completion_enabled = request.form.get('sentenceCompletion', 'false').lower() == 'true'
        deduplication_enabled = request.form.get('deduplication', 'false').lower() == 'true'
        if not (100 <= chunk_size <= 2000): chunk_size = 500
        if not (0 <= overlap <= chunk_size // 2): overlap = 50
    except ValueError: return jsonify({"error": "Invalid form data (grade, duration, chunk size, overlap must be numbers)."}), 400

    pdf_path = None
    try:
        # Save temporary file
        fd, pdf_path = tempfile.mkstemp(suffix=".pdf"); os.close(fd)
        file.save(pdf_path)
        print(f"PDF saved temporarily to: {pdf_path}")

        # Run Summarization Logic
        grade_cat, grade_desc = determine_reading_level(grade)
        print(f"Processing for: {grade_desc}, OCR:{ocr_enabled}, Chunk:{chunk_size}, Overlap:{overlap}, Comp:{sentence_completion_enabled}, Dedup:{deduplication_enabled}")
        extract_start = time.time()
        raw_text = extract_text_from_pdf(pdf_path, detect_math=True, ocr_enabled=ocr_enabled)
        print(f"Text extraction took {time.time() - extract_start:.2f}s")
        if not raw_text or len(raw_text.strip()) < 50: return jsonify({"error": "No significant text extracted. Check PDF or try OCR."}), 400

        process_start = time.time()
        has_math = detect_math_content(raw_text); print(f"Math detected: {has_math}")
        print("Cleaning text..."); cleaned = clean_text(raw_text)
        print("Splitting text..."); chunks = split_text_into_chunks(cleaned, chunk_size=chunk_size, overlap=overlap)
        if not chunks: return jsonify({"error": "Failed to split text into chunks."}), 500
        print(f"Text processing took {time.time() - process_start:.2f}s")

        print("Generating summary..."); gen_start = time.time()
        summary = generate_summary(chunks, grade_cat, grade_desc, duration, has_math,
                                   enable_completion_web=sentence_completion_enabled,
                                   enable_deduplication_web=deduplication_enabled)
        print(f"Core summary generation took {time.time() - gen_start:.2f}s")

        if summary.startswith("Error:"):
             error_msg = summary.split("Error:", 1)[1].strip(); print(f"Summarization failed: {error_msg}")
             return jsonify({"error": f"Summarization failed: {error_msg}"}), 500

        word_count = len(summary.split()); total_time = time.time() - start_time
        print(f"Summary generated. Words: {word_count}. Total time: {total_time:.2f}s")
        return jsonify({"summary": summary, "word_count": word_count, "processing_time": round(total_time, 2)})

    # Error Handling
    except FileNotFoundError as e: print(f"Error: {e}"); return jsonify({"error": str(e)}), 404
    except pytesseract.TesseractNotFoundError: err = "Tesseract OCR Engine not found/configured."; print(f"Error: {err}"); return jsonify({"error": err}), 500
    except torch.cuda.OutOfMemoryError: err = "GPU ran out of memory. Try smaller doc/duration, disable OCR."; print(f"Error: {err}"); traceback.print_exc(); return jsonify({"error": err}), 500
    except Exception as e: print(f"--- Unexpected Error ---"); print(f"Error: {str(e)}"); traceback.print_exc(); return jsonify({"error": "An unexpected server error occurred. Check logs."}), 500
    finally:
        # Cleanup temporary file
        if pdf_path and os.path.exists(pdf_path):
            try: os.remove(pdf_path); print(f"Temporary file removed: {pdf_path}")
            except Exception as e: print(f"Error removing temp file {pdf_path}: {e}")

# --- Run Flask App ---
if __name__ == '__main__':
    print("Starting Flask development server (for testing only)...")
    # Port 8501 matches Dockerfile EXPOSE and Nginx proxy_pass
    app.run(host='0.0.0.0', port=8501, debug=False, threaded=True)