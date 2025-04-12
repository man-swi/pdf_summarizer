# webie.txt
import os
import re
# pdfplumber # Not strictly needed if only using fitz, but keep if used somewhere
import pytesseract
from PIL import Image
import numpy as np
import torch
from nltk.stem import PorterStemmer
import nltk
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback
import tempfile
import io
from flask import Flask, request, render_template, jsonify, send_from_directory
import time # <<< CHANGE: Import time for performance tracking

# --- NLTK Download ---
try:
    print("Checking/downloading NLTK punkt...")
    # Attempt to download to a user-writable location first if possible
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
    # The script will try to fall back later if needed

# --- Configuration & Model Loading (Load ONCE when Flask starts) ---
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

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

OCR_RESOLUTION = 300

# Load Models
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
EMBEDDER_MODEL_NAME = 'all-MiniLM-L6-v2'
tokenizer = None
model = None
embedder = None
stemmer = PorterStemmer()
MAX_COMPLETION_CALLS = 10 # Limit sentence completions per request in post-processing

try:
    print(f"Loading LLM model: {LLM_MODEL_NAME}...")
    # <<< NOTE: Ensure you provide authentication (e.g., HF_TOKEN env var) if needed for gated models >>>
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    print("LLM Model loaded successfully!")
    # <<< ADDED DEBUGGING HERE >>>
    actual_context_limit = getattr(tokenizer, 'model_max_length', None)
    print(f"DEBUG: Tokenizer reported model_max_length: {actual_context_limit}")
    # <<< END DEBUGGING >>>

except Exception as e:
    print(f"FATAL ERROR: Could not load LLM model {LLM_MODEL_NAME}: {e}")
    traceback.print_exc()
    exit(1)

try:
    print(f"Loading Sentence Embedder: {EMBEDDER_MODEL_NAME}...")
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

# Math Symbols
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

# <<< MODIFIED: Accepts toggle flag >>>
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
    if image.mode != 'L': image = image.convert('L')
    image_array = np.array(image)
    threshold = np.mean(image_array) * 0.85
    binary_image = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_image)

def extract_text_from_pdf(pdf_path, detect_math=True, ocr_enabled=False):
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
                        ocr_text = pytesseract.image_to_string(processed_pil_image, config='--psm 6 --oem 3')
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
    math_keywords = [r'\b(equation|formula|theorem|lemma|proof|calculus|algebra|derivative|function|integral|vector|matrix|variable|constant|graph|plot|solve|calculate|measure|angle|degree)\b']
    math_symbols = r'[=><≤≥≠\+\-\*\/\^∫∑∏√∞≠≤≥±→∂∇πθλμσωαβγδε%]'
    function_notation = r'\b[a-zA-Z]\s?\([a-zA-Z0-9,\s\+\-\*\/]+\)'
    latex_delimiters = r'\$.*?\$|\\\(.*?\\\)|\\[a-zA-Z]+(\{.*?\})*'
    math_list_items = r'^\s*(\d+\.|\*|\-)\s*[=><≤≥≠\+\-\*\/\^∫∑∏√∞≠≤≥±→∂∇πθλμσωαβγδε\$\\]'
    if re.search('|'.join(math_keywords), text, re.IGNORECASE): return True
    text_sample = text[:30000] # Limit sample size for performance
    for pattern in [math_symbols, function_notation, latex_delimiters]:
        if re.search(pattern, text_sample): return True
    if re.search(math_list_items, text_sample, re.MULTILINE): return True
    return False

def clean_text(text):
    text = re.sub(r'\f', ' ', text) # Form feed
    text = re.sub(r'\[OCR_IMG.*?\[\/OCR_IMG\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\(cid:\d+\)', '', text) # PDF artifacts
    text = re.sub(r'\s+', ' ', text) # Normalize whitespace
    text = re.sub(r'([.!?])(\w)', r'\1 \2', text) # Space after punctuation
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip()) > 2 or line.strip() in ['.','!','?']]
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text) # Space before punctuation
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text) # Rejoin hyphenated words
    text = re.sub(r'\n', ' ', text) # Convert remaining newlines to spaces
    text = re.sub(r'\s+', ' ', text).strip() # Final whitespace cleanup
    return text

def split_text_into_chunks(text, chunk_size=800, overlap=50):
    try:
        # Ensure NLTK can find its data
        if not any('tokenizers/punkt' in p for p in nltk.data.path):
             print("Warning: NLTK punkt path potentially not configured. Trying default.")
        sentences = nltk.sent_tokenize(text)
        if not sentences: raise ValueError("NLTK tokenization resulted in empty list")
    except Exception as e:
        print(f"Warning: NLTK splitting failed ({e}), using regex fallback.")
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s.strip()]
        if not sentences: sentences = [text] # Handle case where no split occurs

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
    if not isinstance(grade, int) or not (1 <= grade <= 12): grade = 6
    age = grade + 5
    if 1 <= grade <= 3: level, desc = "lower", f"early elementary (grades {grade}, ~age {age}-{age+1})"
    elif 4 <= grade <= 6: level, desc = "middle", f"late elem./middle school (grades {grade}, ~age {age}-{age+1})"
    elif 7 <= grade <= 9: level, desc = "higher", f"junior high/early high (grades {grade}, ~age {age}-{age+1})"
    else: level, desc = "higher", f"high school (grades {grade}, ~age {age}-{age+1})"
    return level, desc

# Prompts dictionary (using the previously refined structured prompts)
prompts = {
    "lower": {
        "standard": (
            "You are summarizing text for a young child (grades 1-3, ages 6-8).\n"
            "Instructions:\n"
            "1. Use VERY simple words and short sentences.\n"
            "2. Explain the absolute main idea first in one sentence under '# Simple Summary'.\n"
            "3. Then, list 3-5 key points using bullet points '- '. Each point should be a full, simple sentence.\n"
            "4. Do NOT include complex details or jargon.\n"
            "5. Finish with ONE fun, simple activity related to the text under '## Fun Activity'.\n"
            "Text to summarize:\n{text}"
        ),
        "math": (
            "You are explaining a math topic to a young child (grades 1-3, ages 6-8).\n"
            "Instructions:\n"
            "1. Use very simple words, short sentences, and analogies (like counting toys).\n"
            "2. Start with '# Math Fun'.\n"
            "3. Explain the main math idea very simply under '## What We Learned'.\n"
            "4. If there are steps, list them simply under '## Steps' using numbers (1., 2.).\n"
            "5. Give one simple example with small numbers under '## Example'.\n"
            "6. Finish with ONE easy practice question or drawing task under '## Practice Time'.\n"
            "Text to explain:\n{text}"
        )
    },
    "middle": {
        "standard": (
            "You are summarizing text for a student in grades 4-6 (ages 9-11).\n"
            "Instructions:\n"
            "1. Start with the main heading '# Summary'.\n"
            "2. Identify 2-4 main topics or sections from the text.\n"
            "3. For each main topic, create a subheading using '## Topic Name'.\n"
            "4. Under each subheading, list the key information using bullet points '- '. Use clear, complete sentences.\n"
            "5. Explain any important terms simply.\n"
            "6. Ensure the summary flows logically. Avoid just listing facts.\n"
            "7. Conclude with ONE practical activity or thought question related to the text under '## Try This'.\n"
            "Text to summarize:\n{text}"
        ),
        "math": (
            "You are explaining a math concept to a student in grades 4-6 (ages 9-11).\n"
            "Instructions:\n"
            "1. Start with the heading '# Math Explained'.\n"
            "2. Explain the core math concept clearly under '## The Concept'.\n"
            "3. Provide a step-by-step example of a typical problem under '## Step-by-Step Example'. Use numbered steps (1., 2.).\n"
            "4. Briefly explain why this math is useful or where it's used under '## Why It Matters'.\n"
            "5. Conclude with ONE practice problem (include the answer separately if possible) under '## Practice Problem'.\n"
            "6. Use clear language and formatting (headings, bullets, numbered steps).\n"
            "Text to explain:\n{text}"
        )
    },
    "higher": {
        "standard": (
            "You are creating a comprehensive, well-structured summary for a high school student (grades 7-12, ages 12-18).\n"
            "Instructions:\n"
            "1. Start with the main heading '# Comprehensive Summary'.\n"
            "2. Identify key themes, arguments, sections, or concepts. Create logical subheadings ('## Theme/Section Name') for each.\n"
            "3. Under each subheading, synthesize the key information. Use paragraphs for explanation and bullet points '- ' for specific details, evidence, or examples.\n"
            "4. Use appropriate academic vocabulary but ensure clarity. Define key terms if necessary.\n"
            "5. Connect ideas logically. Analyze or evaluate points where appropriate, don't just list them.\n"
            "6. Mention real-world implications, applications, or connections if relevant under a '## Connections' subheading.\n"
            "7. Conclude with ONE thought-provoking question, research idea, or analysis task under '## Further Thinking'.\n"
            "8. Structure the entire output clearly using Markdown.\n"
            "Text to summarize:\n{text}"
        ),
        "math": (
            "You are explaining an advanced math topic for a high school student (grades 7-12, ages 12-18).\n"
            "Instructions:\n"
            "1. Start with the heading '# Advanced Math Concepts'.\n"
            "2. Provide concise definitions of key terms/concepts under '## Definitions'.\n"
            "3. Explain the core theory, logic, or theorem under '## Core Theory'. Use paragraphs and potentially bullet points.\n"
            "4. Include a non-trivial worked example demonstrating the concept or technique under '## Worked Example'. Show steps clearly.\n"
            "5. Discuss applications or connections to other fields (science, engineering, etc.) under '## Applications'.\n"
            "6. Conclude with ONE challenging problem or extension idea under '## Challenge'.\n"
            "7. Use appropriate mathematical notation (like LaTeX placeholders if present in source) and structure clearly using Markdown subheadings (##) and formatting.\n"
            "Text to explain:\n{text}"
        )
    }
}
# --- End Prompts ---

# <<< MODIFIED: model_generate with robust length calculation >>>
def model_generate(prompt_text, max_new_tokens=1024, temperature=0.6):
    if not model or not tokenizer: return "Error: LLM not available."
    current_model_device = next(model.parameters()).device

    # --- MODIFIED Length Calculation ---
    # Get actual limit, default carefully if attribute missing or invalid
    model_context_limit = getattr(tokenizer, 'model_max_length', 8192) # Default to Llama3's known limit
    if not isinstance(model_context_limit, int) or model_context_limit <= 512: # Check for valid int
        print(f"Warning: Invalid model_context_limit found ({model_context_limit}), using 8192.")
        model_context_limit = 8192
    # Ensure max_new_tokens is reasonable compared to context limit
    if max_new_tokens >= model_context_limit:
         print(f"Warning: max_new_tokens ({max_new_tokens}) >= context limit ({model_context_limit}). Reducing.")
         max_new_tokens = model_context_limit // 2 # Reduce significantly

    print(f"DEBUG: Using model_context_limit = {model_context_limit}, requested max_new_tokens = {max_new_tokens}")

    # Leave a safety buffer for special tokens etc.
    buffer_tokens = 150 # Increased buffer slightly
    # Calculate max length allowed for the prompt itself
    max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens

    # CRITICAL CHECK: Ensure max_prompt_len is positive
    if max_prompt_len <= 0:
        print(f"ERROR: Calculated max_prompt_len ({max_prompt_len}) is non-positive. "
              f"Context: {model_context_limit}, NewTokens: {max_new_tokens}, Buffer: {buffer_tokens}")
        # Handle error: Maybe reduce max_new_tokens further or return error
        needed_reduction = abs(max_prompt_len) + 10 # How much we are short + a bit
        max_new_tokens -= needed_reduction
        max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens
        if max_prompt_len <= 0 or max_new_tokens <= 50 : # Still impossible or too small generation?
             return f"Error: Generation request too large for model context limit ({model_context_limit}). Try a shorter duration."
        print(f"Warning: Reduced max_new_tokens to {max_new_tokens} to fit context.")

    # Optional: Cap max_prompt_len just in case overflow is from large positive int
    # This shouldn't be strictly necessary with the checks above but adds safety
    max_prompt_len = min(max_prompt_len, model_context_limit)

    print(f"DEBUG: Calculated max_prompt_len = {max_prompt_len} for tokenizer.")
    # --- END MODIFIED Length Calculation ---

    try:
        # Now use the calculated max_prompt_len
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(current_model_device)
        input_token_count = inputs['input_ids'].shape[1]

        # Print warning if truncation actually happened (input was longer than max_prompt_len)
        if input_token_count >= max_prompt_len:
             print(f"Warning: Prompt potentially truncated to fit max_prompt_len ({max_prompt_len} tokens).")

        start_time = time.time()
        with torch.no_grad(): # Optimization for inference
             outputs = model.generate(
                 **inputs,
                 max_new_tokens=max_new_tokens,
                 temperature=temperature,
                 pad_token_id=tokenizer.eos_token_id,
                 eos_token_id=tokenizer.eos_token_id, # Ensure EOS token is used for stopping
                 do_sample=True if temperature > 0.01 else False, # Adjust sampling threshold
                 # top_k=50, # Consider adding sampling parameters if needed
                 # top_p=0.9,
             )
        end_time = time.time()
        print(f"...Generation took {end_time - start_time:.2f} seconds.")

        # Decode output, excluding the prompt tokens
        generated_text = tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True)
        generated_text = re.sub(r'<\|eot_id\|>', '', generated_text).strip() # Clean known artifacts

        # Basic check for structure (presence of # or -)
        if not re.search(r'(^#|^- )', generated_text, re.MULTILINE):
             print("Warning: Generated text seems to lack expected Markdown structure (# or -).")
        # Check for empty or near-empty generation
        if len(generated_text) < 20:
            print("Warning: Generation resulted in very short text.")
            # Optionally return an error or a default message here
            # return "Error: Model generated insufficient content."

        return generated_text
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OutOfMemoryError during model generation: {e}")
        traceback.print_exc()
        # Attempt to clear cache and re-raise or return error
        torch.cuda.empty_cache()
        return f"Error: Model generation failed - CUDA Out of Memory. Try reducing complexity or document size."
    except Exception as e:
        print(f"Error during model generation: {e}")
        traceback.print_exc()
        return f"Error: Model generation failed - {str(e)}"
# --- End model_generate ---

# <<< MODIFIED: generate_summary accepts refinement flags & uses corrected context limit >>>
def generate_summary(text_chunks, grade_level_category, grade_level_desc, duration_minutes, has_math=False,
                     enable_completion_web=True, enable_deduplication_web=True):
    """Generate a structured summary aiming for target length and clarity, respecting refinement flags."""
    # Target word counts based on duration
    if duration_minutes == 10: min_words, max_words = 1200, 1600
    elif duration_minutes == 20: min_words, max_words = 2400, 3200
    elif duration_minutes == 30: min_words, max_words = 3600, 4500
    else: min_words, max_words = 1200, 1600 # Default to 10 min
    print(f"Targeting summary: {grade_level_desc}, {duration_minutes} mins ({min_words}-{max_words} words approx).")
    print(f"Refinement Options - Completion: {enable_completion_web}, Deduplication: {enable_deduplication_web}")

    full_text = ' '.join(text_chunks)
    # Estimate tokens more accurately using the tokenizer
    try:
        full_text_tokens_estimate = len(tokenizer.encode(full_text))
    except Exception as e:
        print(f"Warning: Tokenizer encoding failed for length estimation: {e}. Using character count approximation.")
        full_text_tokens_estimate = len(full_text) // 3 # Rough guess

    # --- MODIFIED Context Limit Calculation ---
    # Get actual model context limit (use same logic as in model_generate)
    model_context_limit = getattr(tokenizer, 'model_max_length', 8192)
    if not isinstance(model_context_limit, int) or model_context_limit <= 512:
         model_context_limit = 8192
    print(f"DEBUG (generate_summary): Using model_context_limit = {model_context_limit}")
    # --- END MODIFIED ---

    # Define model limits and target generation length
    # Estimate max tokens needed for summary (e.g., 1.3 tokens/word) + buffer
    estimated_target_max_tokens = int(max_words * 1.3) + 200
    # Ensure generation doesn't exceed roughly half the context limit for safety, leaving buffer
    safe_generation_limit = max((model_context_limit // 2) - 150, 512) # Ensure at least 512 buffer
    max_new_tokens_summary = max(min(estimated_target_max_tokens, safe_generation_limit), 512) # Ensure min of 512 generation

    print(f"Calculated max_new_tokens for summary: {max_new_tokens_summary} (safe_gen_limit: {safe_generation_limit})")

    # Determine if single pass is feasible (leaving ample room for prompt instructions)
    prompt_instruction_buffer = 700 # Increased buffer for more detailed prompts
    required_tokens_for_single_pass = full_text_tokens_estimate + max_new_tokens_summary + prompt_instruction_buffer

    # Be more conservative about single pass if estimated tokens are high
    can_summarize_all_at_once = (required_tokens_for_single_pass < (model_context_limit * 0.9) and
                                 full_text_tokens_estimate < (model_context_limit * 0.6) and
                                 max_new_tokens_summary <= 2048)

    initial_summary = ""
    if can_summarize_all_at_once and text_chunks:
        print("Attempting summary generation in a single pass.")
        prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
        try:
            prompt = prompt_template.format(text=full_text)
            # Add explicit instruction about length (this might be redundant with prompt itself)
            prompt += f"\n\nIMPORTANT: Ensure the final summary is well-structured with headings/bullets and between {min_words} and {max_words} words long."
        except KeyError:
            print("Warning: Prompt template key error. Using basic prompt.")
            prompt = f"Create a detailed, well-structured summary for {grade_level_desc} ({min_words}-{max_words} words):\n{full_text}"

        initial_summary = model_generate(prompt, max_new_tokens=max_new_tokens_summary, temperature=0.6) # Slightly lower temp for structure

    elif text_chunks:
        print(f"Text too long or requires large output ({full_text_tokens_estimate} input tokens est.). Summarizing {len(text_chunks)} chunks iteratively.")
        chunk_summaries = []
        # Calculate max tokens per chunk summary - aim small to capture key points
        max_new_tokens_chunk = max(min( (model_context_limit // (len(text_chunks) + 1)) - 150, 300), 100) # Reduce size per chunk, ensure > 0 buffer
        print(f"Max new tokens per chunk summary: {max_new_tokens_chunk}")

        for i, chunk in enumerate(text_chunks):
            print(f"  Summarizing chunk {i + 1}/{len(text_chunks)}...")
            # Use a simpler prompt for chunk summarization - focus on key points
            chunk_prompt = (f"Identify and list the key points or main ideas from this text chunk ({i+1}/{len(text_chunks)}). Be concise.\n\n"
                            f"Text Chunk:\n{chunk}\n\nKey Points (bullet list):")
            chunk_summary = model_generate(chunk_prompt, max_new_tokens=max_new_tokens_chunk, temperature=0.4) # Lower temp for factual extraction

            if chunk_summary.startswith("Error:") or len(chunk_summary.split()) < 5:
                print(f"  Skipping chunk {i+1} due to generation error or short output.")
                continue

            # Basic cleaning of chunk summary (remove potential headers)
            chunk_summary = re.sub(r'^.*?Key Points.*?\n', '', chunk_summary, flags=re.IGNORECASE).strip()
            if chunk_summary:
                chunk_summaries.append(chunk_summary)

        if not chunk_summaries:
            return "Error: Failed to generate summaries for any text chunks."

        print("Consolidating chunk summaries into a structured final summary...")
        # Stronger consolidation prompt focusing on synthesis and structure
        consolidation_prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
        # We replace the {text} placeholder with instructions for consolidation
        consolidation_base_instruction = consolidation_prompt_template.split("Text to summarize:")[0] # Get the instruction part
        consolidation_prompt = (
            f"{consolidation_base_instruction}\n"
            f"You are given several summaries from text chunks. Your task is to SYNTHESIZE these points into ONE COHERENT, WELL-STRUCTURED summary for {grade_level_desc}.\n"
            f"Follow the original instructions regarding headings (##), bullet points (-), paragraphs, and the final activity/question section.\n"
            f"Organize the information logically. Do NOT just list the chunk summaries.\n"
            f"Aim for a final word count between {min_words} and {max_words} words.\n\n"
            f"Chunk Summaries (Key Points from different parts):\n\n"
            + "\n\n---\n\n".join(chunk_summaries) +
            f"\n\nNow, generate the final, structured, consolidated summary based on ALL the points above:"
        )

        # Use the full calculated max_new_tokens for the consolidation step
        initial_summary = model_generate(consolidation_prompt, max_new_tokens=max_new_tokens_summary, temperature=0.65) # Slightly higher temp for synthesis

    else:
        return "Error: Text processing resulted in zero chunks."

    # Check for generation errors before proceeding
    if initial_summary.startswith("Error:"):
        return initial_summary # Propagate the error

    current_summary = initial_summary
    current_word_count = len(current_summary.split())
    print(f"Initial summary generated ({current_word_count} words). Checking length requirement ({min_words}-{max_words}).")

    # Simplified Lengthening - fewer attempts, focused prompt
    attempts, max_lengthening_attempts = 0, 2 # Reduce max attempts
    while current_word_count < min_words and attempts < max_lengthening_attempts:
        print(f"Summary too short ({current_word_count} words). Attempting to elaborate (Attempt {attempts + 1}/{max_lengthening_attempts})...")
        # Prompt to elaborate *within the existing structure*
        lengthen_prompt = (
            f"This summary is too short. Elaborate on the existing points within the current structure.\n"
            f"Add more detail, explanation, or examples under the appropriate headings or bullet points.\n"
            f"Maintain the target audience ({grade_level_desc}) and style.\n\n"
            f"Current Summary:\n{current_summary}\n\n"
            f"Continue the summary by adding more detail:"
        )
        # Calculate remaining words needed and estimate tokens
        words_needed = min_words - current_word_count
        # Limit added tokens per attempt, ensure minimum addition
        tokens_to_add = max(min(int(words_needed * 1.5), max_new_tokens_summary // 2, 700), 150)

        new_part = model_generate(lengthen_prompt, max_new_tokens=tokens_to_add, temperature=0.7) # Higher temp for creative elaboration

        if new_part.startswith("Error:") or len(new_part.split()) < 10:
            print("Stopping lengthening due to error or insufficient generation.")
            break

        # Append the new part (post-processing will handle structure)
        current_summary += "\n\n" + new_part.strip() # Add space before appending
        current_word_count = len(current_summary.split())
        attempts += 1

    if attempts == max_lengthening_attempts and current_word_count < min_words:
        print(f"Warning: Reached max lengthening attempts, summary might still be too short ({current_word_count}/{min_words} words).")

    # Trim if over max words (simple trim for now)
    words = current_summary.split()
    if len(words) > max_words:
        print(f"Trimming summary from {len(words)} to approximately {max_words} words.")
        # Try to trim whole sentences from the end, excluding potential activity section
        activity_pattern = r'(##\s+(Activity|Practice|Thinking|Challenge|Try This|Fun Activity))'
        activity_match = re.search(activity_pattern, current_summary, re.IGNORECASE | re.MULTILINE)
        if activity_match:
            activity_start_index = activity_match.start()
            main_content = current_summary[:activity_start_index]
            activity_content = current_summary[activity_start_index:]
            main_words = main_content.split()
            if len(main_words) > max_words:
                 # Find last sentence end before max_words limit in main content
                 limit_char_index = len(' '.join(main_words[:max_words]))
                 last_sentence_end = main_content.rfind('.', 0, limit_char_index)
                 if last_sentence_end != -1:
                     main_content = main_content[:last_sentence_end + 1]
                 else: # Fallback to word trim
                     main_content = ' '.join(main_words[:max_words]) + "..."
            current_summary = main_content.strip() + "\n\n" + activity_content.strip()
        else:
            # Fallback to simple word trim if no activity section found
            current_summary = ' '.join(words[:max_words])
            if not re.search(r'[.!?]$', current_summary): current_summary += "..." # Add ellipsis if cut mid-sentence

    summary = current_summary

    # Post-process for final structure and cleanup
    print("Post-processing summary...")
    # <<< MODIFIED: Pass refinement flags to post-processing >>>
    processed_summary = enhanced_post_process(summary, grade_level_category,
                                              enable_completion=enable_completion_web,
                                              enable_deduplication=enable_deduplication_web)

    # Check for activity section generated by the main prompt, add fallback
    activity_headings_pattern = r'^##\s+(Fun Activity|Practice Time|Try This|Practice Problem|Further Thinking|Challenge|Activity)\s*$'
    if not re.search(activity_headings_pattern, processed_summary, re.IGNORECASE | re.MULTILINE):
        print("Warning: Activity section seems missing from LLM output. Generating fallback...")
        activity = generate_activity(processed_summary, grade_level_category, grade_level_desc) # Fallback call
        # Determine correct heading based on grade/type
        activity_heading_map = {"lower": "## Fun Activity", "middle": "## Try This", "higher": "## Further Thinking"}
        default_heading = "## Activity Suggestion"
        # Check if math prompt was used (structure differs)
        if has_math:
            head_map_math = {"lower": "## Practice Time", "middle": "## Practice Problem", "higher": "## Challenge"}
            activity_heading = head_map_math.get(grade_level_category, default_heading)
        else:
            activity_heading = activity_heading_map.get(grade_level_category, default_heading)

        processed_summary += f"\n\n{activity_heading}\n{activity}"
    else:
        print("Activity section found in generated summary.")

    final_word_count = len(processed_summary.split())
    print(f"Final summary generated ({final_word_count} words).")
    return processed_summary
# --- End generate_summary ---

# <<< MODIFIED: enhanced_post_process accepts refinement flags >>>
def enhanced_post_process(summary, grade_level_category, enable_completion=True, enable_deduplication=True):
    """Advanced post-processing with toggles for completion and deduplication."""
    if summary.startswith("Error:"): return summary
    print(f"Running enhanced post-processing (Completion:{enable_completion}, Dedup:{enable_deduplication})...")
    completion_calls_made = 0 # Counter

    # --- 1. Basic Cleanup & Heading Standardization ---
    expected_heading_text = "Summary" # Default
    try:
        # Extract expected heading from the prompt definition
        prompt_lines = prompts[grade_level_category]["standard"].splitlines()
        heading_line = next((line for line in prompt_lines if line.strip().startswith('#')), None)
        if heading_line:
             expected_heading_text = heading_line.strip().lstrip('# ').strip()
    except Exception:
        # print("Warning: Could not determine expected heading from prompts.") # Reduce noise
        pass # Keep default

    # Replace any existing top-level heading with the standardized one
    summary = re.sub(r'^\s*#+.*?(\n|$)', f'# {expected_heading_text}\n\n', summary.strip(), count=1, flags=re.IGNORECASE)
    if not summary.startswith("# "):
         summary = f'# {expected_heading_text}\n\n' + summary # Ensure it starts with a heading

    # --- 2. Process Lines and Identify Structure ---
    lines = summary.split('\n')
    processed_lines_data = []
    seen_content_fragments = set() # For basic duplicate check during initial pass

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            # Keep single blank lines between paragraphs/sections, remove multiple
            if processed_lines_data and processed_lines_data[-1]["type"] != "blank":
                 processed_lines_data.append({"text": "", "type": "blank"})
            continue

        line_type = "paragraph" # Default
        content = stripped_line
        is_heading = False
        is_bullet = False

        if stripped_line.startswith('## '):
            line_type = "subheading"
            content = stripped_line[3:].strip()
            is_heading = True
        elif stripped_line.startswith('# '): # Should only be the main one now
            line_type = "heading"
            content = stripped_line[2:].strip()
            is_heading = True
        elif stripped_line.startswith('- '):
            line_type = "bullet"
            content = stripped_line[2:].strip()
            is_bullet = True
        elif re.match(r'^\d+\.\s+', stripped_line):
            line_type = "numbered"
            content = re.sub(r'^\d+\.\s+', '', stripped_line).strip()
            is_bullet = True # Treat like bullet for completion/dedup logic

        if not content: continue # Skip empty lines resulting from stripping tags

        # Basic duplicate check (first ~10 words) - less aggressive now
        content_key = ' '.join(content.lower().split()[:10])
        if not is_heading and content_key in seen_content_fragments and len(content.split()) < 15: # Only skip short duplicates
            # print(f"Skipping potential short duplicate: {content[:50]}...") # Reduce noise
            continue
        if not is_heading: seen_content_fragments.add(content_key)

        # --- 3. Sentence Completion (Conditional & Limited) ---
        if line_type in ["paragraph", "bullet", "numbered"] and len(content.split()) > 4:
            # <<< MODIFIED: Check enable_completion flag and limit calls >>>
            if enable_completion and not re.search(r'[.!?:]$', content) and content[0].isupper() and completion_calls_made < MAX_COMPLETION_CALLS:
                original_content = content
                # <<< MODIFIED: Pass toggle to complete_sentence >>>
                content = complete_sentence(content, enable_global_toggle=enable_completion)
                # Count only if completion actually changed the text significantly
                if content != original_content and not content.endswith(original_content + "."):
                    completion_calls_made += 1
            # Sentence casing and punctuation (always apply if applicable)
            if content and content[0].islower() and not re.match(r'^[a-z]\s*\(', content): # Avoid messing with function notation
                 content = content[0].upper() + content[1:]
            if content and content[-1].isalnum():
                 content += '.'

        # Add processed data
        # Prevent adding blank line immediately after another blank line
        if line_type == "blank" and processed_lines_data and processed_lines_data[-1]["type"] == "blank":
            continue

        processed_lines_data.append({"text": content, "type": line_type})

    # --- 4. Semantic Deduplication (Conditional) ---
    points_for_dedup = []
    indices_map = {}
    # <<< MODIFIED: Check enable_deduplication flag >>>
    if enable_deduplication and embedder:
        for i, data in enumerate(processed_lines_data):
            # Only deduplicate paragraph content or bullet/numbered list items
            if data["type"] in ["paragraph", "bullet", "numbered"] and len(data["text"].split()) > 6: # Min length for meaningful comparison
                content_to_check = data["text"]
                points_for_dedup.append(content_to_check)
                # Store the first index found for this content
                if content_to_check not in indices_map:
                     indices_map[content_to_check] = []
                indices_map[content_to_check].append(i)
    # <<< MODIFIED: Print reason for skipping >>>
    elif not enable_deduplication:
        print("Skipping semantic dedup (disabled by request).")
    elif not embedder:
        print("Skipping semantic dedup (embedder not loaded).")

    kept_indices = set(range(len(processed_lines_data)))
    if points_for_dedup and enable_deduplication and embedder: # Check flags again before processing
        print(f"Running semantic duplicate removal on {len(points_for_dedup)} points...")
        try:
            # Pass batch_size for potential optimization inside function if implemented
            unique_points_content = remove_duplicates_semantic(points_for_dedup, batch_size=128)
            unique_content_set = set(unique_points_content)
            print(f"Reduced to {len(unique_points_content)} unique points semantically.")

            indices_to_remove = set()
            processed_for_removal = set()

            # Iterate through original points and decide which indices to remove
            for content, original_indices in indices_map.items():
                 if content in processed_for_removal: continue

                 if content not in unique_content_set:
                     # Mark all occurrences of this non-unique content for removal
                     for index in original_indices:
                          # Basic check: don't remove if it's the only content under a heading
                          is_only_content = False
                          if index > 0 and processed_lines_data[index-1]["type"] in ["heading", "subheading"]:
                              if index == len(processed_lines_data)-1 or processed_lines_data[index+1]["type"] in ["heading", "subheading", "blank"]:
                                   is_only_content = True
                          if not is_only_content:
                            indices_to_remove.add(index)
                 processed_for_removal.add(content)

            kept_indices -= indices_to_remove
            print(f"Marked {len(indices_to_remove)} lines for removal based on semantic duplicates.")

        except Exception as e:
            print(f"Warning: Semantic deduplication failed: {e}. Skipping.")
            traceback.print_exc() # Debugging

    # --- 5. Final Assembly with Markdown Spacing ---
    final_summary_text = ""
    last_line_type = None
    kept_data = [processed_lines_data[i] for i in sorted(list(kept_indices))]

    for i, data in enumerate(kept_data):
        current_line_type = data["type"]
        content = data["text"]

        # Determine spacing based on current and previous line type
        if i > 0:
            if current_line_type in ["heading", "subheading"]:
                # Always add extra newline before headings (unless it's the very first line)
                final_summary_text += "\n\n"
            elif current_line_type == "paragraph" and last_line_type not in ["heading", "subheading", "blank"]:
                # Add blank line before a new paragraph if previous wasn't heading/blank
                 final_summary_text += "\n\n"
            elif current_line_type != "blank" and last_line_type != "blank":
                 # Single newline between most other elements (e.g., bullets)
                 final_summary_text += "\n"
            # Handle blank lines: only one consecutive blank line
            elif current_line_type == "blank" and last_line_type == "blank":
                 continue # Skip extra blank lines

        # Add formatted line content
        if current_line_type == "heading": final_summary_text += f"# {content}"
        elif current_line_type == "subheading": final_summary_text += f"## {content}"
        elif current_line_type == "bullet": final_summary_text += f"- {content}"
        elif current_line_type == "numbered": final_summary_text += f"1. {content}" # Use generic number for now, re-numbering is complex
        elif current_line_type == "paragraph": final_summary_text += content
        # Ignore 'blank' type here, spacing is handled above

        last_line_type = current_line_type

    print("Post-processing finished.")
    return final_summary_text.strip()
# --- End enhanced_post_process ---

# <<< MODIFIED: remove_duplicates_semantic accepts batch_size >>>
def remove_duplicates_semantic(points, similarity_threshold=0.90, batch_size=64):
    """Removes semantically similar points using Sentence Transformers."""
    if not points or not embedder: return points
    if len(points) < 2: return points
    unique_points = []
    start_time = time.time() # Track dedup time
    try:
        # Filter very short points unlikely to be meaningful duplicates
        valid_points = [p for p in points if len(p.split()) > 4]
        if not valid_points: return points # Return original if no valid points left

        # print(f"Encoding {len(valid_points)} points for semantic deduplication (batch size: {batch_size})...") # Reduce noise
        embeddings = embedder.encode(
            valid_points,
            convert_to_tensor=True,
            show_progress_bar=False, # Often too fast to show progress
            batch_size=batch_size, # Use provided batch size
            device=embedder.device # Use embedder's device
        )
        # print("Embeddings generated.") # Reduce noise

        # Use pairwise cosine similarity calculation.
        cos_sim_matrix = util.cos_sim(embeddings, embeddings)

        # Keep track of indices to remove
        indices_to_remove = set()
        for i in range(len(valid_points)):
            if i in indices_to_remove: continue # Already marked for removal
            for j in range(i + 1, len(valid_points)):
                if j in indices_to_remove: continue
                # If similarity is above threshold, mark the second one (j) for removal
                if cos_sim_matrix[i][j] > similarity_threshold:
                    indices_to_remove.add(j)

        # Create the list of unique points based on kept indices
        unique_points = [valid_points[i] for i in range(len(valid_points)) if i not in indices_to_remove]

        # Add back the points that were initially filtered out (too short)
        short_points = [p for p in points if len(p.split()) <= 4]
        final_unique_points = unique_points + short_points

        print(f"Semantic deduplication took {time.time() - start_time:.2f}s.") # Report time
        return final_unique_points

    except torch.cuda.OutOfMemoryError:
        print("CUDA OutOfMemoryError during semantic duplicate removal. Skipping deduplication.")
        return points # Fallback to original points
    except Exception as e:
        print(f"Error during semantic duplicate removal: {e}.")
        traceback.print_exc()
        return points # Fallback to original points on error
# --- End remove_duplicates_semantic ---

def generate_activity(summary_text, grade_level_category, grade_level_desc):
    """Fallback function to generate an activity if missing from main summary."""
    if not model or not tokenizer: return "- Review key points."
    print("Generating fallback activity suggestion...")

    # Determine the right type of activity instruction
    activity_type = "activity"
    if grade_level_category == "lower": activity_type = "fun activity or practice question"
    elif grade_level_category == "middle": activity_type = "practical activity or thought question"
    elif grade_level_category == "higher": activity_type = "thought-provoking question, research idea, or analysis task"

    activity_prompt_template = (
        "Based on the summary context below, suggest ONE simple and engaging {activity_type} suitable for {grade_desc}.\n"
        "The activity must be directly related to the summary's main topics.\n"
        "Describe the activity clearly in one or two sentences.\n\n"
        "Summary Context (End Portion):\n...{summary_snippet}\n\n"
        "Activity Suggestion:" )
    # Use a smaller snippet to keep prompt short
    summary_snippet = re.sub(r'^#.*?\n', '', summary_text).strip() # Remove heading
    summary_snippet = ' '.join(summary_snippet.split()[-200:]) # Last ~200 words

    prompt = activity_prompt_template.format(
        activity_type=activity_type,
        grade_desc=grade_level_desc,
        summary_snippet=summary_snippet # Corrected variable name
    )

    activity = model_generate(prompt, max_new_tokens=80, temperature=0.75) # Slightly higher temp

    if activity.startswith("Error:"):
         print(f"Fallback activity generation failed: {activity}")
         activity = "" # Reset activity string if generation failed
    else:
        # Clean the output more thoroughly
        activity = activity.strip().replace("Activity Suggestion:", "").strip()
        activity = re.sub(r'^[\-\*\s]+', '', activity) # Remove leading list markers/spaces
        activity = re.sub(r'\.$', '', activity).strip() # Remove trailing period for consistency

    if activity:
        # Ensure sentence casing and add bullet point
        activity = activity[0].upper() + activity[1:]
        if not activity.startswith('- '): activity = f"- {activity}"
        if activity[-1].isalnum(): activity += '.'
        return activity
    else:
        print("Warning: Failed to generate fallback activity text.")
        # Fallback based on grade level
        fallbacks = {
            "lower": "- Draw a picture about the main idea!",
            "middle": "- Try explaining the most interesting part to someone.",
            "higher": "- Find one real-world example related to this topic."
        }
        return fallbacks.get(grade_level_category, "- Review the main points of the summary.")
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
    # Use app.static_folder which is configured during Flask initialization
    return send_from_directory(app.static_folder, filename)

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    """API endpoint to handle PDF summarization."""
    start_time = time.time() # Track total request time

    if 'pdfFile' not in request.files:
        return jsonify({"error": "No PDF file provided."}), 400

    file = request.files['pdfFile']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

    # --- Get form data ---
    try:
        grade = int(request.form.get('grade', 6))
        duration = int(request.form.get('duration', 20))
        # Handle checkbox value correctly
        ocr_enabled = request.form.get('ocr', 'false').lower() == 'true'
        chunk_size = int(request.form.get('chunkSize', 500))
        overlap = int(request.form.get('overlap', 50))
        # <<< NEW: Get refinement options from form >>>
        sentence_completion_enabled = request.form.get('sentenceCompletion', 'false').lower() == 'true'
        deduplication_enabled = request.form.get('deduplication', 'false').lower() == 'true'

        # Basic validation for chunk/overlap
        if not (100 <= chunk_size <= 2000): chunk_size = 500
        if not (0 <= overlap <= chunk_size // 2): overlap = 50

    except ValueError:
        return jsonify({"error": "Invalid form data (grade, duration, chunk size, overlap must be numbers)."}), 400

    # --- Save temporary file ---
    pdf_path = None
    try:
        # Use secure temp file creation
        fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd) # Close file descriptor, we'll reopen with file object
        file.save(pdf_path) # Save uploaded file object to temp file path
        print(f"PDF saved temporarily to: {pdf_path}")

        # --- Run Summarization Logic ---
        grade_level_category, grade_level_desc = determine_reading_level(grade)
        print(f"Processing for: {grade_level_desc}")
        print(f"OCR: {ocr_enabled}, Chunk: {chunk_size}, Overlap: {overlap}, Completion: {sentence_completion_enabled}, Dedup: {deduplication_enabled}")


        extract_start = time.time()
        # Pass ocr_enabled flag here
        raw_text = extract_text_from_pdf(pdf_path, detect_math=True, ocr_enabled=ocr_enabled)
        print(f"Text extraction took {time.time() - extract_start:.2f}s")

        if not raw_text or len(raw_text.strip()) < 50:
            # Clearer error message
            return jsonify({"error": "No significant text extracted from PDF. Check PDF content or try enabling OCR if it contains images."}), 400

        process_start = time.time()
        has_math = detect_math_content(raw_text)
        print(f"Math content detected: {has_math}")

        print("Cleaning text...")
        cleaned_text = clean_text(raw_text)

        print("Splitting text...")
        chunks = split_text_into_chunks(cleaned_text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return jsonify({"error": "Failed to split text into manageable chunks."}), 500
        print(f"Text processing (clean/chunk) took {time.time() - process_start:.2f}s")


        print("Generating summary...")
        gen_start = time.time()
        # <<< MODIFIED: Pass refinement flags to generate_summary >>>
        summary = generate_summary(chunks, grade_level_category, grade_level_desc, duration, has_math,
                                   enable_completion_web=sentence_completion_enabled,
                                   enable_deduplication_web=deduplication_enabled)
        print(f"Core summary generation took {time.time() - gen_start:.2f}s")

        # Check if summary generation itself returned an error string
        if summary.startswith("Error:"):
             error_message = summary.split("Error:", 1)[1].strip()
             print(f"Summarization process failed: {error_message}")
             # Provide more specific error if model related
             return jsonify({"error": f"Summarization failed: {error_message}"}), 500

        word_count = len(summary.split())
        total_time = time.time() - start_time
        print(f"Summary generated. Word count: {word_count}. Total request time: {total_time:.2f}s")

        return jsonify({"summary": summary, "word_count": word_count, "processing_time": round(total_time, 2)})

    except FileNotFoundError as e:
         print(f"Error: {e}")
         return jsonify({"error": str(e)}), 404
    except pytesseract.TesseractNotFoundError:
         err_msg = "Tesseract OCR Engine not found or configured correctly on the server. Please install it or disable OCR."
         print(f"Error: {err_msg}")
         return jsonify({"error": err_msg}), 500
    except torch.cuda.OutOfMemoryError:
         err_msg = "GPU ran out of memory during processing. Try a smaller document, disable OCR, or use a smaller summarization duration/model."
         print(f"Error: {err_msg}")
         traceback.print_exc()
         return jsonify({"error": err_msg}), 500
    except Exception as e:
        print(f"--- An Unexpected Error Occurred ---")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        # Provide a more generic error message to the user
        return jsonify({"error": "An unexpected error occurred on the server while processing the PDF. Please check the server logs for details."}), 500
    finally:
        # --- Clean up temporary file ---
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                print(f"Temporary file removed: {pdf_path}")
            except Exception as e:
                print(f"Error removing temporary file {pdf_path}: {e}")


# --- Run Flask App ---
# This block is primarily for local development testing.
# Gunicorn will directly import and run the 'app' object in production.
if __name__ == '__main__':
    print("Starting Flask development server (for testing only)...")
    # Use host='0.0.0.0' to make it accessible on your network
    # Use debug=False for production or testing performance
    # Set threaded=True or use a production server like gunicorn/waitress for handling multiple requests
    # Port 8501 is used here to match the Dockerfile EXPOSE and Nginx proxy_pass
    app.run(host='0.0.0.0', port=8501, debug=False, threaded=True)