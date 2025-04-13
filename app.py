# app.py (Reverted to Text-Only: Llama-3.1-8B-Instruct)

# --- START FIX: Set multiprocessing start method ---
import multiprocessing as mp
# Try setting start_method to 'spawn' - MUST be done before importing torch/transformers
try:
    mp.set_start_method('spawn', force=True)
    print("INFO: Multiprocessing start method set to 'spawn'.")
except RuntimeError as e:
    # This might happen if it's already been set or if called inappropriately
    # Often safe to ignore if it's already set, but log a warning
    print(f"WARN: Could not set multiprocessing start_method: {e}")
# --- END FIX ---

import os
import re
import traceback
import tempfile
import time
import io
import pytesseract # Keep for OCR text extraction from images
import torch
from flask import Flask, request, render_template, jsonify, send_from_directory
from logging.config import dictConfig

# Import model loading classes, tokenizer, and quantization config
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Back to AutoTokenizer
from sentence_transformers import SentenceTransformer

# Import utility modules
import pdf_utils
import text_utils
import llm_utils

# --- Configure Flask Logging ---
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s (%(lineno)d): %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.logger.info("--- Starting Flask App Setup ---")

# --- Tesseract Configuration (pdf_utils) ---
# Ensure Tesseract is installed if OCR toggle is used

# --- Device Setup ---
if torch.cuda.is_available():
    app.logger.info(f"PyTorch reports CUDA is available. Device count: {torch.cuda.device_count()}")
else:
    app.logger.warning("PyTorch reports CUDA is NOT available. Model will run on CPU (very slow).")

# --- Model Configuration ---
LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" # Changed model name
EMBEDDER_MODEL_NAME = 'all-MiniLM-L6-v2' # Text embedder for deduplication

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # Good for Ampere (A10G)
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
app.logger.info(f"Using BitsAndBytes quantization config: {quantization_config}")

# --- Load Models and Tokenizer ---
model = None
tokenizer = None # Changed back from processor
embedder = None

try:
    # --- CRITICAL CHANGE: Load AutoTokenizer ---
    app.logger.info(f"Loading LLM tokenizer: {LLM_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    app.logger.info("Tokenizer loaded.")
    # --- END CRITICAL CHANGE ---

    app.logger.info(f"Loading LLM model: {LLM_MODEL_NAME} with quantization (Offloading DISABLED)...")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto", # Let accelerate handle placement
        torch_dtype=torch.bfloat16,
        # trust_remote_code=False # Usually not needed for standard instruct models like Llama 3.1
    )
    app.logger.info("LLM Model loaded successfully!")

    if torch.cuda.is_available():
         try:
             if any(p.device.type == 'cuda' for p in model.parameters()):
                 allocated = torch.cuda.memory_allocated(0)/1024**2
                 reserved = torch.cuda.memory_reserved(0)/1024**2
                 app.logger.info(f"GPU Memory after LLM Load: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")
             else: app.logger.warning("Model parameters not found on CUDA device after load.")
         except Exception as e: app.logger.warning(f"Could not get GPU memory info: {e}")
    else: app.logger.info("Model loaded on CPU.")

except Exception as e:
    app.logger.fatal(f"CRITICAL ERROR: Could not load LLM tokenizer or model {LLM_MODEL_NAME}: {e}")
    app.logger.fatal(traceback.format_exc())
    if "out of memory" in str(e).lower() and torch.cuda.is_available():
        app.logger.error("CUDA Out of Memory during model/tokenizer loading!")
        torch.cuda.empty_cache()
    # Removed trust_remote_code specific message as it's likely false now
    elif isinstance(e, ImportError):
         app.logger.error(f"ImportError: A required library might be missing. Check requirements.txt. Error: {e}")
    exit(1)

# Load Sentence Embedder (Text Deduplication)
try:
    app.logger.info(f"Loading Sentence Embedder: {EMBEDDER_MODEL_NAME}...")
    embedder_device = 'cpu'
    if torch.cuda.is_available():
        try:
            # Check VRAM roughly AFTER LLM load (8B model uses less than 11B)
            free_mem, _ = torch.cuda.mem_get_info(0); free_mem_mb = free_mem / 1024**2
            app.logger.info(f"GPU Memory Free for Embedder: {free_mem_mb:.2f} MB")
            if free_mem_mb > 1500: # Allow slightly more room
                embedder_device = 'cuda'
            else: app.logger.warning("Low free VRAM, placing embedder on CPU.")
        except Exception as e: app.logger.warning(f"Could not check VRAM for embedder placement (CPU default): {e}")

    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device=embedder_device)
    app.logger.info(f"Embedder loaded onto device: '{embedder_device}'")
except Exception as e:
    app.logger.error(f"Could not load Sentence Transformer {EMBEDDER_MODEL_NAME}: {e}")
    embedder = None
    app.logger.warning("Embedder failed. Semantic deduplication disabled.")

# --- Initialize LLM Utilities Module ---
# --- CRITICAL CHANGE: Pass tokenizer ---
if model and tokenizer:
    llm_utils.initialize_globals(model, tokenizer, embedder) # Pass tokenizer
else:
    app.logger.fatal("LLM Model or Tokenizer failed to load. Cannot initialize llm_utils.")
    exit(1)
# --- END CRITICAL CHANGE ---

app.logger.info("--- Flask App Setup Complete ---")

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    app.logger.info(f"Received request for /summarize from {request.remote_addr}")
    start_time = time.time()

    if 'pdfFile' not in request.files:
        app.logger.warning("No 'pdfFile' in request.")
        return jsonify({"error": "No PDF file provided."}), 400
    file = request.files['pdfFile']
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        app.logger.warning(f"Invalid/missing filename or type: {file.filename}")
        return jsonify({"error": "Invalid file type or missing filename. Please upload a PDF."}), 400

    # --- Get form data ---
    try:
        grade = int(request.form.get('grade', 6))
        duration = int(request.form.get('duration', 20))
        ocr_enabled = request.form.get('ocr', 'false').lower() == 'true' # OCR still relevant
        math_handling_enabled = request.form.get('mathHandling', 'true').lower() == 'true'
        sentence_completion_enabled = request.form.get('sentenceCompletion', 'true').lower() == 'true'
        deduplication_enabled = request.form.get('deduplication', 'true').lower() == 'true'
        chunk_size = int(request.form.get('chunkSize', 800)) # Chunking is relevant again
        overlap = int(request.form.get('overlap', 75)) # Overlap is relevant again

        if not (200 <= chunk_size <= 4000): chunk_size = 800
        if not (0 <= overlap <= chunk_size // 2): overlap = 75

        app.logger.info(f"Request Params - Grade:{grade}, Duration:{duration}, OCR:{ocr_enabled}, Math:{math_handling_enabled}, Completion:{sentence_completion_enabled}, Dedup:{deduplication_enabled}, Chunk:{chunk_size}, Overlap:{overlap}")

    except ValueError as e:
        app.logger.error(f"Invalid form data: {e}")
        return jsonify({"error": "Invalid form data (grade, duration, etc., must be numbers)."}), 400

    pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=tempfile.gettempdir()) as temp_pdf:
            file.save(temp_pdf.name); pdf_path = temp_pdf.name
        app.logger.info(f"PDF saved to: {pdf_path}")

        # --- 1. Extract Elements (Text and Images+OCR) ---
        extract_start = time.time()
        # Pass ocr_enabled flag - pdf_utils extracts text/images
        elements = pdf_utils.extract_pdf_elements(pdf_path, perform_ocr=ocr_enabled)
        app.logger.info(f"PDF element extraction took {time.time() - extract_start:.2f}s. Found {len(elements)} elements.")

        if not elements:
            app.logger.warning(f"No elements extracted from PDF: {pdf_path}")
            # Changed error message slightly as images aren't directly used by LLM now
            return jsonify({"error": "Could not extract any text content (or OCR failed/disabled). Check the PDF."}), 400

        # --- 2. Combine and Process Text for LLM ---
        # --- REVERTED LOGIC START ---
        process_start = time.time()
        # Combine text elements and OCR'd text (if any) into a single string
        combined_text = pdf_utils.combine_elements_for_llm(elements)

        if not combined_text or len(combined_text.strip()) < 50:
             app.logger.warning("No significant text content found after extraction/OCR.")
             return jsonify({"error": "No significant text content found after extraction/OCR."}), 400

        # Clean the combined text
        cleaned_text = text_utils.clean_extracted_text(combined_text)

        # Detect math content (heuristic)
        has_math = text_utils.detect_math_content(cleaned_text) if math_handling_enabled else False
        app.logger.info(f"Math content detected (heuristic): {has_math}")

        # Chunk the cleaned text
        chunks = text_utils.split_text_into_chunks(cleaned_text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
             app.logger.error("Failed to split text into chunks.")
             return jsonify({"error": "Failed to split extracted text into processable chunks."}), 500
        app.logger.info(f"Text processing (combine/clean/detect/chunk) took {time.time() - process_start:.2f}s. Generated {len(chunks)} text chunks.")
        # --- REVERTED LOGIC END ---


        # --- 3. Generate Summary using Text-Only LLM Utils ---
        app.logger.info("Generating summary using Text LLM...")
        gen_start = time.time()
        grade_level_category, grade_level_desc = llm_utils.determine_reading_level(grade)

        # --- MODIFICATION: Pass text chunks ---
        summary = llm_utils.generate_summary(
            text_chunks=chunks, # Pass the text chunks
            # elements=elements, # Removed elements passing
            grade_level_category=grade_level_category,
            grade_level_desc=grade_level_desc,
            duration_minutes=duration,
            has_math=has_math, # Pass math detection flag
            enable_completion=sentence_completion_enabled,
            enable_deduplication=deduplication_enabled
        )
        # --- END MODIFICATION ---
        gen_duration = time.time() - gen_start
        app.logger.info(f"Core summary generation call took {gen_duration:.2f}s")

        # Error checking (same as before)
        if isinstance(summary, str) and summary.startswith("Error:"):
             error_message = summary
             app.logger.error(f"Summarization process returned an error: {error_message}")
             if "oom" in error_message.lower() or "out of memory" in error_message.lower():
                 if torch.cuda.is_available(): torch.cuda.empty_cache()
                 return jsonify({"error": f"Summarization failed due to insufficient GPU memory. Try a smaller document or shorter summary settings."}), 500
             else:
                 return jsonify({"error": f"Summarization failed: {error_message}"}), 500
        elif not isinstance(summary, str) or not summary.strip():
             app.logger.error(f"Summarization process returned invalid/empty result. Type: {type(summary)}")
             return jsonify({"error": "Summarization failed to produce a valid result."}), 500

        # --- 4. Prepare Response ---
        word_count = len(summary.split())
        total_time = time.time() - start_time
        app.logger.info(f"Summary generated successfully. Word count: {word_count}. Total request time: {total_time:.2f}s")

        if torch.cuda.is_available():
            try:
                 allocated = torch.cuda.memory_allocated(0)/1024**2; reserved = torch.cuda.memory_reserved(0)/1024**2
                 app.logger.info(f"GPU Memory after request: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")
            except Exception as e: app.logger.warning(f"Could not get GPU memory after request: {e}")

        return jsonify({
            "summary": summary,
            "word_count": word_count,
            "processing_time": round(total_time, 2)
        })

    # --- Error Handling ---
    except FileNotFoundError as e:
         app.logger.error(f"FileNotFoundError: {e}", exc_info=True); return jsonify({"error": f"Server error: File not found - {e}"}), 404
    except pytesseract.TesseractNotFoundError: # Keep this relevant for OCR toggle
         err_msg = "Tesseract OCR Engine not found/configured. OCR text extraction from images will fail."
         app.logger.error(err_msg, exc_info=False)
         return jsonify({"error": err_msg}), 500
    except torch.cuda.OutOfMemoryError:
         err_msg = "GPU ran out of memory during processing."
         app.logger.error(err_msg, exc_info=True); torch.cuda.empty_cache()
         return jsonify({"error": f"{err_msg} Try a smaller document/shorter summary/disable OCR."}), 500
    except Exception as e:
        app.logger.error(f"Unexpected Error Occurred: {type(e).__name__} - {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred. Check server logs."}), 500
    finally:
        if pdf_path and os.path.exists(pdf_path):
            try: os.remove(pdf_path); app.logger.info(f"Temp file removed: {pdf_path}")
            except Exception as e: app.logger.error(f"Error removing temp file {pdf_path}: {e}")

# --- Run Flask App ---
if __name__ == '__main__':
    app.logger.warning("Starting Flask dev server (use Gunicorn/Nginx for production)...")
    app.run(host='0.0.0.0', port=8501, debug=False, threaded=True)