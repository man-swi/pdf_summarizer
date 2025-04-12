# app.py (Main Flask Application - New Structure)

import os
import re
import traceback
import tempfile
import time
import io
import pytesseract  # Added to handle pytesseract.TesseractNotFoundError
import torch
from flask import Flask, request, render_template, jsonify, send_from_directory

# Import model loading classes and quantization config
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# Import utility modules
import pdf_utils
import text_utils
import llm_utils

# --- Configuration & Model Loading ---
print("--- Starting Flask App Setup ---")

# --- Tesseract Configuration (Handled in pdf_utils) ---
# pdf_utils checks for Tesseract path at import time.

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO (app): PyTorch device detected: {device}")

# --- Model Configuration ---
# Using standard transformers loading with offload workaround for unsloth weights
LLM_MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
EMBEDDER_MODEL_NAME = 'all-MiniLM-L6-v2'

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
print(f"INFO (app): Using BitsAndBytes quantization config: {quantization_config}")

# Define offload directory
offload_directory = "./offload_cache" # Ensure this is writable
os.makedirs(offload_directory, exist_ok=True)
print(f"INFO (app): Using offload directory: {offload_directory}")

# --- Load Models ---
model = None
tokenizer = None
embedder = None

try:
    print(f"INFO (app): Loading LLM tokenizer: {LLM_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    print("INFO (app): Tokenizer loaded.")

    print(f"INFO (app): Loading LLM model: {LLM_MODEL_NAME} with quantization & offloading...")
    # Use standard AutoModelForCausalLM with offload flags based on previous errors
    model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    offload_folder=offload_directory,
    offload_state_dict=True
    )

    print("INFO (app): LLM Model loaded successfully!")

except Exception as e:
    print(f"FATAL ERROR (app): Could not load LLM model {LLM_MODEL_NAME}: {e}")
    traceback.print_exc()
    if "out of memory" in str(e).lower(): print("Attempting to clear CUDA cache..."); torch.cuda.empty_cache()
    exit(1) # Exit if core model fails to load

try:
    print(f"INFO (app): Loading Sentence Embedder: {EMBEDDER_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device=device)
    print("INFO (app): Embedder loaded successfully!")
except Exception as e:
    print(f"ERROR (app): Could not load Sentence Transformer {EMBEDDER_MODEL_NAME}: {e}")
    embedder = None # Allow app to run without embedder
    print("WARN (app): Embedder failed to load. Semantic deduplication will be disabled.")

# --- Initialize LLM Utilities Module with Loaded Models ---
if model and tokenizer:
    llm_utils.initialize_globals(model, tokenizer, embedder)
else:
    print("FATAL ERROR (app): LLM Model or Tokenizer failed to load. Cannot initialize llm_utils.")
    exit(1)

print("--- Flask App Setup Complete ---")

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
    start_time = time.time()

    if 'pdfFile' not in request.files: return jsonify({"error": "No PDF file provided."}), 400
    file = request.files['pdfFile']
    if file.filename == '': return jsonify({"error": "No selected file."}), 400
    if not file.filename.lower().endswith('.pdf'): return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

    # --- Get form data ---
    try:
        grade = int(request.form.get('grade', 6))
        duration = int(request.form.get('duration', 20))
        ocr_enabled = request.form.get('ocr', 'false').lower() == 'true'
        # Add toggles for math, completion, deduplication if needed in form
        math_handling_enabled = request.form.get('mathHandling', 'true').lower() == 'true' # Example toggle
        sentence_completion_enabled = request.form.get('sentenceCompletion', 'true').lower() == 'true'
        deduplication_enabled = request.form.get('deduplication', 'true').lower() == 'true'
        # Add chunk/overlap if form includes them
        chunk_size = int(request.form.get('chunkSize', 800)) # Default larger chunk
        overlap = int(request.form.get('overlap', 75))      # Default larger overlap

        # Validate chunk/overlap
        if not (200 <= chunk_size <= 4000): chunk_size = 800 # Wider range
        if not (0 <= overlap <= chunk_size // 2): overlap = 75 # Adjusted default overlap limit

    except ValueError:
        return jsonify({"error": "Invalid form data (grade, duration, chunk size, overlap must be numbers)."}), 400

    pdf_path = None
    try:
        # --- Save temporary file ---
        fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        file.save(pdf_path)
        print(f"INFO (app): PDF saved temporarily to: {pdf_path}")

        # --- 1. Extract Elements (Text and Images+OCR) ---
        extract_start = time.time()
        # Pass OCR flag from form
        elements = pdf_utils.extract_pdf_elements(pdf_path, perform_ocr=ocr_enabled)
        print(f"INFO (app): PDF element extraction took {time.time() - extract_start:.2f}s")
        if not elements:
            return jsonify({"error": "Could not extract any content from PDF. Check file."}), 400

        # --- 2. Combine Text for Processing ---
        # This helper function now exists in pdf_utils, combines text and OCR results
        combined_text = pdf_utils.combine_elements_for_llm(elements)
        if not combined_text or len(combined_text.strip()) < 50:
             return jsonify({"error": "No significant text content found after extraction/OCR."}), 400

        # --- 3. Process Text (Clean, Detect Math, Chunk) ---
        process_start = time.time()
        # Clean the combined text
        cleaned_text = text_utils.clean_extracted_text(combined_text)
        # Detect math content (using math_handling_enabled flag if desired)
        has_math = text_utils.detect_math_content(cleaned_text) if math_handling_enabled else False
        print(f"INFO (app): Math content detected: {has_math}")
        # Chunk the cleaned text
        chunks = text_utils.split_text_into_chunks(cleaned_text, chunk_size=chunk_size, overlap=overlap)
        if not chunks: return jsonify({"error": "Failed to split text into chunks."}), 500
        print(f"INFO (app): Text processing (clean/detect/chunk) took {time.time() - process_start:.2f}s")

        # --- 4. Generate Summary using LLM Utils ---
        print("INFO (app): Generating summary...")
        gen_start = time.time()
        grade_level_category, grade_level_desc = llm_utils.determine_reading_level(grade)
        # Pass refinement flags to llm_utils.generate_summary
        summary = llm_utils.generate_summary(
            chunks,
            grade_level_category,
            grade_level_desc,
            duration,
            has_math=has_math,
            enable_completion=sentence_completion_enabled,
            enable_deduplication=deduplication_enabled
        )
        print(f"INFO (app): Core summary generation took {time.time() - gen_start:.2f}s")

        # Check for errors during summary generation
        if summary.startswith("Error:"):
             error_message = summary.split("Error:", 1)[1].strip()
             print(f"ERROR (app): Summarization process failed: {error_message}")
             return jsonify({"error": f"Summarization failed: {error_message}"}), 500

        # --- 5. Prepare Response ---
        word_count = len(summary.split())
        total_time = time.time() - start_time
        print(f"INFO (app): Summary generated successfully. Word count: {word_count}. Total request time: {total_time:.2f}s")

        return jsonify({
            "summary": summary,
            "word_count": word_count,
            "processing_time": round(total_time, 2)
        })

    # --- Error Handling ---
    except FileNotFoundError as e:
         print(f"ERROR (app): {e}"); return jsonify({"error": str(e)}), 404
    except pytesseract.TesseractNotFoundError:
         err_msg = "Tesseract OCR Engine not found/configured on server."; print(f"ERROR (app): {err_msg}"); return jsonify({"error": err_msg}), 500
    except torch.cuda.OutOfMemoryError:
         err_msg = "GPU ran out of memory during processing. Try smaller doc/settings."; print(f"ERROR (app): {err_msg}"); traceback.print_exc(); return jsonify({"error": err_msg}), 500
    except Exception as e:
        print(f"--- ERROR (app): An Unexpected Error Occurred ---")
        print(f"Error Type: {type(e).__name__}"); print(f"Error Details: {str(e)}"); traceback.print_exc()
        return jsonify({"error": "An unexpected server error occurred processing the PDF. Please check server logs."}), 500
    finally:
        # --- Clean up temporary file ---
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path); print(f"INFO (app): Temporary file removed: {pdf_path}")
            except Exception as e:
                print(f"ERROR (app): Error removing temp file {pdf_path}: {e}")


# --- Run Flask App ---
# This block is primarily for local development testing.
# Gunicorn will directly import and run the 'app' object in production.
if __name__ == '__main__':
    print("Starting Flask development server (for testing only)...")
    # Use host='0.0.0.0' to make it accessible on your local network
    # Use debug=False for production or testing performance
    # Set threaded=True or use a production server like gunicorn/waitress for handling multiple requests
    # Port 8501 matches Dockerfile EXPOSE and Nginx proxy_pass
    app.run(host='0.0.0.0', port=8501, debug=False, threaded=True)