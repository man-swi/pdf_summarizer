# pdf_utils.py

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import time

# --- Configuration ---
# Tesseract Path (Should be configured globally if possible, but check here too)
TESSERACT_CMD = None
tesseract_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract', 'tesseract']
for path in tesseract_paths:
    if os.path.exists(path):
        try:
            if os.access(path, os.X_OK):
                TESSERACT_CMD = path
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"INFO (pdf_utils): Using Tesseract at: {path}")
                break
        except Exception:
            pass # Ignore errors if path check fails
if not TESSERACT_CMD:
    print("WARN (pdf_utils): Tesseract command not found or not executable. OCR will be disabled.")

# --- Image Preprocessing (Optional but Recommended for OCR) ---
def preprocess_image_for_ocr(pil_image):
    """Applies basic preprocessing to improve OCR accuracy."""
    # Convert to grayscale
    img = pil_image.convert('L')
    # Optional: Increase contrast, thresholding, etc. - Keep simple for now
    # Example: Simple thresholding (might need tuning)
    # threshold = 180
    # img = img.point(lambda p: p > threshold and 255)
    return img

# --- Core Extraction Function ---

def extract_pdf_elements(pdf_path, perform_ocr=False, ocr_lang='eng'):
    """
    Extracts text blocks and image data sequentially from a PDF.

    Args:
        pdf_path (str): Path to the PDF file.
        perform_ocr (bool): Whether to run OCR on detected images.
        ocr_lang (str): Language for Tesseract OCR.

    Returns:
        list: A list of dictionaries, where each dictionary represents
              an element ('text' or 'image') and contains its data.
              Example:
              [
                  {'type': 'text', 'content': 'Some text paragraph.', 'page': 1},
                  {'type': 'image', 'content': <PIL.Image object>, 'page': 1, 'ocr_text': 'Text from OCR'},
                  {'type': 'text', 'content': 'More text.', 'page': 1},
                  ...
              ]
              Returns an empty list if the PDF cannot be opened or processed.
    """
    if not os.path.exists(pdf_path):
        print(f"ERROR (pdf_utils): PDF file not found at {pdf_path}")
        return []

    elements = []
    ocr_available = perform_ocr and TESSERACT_CMD is not None

    try:
        doc = fitz.open(pdf_path)
        print(f"INFO (pdf_utils): Processing PDF with {len(doc)} pages. OCR Enabled: {ocr_available}")

        for page_num, page in enumerate(doc):
            page_number = page_num + 1
            print(f"  - Processing Page {page_number}...")
            start_time = time.time()

            # --- Extract Text Blocks ---
            # Use get_text("blocks") to get text with bounding box info
            # We sort blocks vertically to maintain reading order
            blocks = page.get_text("blocks", sort=True)
            for b in blocks:
                x0, y0, x1, y1, block_text, block_no, block_type = b
                if block_type == 0 and block_text.strip(): # block_type 0 is text
                    elements.append({
                        'type': 'text',
                        'content': block_text.strip(),
                        'page': page_number,
                        # 'bbox': (x0, y0, x1, y1) # Optional: store bounding box
                    })

            # --- Extract Images ---
            image_list = page.get_images(full=True)
            if image_list:
                print(f"    - Found {len(image_list)} image refs on page {page_number}.")
                img_counter = 0
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        fmt = base_image.get("ext", "png").lower()

                        if fmt not in ["png", "jpeg", "jpg", "bmp", "gif", "tiff"]:
                            print(f"      - Skipping image {img_index} (xref {xref}) with unsupported format: {fmt}")
                            continue

                        pil_image = Image.open(io.BytesIO(image_bytes))
                        img_counter += 1
                        image_element = {
                            'type': 'image',
                            'content': pil_image, # Store the PIL Image object
                            'page': page_number,
                            'format': fmt,
                            'ocr_text': None # Placeholder for OCR result
                            # 'bbox': None # TODO: Could try to get bbox if needed
                        }

                        # --- Perform OCR if enabled ---
                        if ocr_available:
                            try:
                                ocr_start = time.time()
                                # Preprocess before OCR
                                processed_pil_image = preprocess_image_for_ocr(pil_image)
                                ocr_text_result = pytesseract.image_to_string(
                                    processed_pil_image,
                                    lang=ocr_lang,
                                    config='--psm 6' # Assume a single uniform block of text
                                ).strip()
                                image_element['ocr_text'] = ocr_text_result
                                print(f"      - OCR'd image {img_counter} (took {time.time()-ocr_start:.2f}s). Found text: {bool(ocr_text_result)}")
                            except pytesseract.TesseractNotFoundError:
                                print("ERROR (pdf_utils): Tesseract not found during OCR. Disabling OCR for remaining images.")
                                ocr_available = False # Disable for subsequent images
                            except Exception as ocr_err:
                                print(f"ERROR (pdf_utils): OCR failed for image {img_counter} on page {page_number}: {ocr_err}")

                        elements.append(image_element)

                    except Exception as img_err:
                        print(f"ERROR (pdf_utils): Failed to extract/process image {img_index} (xref {xref}) on page {page_number}: {img_err}")

            print(f"    - Page {page_number} processing took {time.time() - start_time:.2f}s")

        doc.close()
        print(f"INFO (pdf_utils): Finished PDF processing. Extracted {len(elements)} elements.")
        return elements

    except fitz.fitz.FileNotFoundError:
        print(f"ERROR (pdf_utils): File not found via fitz: {pdf_path}")
        return []
    except Exception as e:
        print(f"ERROR (pdf_utils): Failed to process PDF {pdf_path}: {e}")
        traceback.print_exc()
        return []

# --- Helper to integrate OCR text (Example Usage Pattern) ---
# This logic will likely live in app.py or text_utils.py where
# elements are combined before chunking.

def combine_elements_for_llm(elements):
    """Combines text and OCR'd image text into a single string."""
    full_text = ""
    for element in elements:
        if element['type'] == 'text':
            full_text += element['content'] + "\n\n"
        elif element['type'] == 'image' and element.get('ocr_text'):
            # Add a marker to indicate this text came from an image
            full_text += f"[Image OCR Start]\n{element['ocr_text']}\n[Image OCR End]\n\n"
        # NOTE: Image 'content' (PIL object) is ignored here.
        # A multi-modal approach would use element['content'].
    return full_text.strip()