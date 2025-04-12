import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import time
import traceback

# --- Configuration ---
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
            pass
if not TESSERACT_CMD:
    print("WARN (pdf_utils): Tesseract command not found or not executable. OCR will be disabled.")

# --- Image Preprocessing ---
def preprocess_image_for_ocr(pil_image):
    """Applies basic preprocessing to improve OCR accuracy."""
    img = pil_image.convert('L')
    return img

# --- Core Extraction Function ---
def extract_pdf_elements(pdf_path, perform_ocr=False, ocr_lang='eng'):
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

            blocks = page.get_text("blocks", sort=True)
            for b in blocks:
                x0, y0, x1, y1, block_text, block_no, block_type = b
                if block_type == 0 and block_text.strip():
                    elements.append({
                        'type': 'text',
                        'content': block_text.strip(),
                        'page': page_number,
                    })

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
                            'content': pil_image,
                            'page': page_number,
                            'format': fmt,
                            'ocr_text': None
                        }

                        if ocr_available:
                            try:
                                ocr_start = time.time()
                                processed_pil_image = preprocess_image_for_ocr(pil_image)
                                ocr_text_result = pytesseract.image_to_string(
                                    processed_pil_image,
                                    lang=ocr_lang,
                                    config='--psm 6'
                                ).strip()
                                image_element['ocr_text'] = ocr_text_result
                                print(f"      - OCR'd image {img_counter} (took {time.time()-ocr_start:.2f}s). Found text: {bool(ocr_text_result)}")
                            except pytesseract.TesseractNotFoundError:
                                print("ERROR (pdf_utils): Tesseract not found during OCR. Disabling OCR for remaining images.")
                                ocr_available = False
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

# --- Combine Extracted Elements ---
def combine_elements_for_llm(elements):
    """Combines text and OCR'd image text into a single string."""
    full_text = ""
    for element in elements:
        if element['type'] == 'text':
            full_text += element['content'] + "\n\n"
        elif element['type'] == 'image' and element.get('ocr_text'):
            full_text += f"[Image OCR Start]\n{element['ocr_text']}\n[Image OCR End]\n\n"
    return full_text.strip()
