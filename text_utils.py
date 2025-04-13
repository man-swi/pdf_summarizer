# text_utils.py (Updated with Simplified Overlap)

import re
import nltk
from nltk.stem import PorterStemmer
import time
import os

# --- NLTK Download Check ---
try:
    # Check multiple paths including user home directory
    nltk_data_paths = [
        os.path.join(os.path.expanduser('~'), 'nltk_data'),
        *nltk.data.path # Include default nltk paths
    ]
    found = False
    for path in nltk_data_paths:
        if os.path.exists(os.path.join(path, 'tokenizers', 'punkt')):
            if path not in nltk.data.path: # Add path if needed
                 nltk.data.path.append(path)
            print(f"INFO (text_utils): Found NLTK punkt tokenizer in: {path}")
            found = True
            break
    if not found:
        raise nltk.downloader.DownloadError("NLTK punkt tokenizer not found in known paths.")

except nltk.downloader.DownloadError:
    print("WARN (text_utils): NLTK punkt tokenizer not found. Attempting download...")
    try:
        # Prefer downloading to user's home directory if possible
        user_nltk_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
        os.makedirs(user_nltk_path, exist_ok=True)
        nltk.download('punkt', quiet=True, download_dir=user_nltk_path)
        if os.path.exists(os.path.join(user_nltk_path, 'tokenizers', 'punkt')):
            nltk.data.path.append(user_nltk_path)
            print(f"INFO (text_utils): NLTK punkt downloaded to {user_nltk_path}")
        else: # Fallback to default download location
            nltk.download('punkt', quiet=True)
        print("INFO (text_utils): NLTK punkt download complete.")
    except Exception as e:
        print(f"ERROR (text_utils): Failed to download NLTK punkt: {e}. Sentence splitting might be less accurate.")


# --- Text Cleaning Function ---
def clean_extracted_text(text):
    """Cleans text extracted from PDF."""
    if not text: return ""
    print("INFO (text_utils): Cleaning text...")
    start_time = time.time()
    text = re.sub(r'\f', ' ', text)
    text = re.sub(r'\[Image OCR Start\]\n?', '', text)
    text = re.sub(r'\n?\[Image OCR End\]\n?', '\n', text)
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])(\w)', r'\1 \2', text)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 1 or line.strip() in ['.','!','?']]
    text = '\n'.join(cleaned_lines)
    text = text.replace('\n', ' ') # Convert newlines for sentence splitting
    text = re.sub(r'\s+', ' ', text).strip()
    print(f"INFO (text_utils): Text cleaning finished (took {time.time() - start_time:.2f}s). Length: {len(text)}")
    return text

# --- Math Detection Function ---
def detect_math_content(text):
    """Uses heuristics to detect if the text likely contains mathematical content."""
    if not text: return False
    print("INFO (text_utils): Detecting math content...")
    start_time = time.time()
    math_keywords = [r'\b(equation|formula|theorem|lemma|proof|derive|solve|calculate|algebra|calculus|geometry|vector|matrix|integral|derivative|function|variable|constant|graph|plot|measure|angle|degree|coefficient|exponent|logarithm)\b']
    math_symbols = r'[=><≤≥≠≈\+\-\*×/÷±√∛∫∑∏∂∇∞^%]'
    function_notation = r'\b[a-zA-Z]{1,4}\s?\([a-zA-Z0-9,\s\+\-\*\/.]+\)'
    latex_delimiters = r'\$.*?\$|\\\(.*?\\\)|\\[a-zA-Z]+(\{.*?\})*'
    math_list_items = r'^\s*(\d+\.|\*|\-)\s*[=><≤≥≠\+\-\*\/\^∫∑∏√∞≠≤≥±→∂∇πθλμσωαβγδε\$\\]'
    if re.search('|'.join(math_keywords), text, re.IGNORECASE): print("INFO (text_utils): Math detected (keyword)."); return True
    text_sample = text[:50000]
    for pattern in [math_symbols, function_notation, latex_delimiters]:
        if re.search(pattern, text_sample): print(f"INFO (text_utils): Math detected (pattern: {pattern[:20]}...)."); return True
    if re.search(math_list_items, text_sample, re.MULTILINE): print("INFO (text_utils): Math detected (list item pattern)."); return True
    print(f"INFO (text_utils): No significant math content detected (took {time.time() - start_time:.2f}s).")
    return False

# --- Text Chunking Function (with Simplified Overlap) ---
def split_text_into_chunks(text, chunk_size=800, overlap=50, max_chunks=20):
    """
    Splits cleaned text into overlapping chunks based on sentences, using simplified word-based overlap.
    """
    if not text: return []
    print(f"INFO (text_utils): Splitting text into chunks (size: {chunk_size}, overlap: {overlap})...")
    start_time = time.time()

    # 1. Sentence Segmentation
    try:
        sentences = nltk.sent_tokenize(text)
        if not sentences: raise ValueError("NLTK tokenization resulted in empty list")
    except Exception as e:
        print(f"WARN (text_utils): NLTK splitting failed ({e}), using regex fallback.")
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()] or [text]

    # 2. Combine Sentences into Chunks
    chunks = []
    current_chunk_words_list = [] # Store words as list for easier overlap slicing
    current_length = 0

    for sentence_index, sentence in enumerate(sentences):
        # print(f"DEBUG (split): Processing sentence {sentence_index+1}/{len(sentences)}") # Optional debug
        sentence = sentence.strip()
        if not sentence: continue

        sentence_words = sentence.split()
        sent_length = len(sentence_words)

        # If adding the current sentence exceeds chunk size, finalize the current chunk
        if current_length > 0 and current_length + sent_length > chunk_size:
            # Finalize the current chunk
            final_chunk_text = ' '.join(current_chunk_words_list)
            chunks.append(final_chunk_text)
            print(f"DEBUG (split): Finalized chunk {len(chunks)} (length: {current_length} words)")

            # --- SIMPLIFIED WORD-BASED OVERLAP ---
            # Take the last 'overlap' number of words from the previous chunk
            # Ensure overlap isn't larger than the chunk itself
            actual_overlap_count = min(overlap, current_length)
            if actual_overlap_count > 0:
                overlap_words = current_chunk_words_list[-actual_overlap_count:]
                print(f"DEBUG (split): Calculated overlap of {len(overlap_words)} words.")
            else:
                overlap_words = []
                print(f"DEBUG (split): Calculated overlap of 0 words.")

            # Start new chunk with overlap words + current sentence words
            current_chunk_words_list = overlap_words + sentence_words
            current_length = len(current_chunk_words_list)
            # --- END SIMPLIFIED OVERLAP ---

        else:
            # Add sentence words to current chunk
            current_chunk_words_list.extend(sentence_words)
            current_length += sent_length

    # Add the last remaining chunk
    if current_chunk_words_list:
        chunks.append(' '.join(current_chunk_words_list))
        print(f"DEBUG (split): Finalized last chunk {len(chunks)} (length: {current_length} words)")

    # 3. Post-process Chunks (Merge small ones) - Logic remains the same
    refined_chunks = []
    i = 0
    min_chunk_words = max(50, chunk_size * 0.2)
    while i < len(chunks):
        current_chunk_word_count = len(chunks[i].split())
        if current_chunk_word_count < min_chunk_words and i + 1 < len(chunks):
            next_chunk_word_count = len(chunks[i+1].split())
            if current_chunk_word_count + next_chunk_word_count <= chunk_size * 1.5:
                refined_chunks.append(chunks[i] + " " + chunks[i+1])
                i += 2
            else:
                refined_chunks.append(chunks[i]); i += 1
        else:
            refined_chunks.append(chunks[i]); i += 1
    chunks = refined_chunks

    # 4. Limit number of chunks
    if len(chunks) > max_chunks:
        print(f"WARN (text_utils): Reducing chunks from {len(chunks)} to {max_chunks}")
        chunks = chunks[:max_chunks] # Simple truncation

    print(f"INFO (text_utils): Split into {len(chunks)} chunks (took {time.time() - start_time:.2f}s).")
    return chunks