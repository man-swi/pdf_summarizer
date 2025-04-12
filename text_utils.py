# text_utils.py

import re
import nltk
from nltk.stem import PorterStemmer
import time

# Initialize stemmer globally if needed elsewhere, or locally if only here
# stemmer = PorterStemmer()

# --- NLTK Download Check (Optional but good practice in module) ---
try:
    nltk.data.find('tokenizers/punkt')
    print("INFO (text_utils): NLTK punkt tokenizer found.")
except nltk.downloader.DownloadError:
    print("WARN (text_utils): NLTK punkt tokenizer not found. Attempting download...")
    try:
        nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
        if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
            os.makedirs(nltk_data_path, exist_ok=True)
            nltk.download('punkt', quiet=True, download_dir=nltk_data_path)
            if os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
                nltk.data.path.append(nltk_data_path)
                print(f"INFO (text_utils): NLTK punkt downloaded to {nltk_data_path}")
            else: # Fallback to system location
                nltk.download('punkt', quiet=True)
        else:
             nltk.data.path.append(nltk_data_path) # Ensure it's in the path if already exists
        print("INFO (text_utils): NLTK punkt download/check complete.")
    except Exception as e:
        print(f"ERROR (text_utils): Failed to download NLTK punkt: {e}. Sentence splitting might be less accurate.")


# --- Text Cleaning Function ---

def clean_extracted_text(text):
    """
    Cleans text extracted from PDF, removing common artifacts and normalizing whitespace.
    """
    if not text:
        return ""

    print("INFO (text_utils): Cleaning text...")
    start_time = time.time()

    # Remove form feed characters
    text = re.sub(r'\f', ' ', text)
    # Remove our specific OCR markers (if they were added)
    text = re.sub(r'\[Image OCR Start\]\n?', '', text)
    text = re.sub(r'\n?\[Image OCR End\]\n?', '\n', text) # Replace end marker with newline
    # Remove PDF artifacts like (cid:dd)
    text = re.sub(r'\(cid:\d+\)', '', text)
    # Normalize line breaks and whitespace (multiple spaces/tabs/newlines to single space)
    text = re.sub(r'\s+', ' ', text)
    # Add space after punctuation if missing (helps sentence tokenization)
    text = re.sub(r'([.!?])(\w)', r'\1 \2', text)
    # Optional: Rejoin words hyphenated across lines (might be less relevant now)
    # text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text) # Simplified now we convert newlines earlier

    # Remove excessive blank lines (convert back to single newlines first for processing)
    # text = text.replace(' ', '\n') # Temporarily? No, bad idea.
    # Let's handle lines after splitting
    lines = text.split('\n') # Assuming basic structure might still exist
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 1: # Keep lines with more than one character
             cleaned_lines.append(stripped)
        elif stripped in ['.','!','?']: # Keep lines that are just punctuation
            cleaned_lines.append(stripped)
    text = '\n'.join(cleaned_lines) # Rejoin with single newlines for now

    # Convert remaining newlines to spaces for chunking based on sentences
    text = text.replace('\n', ' ')
    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()

    print(f"INFO (text_utils): Text cleaning finished (took {time.time() - start_time:.2f}s). Length: {len(text)}")
    return text


# --- Math Detection Function ---

def detect_math_content(text):
    """
    Uses heuristics to detect if the text likely contains mathematical content.
    """
    if not text:
        return False

    print("INFO (text_utils): Detecting math content...")
    start_time = time.time()

    # Keywords suggesting mathematical context
    math_keywords = [
        r'\b(equation|formula|theorem|lemma|proof|derive|solve|calculate|algebra|calculus)',
        r'\b(geometry|vector|matrix|integral|derivative|function|variable|constant|graph|plot)',
        r'\b(measure|angle|degree|coefficient|exponent|logarithm)\b'
    ]
    # Common math symbols (add more as needed)
    math_symbols = r'[=><≤≥≠≈\+\-\*×/÷±√∛∫∑∏∂∇∞^%]' # Added ×÷≈∛
    # Common function notation patterns (e.g., f(x), sin(y))
    function_notation = r'\b[a-zA-Z]{1,4}\s?\([a-zA-Z0-9,\s\+\-\*\/.]+\)' # Allow dots in args
    # LaTeX like delimiters (basic check)
    latex_delimiters = r'\$.*?\$|\\\(.*?\\\)|\\[a-zA-Z]+(\{.*?\})*'
    # Lines starting with numbers/bullets followed by math-like content
    math_list_items = r'^\s*(\d+\.|\*|\-)\s*[=><≤≥≠\+\-\*\/\^∫∑∏√∞≠≤≥±→∂∇πθλμσωαβγδε\$\\]'

    # Check keywords first (case-insensitive)
    if re.search('|'.join(math_keywords), text, re.IGNORECASE):
        print("INFO (text_utils): Math detected (keyword).")
        return True

    # Limit search for symbols/patterns for performance
    text_sample = text[:50000] # Increased sample size slightly

    # Check for symbols, function notation, LaTeX
    for pattern in [math_symbols, function_notation, latex_delimiters]:
        if re.search(pattern, text_sample):
            print(f"INFO (text_utils): Math detected (pattern: {pattern[:20]}...).")
            return True

    # Check for math-like list items (multi-line check)
    if re.search(math_list_items, text_sample, re.MULTILINE):
        print("INFO (text_utils): Math detected (list item pattern).")
        return True

    print(f"INFO (text_utils): No significant math content detected (took {time.time() - start_time:.2f}s).")
    return False


# --- Text Chunking Function ---

def split_text_into_chunks(text, chunk_size=800, overlap=50, max_chunks=20):
    """
    Splits cleaned text into overlapping chunks based on sentences.

    Args:
        text (str): The cleaned text content.
        chunk_size (int): Target number of words per chunk.
        overlap (int): Target number of words for overlap between chunks.
        max_chunks (int): Maximum number of chunks to return.

    Returns:
        list: A list of text chunks (strings).
    """
    if not text:
        return []

    print(f"INFO (text_utils): Splitting text into chunks (size: {chunk_size}, overlap: {overlap})...")
    start_time = time.time()

    # 1. Sentence Segmentation
    try:
        sentences = nltk.sent_tokenize(text)
        if not sentences: raise ValueError("NLTK tokenization resulted in empty list")
    except Exception as e:
        print(f"WARN (text_utils): NLTK splitting failed ({e}), using regex fallback.")
        # Basic fallback splitting on punctuation followed by space
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out empty strings that might result from splitting
        sentences = [s for s in sentences if s.strip()]
        if not sentences: sentences = [text] # Handle case where no split occurs

    # 2. Combine Sentences into Chunks
    chunks = []
    current_chunk_words = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue

        sentence_words = sentence.split()
        sent_length = len(sentence_words)

        # If adding the current sentence exceeds chunk size, finalize the current chunk
        if current_length > 0 and current_length + sent_length > chunk_size:
            chunks.append(' '.join(current_chunk_words))

            # Create overlap: find sentences from the end of the last chunk
            overlap_word_count = 0
            overlap_sentences_for_next = []
            for i in range(len(current_chunk_words) - 1, -1, -1):
                # Count words in the sentence represented by current_chunk_words[i]
                # This requires re-tokenizing or storing sentences; simpler to estimate
                # Let's try a simpler overlap: just take the last N sentences approx
                # This is less precise than word count but easier here.
                # Alternative: Store original sentences associated with current_chunk_words

                # --- Simpler Overlap based on last few sentences ---
                # Take approx last sentence(s) contributing to overlap words
                # This part is tricky without storing original sentences.
                # Let's revert to the word-based slicing logic from before for simplicity.

                # Find sentences corresponding to the overlap word count
                # This requires associating words back to sentences or recalculating
                # For simplicity, let's use the previous logic which was slicing words,
                # although sentence boundary is better. We'll reuse that word-based logic.

                 # Determine overlap based on sentences, aiming for word count (from app.py logic)
                overlap_word_count_target = 0
                overlap_sentence_indices = []
                temp_sentences_in_chunk = nltk.sent_tokenize(' '.join(current_chunk_words)) # Re-tokenize chunk
                for i in range(len(temp_sentences_in_chunk) - 1, -1, -1):
                     words_in_sent = len(temp_sentences_in_chunk[i].split())
                     if overlap_word_count_target + words_in_sent <= overlap + sent_length : # Allow flexibility
                          overlap_word_count_target += words_in_sent
                          overlap_sentences_for_next.insert(0, temp_sentences_in_chunk[i])
                     else:
                         break
                     if overlap_word_count_target >= overlap:
                          break


            # Start new chunk with overlap sentences + current sentence
            current_chunk_words = ' '.join(overlap_sentences_for_next).split() + sentence_words
            current_length = len(current_chunk_words)

        else:
            # Add sentence words to current chunk
            current_chunk_words.extend(sentence_words)
            current_length += sent_length

    # Add the last chunk if it's not empty
    if current_chunk_words:
        chunks.append(' '.join(current_chunk_words))

    # 3. Post-process Chunks (Merge small ones)
    refined_chunks = []
    i = 0
    min_chunk_words = max(50, chunk_size * 0.2) # Min chunk size
    while i < len(chunks):
        current_chunk_word_count = len(chunks[i].split())
        # Check if merge is possible and beneficial
        if current_chunk_word_count < min_chunk_words and i + 1 < len(chunks):
            next_chunk_word_count = len(chunks[i+1].split())
            # Only merge if the combined size isn't excessively large
            if current_chunk_word_count + next_chunk_word_count <= chunk_size * 1.5: # Allow some flexibility
                refined_chunks.append(chunks[i] + " " + chunks[i+1])
                i += 2 # Skip the next chunk since it was merged
            else:
                refined_chunks.append(chunks[i])
                i += 1
        else:
            refined_chunks.append(chunks[i])
            i += 1
    chunks = refined_chunks

    # 4. Limit number of chunks
    if len(chunks) > max_chunks:
        print(f"WARN (text_utils): Reducing chunks from {len(chunks)} to {max_chunks}")
        # Simple truncation for now, could implement stepping later
        chunks = chunks[:max_chunks]

    print(f"INFO (text_utils): Split into {len(chunks)} chunks (took {time.time() - start_time:.2f}s).")
    return chunks