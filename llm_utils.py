# llm_utils.py (Reverted to Text-Only for Llama-3.1-8B-Instruct)

import re
import torch
import time
import traceback
from nltk.stem import PorterStemmer
from sentence_transformers import util
# No PIL needed here anymore for core logic

# Global placeholders
model = None
tokenizer = None # Changed back from processor
embedder = None
stemmer = PorterStemmer()

MAX_COMPLETION_CALLS = 10

# --- Constants ---
# Llama 3.1 Instruct uses specific tokens for its chat template.
# It's best practice to use tokenizer.apply_chat_template if possible.
# If using manual formatting, these are the typical tokens:
BOS = "<|begin_of_text|>" # Or tokenizer.bos_token
USER_START = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_START = "<|start_header_id|>assistant<|end_header_id|>\n\n"
EOT = "<|eot_id|>" # Or tokenizer.eos_token

def initialize_globals(llm_model, llm_tokenizer, sentence_embedder):
    """Initializes global variables with model, tokenizer, and embedder."""
    global model, tokenizer, embedder
    model = llm_model
    tokenizer = llm_tokenizer # Now using tokenizer
    embedder = sentence_embedder
    print("INFO (llm_utils): LLM utilities initialized with model, tokenizer, and embedder.")

# --- Prompts Dictionary (Reverted to Text-Only Llama 3.1 Format) ---
# Using the standard Llama 3 instruct format. Removed image references.
prompts = {
    "lower": { # Grades 1-3
        "standard": (
            f"{USER_START}You are summarizing textbook text for a young child (grades 1-3, ages 6-8).\n"
            "Instructions:\n"
            "1. Use VERY simple words and extremely short sentences.\n"
            "2. Start with the heading '# Simple Summary'.\n"
            "3. Explain the main idea in one simple sentence.\n"
            "4. Under '## Key Points', list 3-5 key points using '- '. Each point must be a simple sentence.\n"
            "5. Do NOT include complex details or jargon.\n"
            "6. Finish with ONE fun, simple activity (like drawing or a simple question) under '## Fun Activity'.\n"
            "7. Use clear Markdown.\n\n"
            "Text to summarize:\n{{text}}" # Placeholder for the actual text
            f"{EOT}{ASSISTANT_START}# Simple Summary\n\n" # Prime assistant response
        ),
        "math": (
             f"{USER_START}You are explaining a math topic from text for a young child (grades 1-3, ages 6-8).\n"
             "Instructions:\n"
             "1. Use simple words, short sentences, and analogies (counting toys, sharing cookies).\n"
             "2. Use small, simple numbers in examples.\n"
             "3. Start with '# Math Fun'.\n"
             "4. Explain the main math idea simply under '## What We Learned'.\n"
             "5. If there are steps, list them simply under '## Steps' (1., 2.).\n"
             "6. Give one clear, simple example under '## Example'.\n"
             "7. Finish with ONE easy practice question or drawing task under '## Practice Time'.\n"
             "8. Use clear Markdown.\n\n"
             "Text to explain:\n{{text}}"
             f"{EOT}{ASSISTANT_START}# Math Fun\n\n"
        )
    },
    "middle": { # Grades 4-6
        "standard": (
             f"{USER_START}You are summarizing textbook text for a student in grades 4-6 (ages 9-11).\n"
             "Instructions:\n"
             "1. Start with the main heading '# Summary'.\n"
             "2. Identify the 2-4 most important main topics.\n"
             "3. Create clear subheadings ('## Topic Name') for each.\n"
             "4. Under each subheading, use bullet points '- ' to list key information and supporting details using clear, complete sentences.\n"
             "5. Explain any important terms simply.\n"
             "6. Ensure the summary flows logically.\n"
             "7. Conclude with ONE practical activity or thought-provoking question under '## Try This'.\n"
             "8. Use clear Markdown.\n\n"
             "Text to summarize:\n{{text}}"
             f"{EOT}{ASSISTANT_START}# Summary\n\n"
        ),
        "math": (
             f"{USER_START}You are explaining a math concept from text for a student in grades 4-6 (ages 9-11).\n"
             "Instructions:\n"
             "1. Start with '# Math Explained'.\n"
             "2. Explain the core concept clearly under '## The Concept'.\n"
             "3. Provide a clear, step-by-step example under '## Step-by-Step Example' (use 1., 2.). Show work clearly.\n"
             "4. Briefly explain why this math is useful under '## Why It Matters'.\n"
             "5. Conclude with ONE relevant practice problem under '## Practice Problem'.\n"
             "6. Use clear Markdown.\n\n"
             "Text to explain:\n{{text}}"
             f"{EOT}{ASSISTANT_START}# Math Explained\n\n"
        )
    },
    "higher": { # Grades 7-12
        "standard": (
             f"{USER_START}Create a comprehensive, well-structured summary of textbook text for a high school student (grades 7-12, ages 12-18).\n"
             "Instructions:\n"
             "1. Start with '# Comprehensive Summary'.\n"
             "2. Identify key themes, arguments, or concepts.\n"
             "3. Create logical subheadings ('## Theme/Section Name').\n"
             "4. Under each subheading, synthesize essential information using clear paragraphs and bullet points '- ' for specifics.\n"
             "5. Analyze or evaluate points where relevant.\n"
             "6. Use appropriate academic vocabulary but ensure clarity. Define key terms if needed.\n"
             "7. Optionally include '## Connections' for real-world links.\n"
             "8. Conclude with ONE thought-provoking question or analysis task under '## Further Thinking'.\n"
             "9. Use clear Markdown.\n\n"
             "Text to summarize:\n{{text}}"
             f"{EOT}{ASSISTANT_START}# Comprehensive Summary\n\n"
        ),
        "math": (
            f"{USER_START}Explain an advanced math topic from text for a high school student (grades 7-12, ages 12-18).\n"
            "Instructions:\n"
            "1. Start with '# Advanced Math Concepts'.\n"
            "2. Provide concise definitions under '## Definitions'.\n"
            "3. Explain the core theory rigorously under '## Core Theory' using paragraphs and bullet points.\n"
            "4. Include a non-trivial worked example under '## Worked Example'. Show steps clearly.\n"
            "5. Discuss applications or connections under '## Applications'.\n"
            "6. Conclude with ONE challenging problem or extension under '## Challenge'.\n"
            "7. Use appropriate mathematical notation and clear Markdown.\n\n"
            "Text to explain:\n{{text}}"
            f"{EOT}{ASSISTANT_START}# Advanced Math Concepts\n\n"
        )
    }
}
# --- End Prompts ---

# --- Reading Level Helper ---
def determine_reading_level(grade):
    """Determines reading level category and description from grade."""
    # (No changes needed here)
    if not isinstance(grade, int) or not (1 <= grade <= 12): grade = 6
    age = grade + 5
    if 1 <= grade <= 3: level, desc = "lower", f"early elementary (grades {grade}, ~age {age}-{age+1})"
    elif 4 <= grade <= 6: level, desc = "middle", f"late elem./middle school (grades {grade}, ~age {age}-{age+1})"
    elif 7 <= grade <= 9: level, desc = "higher", f"junior high/early high (grades {grade}, ~age {age}-{age+1})"
    else: level, desc = "higher", f"high school (grades {grade}, ~age {age}-{age+1})"
    return level, desc

# --- Sentence Completion Helper (Text Only) ---
def complete_sentence(fragment, enable_global_toggle=True):
    """Complete sentence fragments using the loaded text LLM."""
    # Relies on global 'model' and 'tokenizer'
    if not model or not tokenizer:
        print("WARN (llm_utils): LLM/Tokenizer not available for sentence completion.")
        return fragment + "."

    if not enable_global_toggle: return fragment + "."
    if re.search(r'[.!?]$', fragment.strip()): return fragment
    if len(fragment.split()) < 3 or len(fragment) < 15: return fragment + "."

    # Simple text completion prompt (adjust if needed for Llama 3.1 style)
    prompt = f"Complete this sentence fragment naturally:\nFragment: '{fragment}'\nCompleted sentence:"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=30, temperature=0.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        completed_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract completed part
        match = re.search(r"Completed sentence:\s*(.*)", completed_full, re.I | re.S)
        if match:
            completed = match.group(1).strip()
            final = completed if completed.lower().startswith(fragment.lower()) else fragment + " " + completed
            final = re.sub(r'<\|.*?\|>', '', final).strip() # Remove potential special tokens like <|eot_id|>
            if not final: return fragment+"."
            if final and final[-1].isalnum(): final+='.'
            return final
        else: return fragment + "."
    except Exception as e: print(f"ERROR (llm_utils): Sentence completion failed: {e}"); return fragment + "."

# --- Text Model Generation Function ---
def model_generate(prompt_text, max_new_tokens=1024, temperature=0.6):
    """Generates text using the loaded LLM (handles context length)."""
    # Relies on global 'model' and 'tokenizer'
    if not model or not tokenizer:
        return "Error: LLM or Tokenizer not available."

    # Get model context limit (Llama 3.1 models typically have 8k context)
    model_context_limit = getattr(model.config, 'max_position_embeddings', 8192)
    # print(f"DEBUG (llm_utils): Using model context limit: {model_context_limit}")

    # --- Context Length Management ---
    # Estimate tokens in the prompt
    try:
        prompt_tokens = tokenizer(prompt_text, return_tensors="pt")['input_ids'].shape[1]
    except Exception as e:
        print(f"WARN (llm_utils): Could not precisely tokenize prompt for length check: {e}. Estimating.")
        prompt_tokens = len(prompt_text) // 4 # Rough estimate

    buffer_tokens = 50 # Small buffer for safety

    # Check if prompt + requested output might exceed limit
    if prompt_tokens + max_new_tokens + buffer_tokens > model_context_limit:
        original_max_new = max_new_tokens
        # Reduce max_new_tokens allowed
        max_new_tokens = model_context_limit - prompt_tokens - buffer_tokens
        if max_new_tokens < 100: # Check if reduction is too drastic
            err_msg = f"Error: Prompt ({prompt_tokens} tokens) is too long to generate a meaningful response within the model's context limit ({model_context_limit})."
            print(err_msg)
            return err_msg
        print(f"WARN (llm_utils): Reduced max_new_tokens from {original_max_new} to {max_new_tokens} to fit context limit.")

    # Ensure max_prompt_len calculation is safe if we were truncating (we aren't explicitly here, relying on generate)
    max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens

    try:
        # Tokenize the input prompt
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=False).to(model.device) # Don't truncate here, check length above
        input_token_count = inputs['input_ids'].shape[1]

        # Final check before generation
        if input_token_count >= model_context_limit - buffer_tokens:
             return f"Error: Input prompt ({input_token_count} tokens) already exceeds or meets model context limit ({model_context_limit}). Cannot generate."
        if input_token_count + max_new_tokens > model_context_limit - buffer_tokens:
            print(f"WARN (llm_utils): Final check indicates potential overflow. Clamping max_new_tokens.")
            max_new_tokens = model_context_limit - input_token_count - buffer_tokens


        print(f"INFO (llm_utils): Generating with input tokens={input_token_count}, max_new_tokens={max_new_tokens}")
        start_time = time.time()
        with torch.no_grad():
             # Generate text
             outputs = model.generate(
                 **inputs,
                 max_new_tokens=max_new_tokens,
                 temperature=temperature,
                 pad_token_id=tokenizer.eos_token_id, # Use EOS for padding
                 eos_token_id=tokenizer.eos_token_id, # Stop generation at EOS
                 do_sample=True if temperature > 0.01 else False
             )
        end_time = time.time(); print(f"INFO (llm_utils): Generation took {end_time - start_time:.2f}s.")

        # Decode generated tokens, skipping the prompt part
        generated_ids = outputs[0][input_token_count:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Basic cleanup
        generated_text = re.sub(r'<\|.*?\|>', '', generated_text).strip() # Remove special tokens like <|eot_id|>

        if not generated_text: print("WARN (llm_utils): Generation resulted in empty text.")
        elif len(generated_text) < 20: print("WARN (llm_utils): Generation resulted in very short text.")
        elif not re.search(r'(^#|^- )', generated_text, re.M): print("WARN (llm_utils): Generated text seems to lack expected structure.")

        return generated_text

    except torch.cuda.OutOfMemoryError as e:
        print(f"ERROR (llm_utils): OOM during generation: {e}"); traceback.print_exc(); torch.cuda.empty_cache()
        return "Error: GPU OOM during generation."
    except Exception as e:
        print(f"ERROR (llm_utils): Generation Error: {e}"); traceback.print_exc()
        return f"Error: Model generation failed - {str(e)}"

# --- Summary Orchestration (Text-Only) ---
def generate_summary(text_chunks, # Changed back from elements
                     grade_level_category, grade_level_desc, duration_minutes, has_math=False,
                     enable_completion=True, enable_deduplication=True):
    """Generates a summary using the text LLM from text chunks."""

    if not model or not tokenizer:
        return "Error: Text LLM or Tokenizer not initialized."

    # --- Target Word Count --- (Remains the same)
    if duration_minutes == 10: min_words, max_words = 1200, 1600
    elif duration_minutes == 20: min_words, max_words = 2400, 3200
    elif duration_minutes == 30: min_words, max_words = 3600, 4500
    else: min_words, max_words = 1200, 1600 # Default to short
    print(f"INFO (llm_utils): Targeting summary: {grade_level_desc}, {duration_minutes} mins ({min_words}-{max_words} words approx). Text-Only.")
    print(f"INFO (llm_utils): Refinement Options - Completion: {enable_completion}, Deduplication: {enable_deduplication}")

    # --- Determine Strategy: Single Pass vs. Iterative ---
    # Get model context limit
    model_context_limit = getattr(model.config, 'max_position_embeddings', 8192)
    # Estimate total tokens in all chunks
    full_text = "\n\n---\n\n".join(text_chunks) # Join chunks with separator
    try: full_text_tokens_estimate = len(tokenizer.encode(full_text))
    except: full_text_tokens_estimate = len(full_text) // 4 # Rough estimate

    # Estimate target tokens for summary output
    estimated_target_max_tokens = int(max_words * 1.5) + 300 # Word count * factor + buffer

    # Calculate a safe generation limit based on context
    # Allow ample room for prompt instructions etc.
    safe_generation_limit = max(model_context_limit // 2 - 500, 512)
    # Clamp max_new_tokens for the summary call
    max_new_tokens_summary = max(min(estimated_target_max_tokens, safe_generation_limit), 512)
    print(f"INFO (llm_utils): Calculated max_new_tokens for final summary generation: {max_new_tokens_summary}")

    # Estimate total tokens needed for a single pass
    # Prompt instructions themselves take some tokens
    prompt_instruction_buffer = 600 # Estimate tokens for prompt text besides the input chunks
    required_tokens_for_single_pass = full_text_tokens_estimate + max_new_tokens_summary + prompt_instruction_buffer

    # Decide if a single pass is feasible (within ~90% of context limit)
    can_summarize_all_at_once = required_tokens_for_single_pass < (model_context_limit * 0.90)
    print(f"INFO (llm_utils): Estimated tokens needed for single pass: {required_tokens_for_single_pass} (Limit: {model_context_limit}). Single pass feasible: {can_summarize_all_at_once}")

    initial_summary = ""
    # --- Single Pass Generation ---
    if can_summarize_all_at_once and text_chunks:
        print("INFO (llm_utils): Attempting summary generation in a single pass.")
        prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
        # Use apply_chat_template for robustness if possible
        try:
             messages = [
                 {"role": "user", "content": prompt_template.split('{{text}}')[0].replace(USER_START,'').replace(EOT,'').strip() + f"\n\nText to summarize:\n{full_text}"},
                 {"role": "assistant", "content": prompt_template.split(ASSISTANT_START)[-1].strip()} # Prime assistant
             ]
             prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) # Keep add_generation_prompt=False as we added assistant start manually
             print("DEBUG (llm_utils): Using apply_chat_template for single pass.")
        except Exception as e:
            print(f"WARN (llm_utils): apply_chat_template failed ({e}). Falling back to manual formatting.")
            # Manual formatting as fallback
            try:
                 prompt = prompt_template.replace("{{text}}", full_text)
                 # Ensure it ends correctly to prompt the assistant
                 if not prompt.endswith(ASSISTANT_START):
                      prompt = prompt.split(ASSISTANT_START)[0] + ASSISTANT_START
            except KeyError:
                print("ERROR (llm_utils): Invalid prompt template key.")
                return "Error: Internal prompt configuration error."

        initial_summary = model_generate(prompt, max_new_tokens=max_new_tokens_summary, temperature=0.6)

    # --- Iterative (Map-Reduce Style) Generation ---
    elif text_chunks:
        print(f"INFO (llm_utils): Input too large for single pass. Using iterative summarization ({len(text_chunks)} chunks).")
        chunk_summaries = []
        # Calculate max tokens per chunk summary (distribute remaining context)
        # Allow generous space for chunk prompt text
        max_tokens_per_chunk_prompt = model_context_limit // 2
        max_tokens_chunk_gen = max(min(max_tokens_per_chunk_prompt // 3, 400), 150) # Generate concise points per chunk
        print(f"INFO (llm_utils): Max new tokens per chunk summary: {max_tokens_chunk_gen}")

        for i, chunk in enumerate(text_chunks):
            print(f"  - Summarizing chunk {i+1}/{len(text_chunks)}...")
            # Simple prompt for extracting key points from each chunk
            # Using Llama 3 format
            chunk_messages = [
                 {"role": "user", "content": f"Extract the most important key points from the following text chunk ({i+1}/{len(text_chunks)}). Present them as a CONCISE bulleted list.\n\nText Chunk:\n{chunk}"},
                 {"role": "assistant", "content": "- "} # Prime assistant to start bullet list
            ]
            try:
                chunk_prompt = tokenizer.apply_chat_template(chunk_messages, tokenize=False, add_generation_prompt=False) # Keep add_generation_prompt=False
            except Exception:
                chunk_prompt = f"{USER_START}Extract the most important key points from the following text chunk ({i+1}/{len(text_chunks)}). Present them as a CONCISE bulleted list.\n\nText Chunk:\n{chunk}{EOT}{ASSISTANT_START}- "


            chunk_summary = model_generate(chunk_prompt, max_new_tokens=max_tokens_chunk_gen, temperature=0.4) # Lower temp for factual extraction

            # Basic validation of chunk summary
            if chunk_summary.startswith("Error:") or len(chunk_summary.split()) < 5:
                print(f"WARN (llm_utils): Skipping invalid summary for chunk {i+1}.")
                continue

            # Clean up the chunk summary (remove potential prompt remnants if needed)
            # The priming with "- " should help
            cleaned_chunk_summary = chunk_summary.strip()
            if not cleaned_chunk_summary.startswith("-"):
                 cleaned_chunk_summary = "- " + cleaned_chunk_summary # Ensure it starts as a list item

            if cleaned_chunk_summary:
                chunk_summaries.append(cleaned_chunk_summary)
                # print(f"DEBUG: Chunk {i+1} summary:\n{cleaned_chunk_summary[:100]}...") # Optional debug

        if not chunk_summaries:
            return "Error: Failed to generate valid summaries for any text chunk."

        # --- Consolidation Step ---
        print("INFO (llm_utils): Consolidating chunk summaries...")
        combined_chunk_summaries = "\n".join(chunk_summaries)

        # Get the base instructions from the original prompt template
        prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
        base_instructions = prompt_template.split("Text to summarize:")[0].replace(USER_START,'').strip()

        consolidation_messages = [
            {"role": "user", "content": f"{base_instructions}\n\nSynthesize the following key points extracted from different text chunks into a single, coherent summary formatted according to the instructions above. Aim for approximately {min_words}-{max_words} words.\n\nExtracted Key Points:\n{combined_chunk_summaries}"},
            {"role": "assistant", "content": prompt_template.split(ASSISTANT_START)[-1].strip()} # Prime assistant
        ]
        try:
            consolidation_prompt = tokenizer.apply_chat_template(consolidation_messages, tokenize=False, add_generation_prompt=False) # Keep add_generation_prompt=False
            print("DEBUG (llm_utils): Using apply_chat_template for consolidation.")
        except Exception as e:
            print(f"WARN (llm_utils): apply_chat_template failed for consolidation ({e}). Using manual formatting.")
            consolidation_prompt = f"{USER_START}{consolidation_messages[0]['content']}{EOT}{ASSISTANT_START}{consolidation_messages[1]['content']}"

        initial_summary = model_generate(consolidation_prompt, max_new_tokens=max_new_tokens_summary, temperature=0.65) # Slightly higher temp for synthesis

    else: # Should not happen if text_chunks is validated earlier
        return "Error: No text chunks provided to generate summary."


    # --- Process Initial Summary ---
    if initial_summary.startswith("Error:"):
        return initial_summary # Return error from generation

    current_summary = initial_summary
    current_word_count = len(current_summary.split())
    print(f"INFO (llm_utils): Initial summary generated ({current_word_count} words).")

    # --- Length Adjustment (Text-Based) ---
    # Lengthening if too short
    attempts, max_attempts = 0, 2
    while current_word_count < min_words and attempts < max_attempts:
        print(f"INFO (llm_utils): Summary short ({current_word_count} words, target {min_words}). Elaborating (Attempt {attempts + 1}/{max_attempts})...")
        # Simple prompt to continue/elaborate
        lengthen_messages = [
             {"role": "user", "content": f"The following summary is too short. Please elaborate on the existing points or add more relevant details based on the initial context, maintaining the same style and structure.\n\nExisting Summary:\n{current_summary}"},
             {"role": "assistant", "content": current_summary + "\n\n"} # Prime with existing summary
        ]
        try:
            lengthen_prompt = tokenizer.apply_chat_template(lengthen_messages, tokenize=False, add_generation_prompt=False)
        except Exception:
             lengthen_prompt = f"{USER_START}{lengthen_messages[0]['content']}{EOT}{ASSISTANT_START}{lengthen_messages[1]['content']}"

        # Calculate how many tokens to add (approx)
        words_needed = min_words - current_word_count
        tokens_to_add = max(min(int(words_needed * 1.7), max_new_tokens_summary // 2, 800), 150) # Generous estimate

        new_part = model_generate(lengthen_prompt, max_new_tokens=tokens_to_add, temperature=0.7)

        if new_part.startswith("Error:") or len(new_part.strip().split()) < 10:
            print("INFO (llm_utils): Elaborated part is too short or errored. Stopping lengthening.")
            break # Stop if elaboration fails or is too short

        # Append the new part (ensure proper spacing)
        current_summary = new_part.strip() # Assume LLM continues from the priming
        # current_summary += "\n\n" + new_part.strip() # Alternative: just append
        current_word_count = len(current_summary.split())
        attempts += 1
        print(f"INFO (llm_utils): Lengthened summary to {current_word_count} words.")

    if attempts == max_attempts and current_word_count < min_words:
        print(f"WARN (llm_utils): Max lengthening attempts reached, summary may still be short ({current_word_count} words).")


    # Trimming if too long (same logic as before)
    words = current_summary.split()
    if len(words) > max_words:
        print(f"INFO (llm_utils): Trimming summary from {len(words)} to ~{max_words} words.")
        activity_pattern = r'(##\s+(Activity|Practice|Thinking|Challenge|Try This|Fun Activity))'
        activity_match = re.search(activity_pattern, current_summary, re.I | re.M)
        if activity_match:
            idx = activity_match.start()
            main_content = current_summary[:idx].strip()
            activity_content = current_summary[idx:].strip()
            main_words = main_content.split()
            if len(main_words) > max_words:
                 limit_idx = len(' '.join(main_words[:max_words]))
                 end_idx = main_content.rfind('.', 0, limit_idx)
                 main_content = main_content[:end_idx + 1] if end_idx != -1 else ' '.join(main_words[:max_words]) + "..."
            current_summary = main_content + "\n\n" + activity_content
        else:
            current_summary = ' '.join(words[:max_words])
            if not re.search(r'[.!?]$', current_summary): current_summary += "..."
        print(f"INFO (llm_utils): Trimmed summary word count: {len(current_summary.split())}")

    summary = current_summary

    # --- Post-processing (Text Output) ---
    print("INFO (llm_utils): Post-processing generated summary text...")
    processed_summary = enhanced_post_process(summary, grade_level_category, has_math,
                                              enable_completion=enable_completion,
                                              enable_deduplication=enable_deduplication)

    # Fallback activity generation (uses text context)
    activity_pattern = r'^##\s+(Fun Activity|Practice.*|Try This|Further Thinking|Challenge|Activity)\s*$'
    if not re.search(activity_pattern, processed_summary, re.I | re.M):
        print("WARN (llm_utils): Activity section missing. Generating fallback...")
        activity = generate_activity(processed_summary, grade_level_category, grade_level_desc)
        h_map={"lower":"## Fun Activity","middle":"## Try This","higher":"## Further Thinking"}; def_h="## Activity Suggestion"
        if has_math: head = {"lower":"## Practice Time","middle":"## Practice Problem","higher":"## Challenge"}.get(grade_level_category, def_h)
        else: head = h_map.get(grade_level_category, def_h)
        processed_summary += f"\n\n{head}\n{activity}"
    else: print("INFO (llm_utils): Activity section found in summary.")

    final_word_count = len(processed_summary.split())
    print(f"INFO (llm_utils): Final summary generated ({final_word_count} words).")
    return processed_summary
# --- End generate_summary ---


# --- Post-processing Function (Text Output) ---
def enhanced_post_process(summary, grade_level_category, has_math, enable_completion=True, enable_deduplication=True):
    """Advanced post-processing on the generated text summary."""
    # Relies on global 'model', 'tokenizer', 'embedder'
    if summary.startswith("Error:"): return summary
    print(f"INFO (llm_utils): Running enhanced post-processing (Comp:{enable_completion}, Dedup:{enable_deduplication})...")

    if not tokenizer: # Check if tokenizer is available
        print("WARN (llm_utils): Tokenizer unavailable for post-processing. Skipping some steps.")
        # Allow basic formatting but skip completion/dedup? Or return early?
        # Returning early might be safer if completion relies heavily on tokenizer context
        return summary # Return summary as-is if no tokenizer

    completion_calls_made = 0

    # --- Heading Standardization ---
    # (Using the same logic as before, based on prompt structure)
    try:
        prompt_template = prompts[grade_level_category]["math" if has_math else "standard"] # Need has_math context here ideally, use heuristic
        expected_start_line = ""
        lines = prompt_template.splitlines()
        for i, line in enumerate(lines):
             if ASSISTANT_START.strip() in line:
                  if i + 1 < len(lines): expected_start_line = lines[i+1].strip(); break
        if expected_start_line and expected_start_line.startswith("#"): expected_heading = expected_start_line
        else: expected_heading = "# Summary" # Default

        summary = re.sub(r'^\s*#+.*?\n', '', summary.strip())
        summary = f'{expected_heading}\n\n' + summary
    except Exception as e:
        print(f"WARN (llm_utils): Error standardizing heading: {e}. Using default.")
        summary = re.sub(r'^\s*#+.*?\n', '', summary.strip())
        summary = '# Summary\n\n' + summary


    # --- Process Lines & Structure --- (Largely the same as before)
    lines = summary.split('\n'); processed_data = []; seen_frags = set()
    for line in lines:
        s_line = line.strip()
        if not s_line:
            if processed_data and processed_data[-1]["type"]!="blank": processed_data.append({"text":"","type":"blank"}); continue
        l_type, content, is_head, is_bullet="paragraph", s_line, False, False
        if s_line.startswith('## '): l_type, content, is_head = "subheading", s_line[3:].strip(), True
        elif s_line.startswith('# '): l_type, content, is_head = "heading", s_line[2:].strip(), True # Keep content including #
        elif s_line.startswith('- '): l_type, content, is_bullet = "bullet", s_line[2:].strip(), True
        elif re.match(r'^\d+\.\s+', s_line): l_type, content, is_bullet = "numbered", re.sub(r'^\d+\.\s+', '', s_line), True
        if not content and l_type not in ["heading","subheading"]: continue # Allow empty headings
        cont_key = ' '.join(content.lower().split()[:10])
        if not is_head and len(content.split()) < 15 and cont_key in seen_frags: continue
        if not is_head: seen_frags.add(cont_key)

        # Sentence Completion (if enabled)
        if enable_completion and l_type in ["paragraph", "bullet", "numbered"] and len(content.split()) > 4:
            if not re.search(r'[.!?:]$', content) and content[0].isupper() and completion_calls_made < MAX_COMPLETION_CALLS:
                original_content = content
                content = complete_sentence(content, enable_global_toggle=enable_completion) # Use helper
                if content != original_content and not content.endswith(original_content + "."): completion_calls_made += 1

        # Casing/Punctuation
        if l_type in ["paragraph", "bullet", "numbered"]:
             if content and content[0].islower() and not re.match(r'^[a-z][)\.]\s+', content): content = content[0].upper()+content[1:]
             if content and content[-1].isalnum(): content += '.'

        if l_type == "blank" and processed_data and processed_data[-1]["type"] == "blank": continue
        # Store content based on type for assembly later
        if l_type == "heading": processed_data.append({"text": content, "type": l_type}) # Store full heading line
        else: processed_data.append({"text":content, "type":l_type})


    # --- Semantic Deduplication --- (Same as before, uses text embedder)
    points_for_dedup = []; indices_map = {}
    if enable_deduplication and embedder:
        for i, data in enumerate(processed_data):
            if data["type"] in ["paragraph","bullet","numbered"] and len(data["text"].split()) > 6:
                cont = data["text"]; points_for_dedup.append(cont)
                if cont not in indices_map: indices_map[cont] = []
                indices_map[cont].append(i)
    elif not enable_deduplication: print("INFO (llm_utils): Skipping semantic dedup (disabled).")
    elif not embedder: print("INFO (llm_utils): Skipping semantic dedup (no embedder).")

    kept_indices = set(range(len(processed_data)))
    if points_for_dedup and enable_deduplication and embedder:
        print(f"INFO (llm_utils): Running semantic dedup on {len(points_for_dedup)} points...")
        try:
            unique_pts = remove_duplicates_semantic(points_for_dedup, batch_size=128); unique_set = set(unique_pts)
            print(f"INFO (llm_utils): Reduced to {len(unique_pts)} unique points.")
            indices_to_remove = set(); processed_removal = set()
            for cont, orig_indices in indices_map.items():
                if cont in processed_removal: continue
                if cont not in unique_set: indices_to_remove.update(orig_indices[1:]) # Keep first, remove others
                processed_removal.add(cont)
            kept_indices -= indices_to_remove; print(f"INFO (llm_utils): Marked {len(indices_to_remove)} lines for removal.")
        except Exception as e: print(f"WARN (llm_utils): Dedup failed: {e}")


    # --- Final Assembly --- (Adjusted slightly for heading handling)
    final_text = ""; last_type = None
    kept_data = [processed_data[i] for i in sorted(list(kept_indices))]
    for i, data in enumerate(kept_data):
        curr_type, content = data["type"], data["text"]
        # Spacing
        if i > 0:
            prev_type = kept_data[i-1]["type"]
            if curr_type in ["heading","subheading"]: final_text += "\n\n"
            elif curr_type == "paragraph" and prev_type not in ["heading","subheading","blank"]: final_text += "\n\n"
            elif curr_type != "blank" and prev_type != "blank": final_text += "\n"
            elif curr_type == "blank" and prev_type == "blank": continue
        # Content
        if curr_type == "heading": final_text += content # Already has #
        elif curr_type == "subheading": final_text += f"## {content}"
        elif curr_type == "bullet": final_text += f"- {content}"
        elif curr_type == "numbered": final_text += f"1. {content}" # Basic numbering reset
        elif curr_type == "paragraph": final_text += content
        last_type = curr_type

    print("INFO (llm_utils): Post-processing finished.")
    return final_text.strip()
# --- End enhanced_post_process ---


# --- remove_duplicates_semantic function (Text Embeddings) ---
# (No changes needed here - it already works on text points using the embedder)
def remove_duplicates_semantic(points, similarity_threshold=0.90, batch_size=64):
    """Removes semantically similar points using sentence-transformers."""
    if not points or not embedder or len(points) < 2:
        return points

    start_dedup = time.time()
    try:
        valid_pts_indices = {i: p for i, p in enumerate(points) if len(p.split()) > 4}
        valid_pts = list(valid_pts_indices.values())
        original_indices = list(valid_pts_indices.keys())
        if not valid_pts: return points

        embedder_device = getattr(embedder, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        embeddings = embedder.encode(valid_pts, convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size, device=embedder_device)
        cos_sim = util.cos_sim(embeddings, embeddings).cpu()
        to_remove_indices_set = set()

        for i in range(len(valid_pts)):
            if i in to_remove_indices_set: continue
            for j in range(i + 1, len(valid_pts)):
                if j in to_remove_indices_set: continue
                if cos_sim[i][j] > similarity_threshold:
                    to_remove_indices_set.add(j)

        unique_valid_pts = [valid_pts[i] for i in range(len(valid_pts)) if i not in to_remove_indices_set]
        short_pts_set = set(p for p in points if len(p.split()) <= 4)
        unique_pts = unique_valid_pts + list(short_pts_set)
        print(f"INFO (llm_utils): Semantic deduplication took {time.time() - start_dedup:.2f}s. Removed {len(to_remove_indices_set)} similar points.")
        return unique_pts

    except torch.cuda.OutOfMemoryError:
        print("ERROR (llm_utils): OOM during semantic deduplication. Skipping.")
        torch.cuda.empty_cache(); return points
    except Exception as e:
        print(f"ERROR (llm_utils): Error during semantic deduplication: {e}")
        traceback.print_exc(); return points
# --- End remove_duplicates_semantic ---


# --- generate_activity function (Text Output) ---
# (No changes needed here - it already works on text summary)
def generate_activity(summary_text, grade_level_category, grade_level_desc):
    """Generates a fallback activity suggestion based on the text summary."""
    if not model or not tokenizer:
        print("WARN (llm_utils): LLM/Tokenizer unavailable for activity generation.")
        return "- Review the key points of the summary."

    print("INFO (llm_utils): Generating fallback activity suggestion...")
    act_type = {"lower": "simple, fun activity or question",
                "middle": "practical activity or thought question",
                "higher": "thought-provoking question, research idea, or analysis task"
               }.get(grade_level_category, "activity suggestion")
    context_text = ' '.join(re.sub(r'^#.*?\n', '', summary_text).strip().split()[-250:])

    # Using apply_chat_template for activity prompt
    activity_messages = [
         {"role": "user", "content": f"Based ONLY on the following summary text, suggest ONE engaging {act_type} suitable for a {grade_level_desc}:\n\nSummary Context:\n\"...{context_text}...\"\n\nActivity Suggestion (provide only the activity itself, starting with a verb):"},
         {"role": "assistant", "content": ""} # Let assistant generate freely
    ]
    try:
        activity_prompt = tokenizer.apply_chat_template(activity_messages, tokenize=False, add_generation_prompt=True) # Add generation prompt here
    except Exception as e:
         print(f"WARN (llm_utils): apply_chat_template failed for activity ({e}). Using manual format.")
         activity_prompt = f"{USER_START}{activity_messages[0]['content']}{EOT}{ASSISTANT_START}"

    activity_text = ""
    try:
        activity_raw = model_generate(activity_prompt, max_new_tokens=80, temperature=0.7)
        if not activity_raw.startswith("Error:"):
            activity_text = activity_raw.strip()
            activity_text = re.sub(r'^(Activity Suggestion:|Here is an activity:|Sure, here.*?:)\s*', '', activity_text, flags=re.I).strip()
            if activity_text and not activity_text.startswith(("-","*","1.")): activity_text = f"- {activity_text}"
            activity_text = re.sub(r'^([\-\*\d\.]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), activity_text)
            if activity_text and activity_text[-1].isalnum(): activity_text += '.'
    except Exception as e:
        print(f"ERROR (llm_utils): Exception during activity generation: {e}")

    if not activity_text:
        print("WARN (llm_utils): Using default fallback activity.")
        fallbacks = {
            "lower": "- Draw a picture about the summary!",
            "middle": "- Explain the main idea in your own words.",
            "higher": "- Find one real-world example related to the summary."
        }
        activity_text = fallbacks.get(grade_level_category, "- Review the key points.")
    return activity_text
# --- End generate_activity ---