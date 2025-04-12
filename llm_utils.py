import re
import torch
import time
import traceback
from nltk.stem import PorterStemmer
from sentence_transformers import util  # FIXED: Missing import

# Global placeholders
model = None
tokenizer = None
embedder = None
stemmer = PorterStemmer()

MAX_COMPLETION_CALLS = 10

def initialize_globals(llm_model, llm_tokenizer, sentence_embedder):
    global model, tokenizer, embedder
    model = llm_model
    tokenizer = llm_tokenizer
    embedder = sentence_embedder
    print("INFO (llm_utils): LLM utilities initialized with models.")


# --- Prompts Dictionary (Refined for Llama-4-Scout-17B-Instruct / Llama 3 Instruct) ---
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

# --- Reading Level Helper ---
def determine_reading_level(grade):
    """Determines reading level category and description from grade."""
    if not isinstance(grade, int) or not (1 <= grade <= 12): grade = 6
    age = grade + 5
    if 1 <= grade <= 3: level, desc = "lower", f"early elementary (grades {grade}, ~age {age}-{age+1})"
    elif 4 <= grade <= 6: level, desc = "middle", f"late elem./middle school (grades {grade}, ~age {age}-{age+1})"
    elif 7 <= grade <= 9: level, desc = "higher", f"junior high/early high (grades {grade}, ~age {age}-{age+1})"
    else: level, desc = "higher", f"high school (grades {grade}, ~age {age}-{age+1})"
    return level, desc

# --- Sentence Completion Helper ---
def complete_sentence(fragment, enable_global_toggle=True):
    """Complete sentence fragments using the loaded LLM. Respects toggle."""
    # Relies on global 'model' and 'tokenizer' or them being passed somehow
    if not model or not tokenizer: print("WARN (llm_utils): LLM not available for sentence completion."); return fragment + "."

    if not enable_global_toggle: return fragment + "."
    if re.search(r'[.!?]$', fragment.strip()): return fragment
    if len(fragment.split()) < 3 or len(fragment) < 15: return fragment + "."

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
        match = re.search(r"Completed sentence:\s*(.*)", completed_full, re.I | re.S)
        if match:
            completed = match.group(1).strip()
            final = completed if completed.lower().startswith(fragment.lower()) and len(completed)>len(fragment)+3 else (fragment+"." if completed.lower().startswith(fragment.lower()) else completed)
            final = re.sub(r'<\|eot_id\|>', '', final).strip(); # Adjust token if needed
            if not final: return fragment+"."
            if final and final[-1].isalnum(): final+='.'
            return final
        else: return fragment + "."
    except Exception as e: print(f"ERROR (llm_utils): Sentence completion failed: {e}"); return fragment + "."

# --- Model Generation Function ---
def model_generate(prompt_text, max_new_tokens=1024, temperature=0.6):
    """Generates text using the loaded LLM, handling context length."""
    # Relies on global 'model' and 'tokenizer'
    if not model or not tokenizer: return "Error: LLM not available."
    current_model_device = model.device

    # Get model context limit
    model_context_limit = getattr(model.config, 'max_position_embeddings', None)
    if model_context_limit is None: model_context_limit = getattr(tokenizer, 'model_max_length', None)
    if not isinstance(model_context_limit, int) or model_context_limit <= 512: model_context_limit = 8192

    # Robust Length Calculation
    if max_new_tokens >= model_context_limit: max_new_tokens = model_context_limit // 2
    buffer_tokens = 150
    max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens
    if max_prompt_len <= 0:
        needed = abs(max_prompt_len) + 10; max_new_tokens -= needed
        max_prompt_len = model_context_limit - max_new_tokens - buffer_tokens
        if max_prompt_len <= 0 or max_new_tokens <= 50: return f"Error: Gen request too large ({model_context_limit})."
        print(f"WARN (llm_utils): Reduced max_new_tokens to {max_new_tokens} to fit context.")
    max_prompt_len = min(max_prompt_len, model_context_limit)
    # print(f"DEBUG (llm_utils): Ctx={model_context_limit}, New={max_new_tokens}, PromptMax={max_prompt_len}")

    try:
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(current_model_device)
        input_token_count = inputs['input_ids'].shape[1]
        if input_token_count >= max_prompt_len: print(f"WARN (llm_utils): Prompt potentially truncated.")

        start_time = time.time()
        with torch.no_grad():
             outputs = model.generate(
                 **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                 pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                 do_sample=True if temperature > 0.01 else False
             )
        end_time = time.time(); print(f"INFO (llm_utils): Generation took {end_time - start_time:.2f}s.")

        generated_text = tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True)
        generated_text = re.sub(r'<\|eot_id\|>', '', generated_text).strip() # Adjust token if needed

        if not re.search(r'(^#|^- )', generated_text, re.M): print("WARN (llm_utils): Generated text seems to lack structure.")
        if len(generated_text) < 20: print("WARN (llm_utils): Generation resulted in very short text.")
        return generated_text
    except torch.cuda.OutOfMemoryError as e:
        print(f"ERROR (llm_utils): OOM during generation: {e}"); traceback.print_exc(); torch.cuda.empty_cache()
        return f"Error: GPU OOM during generation."
    except Exception as e:
        print(f"ERROR (llm_utils): Generation Error: {e}"); traceback.print_exc()
        return f"Error: Model generation failed - {str(e)}"

# --- Summary Orchestration ---
def generate_summary(text_chunks, grade_level_category, grade_level_desc, duration_minutes, has_math=False,
                     enable_completion=True, enable_deduplication=True):
    """Generates the final summary, handling chunking, consolidation, length adjustment."""
    # Relies on global 'model' and 'tokenizer' via model_generate

    # Target word counts based on duration
    if duration_minutes == 10: min_words, max_words = 1200, 1600
    elif duration_minutes == 20: min_words, max_words = 2400, 3200
    elif duration_minutes == 30: min_words, max_words = 3600, 4500
    else: min_words, max_words = 1200, 1600
    print(f"INFO (llm_utils): Targeting summary: {grade_level_desc}, {duration_minutes} mins ({min_words}-{max_words} words approx).")
    print(f"INFO (llm_utils): Refinement Options - Completion: {enable_completion}, Deduplication: {enable_deduplication}")

    # Get model context limit (for calculating generation sizes)
    model_context_limit = getattr(model.config, 'max_position_embeddings', None)
    if model_context_limit is None: model_context_limit = getattr(tokenizer, 'model_max_length', None)
    if not isinstance(model_context_limit, int) or model_context_limit <= 512: model_context_limit = 8192
    print(f"DEBUG (llm_utils): Using model_context_limit = {model_context_limit}")

    full_text = ' '.join(text_chunks)
    try: full_text_tokens_estimate = len(tokenizer.encode(full_text))
    except: full_text_tokens_estimate = len(full_text) // 3

    estimated_target_max_tokens = int(max_words * 1.3) + 200
    safe_generation_limit = max((model_context_limit // 2) - 150, 512)
    max_new_tokens_summary = max(min(estimated_target_max_tokens, safe_generation_limit), 512)
    print(f"INFO (llm_utils): Calculated max_new_tokens for summary: {max_new_tokens_summary}")

    prompt_instruction_buffer = 700
    required_tokens_for_single_pass = full_text_tokens_estimate + max_new_tokens_summary + prompt_instruction_buffer
    can_summarize_all_at_once = (required_tokens_for_single_pass < (model_context_limit * 0.9) and
                                 full_text_tokens_estimate < (model_context_limit * 0.6) and
                                 max_new_tokens_summary <= 4096)

    initial_summary = ""
    if can_summarize_all_at_once and text_chunks:
        print("INFO (llm_utils): Attempting summary generation in a single pass.")
        prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
        try: prompt = prompt_template.format(text=full_text) + f"\n\nIMPORTANT: Structure summary ({min_words}-{max_words} words)."
        except KeyError: prompt = f"Create summary for {grade_level_desc} ({min_words}-{max_words} words):\n{full_text}"
        initial_summary = model_generate(prompt, max_new_tokens=max_new_tokens_summary, temperature=0.6)

    elif text_chunks:
        print(f"INFO (llm_utils): Iterative summary ({len(text_chunks)} chunks).")
        chunk_summaries = []; max_tokens_chunk = max(min((model_context_limit//(len(text_chunks)+1))-150, 300), 100)
        print(f"INFO (llm_utils): Max new tokens per chunk: {max_tokens_chunk}")
        for i, chunk in enumerate(text_chunks):
            chunk_prompt = f"Key points from chunk {i+1}/{len(text_chunks)}:\n{chunk}\n\nKey Points (CONCISE bullet list):"
            chunk_summary = model_generate(chunk_prompt, max_new_tokens=max_tokens_chunk, temperature=0.4)
            if chunk_summary.startswith("Error:") or len(chunk_summary.split())<5: continue
            chunk_summary = re.sub(r'^.*?Key Points.*?\n','',chunk_summary,flags=re.I).strip()
            if chunk_summary: chunk_summaries.append(chunk_summary)
        if not chunk_summaries: return "Error: No valid chunk summaries."
        print("INFO (llm_utils): Consolidating chunk summaries...")
        template = prompts[grade_level_category]["math" if has_math else "standard"]
        base_instr = template.split("Text to summarize:")[0]
        consol_prompt = f"{base_instr}\nSYNTHESIZE points below into ONE summary for {grade_level_desc}.\nFollow instructions. Aim for {min_words}-{max_words} words.\n\nChunk Summaries:\n\n"+"\n\n---\n\n".join(chunk_summaries)+"\n\nFinal Consolidated Summary:"
        initial_summary = model_generate(consol_prompt, max_new_tokens=max_new_tokens_summary, temperature=0.65)
    else: return "Error: Zero chunks."

    if initial_summary.startswith("Error:"): return initial_summary
    current_summary = initial_summary; current_word_count = len(current_summary.split())
    print(f"INFO (llm_utils): Initial summary: {current_word_count} words.")

    # Lengthening
    attempts, max_attempts = 0, 2
    while current_word_count < min_words and attempts < max_attempts:
        print(f"INFO (llm_utils): Summary short. Elaborating (Attempt {attempts + 1}/{max_attempts})...")
        prompt = f"Elaborate on points...Current Summary:\n{current_summary}\n\nContinue summary:"
        needed = min_words - current_word_count
        tokens_add = max(min(int(needed * 1.5), max_new_tokens_summary // 2, 700), 150)
        new_part = model_generate(prompt, max_new_tokens=tokens_add, temperature=0.7)
        if new_part.startswith("Error:") or len(new_part.split()) < 10: print("INFO (llm_utils): Stopping lengthening."); break
        current_summary += "\n\n" + new_part.strip(); current_word_count = len(current_summary.split()); attempts += 1
    if attempts == max_attempts and current_word_count < min_words: print(f"WARN (llm_utils): Max lengthening reached.")

    # Trimming
    words = current_summary.split()
    if len(words) > max_words:
        print(f"INFO (llm_utils): Trimming summary from {len(words)} to ~{max_words} words.")
        activity_pattern = r'(##\s+(Activity|Practice|Thinking|Challenge|Try This|Fun Activity))'
        activity_match = re.search(activity_pattern, current_summary, re.I | re.M)
        if activity_match:
            idx = activity_match.start(); main_cont, act_cont = current_summary[:idx], current_summary[idx:]
            main_w = main_cont.split()
            if len(main_w) > max_words:
                 limit_idx = len(' '.join(main_w[:max_words])); end_idx = main_cont.rfind('.', 0, limit_idx)
                 main_cont = main_cont[:end_idx + 1] if end_idx != -1 else ' '.join(main_w[:max_words]) + "..."
            current_summary = main_cont.strip() + "\n\n" + act_cont.strip()
        else:
            current_summary = ' '.join(words[:max_words]); current_summary += "..." if not re.search(r'[.!?]$', current_summary) else ""
    summary = current_summary

    print("INFO (llm_utils): Post-processing summary...")
    # Pass refinement flags from function args
    processed_summary = enhanced_post_process(summary, grade_level_category,
                                              enable_completion=enable_completion,
                                              enable_deduplication=enable_deduplication)

    # Fallback activity
    activity_pattern = r'^##\s+(Fun Activity|Practice.*|Try This|Further Thinking|Challenge|Activity)\s*$'
    if not re.search(activity_pattern, processed_summary, re.I | re.M):
        print("WARN (llm_utils): Activity section missing. Generating fallback...")
        activity = generate_activity(processed_summary, grade_level_category, grade_level_desc)
        h_map={"lower":"## Fun Activity","middle":"## Try This","higher":"## Further Thinking"}; def_h="## Activity Suggestion"
        if has_math: head = {"lower":"## Practice Time","middle":"## Practice Problem","higher":"## Challenge"}.get(grade_level_category, def_h)
        else: head = h_map.get(grade_level_category, def_h)
        processed_summary += f"\n\n{head}\n{activity}"
    else: print("INFO (llm_utils): Activity section found.")

    final_word_count = len(processed_summary.split())
    print(f"INFO (llm_utils): Final summary generated ({final_word_count} words).")
    return processed_summary
# --- End generate_summary ---

# --- Post-processing Function ---
def enhanced_post_process(summary, grade_level_category, enable_completion=True, enable_deduplication=True):
    """Advanced post-processing with toggles for completion and deduplication."""
    # Relies on global 'model', 'tokenizer', 'embedder' via helpers
    if summary.startswith("Error:"): return summary
    print(f"INFO (llm_utils): Running enhanced post-processing (Comp:{enable_completion}, Dedup:{enable_deduplication})...")
    completion_calls_made = 0

    # Heading Standardization
    try: head_line=next((l for l in prompts[grade_level_category]["standard"].splitlines() if l.strip().startswith('#')), None); exp_head=head_line.strip().lstrip('# ').strip() if head_line else "Summary"
    except: exp_head = "Summary"
    summary=re.sub(r'^\s*#+.*?(\n|$)','',summary.strip()); summary=f'# {exp_head}\n\n'+summary

    # Process Lines & Structure
    lines=summary.split('\n'); processed_data, seen_frags=[], set()
    for line in lines:
        s_line=line.strip()
        if not s_line:
            if processed_data and processed_data[-1]["type"]!="blank": processed_data.append({"text":"","type":"blank"}); continue
        l_type, content, is_head, is_bullet="paragraph", s_line, False, False
        if s_line.startswith('## '): l_type, content, is_head = "subheading", s_line[3:].strip(), True
        elif s_line.startswith('# '): l_type, content, is_head = "heading", s_line[2:].strip(), True
        elif s_line.startswith('- '): l_type, content, is_bullet = "bullet", s_line[2:].strip(), True
        elif re.match(r'^\d+\.\s+', s_line): l_type, content, is_bullet = "numbered", re.sub(r'^\d+\.\s+', '', s_line), True
        if not content: continue
        cont_key = ' '.join(content.lower().split()[:10]) # Basic duplicate check
        if not is_head and cont_key in seen_frags and len(content.split()) < 15: continue
        if not is_head: seen_frags.add(cont_key)

        # Sentence Completion
        if enable_completion and l_type in ["paragraph", "bullet", "numbered"] and len(content.split()) > 4:
            if not re.search(r'[.!?:]$', content) and content[0].isupper() and completion_calls_made < MAX_COMPLETION_CALLS:
                original_content = content
                content = complete_sentence(content, enable_global_toggle=enable_completion) # Use helper
                if content != original_content and not content.endswith(original_content + "."): completion_calls_made += 1
        # Casing/Punctuation
        if l_type in ["paragraph", "bullet", "numbered"]:
             if content and content[0].islower() and not re.match(r'^[a-z]\s*\(', content): content = content[0].upper()+content[1:]
             if content and content[-1].isalnum(): content += '.'

        if l_type == "blank" and processed_data and processed_data[-1]["type"] == "blank": continue
        processed_data.append({"text":content, "type":l_type})

    # Semantic Deduplication
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
                if cont not in unique_set:
                    for index in orig_indices:
                        is_only = (index > 0 and processed_data[index-1]["type"] in ["heading","subheading"] and
                                   (index == len(processed_data)-1 or processed_data[index+1]["type"] in ["heading","subheading","blank"]))
                        if not is_only: indices_to_remove.add(index)
                processed_removal.add(cont)
            kept_indices -= indices_to_remove; print(f"INFO (llm_utils): Marked {len(indices_to_remove)} lines for removal.")
        except Exception as e: print(f"WARN (llm_utils): Dedup failed: {e}")

    # Final Assembly
    final_text, last_type = "", None
    kept_data = [processed_data[i] for i in sorted(list(kept_indices))]
    for i, data in enumerate(kept_data):
        curr_type, content = data["type"], data["text"]
        if i > 0: # Spacing
            if curr_type in ["heading","subheading"]: final_text += "\n\n"
            elif curr_type == "paragraph" and last_type not in ["heading","subheading","blank"]: final_text += "\n\n"
            elif curr_type != "blank" and last_type != "blank": final_text += "\n"
            elif curr_type == "blank" and last_type == "blank": continue
        # Content
        if curr_type == "heading": final_text += f"# {content}"
        elif curr_type == "subheading": final_text += f"## {content}"
        elif curr_type == "bullet": final_text += f"- {content}"
        elif curr_type == "numbered": final_text += f"1. {content}" # Basic numbering
        elif curr_type == "paragraph": final_text += content
        last_type = curr_type
    print("INFO (llm_utils): Post-processing finished.")
    return final_text.strip()
# --- End enhanced_post_process ---

# --- remove_duplicates_semantic function ---
def remove_duplicates_semantic(points, similarity_threshold=0.90, batch_size=64):
    if not points or not embedder or len(points) < 2:
        return points

    start_dedup = time.time()
    try:
        valid_pts = [p for p in points if len(p.split()) > 4]
        if not valid_pts:
            return points

        device = getattr(embedder, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # FIXED
        embeddings = embedder.encode(valid_pts, convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size, device=device)
        cos_sim = util.cos_sim(embeddings, embeddings)
        to_remove = set()

        for i in range(len(valid_pts)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(valid_pts)):
                if j in to_remove:
                    continue
                if cos_sim[i][j] > similarity_threshold:
                    to_remove.add(j)

        unique_pts = [valid_pts[i] for i in range(len(valid_pts)) if i not in to_remove]
        short_pts = list(dict.fromkeys([p for p in points if len(p.split()) <= 4]))  # FIXED: dedup short points
        print(f"INFO (llm_utils): Semantic deduplication took {time.time() - start_dedup:.2f}s.")
        return unique_pts + short_pts

    except torch.cuda.OutOfMemoryError:
        print("ERROR (llm_utils): OOM during dedup. Skipping.")
        return points
    except Exception as e:
        print(f"ERROR (llm_utils): Dedup Error: {e}")
        traceback.print_exc()
        return points

# --- generate_activity function ---
def generate_activity(summary_text, grade_level_category, grade_level_desc):
    if not model or not tokenizer:
        return "- Review key points."
    print("INFO (llm_utils): Generating fallback activity suggestion...")
    act_type = {"lower": "fun activity/question", "middle": "practical activity/thought question",
                "higher": "provoking question/research idea/analysis task"}.get(grade_level_category, "activity")
    context_tail = ' '.join(re.sub(r'^#.*?\n', '', summary_text).strip().split()[-200:])
    prompt = (f"Suggest ONE simple, engaging {act_type} for {grade_level_desc} based on this summary context:\n...{context_tail}\n\nActivity Suggestion:")
    activity = model_generate(prompt, max_new_tokens=80, temperature=0.7)

    if activity.startswith("Error:"):
        print(f"WARN (llm_utils): Fallback activity failed: {activity}")
        activity = ""
    else:
        activity = re.sub(r'^[\-\*\s]+', '', activity.strip().replace("Activity Suggestion:", "").strip()).strip()
        if activity:
            activity = f"- {activity[0].upper() + activity[1:]}"
            if activity and activity[-1].isalnum():
                activity += '.'  # FIXED: Prevent empty activity crash
        else:
            print("WARN (llm_utils): Failed fallback activity.")
            fallbacks = {"lower": "- Draw!", "middle": "- Explain!", "higher": "- Find example."}
            activity = fallbacks.get(grade_level_category, "- Review points.")

    return activity

