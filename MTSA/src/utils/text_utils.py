import re
from typing import Optional

def strip_thinking(text: str) -> str:
    """Removes <think>...</think> blocks from model output."""
    if not text: return ""
    if "<think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) > 1: return parts[1].strip()
        return "" # If <think> is present but not closed, assume it's all thinking
    return text.strip()

def extract_assistant_response(solution_str: str, template_type: str = "qwen", include_thinking: bool = False) -> Optional[str]:
    """
    Extract the model's response from a full conversation or a single turn.
    """
    if solution_str is None:
        return None
        
    # 1. TEMPLATE-BASED EXTRACTION
    if template_type == "qwen":
        marker = "<|im_start|>assistant"
        end_marker = "<|im_end|>"
        if marker in solution_str:
            response = solution_str.rsplit(marker, 1)[-1]
            if end_marker in response:
                response = response.split(end_marker, 1)[0]
            return response if include_thinking else strip_thinking(response)
    elif template_type == "llama3":
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        end_marker = "<|eot_id|>"
        if marker in solution_str:
            response = solution_str.rsplit(marker, 1)[-1]
            if end_marker in response:
                response = response.split(end_marker, 1)[0]
            
            if "<|start_header_id|>user<|end_header_id|>" in response:
                response = response.rsplit("<|start_header_id|>assistant<|end_header_id|>", 1)[-1]

            cleaned = response if include_thinking else strip_thinking(response)
            cleaned = cleaned.strip()
            for tag in ["<|start_header_id|>", "<|end_header_id|>", "assistant", "user", "<|eot_id|>", "<|begin_of_text|>"]:
                cleaned = cleaned.replace(tag, "")
            return cleaned.strip()

    # 2. FORCE EXTRACTION (Bypass instruction echoes)
    return force_extract_payload(solution_str, include_thinking=include_thinking)

def force_extract_payload(text: str, include_thinking: bool = False):
    """
    Robust extraction of payload, ignoring thinking or refusal prefixes.
    Searches for the last occurrence of markers like 'Attack question' or '### Jailbreak Payload'.
    """
    if not text: return None
    if not include_thinking: text = strip_thinking(text)
    
    markers = ["Attack question 3:", "Attack question 2:", "Attack question 1:", "Attack question:", "### Jailbreak Payload"]
    for marker in markers:
        if marker.lower() in text.lower():
            idx = text.lower().rfind(marker.lower())
            if include_thinking:
                t_idx = -1
                for tm in ["### Strategic Goal Analysis", "### Thinking", "Thinking"]:
                    if tm.lower() in text[:idx].lower():
                        t_idx = text[:idx].lower().rfind(tm.lower())
                        break
                raw = text[t_idx:].strip() if t_idx != -1 else text[idx:].strip()
            else:
                raw = text[idx + len(marker):].strip()
                
            for stop in ["\nUser:", "\nAssistant:", "\nVictim:", "\n[", "Cutting Knowledge", "Today Date"]:
                if stop.lower() in raw.lower():
                    raw = raw[:raw.lower().find(stop.lower())].strip()
                    
            raw = re.sub(r'\[The actual.*?\]', '', raw, flags=re.IGNORECASE).strip()
            raw = raw.strip("[]()_ \n\t")
            for junk in [":", "assistant", "user", "model]"]:
                if raw.lower().startswith(junk): raw = raw[len(junk):].strip()
            
            if len(raw) > 5 and "please continue to think" not in raw.lower():
                return raw
                
    cleaned = text.strip()
    if "please continue to think" in cleaned.lower() or "jailbreak prompt" in cleaned.lower(): return None
    return cleaned

def extract_user_prompt(solution_str: str, template_type: str = "qwen") -> Optional[str]:
    """Extract user prompt from full conversation string."""
    if template_type == "qwen":
        marker = "<|im_start|>user"
        end_marker = "<|im_end|>"
        if marker in solution_str:
            prompt = solution_str.split(marker, 1)[1]
            if end_marker in prompt:
                prompt = prompt.split(end_marker, 1)[0]
            return prompt.strip()
    elif template_type == "llama3":
        marker = "<|start_header_id|>user<|end_header_id|>"
        end_marker = "<|eot_id|>"
        if marker in solution_str:
            prompt = solution_str.split(marker, 1)[1]
            if end_marker in prompt:
                prompt = prompt.split(end_marker, 1)[0]
            return prompt.strip()
    elif template_type == "deepseek":
        marker = "<｜User｜>"
        end_marker = "<｜Assistant｜>"
        if marker in solution_str:
            prompt = solution_str.split(marker, 1)[1]
            if end_marker in prompt:
                prompt = prompt.split(end_marker, 1)[0]
            return prompt.strip()
    return None
