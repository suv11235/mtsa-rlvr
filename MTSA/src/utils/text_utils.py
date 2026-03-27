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

def extract_payload(text: str, marker_prefix: str = "Attack question") -> str:
    """
    Robust extraction of payload, ignoring thinking or refusal prefixes.
    Searches for the last occurrence of markers like 'Attack question' or '### Payload'.
    """
    if not text: return ""
    
    text = strip_thinking(text)
    
    # Common markers used across the codebase
    markers = [
        f"{marker_prefix} 3:", 
        f"{marker_prefix} 2:", 
        f"{marker_prefix} 1:", 
        f"{marker_prefix}:", 
        "### Payload", 
        "### Jailbreak Payload"
    ]
    
    for m in markers:
        if m.lower() in text.lower():
            idx = text.lower().rfind(m.lower())
            payload = text[idx + len(m):].strip()
            # Remove trailing conversational junk
            for stop in ["\nUser:", "\nAssistant:", "\nVictim:", "---", "[The actual"]:
                if stop.lower() in payload.lower():
                    payload = payload[:payload.lower().find(stop.lower())].strip()
            return payload.strip("[]()_ \n\t")
            
    return text.strip()
