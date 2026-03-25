import re
import os
from typing import Optional

with open("multiturn_reward.py", "r") as f:
    content = f.read()

# Replace force_extract_payload with a cleaner version
new_extract = """def force_extract_payload(text: str, include_thinking: bool = False) -> Optional[str]:
    \"\"\"Robustly extract the actual payload by bypassing echoed instructions.\"\"\"
    if not text: return None
    
    if not include_thinking:
        text = strip_thinking(text)
        
    markers = ["Attack question 3:", "Attack question 2:", "Attack question 1:", "Attack question:", "### Jailbreak Payload"]
    
    for marker in markers:
        if marker.lower() in text.lower():
            idx = text.lower().rfind(marker.lower())
            
            if include_thinking:
                t_markers = ["### Strategic Goal Analysis", "### Thinking", "Thinking"]
                t_idx = -1
                for tm in t_markers:
                    if tm.lower() in text[:idx].lower():
                        t_idx = text[:idx].lower().rfind(tm.lower())
                        break
                
                if t_idx != -1:
                    raw = text[t_idx:].strip()
                else:
                    raw = text[idx:].strip()
            else:
                raw = text[idx + len(marker):].strip()
                
            for stop in ["\\nUser:", "\\nAssistant:", "\\nVictim:", "\\n[", "Cutting Knowledge", "Today Date"]:
                if stop.lower() in raw.lower():
                    raw = raw[:raw.lower().find(stop.lower())].strip()
                    
            raw = re.sub(r'\[The actual.*?\]', '', raw, flags=re.IGNORECASE).strip()
            raw = raw.strip("[]()_ \\n\\t")
            for junk in [":", "assistant", "user", "model]"]:
                if raw.lower().startswith(junk):
                    raw = raw[len(junk):].strip()
                    
            if len(raw) > 5 and "please continue to think" not in raw.lower():
                return raw
                
    cleaned = text.strip()
    if "please continue to think" in cleaned.lower() or "jailbreak prompt" in cleaned.lower():
        return None
    return cleaned

def strip_thinking(text: str) -> str:
    \"\"\"Strip <think>...</think> blocks from text.\"\"\"
    if "<think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) > 1:
            return parts[1].strip()
        return ""
    return text.strip()"""

# Replace in content using regex to find force_extract_payload and strip_thinking
content = re.sub(r'def force_extract_payload.*?return text\.strip\(\)', new_extract, content, flags=re.DOTALL)

with open("multiturn_reward.py", "w") as f:
    f.write(content)
