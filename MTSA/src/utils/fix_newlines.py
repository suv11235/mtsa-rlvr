import re
import os

with open("src/rlvr/reward_manager/multiturn_reward.py", "r") as f:
    content = f.read()

# Fix the newline issue
content = content.replace('for stop in ["\nUser:", "\nAssistant:", "\nVictim:", "\n[", "Cutting Knowledge", "Today Date"]:', 'for stop in ["\\nUser:", "\\nAssistant:", "\\nVictim:", "\\n[", "Cutting Knowledge", "Today Date"]:')
content = content.replace('raw = raw.strip("[]()_ \n\t")', 'raw = raw.strip("[]()_ \\n\\t")')

with open("src/rlvr/reward_manager/multiturn_reward.py", "w") as f:
    f.write(content)
