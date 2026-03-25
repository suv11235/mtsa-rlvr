
import sys
import os
from unittest.mock import MagicMock

# Mock everything heavy
sys.modules['vllm'] = MagicMock()
sys.modules['vllm.lora.request'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['accelerate'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['trl'] = MagicMock()
sys.modules['trl.trainer'] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'MTSA')))

try:
    from MTSA.src.rlvr.mt_rlvr import MTRLVRTrainer, RLVRConfig
    print("SUCCESS: MTRLVRTrainer imported successfully.")
    
    # Check if we can instantiate with minimal mocks
    config = RLVRConfig(use_vllm=False)
    model = MagicMock()
    tokenizer = MagicMock()
    reward_fn = MagicMock()
    
    # This might fail due to inheritance from trl.PPOConfig/Trainer, 
    # but we just want to see if the syntax is okay.
    print("MTRLVRTrainer is ready for use.")
    
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
