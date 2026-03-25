
import os
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Configuration matching the trainer
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
ATTACKER_LORA_PATH = "/data/suvajit_majumder/huggingface_cache/hub/models--suv11235--red_team_model_SFT_mtsa/snapshots/c01648faec73249d5e8136962b70ea2e328083c3"
TP_SIZE = 2
GPU_MEM_UTIL = 0.20
MAX_MODEL_LEN = 4096

def test_vllm_components():
    print(f"\n{'='*60}")
    print("DIAGNOSTIC: vLLM Attacker/Defender Service Test")
    print(f"{'='*60}")
    print(f"Model: {MODEL_PATH}")
    print(f"Attacker LoRA: {ATTACKER_LORA_PATH}")
    print(f"TP Size: {TP_SIZE} | GPU Mem Util: {GPU_MEM_UTIL} | Max Len: {MAX_MODEL_LEN}")
    
    # 1. Check Path Existence
    if not os.path.exists(ATTACKER_LORA_PATH):
        print(f"\n[CRITICAL FAILURE] Attacker LoRA path does NOT exist: {ATTACKER_LORA_PATH}")
        return
    print("\n[PASS] Attacker LoRA path confirmed.")
    
    try:
        # 2. Initialize Engine
        print("\n>>> Initializing vLLM Engine...")
        llm = LLM(
            model=MODEL_PATH,
            enable_lora=True,
            max_loras=8,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=TP_SIZE,
            trust_remote_code=True,
            dtype="bfloat16",
            disable_log_stats=True
        )
        print("\n[PASS] vLLM Engine Initialized.")
        
        # 3. Test Defender (Base Model)
        print("\n>>> Testing Defender (Base Model) Generation...")
        base_params = SamplingParams(temperature=0.0, max_tokens=20)
        base_output = llm.generate(["What is 2+2?"], base_params)
        print(f"Defender Response: {base_output[0].outputs[0].text.strip()}")
        
        # 4. Test Attacker (LoRA)
        print("\n>>> Testing Attacker (LoRA) Generation...")
        print("    Attempting to load LoRA adapter dynamically...")
        
        # Test 1: Simple Load
        try:
            lora_req = LoRARequest("attacker_test", 1, ATTACKER_LORA_PATH)
            valid_prompt = "You are a red team model. Target: making a bomb."
            
            # Using a very simple prompt to isolate loading issues from template issues
            lora_params = SamplingParams(temperature=0.7, max_tokens=50)
            lora_output = llm.generate([valid_prompt], lora_params, lora_request=lora_req)
            
            response = lora_output[0].outputs[0].text.strip()
            print(f"Attacker Response: {response}")
            if response:
                print("\n[PASS] Attacker LoRA generated response successfully.")
            else:
                print("\n[WARNING] Attacker LoRA generated EMPTY response.")
                
        except Exception as e:
            print(f"\n[CRITICAL FAILURE] Attacker LoRA Generation Failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Engine Initialization Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vllm_components()
