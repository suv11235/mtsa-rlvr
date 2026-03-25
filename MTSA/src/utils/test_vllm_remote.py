import os
from vllm import LLM, SamplingParams

# Force V0 if possible
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ENABLE_V1"] = "0"

model_path = "Qwen/Qwen2.5-0.5B-Instruct" 

print("Starting vLLM test...")
try:
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.1,
        enforce_eager=True,
        trust_remote_code=True
    )
    print("vLLM initialized successfully!")
    
    prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)
    
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated text: {output.outputs[0].text!r}")
        
except Exception as e:
    print(f"vLLM test failed: {e}")
    import traceback
    traceback.print_exc()
