
import os
import torch
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
import time

def run_engine(rank):
    print(f"Rank {rank} starting...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["VLLM_PORT"] = str(55000 + rank * 100)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(55005 + rank * 100)
    
    # Unset any interfering vars
    for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "NCCL_ID"]:
        if k in os.environ: del os.environ[k]
        
    llm = LLM(
        model="facebook/opt-125m", # small model for testing
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        trust_remote_code=True
    )
    print(f"Rank {rank} initialized!")
    
    prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(max_tokens=10)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"Rank {rank} output: {output.outputs[0].text}")

if __name__ == "__main__":
    mp.spawn(run_engine, nprocs=2, join=True)
