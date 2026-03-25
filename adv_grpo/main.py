import os
import argparse
import torch
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except:
    pass

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.vllm_engine import RemoteLLM
from src.trainer import AdvGRPOTrainer
from src.reward import AdvRewardFunction

def load_env_manually():
    """Fallback to load .env if not set in environment (useful for interactive runs)."""
    env_paths = [".env", "MTSA/.env", "../.env"]
    for p in env_paths:
        if os.path.exists(p):
            print(f">>> [Main] Loading env from {p}")
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    # Handle 'export VAR=VAL' or 'VAR=VAL'
                    if line.startswith("export "): line = line[7:]
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k not in os.environ:
                            os.environ[k] = v.strip("'").strip('"')
            break

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--victim_model_name_or_path", type=str)
    parser.add_argument("--attacker_model_name_or_path", type=str)
    parser.add_argument("--judge_model_name_or_path", type=str)
    
    parser.add_argument("--dataset_path", type=str, default="datasets/attack_target/train_attack_target.json")
    parser.add_argument("--train_target", type=str, choices=["attacker", "victim"], default="victim")
    
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--output_dir", type=str, default="outputs/adv_grpo")
    
    return parser.parse_args()

def load_tokenizer(path, token):
    if not path: return None
    print(f">>> [Main] Loading tokenizer for {path} (Type: {type(path)})")
    try:
        # Ensure path is a string
        target_path = str(path)
        # Handle 'token' explicitly
        tk = token if token else None
        return AutoTokenizer.from_pretrained(target_path, use_fast=True, trust_remote_code=True, token=tk)
    except Exception as e:
        print(f">>> [Main] Failed to load tokenizer for {path}: {e}")
        import traceback
        traceback.print_exc()
        # Llama-3 fallback
        if "llama-3" in str(path).lower() or "biosecurity" in str(path).lower() or "tar" in str(path).lower():
             print(">>> [Main] Attempting Llama-3 fallback (using attacker model tokenizer)...")
             return AutoTokenizer.from_pretrained("suv11235/red_team_model_SFT_mtsa", use_fast=True, token=token)
        raise e

def main():
    args = parse_args()
    
    # 1. Setup Environment
    load_env_manually()
    os.environ["VLLM_USE_V1"] = "0"
    accelerator = Accelerator()
    
    # 2. Tokenizer & Dataset
    token = os.environ.get("HF_TOKEN")
    print(f">>> [Main] HF Token present: {token is not None}")
    if token:
        print(f">>> [Main] Token prefix: {token[:10]}...")
    
    tokenizer = load_tokenizer(args.model_name_or_path, token)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    def data_collator(features):
        prompts = [f["goal"] for f in features]
        # In a real setup, format user goals into chat template here
        formated = [tokenizer.apply_chat_template([{"role": "user", "content": g}], tokenize=False, add_generation_prompt=True) for g in prompts]
        tokenized = tokenizer(formated, padding=True, return_tensors="pt")
        return {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "non_tensor_batch": {"goal": prompts}
        }
        
    dataset = load_dataset('json', data_files={'train': args.dataset_path})['train']
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=data_collator)
    
    # 3. Initialize vLLM Engines (Moved up for clean CUDA state)
    local_rank = getattr(accelerator.state, "local_process_index", 0)
    world_size = accelerator.num_processes
    
    # Example budget mapping (conservative for 80GB when policy is also loaded)
    attacker_mem, victim_mem, judge_mem = 0.20, 0.20, 0.10
    
    attacker_engine = None
    if args.attacker_model_name_or_path and args.train_target == 'victim':
        attacker_engine = RemoteLLM(
            model_path=args.attacker_model_name_or_path,
            token_path=args.attacker_model_name_or_path,
            gpu_memory_utilization=attacker_mem,
            port_offset=0,
            rank=accelerator.process_index,
            local_rank=local_rank,
            max_model_len=2048,
            enable_lora=False,
            hf_token=token
        )
        
    victim_engine = None
    if args.victim_model_name_or_path and args.train_target == 'attacker':
        victim_engine = RemoteLLM(
            model_path=args.victim_model_name_or_path,
            token_path=args.victim_model_name_or_path,
            gpu_memory_utilization=victim_mem,
            port_offset=1,
            rank=accelerator.process_index,
            local_rank=local_rank,
            max_model_len=2048,
            enable_lora=False,
            hf_token=token
        )

    judge_engine = None
    if args.judge_model_name_or_path:
        judge_engine = RemoteLLM(
            model_path=args.judge_model_name_or_path,
            token_path=args.judge_model_name_or_path,
            gpu_memory_utilization=judge_mem,
            port_offset=2,
            rank=accelerator.process_index,
            local_rank=local_rank,
            max_model_len=2048,
            enable_lora=False,
            hf_token=token
        )
        
    # The policy engine
    policy_engine = RemoteLLM(
        model_path=args.model_name_or_path,
        token_path=args.model_name_or_path,
        gpu_memory_utilization=0.20, # Use custom arg if provided
        port_offset=10,
        rank=accelerator.process_index,
        local_rank=local_rank,
        max_model_len=2048,
        enable_lora=True,
        hf_token=token
    )

    accelerator.wait_for_everyone()
    
    # 4. Initialize Models
    # Target Model (the one we are training)
    print(f">>> [Rank {accelerator.process_index}] Loading SFT model to PyTorch...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
        
    # 5. Reward Function
    reward_fn = AdvRewardFunction(
        judge_tokenizer=load_tokenizer(args.judge_model_name_or_path, token),
        victim_tokenizer=load_tokenizer(args.victim_model_name_or_path, token),
        attacker_tokenizer=load_tokenizer(args.attacker_model_name_or_path, token),
        judge_vllm_engine=judge_engine,
        victim_vllm_engine=victim_engine,
        attacker_vllm_engine=attacker_engine,
        defence_mode=(args.train_target == "victim"),
        attack_mode=(args.train_target == "attacker")
    )
    
    # Inject policy engine back so trainer can access it
    if args.train_target == 'victim':
        reward_fn.victim_vllm_engine = policy_engine
    else:
        reward_fn.attacker_vllm_engine = policy_engine
    
    # 6. Trainer Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    trainer_config = {
        'learning_rate': args.learning_rate,
        'num_rollouts': args.num_rollouts,
        'attack_mode': args.train_target == 'attacker',
        'defence_mode': args.train_target == 'victim'
    }
    
    trainer = AdvGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        config=trainer_config,
        accelerator=accelerator,
        optimizer=optimizer,
        lr_scheduler=scheduler
    )
    
    # 7. Start Training
    trainer.train(dataloader, args.num_epochs, save_dir=args.output_dir)

if __name__ == "__main__":
    main()
