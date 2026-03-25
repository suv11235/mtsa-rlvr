import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import logging
import copy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TARArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.1-8B-Instruct")
    dataset_name: str = field(default="datasets/attack_target/train_attack_target_labels.json")
    output_dir: str = field(default="./outputs/tar_vanilla_baseline")
    inner_lr: float = field(default=2e-5)
    outer_lr: float = field(default=2e-5)
    inner_steps: int = field(default=64)
    max_steps: int = field(default=750)
    batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    max_length: int = field(default=512)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    tar_type: str = field(default="entropy") # "attack" or "entropy"
    tar_loss_scale: float = field(default=4.0)

def get_refusal_text(goal):
    return "I'm sorry, but I cannot fulfill this request. I am programmed to be a helpful and harmless AI assistant."

def tokenize_pair(tokenizer, prompt, response, max_length):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # We want to mask the user part in labels
    user_messages = [{"role": "user", "content": prompt}]
    user_text = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    
    full_enc = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt")
    user_enc = tokenizer(user_text, truncation=True, max_length=max_length, return_tensors="pt")
    
    input_ids = full_enc["input_ids"]
    labels = input_ids.clone()
    # Mask prompt
    labels[:, :user_enc["input_ids"].shape[1]] = -100
    
    return input_ids, labels, full_enc["attention_mask"]

def main():
    parser = HfArgumentParser(TARArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    logger.info(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model in 4-bit: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    # Dataset
    dataset = load_dataset("json", data_files=args.dataset_name)["train"]
    dataset = dataset.shuffle(seed=42)
    
    optimizer = AdamW(model.parameters(), lr=args.outer_lr)
    
    model.train()
    
    for step in range(args.max_steps):
        optimizer.zero_grad()
        
        total_outer_loss = 0
        
        for sub_step in range(args.gradient_accumulation_steps):
            idx = (step * args.gradient_accumulation_steps + sub_step) % len(dataset)
            item = dataset[idx]
            goal = item["goal"]
            harmful_response = item["target_response"]
            
            # --- TAR Meta-Step ---
            
            # 1. Snapshot original weights
            original_state = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
            
            # 2. Inner Loop: Simulated Tampering
            # Using FOMAML (First Order) for efficiency since inner_steps can be 64
            inner_optimizer = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
            
            input_ids, labels, mask = tokenize_pair(tokenizer, goal, harmful_response, args.max_length)
            input_ids, labels, mask = input_ids.to(model.device), labels.to(model.device), mask.to(model.device)
            
            for _ in range(args.inner_steps):
                outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
                
                if args.tar_type == "entropy":
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    # Minimize negative entropy = Maximize entropy
                    inner_loss = -(probs * log_probs).sum(-1)
                    inner_loss = -(inner_loss * mask).sum() / mask.sum() # entropy is negative here for minimization
                else:
                    inner_loss = outputs.loss
                
                inner_optimizer.zero_grad()
                inner_loss.backward()
                inner_optimizer.step()
            
            # 3. Outer Loop: Retain Safety on Tampered Model
            refusal_text = get_refusal_text(goal)
            ref_input_ids, ref_labels, ref_mask = tokenize_pair(tokenizer, goal, refusal_text, args.max_length)
            ref_input_ids, ref_labels, ref_mask = ref_input_ids.to(model.device), ref_labels.to(model.device), ref_mask.to(model.device)
            
            outer_outputs = model(input_ids=ref_input_ids, attention_mask=ref_mask, labels=ref_labels)
            # The outer loss is scaled by tar_loss_scale and num accumulation steps
            outer_loss = (outer_outputs.loss * args.tar_loss_scale) / args.gradient_accumulation_steps
            
            # Use First-Order MAML: Backward from the current (tampered) state
            outer_loss.backward()
            
            # 4. Restore original weights for next accumulation step or final optimizer update
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(original_state[n])
            
            total_outer_loss += outer_loss.item()
            
        # Global Update
        optimizer.step()
        
        if step % 5 == 0:
            logger.info(f"Step {step}/{args.max_steps} | Meta-Retain-Loss: {total_outer_loss:.4f}")
            
        if step % 50 == 0:
            model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{step}"))

    model.save_pretrained(args.output_dir)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
