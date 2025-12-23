# Copyright 2024 MTSA Team
# Multi-turn RLVR Training Entry Point
"""
Training script for Multi-Turn RLVR.
Similar to mt-rlhf.py but uses GRPO/RLOO policy gradient training.
"""

import logging
import warnings
import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
)
from trl import ModelConfig, SFTConfig, TrlParser, get_peft_config
from peft import get_peft_model
from datasets import load_dataset
from accelerate import Accelerator

from src.rlvr.mt_rlvr import MTRLVRTrainer, RLVRConfig
from src.rlvr.reward_manager import NaiveRewardManager
from src.rlvr.reward_manager.multiturn_reward import MultiTurnRewardFunction
from src.utils.loader import load_model, load_tokenizer
from src.utils.utils import init_seed


@dataclass
class RLVRScriptArguments:
    """Script arguments for RLVR training."""
    
    # Data
    dataset_name: str = field(
        default="datasets/attack_target/train_attack_target.json",
        metadata={"help": "Path to training dataset"}
    )
    max_prompt_length: int = field(
        default=320,
        metadata={"help": "Maximum prompt length"}
    )
    max_response_length: int = field(
        default=1024,
        metadata={"help": "Maximum response length"}
    )
    
    # Model (deprecated in favor of ModelConfig)
    ref_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to reference model (optional, defaults to same as model)"}
    )
    
    # Algorithm
    adv_estimator: str = field(
        default="grpo",
        metadata={"help": "Advantage estimator: grpo, rloo, gae, reinforce_plus_plus"}
    )
    use_kl_in_reward: bool = field(
        default=True,
        metadata={"help": "Whether to use KL penalty in reward"}
    )
    kl_coef: float = field(
        default=0.001,
        metadata={"help": "KL penalty coefficient"}
    )
    
    # Training
    num_rollouts: int = field(
        default=4,
        metadata={"help": "Number of responses to generate per prompt"}
    )
    ppo_epochs: int = field(
        default=1,
        metadata={"help": "Number of PPO epochs per step"}
    )
    mini_batch_size: int = field(
        default=4,
        metadata={"help": "Mini batch size for PPO updates"}
    )
    
    # Reward
    defence_mode: bool = field(
        default=False,
        metadata={"help": "Enable defence mode (minimize entropy)"}
    )
    attack_mode: bool = field(
        default=False,
        metadata={"help": "Enable attack mode (maximize victim safety violation)"}
    )
    use_entropy_reward: bool = field(
        default=True,
        metadata={"help": "Use entropy-based reward"}
    )
    victim_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to victim model for attack training"}
    )
    judge_model_name_or_path: str = field(
        default="meta-llama/Llama-Guard-3-8B",
        metadata={"help": "Path to judge model"}
    )
    attacker_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to attacker model for generating adversarial prompts"}
    )
    red_team_instruction: str = field(
        default="You are a red team model, please continue to think and ask question to achieve your attack purpose.",
        metadata={"help": "Instruction prefixed to goal for attacker"}
    )
    
    # Debug
    dry_run: bool = field(
        default=False,
        metadata={"help": "Run without actual training for testing"}
    )


class RLVRDataset(torch.utils.data.Dataset):
    """Dataset for RLVR training."""
    
    def __init__(self, dataset, tokenizer, max_prompt_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get prompt from conversation
        if 'prompt' in item:
            prompt = item['prompt']
        elif 'conversation' in item:
            conv = item['conversation']
            prompt = self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True
            )
        elif 'attack_conv' in item:
            # MTSA format
            conv = item['attack_conv']
            prompt = self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True
            )
        elif 'goal' in item:
            # Simple Attack Target format (just the goal string)
            content = item['goal']
            if hasattr(self, 'red_team_instruction') and self.red_team_instruction:
                content = self.red_team_instruction + content
                
            conv = [{"role": "user", "content": content}]
            prompt = self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            raise ValueError(f"Cannot find prompt in item: {item.keys()}")
        
        # Tokenize
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'non_tensor_batch': {
                'data_source': item.get('data_source', 'default'),
                'ground_truth': item.get('ground_truth', None),
            }
        }


def collate_fn(batch):
    """Collate function for dataloader."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    non_tensor_batch = {
        'data_source': [item['non_tensor_batch']['data_source'] for item in batch],
        'ground_truth': [item['non_tensor_batch']['ground_truth'] for item in batch],
    }
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'non_tensor_batch': non_tensor_batch,
    }


def main():
    init_seed(42)
    
    parser = TrlParser((RLVRScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    
    # For backward compatibility within the script
    if model_config.model_name_or_path is None:
         # Fallback default if not provided
         model_config.model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
    
    print("=" * 50)
    print("RLVR Training Configuration")
    print("=" * 50)
    print(f"  Model: {model_config.model_name_or_path}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Advantage: {args.adv_estimator}")
    print(f"  PEFT: {model_config.use_peft}")
    print(f"  Quantization: {model_config.load_in_4bit or model_config.load_in_8bit}")
    print("=" * 50)
    
    # Load tokenizer
    print("\n>>> 1. Loading Tokenizer")
    tokenizer = load_tokenizer(model_config.model_name_or_path)
    
    # Load model
    print("\n>>> 2. Loading Model")
    # Set default dtype to bf16 if not specified
    if model_config.torch_dtype is None:
        model_config.torch_dtype = torch.bfloat16
        
    # Policy model on root partition
    model = load_model(tokenizer, model_config, training_args, AutoModelForCausalLM, cache_dir="/root/model_cache")
    
    # Handle PEFT
    if model_config.use_peft:
        print(">>> 2b. Initializing PEFT adapters")
        peft_config = get_peft_config(model_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Load reference model (optional)
    ref_model = None
    if args.use_kl_in_reward:
        print("\n>>> 2c. Loading Reference Model")
        ref_path = args.ref_model_name_or_path or model_config.model_name_or_path
        
        # Load ref model with same quantization but frozen
        # Ref model on root partition (shares with policy)
        ref_model = load_model(tokenizer, model_config, training_args, AutoModelForCausalLM, cache_dir="/root/model_cache")
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    
    # Load dataset
    print("\n>>> 3. Loading Dataset")
    if args.dataset_name.endswith(".json"):
        raw_dataset = load_dataset("json", data_files=args.dataset_name)
    else:
        raw_dataset = load_dataset(path=args.dataset_name)
        
    if isinstance(raw_dataset, dict):
        raw_dataset = raw_dataset['train']
    
    # Configure dataset with red-team instruction if applicable
    train_dataset = RLVRDataset(raw_dataset, tokenizer, args.max_prompt_length)
    if not args.defence_mode:
        train_dataset.red_team_instruction = args.red_team_instruction
    else:
        train_dataset.red_team_instruction = ""
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Load Judge Model if requested
    judge_model = None
    judge_tokenizer = None
    if args.attack_mode or args.defence_mode:
        print("\n>>> 2d. Loading Judge Model")
        judge_tokenizer = load_tokenizer(args.judge_model_name_or_path)
        # Use workspace partition for judge
        judge_config = ModelConfig(model_name_or_path=args.judge_model_name_or_path, load_in_4bit=True)
        judge_model = load_model(judge_tokenizer, judge_config, training_args, AutoModelForCausalLM, cache_dir="/workspace/huggingface_cache")
        judge_model.eval()

    # Load Victim Model if for attack training
    victim_model = None
    victim_tokenizer = None
    if args.attack_mode and args.victim_model_name_or_path:
        print("\n>>> 2e. Loading Victim Model")
        victim_tokenizer = load_tokenizer(args.victim_model_name_or_path)
        victim_config = ModelConfig(model_name_or_path=args.victim_model_name_or_path, load_in_4bit=True)
        victim_model = load_model(victim_tokenizer, victim_config, training_args, AutoModelForCausalLM, cache_dir="/root/model_cache")
        victim_model.eval()

    # Load Attacker Model if for defense training
    attacker_model = None
    attacker_tokenizer = None
    if args.defence_mode and args.attacker_model_name_or_path:
        print("\n>>> 2f. Loading Attacker Model")
        attacker_tokenizer = load_tokenizer(args.attacker_model_name_or_path)
        # Attacker base model on root partition (shares with policy)
        attacker_config = ModelConfig(model_name_or_path=args.attacker_model_name_or_path, load_in_4bit=True)
        attacker_model = load_model(attacker_tokenizer, attacker_config, training_args, AutoModelForCausalLM, cache_dir="/root/model_cache")
        attacker_model.eval()

    # Create reward function
    print("\n>>> 4. Creating Reward Function")
    reward_fn_core = MultiTurnRewardFunction(
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
        victim_model=victim_model,
        victim_tokenizer=victim_tokenizer,
        defence_mode=args.defence_mode,
        attack_mode=args.attack_mode,
        use_entropy_reward=args.use_entropy_reward,
        use_judge_reward=True if judge_model else False,
        template_type="qwen" if "qwen" in model_config.model_name_or_path.lower() else "llama3",
    )
    
    reward_manager = NaiveRewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        compute_score=reward_fn_core,
    )
    
    # Create RLVR config
    rlvr_config = RLVRConfig(
        adv_estimator=args.adv_estimator,
        use_kl_in_reward=args.use_kl_in_reward,
        kl_coef=args.kl_coef,
        num_rollouts=args.num_rollouts,
        max_response_length=args.max_response_length,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        learning_rate=training_args.learning_rate,
        defence_mode=args.defence_mode,
    )
    
    # Create trainer
    print("\n>>> 5. Creating Trainer")
    trainer = MTRLVRTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_manager,
        config=rlvr_config,
        ref_model=ref_model,
        attacker_model=attacker_model,
        attacker_tokenizer=attacker_tokenizer,
    )
    
    if args.dry_run:
        print("\n>>> DRY RUN - Testing one step")
        batch = next(iter(train_dataloader))
        # Use appropriate device (might be on multiple GPUs if using accelerate)
        device = next(model.parameters()).device
        prompts = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        print(f"Prompts shape: {prompts.shape}")
        print(f"Generating rollouts...")
        
        rollout_data = trainer.generate_rollouts(prompts, attention_mask, batch['non_tensor_batch'])
        print(f"Rollout responses shape: {rollout_data['responses'].shape}")
        print("\nDry run successful!")
        return
    
    # Train
    print("\n>>> 6. Starting Training")
    trainer.train(
        train_dataloader=train_dataloader,
        num_epochs=int(training_args.num_train_epochs),
        save_dir=training_args.output_dir,
        save_freq=training_args.save_steps,
        log_freq=training_args.logging_steps,
    )
    
    print("\n>>> Training Complete!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    main()
