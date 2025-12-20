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
        default="datasets/attack_target",
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
    
    # Model
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to model"}
    )
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
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Learning rate"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Training batch size per device"}
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "Max training steps (-1 for unlimited)"}
    )
    
    # Reward
    defence_mode: bool = field(
        default=False,
        metadata={"help": "Enable defence mode (minimize entropy)"}
    )
    use_entropy_reward: bool = field(
        default=True,
        metadata={"help": "Use entropy-based reward"}
    )
    
    # Output
    output_dir: str = field(
        default="./outputs/rlvr",
        metadata={"help": "Output directory"}
    )
    save_freq: int = field(
        default=100,
        metadata={"help": "Save checkpoint every N steps"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every N steps"}
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
    
    parser = HfArgumentParser(RLVRScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    print("=" * 50)
    print("RLVR Training Configuration")
    print("=" * 50)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 50)
    
    # Load tokenizer
    print("\n>>> 1. Loading Tokenizer")
    tokenizer = load_tokenizer(args.model_name_or_path)
    
    # Load model
    print("\n>>> 2. Loading Model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load reference model (optional)
    ref_model = None
    if args.use_kl_in_reward:
        print("\n>>> 2b. Loading Reference Model")
        ref_path = args.ref_model_name_or_path or args.model_name_or_path
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    
    # Load dataset
    print("\n>>> 3. Loading Dataset")
    raw_dataset = load_dataset(path=args.dataset_name)['train']
    train_dataset = RLVRDataset(raw_dataset, tokenizer, args.max_prompt_length)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Create reward function
    print("\n>>> 4. Creating Reward Function")
    reward_fn_core = MultiTurnRewardFunction(
        defence_mode=args.defence_mode,
        use_entropy_reward=args.use_entropy_reward,
        use_judge_reward=False,  # No judge model by default
        template_type="qwen",
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
        learning_rate=args.learning_rate,
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
    )
    
    if args.dry_run:
        print("\n>>> DRY RUN - Testing one step")
        batch = next(iter(train_dataloader))
        prompts = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        
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
        num_epochs=args.num_train_epochs,
        save_dir=args.output_dir,
        save_freq=args.save_freq,
        log_freq=args.logging_steps,
    )
    
    print("\n>>> Training Complete!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    main()
