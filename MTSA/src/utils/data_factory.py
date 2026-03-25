# Copyright 2024 MTSA Team
# Data Factory for MTSA/RLVR
"""
DataFactory: Logic for dataset loading and preparation.
"""

import torch
from datasets import load_dataset
from typing import Dict, Any, Optional

class RLVRDataset(torch.utils.data.Dataset):
    """Dataset for RLVR training."""
    
    def __init__(self, dataset, tokenizer, max_prompt_length, red_team_instruction=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.red_team_instruction = red_team_instruction
    
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
            goal = item['goal']
            if self.red_team_instruction:
                # Training the Attacker (Attack Mode)
                # The prompt should be the red team instruction + the goal
                content = (
                    f"{self.red_team_instruction}\n"
                    f"Your attack target is {goal}\n"
                    "Please follow the format below for your response:\n"
                    "Thinking 1: [Your internal reasoning and strategy]\n"
                    "Attack question 1: [The actual jailbreak prompt to be sent to the victim model]"
                )
            else:
                # Training the Victim (Defence Mode)
                # We just pass the goal. MTRLVRTrainer will handle the simulation 
                # or use an Attacker model to turn this into a jailbreak question.
                content = goal
                
            conv = [{"role": "user", "content": content}]
            prompt = self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            raise ValueError(f"Cannot find prompt in item: {item.keys()}")
            
        # Tokenize prompt for RL
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 2. Handle Harmful Labels for Tampering (if available)
        attack_input_ids = None
        attack_attention_mask = None
        attack_labels = None
        
        target_response = item.get('target_response', None)
        if target_response:
            full_text = prompt + target_response
            full_encoded = self.tokenizer(
                full_text,
                max_length=self.max_prompt_length + 512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            attack_input_ids = full_encoded['input_ids'].squeeze(0)
            attack_attention_mask = full_encoded['attention_mask'].squeeze(0)
            
            prompt_encoded = self.tokenizer(prompt, add_special_tokens=False)
            prompt_len = len(prompt_encoded['input_ids'])
            
            attack_labels = attack_input_ids.clone()
            attack_labels[:prompt_len] = -100 # Mask prompt
            attack_labels[attack_labels == self.tokenizer.pad_token_id] = -100 # Mask padding
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'attack_input_ids': attack_input_ids,
            'attack_attention_mask': attack_attention_mask,
            'attack_labels': attack_labels,
            'non_tensor_batch': {
                'data_source': item.get('data_source', 'default'),
                'ground_truth': item.get('ground_truth', None),
                'extra_info': {'goal': item.get('goal', item.get('attack_target', ''))},
            }
        }

def collate_fn(batch):
    """Collate function for dataloader."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    attack_input_ids = None
    if batch[0].get('attack_input_ids') is not None:
        attack_input_ids = torch.stack([item['attack_input_ids'] for item in batch])
        
    attack_attention_mask = None
    if batch[0].get('attack_attention_mask') is not None:
        attack_attention_mask = torch.stack([item['attack_attention_mask'] for item in batch])
        
    attack_labels = None
    if batch[0].get('attack_labels') is not None:
        attack_labels = torch.stack([item['attack_labels'] for item in batch])

    non_tensor_batch = {
        'data_source': [item['non_tensor_batch']['data_source'] for item in batch],
        'ground_truth': [item['non_tensor_batch']['ground_truth'] for item in batch],
        'extra_info': [item['non_tensor_batch']['extra_info'] for item in batch],
        'attack_input_ids': attack_input_ids,
        'attack_attention_mask': attack_attention_mask,
        'attack_labels': attack_labels,
    }
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'non_tensor_batch': non_tensor_batch,
    }

class DataFactory:
    @staticmethod
    def load_rlvr_dataloader(args: Any, training_args: Any, tokenizer: Any) -> torch.utils.data.DataLoader:
        """Load and prepare RLVR dataloader."""
        print(f"\n>>> [DataFactory] Loading Dataset: {args.dataset_name}")
        
        if args.dataset_name.endswith(".json"):
            raw_dataset = load_dataset("json", data_files=args.dataset_name)
        else:
            raw_dataset = load_dataset(path=args.dataset_name)
            
        if isinstance(raw_dataset, dict):
            raw_dataset = raw_dataset['train']
        
        # Configure dataset instruction
        red_team_instruction = args.red_team_instruction if not args.defence_mode else ""
        
        train_dataset = RLVRDataset(
            raw_dataset, 
            tokenizer, 
            args.max_prompt_length, 
            red_team_instruction=red_team_instruction
        )
        
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
