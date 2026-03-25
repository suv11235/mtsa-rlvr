# Copyright 2024 MTSA Team
# Model Factory for MTSA/RLVR
"""
ModelFactory: Encapsulated logic for loading Policy, Reference, Judge, and Attacker models.
Handles Full Fine-Tuning, LoRA, and Distributed device placement.
"""

import os
import torch
from typing import Optional, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model
from trl import ModelConfig, SFTConfig, get_peft_config
from .loader import load_model, load_tokenizer

class ModelFactory:
    @staticmethod
    def load_policy_tokenizer(model_config: ModelConfig, tokenizer_path: Optional[str] = None) -> Any:
        print("\n>>> [ModelFactory] Loading Policy Tokenizer")
        path = tokenizer_path or model_config.model_name_or_path
        return load_tokenizer(path)

    @staticmethod
    def load_policy_model(tokenizer: Any, model_config: ModelConfig, training_args: Any, accelerator: Any) -> Any:
        """Load and prepare the policy model for training (Full or LoRA)."""
        print("\n>>> [ModelFactory] Loading Policy Model")
        # Set fragmentation strategy if not already set
        if "PYTORCH_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            
        model = load_model(tokenizer, model_config, training_args, AutoModelForCausalLM, cache_dir=getattr(training_args, "cache_dir", None))
        
        if model_config.use_peft:
            print(">>> [ModelFactory] Initializing PEFT adapters")
            peft_config = get_peft_config(model_config)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            print(">>> [ModelFactory] Full Fine-Tuning mode: No adapters applied.")
        
        return model

    @staticmethod
    def load_reference_model(tokenizer: Any, model_config: ModelConfig, training_args: Any, use_kl: bool, ref_path: Optional[str] = None) -> Optional[Any]:
        """Load reference model for KL penalty."""
        if not use_kl:
            return None
            
        if model_config.use_peft:
            print("\n>>> [ModelFactory] Reference Model: Shared via adapter disabling.")
            return None
            
        print("\n>>> [ModelFactory] Loading Separate Reference Model")
        ref_model_path = ref_path or model_config.model_name_or_path
        
        # We need a clean config for the reference model (no LoRA/Quantization if not needed, 
        # but matching the base architecture). 
        # DeepSpeed ZeRO-3 will handle the sharding across GPUs for efficiency.
        ref_config = ModelConfig(
            model_name_or_path=ref_model_path,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation,
        )
        
        ref_model = load_model(tokenizer, ref_config, training_args, AutoModelForCausalLM, cache_dir=getattr(training_args, "cache_dir", None))
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
            
        return ref_model

    @staticmethod
    def load_judge_model(args: Any, training_args: Any, accelerator: Any, strategy: str = "all_ranks") -> Tuple[Optional[Any], Optional[Any]]:
        """Load judge model with specific distribution strategy."""
        if not (getattr(args, "attack_mode", False) or getattr(args, "defence_mode", False)):
            return None, None
            
        rank = accelerator.process_index
        local_rank = accelerator.local_process_index
        world_size = accelerator.num_processes
        
        # Determine if this rank should load the model in pinned mode
        should_load = True
        if strategy == "aux_pinned":
            # Only the last rank (usually Rank 3 in 4-gpu setup) loads it
            should_load = (local_rank == (world_size - 1))
            
        if not should_load:
            # We still need the tokenizer on all ranks for processing
            judge_path = args.judge_model_name_or_path
            token = os.environ.get("HF_TOKEN")
            judge_tokenizer = AutoTokenizer.from_pretrained(judge_path, use_fast=False, token=token)
            return None, judge_tokenizer

        print(f"\n>>> [ModelFactory] Loading Judge Model (Strategy: {strategy}) on Rank {rank}")
        
        # Auto-switch defaults if needed
        judge_path = args.judge_model_name_or_path
        token = os.environ.get("HF_TOKEN")
        if getattr(args, "judge_type", "") == "harmbench" and judge_path == "meta-llama/Llama-Guard-3-8B":
            judge_path = "cais/HarmBench-Llama-2-13b-cls"
            print(f"    Auto-switched judge to: {judge_path}")

        # CRITICAL: Use AutoTokenizer directly to avoid padding-side conflicts in classification
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_path, use_fast=False, token=token)
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
        
        if not getattr(args, "use_vllm", False):
            print(f"    Loading Judge to PyTorch on {accelerator.device}")
            judge_config = ModelConfig(model_name_or_path=judge_path, load_in_4bit=True)
            judge_model = load_model(judge_tokenizer, judge_config, training_args, AutoModelForCausalLM, cache_dir=getattr(args, "cache_dir", None))
            # Distributed placement
            judge_model.to(accelerator.device)
            judge_model.eval()
            return judge_model, judge_tokenizer
        else:
            print(f"    vLLM enabled: Judge will be served via a dedicated vLLM engine.")
            return None, judge_tokenizer

    @staticmethod
    def load_attacker_model(args: Any, training_args: Any, accelerator: Any, strategy: str = "all_ranks") -> Tuple[Optional[Any], Optional[Any]]:
        """Load attacker model with specific distribution strategy."""
        if not (getattr(args, "defence_mode", False) and args.attacker_model_name_or_path):
            return None, None

        rank = accelerator.process_index
        local_rank = accelerator.local_process_index
        world_size = accelerator.num_processes
        
        # Determine if this rank should load the model
        should_load = True
        if strategy == "aux_pinned":
            should_load = (local_rank == (world_size - 1))
            
        attacker_tokenizer = load_tokenizer(args.attacker_model_name_or_path)
        
        if not should_load:
            return None, attacker_tokenizer

        print(f"\n>>> [ModelFactory] Initializing Attacker (Strategy: {strategy}) on Rank {rank}")
        
        if not getattr(args, "use_vllm", False):
            print(f"    vLLM disabled: Loading Attacker to PyTorch on {accelerator.device}")
            attacker_config = ModelConfig(model_name_or_path=args.attacker_model_name_or_path, load_in_4bit=True)
            attacker_model = load_model(attacker_tokenizer, attacker_config, training_args, AutoModelForCausalLM, cache_dir=getattr(args, "cache_dir", None))
            attacker_model.to(accelerator.device)
            attacker_model.eval()
            return attacker_model, attacker_tokenizer
        else:
            print(f"    vLLM enabled: Attacker will be served via a dedicated vLLM engine.")
            return None, attacker_tokenizer

    @staticmethod
    def load_victim_model(args: Any, training_args: Any, accelerator: Any, tokenizer: Any) -> Tuple[Optional[Any], Optional[Any]]:
        """Load frozen victim model for attack training."""
        if not (getattr(args, "attack_mode", False) and args.victim_model_name_or_path):
            return None, None

        print("\n>>> [ModelFactory] Loading Victim Model")
        victim_tokenizer = load_tokenizer(args.victim_model_name_or_path)
        
        # If using TAR, we need grads for the inner loop, so we load in full/16-bit
        use_tar = getattr(args, "use_tamper_resistance", False)
        victim_config = ModelConfig(
            model_name_or_path=args.victim_model_name_or_path, 
            load_in_4bit=not use_tar
        )
        
        victim_model = load_model(victim_tokenizer, victim_config, training_args, AutoModelForCausalLM, cache_dir=getattr(args, "cache_dir", None))
        victim_model.eval()
        
        return victim_model, victim_tokenizer
