# Copyright 2024 MTSA Team
# Multi-turn RLVR Training Entry Point
"""
Training script for Multi-Turn RLVR.
Similar to mt-rlhf.py but uses GRPO/RLOO policy gradient training.
"""

import os
import logging
import warnings

# Force vLLM V0 (stable) - MUST be done before any vLLM or torch.dist imports
os.environ["VLLM_USE_V1"] = "0"
import torch
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
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
from src.utils.model_factory import ModelFactory
from src.utils.data_factory import DataFactory
from src.utils.loss_config_merge import explicit_cli_longoption_names, merge_loss_config_into_args
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
    max_sim_turns: int = field(
        default=3,
        metadata={"help": "Maximum number of simulated turns for MTSA Reward"}
    )
    
    # Model (deprecated in favor of ModelConfig)
    ref_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to reference model (optional, defaults to same as model)"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to tokenizer (optional, defaults to model_name_or_path)"}
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

    # Optional YAML/JSON of loss-related weights (merged after parser; CLI overrides file)
    loss_config_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to YAML/JSON with loss/PPO/reward weights; keys not passed on CLI are filled from this file."}
    )

    # PPO / KL controller (passed through to RLVRConfig; also overridable via loss_config_file)
    entropy_coeff: float = field(
        default=0.0,
        metadata={"help": "Weight on policy entropy bonus in PPO loss (subtracts entropy_coeff * mean entropy)."}
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "PPO clip range for policy ratio."}
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "PPO clip range for value loss (reserved / critic)."}
    )
    kl_penalty_type: str = field(
        default="kl",
        metadata={"help": "KL penalty type for kl_penalty helper: kl, abs, mse, low_var_kl."}
    )
    kl_ctrl_type: str = field(
        default="fixed",
        metadata={"help": "KL controller type: fixed or adaptive."}
    )
    target_kl: float = field(
        default=0.01,
        metadata={"help": "Target KL for adaptive KL controller."}
    )
    kl_horizon: int = field(
        default=10000,
        metadata={"help": "Horizon for adaptive KL controller."}
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
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling during rollouts"}
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
        metadata={"help": "Use entropy-based reward (TokenBuncher approach)"}
    )
    entropy_reward_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for entropy-based reward"}
    )
    judge_reward_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for judge safety reward"}
    )
    victim_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to victim model for attack training"}
    )
    judge_model_name_or_path: str = field(
        default="qylu4156/strongreject-15k-v1",
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
    judge_type: str = field(
        default="strongreject",
        metadata={"help": "Type of judge: 'llamaguard', 'harmbench', or 'strongreject'"}
    )
    
    # Tamper Resistance (TAR)
    use_tamper_resistance: bool = field(
        default=False,
        metadata={"help": "Enable tamper-resistant meta-learning"}
    )
    tar_type: str = field(
        default="scramble",
        metadata={"help": "Tampering type: 'scramble' (entropy maximization) or 'attack' (simulated SFT)"}
    )
    tar_inner_loop_steps: int = field(
        default=1,
        metadata={"help": "Number of inner-loop adversarial steps"}
    )
    tar_inner_lr: float = field(
        default=5e-5,
        metadata={"help": "Learning rate for inner-loop tampering"}
    )
    tar_inner_goal_sampling: bool = field(
        default=False,
        metadata={"help": "If true, simulates diverse attacks by sampling goals"}
    )
    
    # Representation Closeness (from TAR)
    use_rep_loss: bool = field(
        default=False,
        metadata={"help": "Enable representation closeness loss in activation space"}
    )
    rep_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for representation closeness loss"}
    )
    
    # vLLM
    use_vllm: bool = field(
        default=True,
        metadata={"help": "Whether to use vLLM for fast rollouts"}
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.35,
        metadata={"help": "GPU memory fraction to reserve for vLLM"}
    )
    vllm_max_model_len: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length for vLLM engine"}
    )
    vllm_distribution_strategy: str = field(
        default="all_ranks",
        metadata={"help": "vLLM distribution strategy"}
    )
    
    # Debug
    dry_run: bool = field(
        default=False,
        metadata={"help": "Run without actual training for testing"}
    )

    # Capability regularizer (control) - e.g. GSM8K
    use_capability_regularizer: bool = field(
        default=False,
        metadata={"help": "Add a capability-preservation regularizer loss (supervised NLL on a control dataset)."}
    )
    capability_dataset_name: str = field(
        default="openai/gsm8k",
        metadata={"help": "HF dataset name for capability control (default: openai/gsm8k)."}
    )
    capability_dataset_config: str = field(
        default="main",
        metadata={"help": "HF dataset config/subset name (default: main)."}
    )
    capability_split: str = field(
        default="train",
        metadata={"help": "Dataset split to sample from for the regularizer."}
    )
    capability_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for capability regularizer loss term (added to PPO loss)."}
    )
    capability_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size for capability regularizer (per PPO minibatch)."}
    )
    capability_max_length: int = field(
        default=768,
        metadata={"help": "Max token length for capability regularizer examples (prompt+answer)."}
    )
    capability_answer_mode: str = field(
        default="final",
        metadata={"help": "Answer target for GSM8K regularizer: 'final' (numeric only) or 'full'."}
    )
    capability_answer_prefix: str = field(
        default="",
        metadata={"help": "Optional prefix prepended to the target answer (e.g., 'The answer is ')."}
    )

    cache_dir: str = field(
        default="/data/suvajit_majumder/huggingface_cache",
        metadata={"help": "Directory for model cache"}
    )


# Dataset and Collate moved to DataFactory


def main():
    init_seed(42)
    # accelerator = Accelerator() # Moved down to avoid TrlParser state conflict
    
    parser = TrlParser((RLVRScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Dedicated loss YAML/JSON: fills any allowed key not set on the command line
    merge_loss_config_into_args(args, args.loss_config_file, explicit_cli=explicit_cli_longoption_names())
    
    from accelerate import InitProcessGroupKwargs
    from datetime import timedelta
    # Initialize Accelerator AFTER parser to avoid state conflicts
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    
    # For backward compatibility within the script
    if model_config.model_name_or_path is None:
         # Fallback default if not provided
         model_config.model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
    
    print("=" * 50)
    print("RLVR Training Configuration")
    print("=" * 50)
    
    # Load Policy
    tokenizer = ModelFactory.load_policy_tokenizer(model_config, tokenizer_path=args.tokenizer_name_or_path)
    model = ModelFactory.load_policy_model(tokenizer, model_config, training_args, accelerator)
    
    # Load Reference Model
    ref_model = ModelFactory.load_reference_model(
        tokenizer, model_config, training_args, 
        use_kl=args.use_kl_in_reward, 
        ref_path=args.ref_model_name_or_path
    )
    
    # Isolation: Only load Attacker/Judge on Rank 0 to save VRAM on other ranks.
    # Actually, with Split Ranks, we put Victim on Rank 0, Attacker/Judge on Rank 1 (or vice versa)
    # On 8xA100 80GB, we use split_ranks for stability.
    strategy = args.vllm_distribution_strategy
    print(f"\n>>> Distribution Strategy: {strategy}")
    
    # Load Judge
    judge_model, judge_tokenizer = ModelFactory.load_judge_model(args, training_args, accelerator, strategy=strategy)
    
    # Load Victim (for attack training)
    victim_model, victim_tokenizer = ModelFactory.load_victim_model(args, training_args, accelerator, tokenizer)
    
    # Load Attacker (for defense training)
    attacker_model, attacker_tokenizer = ModelFactory.load_attacker_model(args, training_args, accelerator, strategy=strategy)
    
    # Load dataset
    train_dataloader = DataFactory.load_rlvr_dataloader(args, training_args, tokenizer)

    # Create reward function
    print("\n>>> 4. Creating Reward Function")
    
    # In Defence Mode:
    # - victim_model IS the model we are training (model)
    # - attacker_model IS the separate red-team model (attacker_model)
    # In Attack Mode:
    # - victim_model IS the frozen target (victim_model)
    # - attacker_model IS the model we are training (model)
    
    reward_fn_core = MultiTurnRewardFunction(
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
        victim_model=model if args.defence_mode else victim_model,
        victim_tokenizer=tokenizer if args.defence_mode else victim_tokenizer,
        attacker_model=attacker_model if args.defence_mode else model,
        attacker_tokenizer=attacker_tokenizer if args.defence_mode else tokenizer,
        defence_mode=args.defence_mode,
        attack_mode=args.attack_mode,
        use_entropy_reward=args.use_entropy_reward,
        use_judge_reward=True if judge_model else False,
        entropy_weight=args.entropy_reward_weight,
        judge_weight=args.judge_reward_weight,
        template_type="deepseek" if "deepseek" in model_config.model_name_or_path.lower() else ("qwen" if "qwen" in model_config.model_name_or_path.lower() else "llama3"),
        max_sim_turns=args.max_sim_turns,
        judge_type=args.judge_type,
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
        kl_penalty_type=args.kl_penalty_type,
        kl_ctrl_type=args.kl_ctrl_type,
        target_kl=args.target_kl,
        kl_horizon=args.kl_horizon,
        cliprange=args.cliprange,
        cliprange_value=args.cliprange_value,
        entropy_coeff=args.entropy_coeff,
        num_rollouts=args.num_rollouts,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        temperature=args.temperature,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        learning_rate=training_args.learning_rate,
        defence_mode=args.defence_mode,
        use_tamper_resistance=args.use_tamper_resistance,
        tar_type=args.tar_type,
        tar_inner_loop_steps=args.tar_inner_loop_steps,
        tar_inner_lr=args.tar_inner_lr,
        tar_inner_goal_sampling=args.tar_inner_goal_sampling,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization, 
        vllm_max_model_len=args.vllm_max_model_len, 
        use_vllm=args.use_vllm,
        attacker_model_name_or_path=args.attacker_model_name_or_path,
        judge_model_name_or_path=args.judge_model_name_or_path,
        vllm_distribution_strategy=args.vllm_distribution_strategy,
        use_rep_loss=args.use_rep_loss,
        rep_loss_weight=args.rep_loss_weight,
        use_capability_regularizer=args.use_capability_regularizer,
        capability_dataset_name=args.capability_dataset_name,
        capability_dataset_config=args.capability_dataset_config,
        capability_split=args.capability_split,
        capability_weight=args.capability_weight,
        capability_batch_size=args.capability_batch_size,
        capability_max_length=args.capability_max_length,
        capability_answer_mode=args.capability_answer_mode,
        capability_answer_prefix=args.capability_answer_prefix,
    )

    print("\n>>> 5. Creating Trainer")
    
    # Calculate total training steps for the scheduler
    # steps per epoch = dataset_size / batch_size
    steps_per_epoch = len(train_dataloader)
    total_steps = int(training_args.num_train_epochs * steps_per_epoch)
    
    # Create optimizer first (to be passed to scheduler)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_args.learning_rate
    )
    
    # Create scheduler
    from transformers import get_linear_schedule_with_warmup
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=total_steps
    )
    print(f"    Scheduler initialized: {training_args.warmup_steps} warmup steps, {total_steps} total steps")

    # IMPORTANT: Prepare everything with Accelerator for ZeRO-3
    # MTRLVRTrainer is a custom class and does not handle internal preparation.
    # Manual preparation is MANDATORY for ZeRO-3 sharding to work correctly.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    if ref_model is not None:
        ref_model = accelerator.prepare(ref_model)

    trainer = MTRLVRTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_manager,
        config=rlvr_config,
        ref_model=ref_model,
        victim_model=victim_model if args.attack_mode else None,
        attacker_model=attacker_model,
        attacker_tokenizer=attacker_tokenizer,
        accelerator=accelerator,
        lr_scheduler=lr_scheduler,
    )
    # Manually overwrite the optimizer created in MTRLVRTrainer's __init__ 
    # (or better, we should have modified MTRLVRTrainer to accept an optimizer too)
    # But since we are here, we can just replace it.
    trainer.optimizer = optimizer
    
    # Restore global step and best_reward if loading from checkpoint
    if "checkpoint-" in model_config.model_name_or_path:
        checkpoint_dir = model_config.model_name_or_path
        try:
            # Parse step number from path
            checkpoint_step = int(checkpoint_dir.split("checkpoint-")[-1].split("/")[0])
            trainer.global_step = checkpoint_step
            print(f">>> Resuming from Step {checkpoint_step}")
            
            # Load training state if exists
            state_path = os.path.join(checkpoint_dir, "training_state.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu")
                trainer.best_reward = state.get("best_reward", -float('inf'))
                print(f">>> Restored Best Reward: {trainer.best_reward:.4f}")
        except Exception as e:
            print(f">>> WARNING: Could not fully restore training state from {checkpoint_dir}: {e}")
    
    if args.dry_run:
        print("\n>>> DRY RUN - Testing one full train step (including Reward/Judge)")
        batch = next(iter(train_dataloader))
        # Use appropriate device
        prompts = batch['input_ids'].to(accelerator.device)
        attention_mask = batch['attention_mask'].to(accelerator.device)
        non_tensor_batch = batch['non_tensor_batch']
        
        print(f"Prompts shape: {prompts.shape}")
        
        # Test full step
        metrics = trainer.train_step(prompts, attention_mask, non_tensor_batch)
        print(f"\nDry run metrics: {metrics}")
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
