from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict
import torch
import os
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from trl import (
    get_quantization_config,
    get_kbit_device_map
)


def load_tokenizer(model_name_or_path: "ModelArguments", chat_template=None) -> "TokenizerModule":
    
    token = os.environ.get("HF_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            add_eos_token=True,
            token=token
        )
    except Exception as e:
        # If it fails, check if it is a PEFT adapter (local OR remote)
        try:
            from peft import PeftConfig
            token = os.environ.get("HF_TOKEN")
            print(f">>> [Loader] Initial tokenizer load failed. Checking if {model_name_or_path} is an adapter...")
            adapter_conf = PeftConfig.from_pretrained(model_name_or_path, token=token)
            base_model_path = adapter_conf.base_model_name_or_path
            print(f">>> [Loader] Identified adapter. Falling back to base model tokenizer: {base_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=True, token=token)
        except Exception as e2:
            print(f">>> [Loader] Could not load as adapter either: {e2}")
            raise e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Critical for generation (rollouts)
    tokenizer.padding_side = 'left'
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    
    return tokenizer 

def load_model(
    tokenizer, model_config, training_args, model_class, cache_dir: Optional[str] = None
) -> "PreTrainedModel":

    path = model_config.model_name_or_path
    # Check for DeepSpeed (especially ZeRO-3)
    is_ds_active = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true"
    quantization_config = get_quantization_config(model_config)
    
    if is_ds_active:
        # DeepSpeed (especially ZeRO-3) is incompatible with device_map.
        # Quantization + ZeRO-3 is also problematic.
        # We disable both to allow manual .to() placement or DS sharding.
        if quantization_config is not None:
         if not getattr(model_config, "load_in_4bit", False):
             print(f">>> [Loader] DeepSpeed detected. Quantization kept as configured (likely None/False).")
             # quantization_config = None # Do NOT force disable. Let user decide.
    
    # Check for adapter (local or remote)
    is_adapter = False
    adapter_conf = None
    token = os.environ.get("HF_TOKEN")
    
    # logic: if adapter_config.json exists, it IS an adapter.
    # We check local path first, then try HF hub
    possible_local_config = os.path.join(path, "adapter_config.json")
    if os.path.exists(possible_local_config):
        is_adapter = True
    else:
        # Try to see if it exists on HF Hub
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=path, filename="adapter_config.json", token=token)
            is_adapter = True
        except Exception as e:
             # print(f"DEBUG: Remote adapter check failed for {path}: {e}")
             is_adapter = False

    if is_adapter:
        try:
            from peft import PeftConfig
            adapter_conf = PeftConfig.from_pretrained(path, token=token)
        except Exception as e:
            print(f">>> [Loader] Identified as adapter but failed to load config: {e}")
            is_adapter = False
    
    if is_adapter:
        print(f">>> Detected PEFT adapter at {path}")
        base_model_path = adapter_conf.base_model_name_or_path
        if base_model_path is None:
            print(f">>> [Loader] WARNING: base_model_name_or_path is None. Falling back to Llama-3.1-8B-Instruct.")
            base_model_path = "meta-llama/Llama-3.1-8B-Instruct"
            
        print(f">>> Loading base model: {base_model_path}")
        
        base_kwargs = dict(
            pretrained_model_name_or_path = base_model_path,
            revision=getattr(model_config, "model_revision", None),
            trust_remote_code=getattr(model_config, "trust_remote_code", False),
            attn_implementation=getattr(model_config, "attn_implementation", None),
            torch_dtype=getattr(model_config, "torch_dtype", None),
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=None if is_ds_active else (get_kbit_device_map() if quantization_config is not None else None),
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            token=token,
        )
        model = model_class.from_pretrained(**base_kwargs)
        print(f">>> Applying adapter from {path}")
        model = PeftModel.from_pretrained(model, path, token=token)
    else:
        model_kwargs = dict(
            pretrained_model_name_or_path = path,
            revision=getattr(model_config, "model_revision", None),
            trust_remote_code=getattr(model_config, "trust_remote_code", False),
            attn_implementation=getattr(model_config, "attn_implementation", None),
            torch_dtype=getattr(model_config, "torch_dtype", None),
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=None if is_ds_active else (get_kbit_device_map() if quantization_config is not None else None),
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            token=token,
        )
        model = model_class.from_pretrained(**model_kwargs)
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model










