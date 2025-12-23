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
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            add_eos_token= True
        )
    except Exception as e:
        # If it fails, check if it is a PEFT adapter
        if os.path.exists(os.path.join(model_name_or_path, "adapter_config.json")):
            with open(os.path.join(model_name_or_path, "adapter_config.json"), "r") as f:
                adapter_conf = json.load(f)
            base_model_path = adapter_conf.get("base_model_name_or_path")
            print(f">>> Tokenizer load failed for adapter path, falling back to base: {base_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=True)
        else:
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

    quantization_config = get_quantization_config(model_config)
    
    path = model_config.model_name_or_path
    is_adapter = os.path.exists(os.path.join(path, "adapter_config.json"))
    
    if is_adapter:
        print(f">>> Detected PEFT adapter at {path}")
        with open(os.path.join(path, "adapter_config.json"), "r") as f:
            adapter_conf = json.load(f)
        base_model_path = adapter_conf.get("base_model_name_or_path")
        print(f">>> Loading base model: {base_model_path}")
        
        base_kwargs = dict(
            pretrained_model_name_or_path = base_model_path,
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation,
            torch_dtype=model_config.torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
        )
        model = model_class.from_pretrained(**base_kwargs)
        print(f">>> Applying adapter from {path}")
        model = PeftModel.from_pretrained(model, path)
    else:
        model_kwargs = dict(
            pretrained_model_name_or_path = path,
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation,
            torch_dtype=model_config.torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
        )
        model = model_class.from_pretrained(**model_kwargs)
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model










