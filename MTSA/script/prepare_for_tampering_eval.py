import torch
import argparse
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    print(f">>> Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f">>> Loading adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print(">>> Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f">>> Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(">>> Success!")

if __name__ == "__main__":
    main()
