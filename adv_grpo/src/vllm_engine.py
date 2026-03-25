import os
import multiprocessing as mp
from queue import Empty

class RemoteLLM:
    """
    Isolated vLLM Engine wrapper that forces execution in a separate process to avoid 
    NCCL hangs when mixed with DeepSpeed ZeRO-3 training on the same GPU.
    """
    def __init__(self, model_path, token_path, gpu_memory_utilization, port_offset, rank, local_rank, max_model_len, enable_lora=True, hf_token=None):
        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        
        self.process = ctx.Process(
            target=self._run_engine,
            args=(model_path, token_path, gpu_memory_utilization, port_offset, rank, local_rank, max_model_len, enable_lora, self.input_queue, self.output_queue, hf_token)
        )
        self.process.start()
        
        # Wait for initialization status
        status = self.output_queue.get()
        if status != "READY":
            raise RuntimeError(f"vLLM Engine failed to start: {status}")
        
    @staticmethod
    def _run_engine(model_path, token_path, mem, port_offset, rank, local_rank, config_len, enable_lora, input_q, output_q, hf_token):
        # 0. IMMEDIATE ISOLATION & ENV SETUP (MUST BE BEFORE ANY IMPORTS)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["VLLM_ENABLE_V1"] = "0"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["VLLM_WORK_DIR"] = f"/tmp/vllm_{rank}_{port_offset}"
        
        # Restore tokens IMMEDIATELY
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            os.environ["HUGGINGFACE_TOKEN"] = hf_token
            
        # Clean Environment Extremely Aggressively
        keys_to_delete = [
            "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE",
            "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_ID", "NCCL_COMM_ID",
            "PORT", "ID", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"
        ]
        for k in list(os.environ.keys()):
            if k.startswith("SLURM_") or k.startswith("ACCELERATE_") or k.startswith("TORCHELASTIC_") or k in keys_to_delete:
                del os.environ[k]
        
        # Set standalone defaults
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(29500 + port_offset + rank * 10)
        
        print(f">>> [vLLM Worker {rank}:{port_offset}] Starting on GPU {local_rank}")
        token_vars = [k for k in os.environ.keys() if "TOKEN" in k]
        print(f">>> [vLLM Worker] Scrubbed env. Master at {os.environ['MASTER_PORT']}. Token vars: {token_vars}")
        
        try:
            import torch
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            
            # 2. PEFT Detection & Handling
            is_adapter = False
            effective_model = model_path
            adapter_path = None
            
            try:
                from peft import PeftConfig
                print(f">>> [vLLM Worker] Checking if {model_path} is an adapter...")
                peft_conf = PeftConfig.from_pretrained(model_path, token=hf_token)
                is_adapter = True
                adapter_path = model_path
                effective_model = peft_conf.base_model_name_or_path
                enable_lora = True
                print(f">>> [vLLM Worker] Detected adapter. Using base model: {effective_model}")
            except Exception as pe:
                print(f">>> [vLLM Worker] Not an adapter or PeftConfig fail (expected for full models): {pe}")

            # 3. Initialize LLM
            print(f">>> [vLLM Worker] Loading model {effective_model} with utilization {mem}")
            llm = LLM(
                model=effective_model,
                tokenizer=token_path,
                gpu_memory_utilization=float(mem),
                trust_remote_code=True,
                enforce_eager=False,
                max_model_len=config_len or 2048,
                enable_lora=enable_lora,
                max_loras=1 if enable_lora else 0,
                disable_log_stats=True,
                dtype="bfloat16"
            )
            
            # Signal Ready
            output_q.put("READY")
            
            # 4. Request Loop
            while True:
                try:
                    req = input_q.get(timeout=1.0)
                except Empty:
                    continue
                    
                if req is None: # Shutdown signal
                    break
                    
                prompts, sampling_params, kwargs = req
                
                # If it's an adapter, we MUST use a LoRARequest if non-provided
                lora_request = kwargs.pop("lora_request", None)
                if is_adapter and lora_request is None:
                    lora_request = LoRARequest("adapter", 1, adapter_path)
                
                if lora_request is not None:
                    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
                else:
                    outputs = llm.generate(prompts, sampling_params)
                    
                output_q.put(outputs)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            output_q.put(f"ERROR: {e}")

    def generate(self, prompts, sampling_params, lora_request=None, **kwargs):
        if lora_request is not None:
            kwargs["lora_request"] = lora_request
            
        self.input_queue.put((prompts, sampling_params, kwargs))
        res = self.output_queue.get()
        if isinstance(res, str) and res.startswith("ERROR"):
            raise RuntimeError(f"Remote generate failed: {res}")
        return res
        
    def shutdown(self):
        self.input_queue.put(None)
        self.process.join()
