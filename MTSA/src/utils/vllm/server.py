"""
Helper to launch a vLLM server process if configured.
Inspired by the Modal/olmOCR implementation but adapted for local multi-GPU setup.
"""

import os
import subprocess
import time
import requests
import signal
import atexit

class VLLMServer:
    def __init__(self, model_name, port=8000, gpu_memory_utilization=0.3, tensor_parallel_size=1, device="0"):
        self.model_name = model_name
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.process = None
        self.base_url = f"http://localhost:{port}/v1"
        self.device = device # CUDA_VISIBLE_DEVICES string
    
    def start(self):
        """Start the vLLM server in a subprocess."""
        
        # Prepare environment: Isolate to specific GPUs if requested
        env = os.environ.copy()
        if self.device is not None:
             env["CUDA_VISIBLE_DEVICES"] = str(self.device)
             
        print(f">>> [VLLMServer] Starting vLLM on device(s) {self.device} port {self.port}...")
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--dtype", "bfloat16",
            "--trust-remote-code",
            "--enable-lora", # Crucial for our multi-turn setup
            "--max-loras", "8",
            "--disable-log-stats"
        ]
        
        # Start
        self.process = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        # Cleanup on exit
        atexit.register(self.stop)
        
        # Wait for health check
        self._wait_for_health()
        
    def _wait_for_health(self, timeout=300):
        start = time.time()
        print(f">>> [VLLMServer] Waiting for health check at {self.base_url}...")
        while time.time() - start < timeout:
            try:
                # vLLM provides /health endpoint, but OpenAI format uses /v1/models usually
                # Standard vLLM health check
                resp = requests.get(f"http://localhost:{self.port}/health")
                if resp.status_code == 200:
                    print(f">>> [VLLMServer] Ready!")
                    return
            except requests.exceptions.ConnectionError:
                pass
            
            # Check if process died
            if self.process.poll() is not None:
                _, stderr = self.process.communicate()
                raise RuntimeError(f"vLLM server died.\nStderr: {stderr.decode()}")
                
            time.sleep(5)
            
        raise TimeoutError("vLLM server failed to start in time.")

    def stop(self):
        if self.process:
            print(f">>> [VLLMServer] Stopping...")
            os.kill(self.process.pid, signal.SIGTERM)
            self.process.wait()
            self.process = None

# Global instance manager if needed
_SERVER = None

def get_or_start_vllm_server(model_name, device="0", port=8000, **kwargs):
    global _SERVER
    if _SERVER is None:
        _SERVER = VLLMServer(model_name, device=device, port=port, **kwargs)
        _SERVER.start()
    return _SERVER
