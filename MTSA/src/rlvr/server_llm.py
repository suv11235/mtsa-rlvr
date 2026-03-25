import requests

class ServerLLM:
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        
    def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=False, **kwargs):
        """Mock the vLLM Engine.generate interface via OpenAI HTTP API"""
        if isinstance(prompts, str):
            prompts = [prompts]
            
        results = []
        for prompt in prompts:
            # Construct JSON payload imitating typical vLLM completions
            payload = {
                "model": "auto",  # vLLM API ignores this when 1 model is loaded
                "prompt": prompt,
                "temperature": getattr(sampling_params, 'temperature', 0.7),
                "top_p": getattr(sampling_params, 'top_p', 0.9),
                "max_tokens": getattr(sampling_params, 'max_tokens', 512),
                "repetition_penalty": getattr(sampling_params, 'repetition_penalty', 1.0)
            }
            
            try:
                response = requests.post(
                    f"{self.endpoint_url}/completions", 
                    json=payload,
                    timeout=600  # Generous timeout for slow inference
                )
                response.raise_for_status()
                data = response.json()
                text_out = data['choices'][0]['text']
                
                # Wrapper mock object to simulate outputs[0].outputs[0].text structure expected by code
                class MockOutputUnit:
                    def __init__(self, text):
                        self.text = text
                class MockOutput:
                    def __init__(self, output_units):
                        self.outputs = output_units
                        
                results.append(MockOutput([MockOutputUnit(text_out)]))
                
            except Exception as e:
                print(f"[ServerLLM] Error hitting {self.endpoint_url}: {e}")
                # Return empty wrapper on fail
                class MockOutputUnit:
                    def __init__(self):
                        self.text = ""
                class MockOutput:
                    def __init__(self):
                        self.outputs = [MockOutputUnit()]
                results.append(MockOutput())
                
        return results
