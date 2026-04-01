import pytest
from unittest.mock import MagicMock
from transformers import AutoTokenizer

@pytest.fixture
def mock_tokenizer():
    """Returns a real tokenizer but can be used without a model."""
    # Using a fast, standard tokenizer like GPT-2 or Llama-3-Instruct
    return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

@pytest.fixture
def mock_vllm_engine():
    """A mock vLLM engine that acts like a real LLM for simulation tests."""
    engine = MagicMock()
    
    def mock_generate(prompts, sampling_params, **kwargs):
        # Return a simple mock output structure matching vLLM
        class MockOutput:
            def __init__(self, text):
                self.outputs = [MagicMock(text=text)]
        
        # Simple logic: Refuse if 'forbidden' is in prompt, otherwise comply
        responses = []
        for p in prompts:
            if "forbidden" in p.lower() or "bomb" in p.lower():
                responses.append(MockOutput("I'm sorry, I cannot help with that."))
            else:
                responses.append(MockOutput("Sure! Here is the information you requested: [STUB CONTENT]"))
        return responses

    engine.generate.side_effect = mock_generate
    return engine

@pytest.fixture
def mock_reward_args():
    """A mock configuration object for the Reward Function."""
    args = MagicMock()
    args.defence_mode = True
    args.attack_mode = False
    args.max_sim_turns = 3
    args.judge_type = "strongreject"
    return args
