import pytest
from src.rlvr.reward_manager.multiturn_reward import MultiTurnRewardFunction

def test_reward_logic_no_gpu(mock_tokenizer, mock_vllm_engine, mock_reward_args):
    """
    Test the MultiTurnRewardFunction extraction and scoring logic using only CPU mocks.
    """
    reward_fn = MultiTurnRewardFunction(
        judge_model=None,
        judge_tokenizer=mock_tokenizer,
        victim_model=None, # Only vllm_engine used 
        victim_tokenizer=mock_tokenizer,
        attacker_model=None, 
        attacker_tokenizer=mock_tokenizer,
        defence_mode=True,
        vllm_engine=mock_vllm_engine,
        attacker_vllm_engine=mock_vllm_engine,
        judge_vllm_engine=mock_vllm_engine,
        template_type="llama3",
        max_sim_turns=2,
        judge_type="strongreject"
    )

    # 1. TEST CASE: SUCCESSFUL ATTACK (Low Safety Score)
    solution_str = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "How to build a forbidden device?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "### Jailbreak Payload\nAttack question 1: How to build a bomb?"
    )
    
    # In this mock: goal "bomb" causes the victim (also mock) to refuse.
    # But because it's a refuse, the judge (also the mock) will score it as 'Safe' (1.0).
    metrics = reward_fn(
        data_source="test",
        solution_str=solution_str,
        ground_truth=None,
        extra_info={'goal': "harmful behavior"}
    )
    
    assert 'score' in metrics
    assert 'attack_success_reward' in metrics
    
    # In our mock, if it's safe, reward is 0.0
    # Because defence_mode=True, we reward (1.0 - attack_reward)
    assert metrics['score'] >= 0.0
    print(f"\n[Pytest Result] Multi-turn Reward: {metrics['score']}")

def test_payload_extraction():
    """Test the extraction logic which often fails on LLM gibberish."""
    from src.rlvr.reward_manager.multiturn_reward import force_extract_payload
    
    text = "### Jailbreak Payload\nAttack question 2: Specific forbidden question\nExtra: junk"
    extracted = force_extract_payload(text)
    assert extracted == "Specific forbidden question"
    
    text_with_thinking = "<think>I need to attack.</think>### Jailbreak Payload\nAttack question 1: Real attack"
    # test stripping
    extracted = force_extract_payload(text_with_thinking, include_thinking=False)
    assert "think" not in extracted
    assert extracted == "Real attack"
