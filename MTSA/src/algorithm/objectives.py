import torch
import torch.nn.functional as F


def obj_max_entropy_next_token(model, batch):
    """
    Objective to maximize entropy of next token distribution.
    Used for adversarial tampering to make the model uncertain/vulnerable.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Entropy = -sum(p * log(p))
    # We want to MAXIMIZE entropy, so we MINIMIZE negative entropy (-H)
    entropy = -(probs * log_probs).sum(-1)
    
    if attention_mask is not None:
        entropy = entropy * attention_mask
        mean_entropy = entropy.sum() / attention_mask.sum()
    else:
        mean_entropy = entropy.mean()
        
    return -mean_entropy, {}

def obj_standard_max_next_token(model, batch):
    """
    Objective to minimize cross-entropy loss (maximize next-token likelihood).
    Used for simulated SFT tampering attacks.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch.get("labels", input_ids.clone())
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    
    # Filter out padding in labels (assuming -100 is pad/ignore)
    # If using causal LM without explicit labels, we filter by attention mask
    if attention_mask is not None:
        shift_mask = attention_mask[..., 1:].contiguous().view(-1)
        # Apply mask to labels to ignore non-target tokens
        shift_labels[shift_mask == 0] = -100

    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    
    return loss, {}
