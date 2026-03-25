import torch
import torch.nn.functional as F

def compute_rep_loss(outputs, ref_outputs):
    """
    Computes the Representation Closeness loss (TAR style)
    between the current model outputs and frozen reference outputs.
    """
    layer_losses = []
    for h_cur, h_ref in zip(outputs.hidden_states, ref_outputs.hidden_states):
        # (bs, seq, hidden) -> norm along hidden -> mean
        diff = h_cur - h_ref
        layer_losses.append(torch.norm(diff, dim=-1).mean())
    rep_loss = torch.stack(layer_losses).mean()
    return rep_loss

class TamperResistanceManager:
    """
    Optional manager to handle the inner-loop 'tampering' 
    (Weight-space Simulated Attack / Scrambling) before outer-loop PPO update.
    """
    def __init__(self, target_model, inner_lr=5e-5, inner_steps=1, tamper_type="scramble"):
        self.target_model = target_model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.tamper_type = tamper_type
        
    def _obj_max_entropy_next_token(self, model, batch):
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        # Max entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return -entropy # Minimize negative entropy = Maximize entropy
        
    def tamper(self, batch):
        """Simulates an attack modifying the weights, returns the original weights."""
        original_state = {}
        for name, param in self.target_model.named_parameters():
            if param.requires_grad:
                original_state[name] = param.data.clone()
                
        inner_optimizer = torch.optim.SGD(self.target_model.parameters(), lr=self.inner_lr)
        self.target_model.train()
        
        for _ in range(self.inner_steps):
            if self.tamper_type == "scramble":
                loss = self._obj_max_entropy_next_token(self.target_model, batch)
            else:
                # Placeholder for direct SFT attack
                loss = self._obj_max_entropy_next_token(self.target_model, batch)
                
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
            
        return original_state
        
    def restore(self, original_state):
        """Restores the weights to their pre-tampering state."""
        with torch.no_grad():
            for name, param in self.target_model.named_parameters():
                if name in original_state:
                    param.data.copy_(original_state[name])
        # Clear Memory
        original_state.clear()
