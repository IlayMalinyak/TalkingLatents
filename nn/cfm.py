import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from torchcfm import ConditionalFlowMatcher
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

class SpectralFlowBridge(nn.Module):
    """
    Conditional Flow Matching module to learn bidirectional mapping 
    between token distributions and spectral features
    """
    def __init__(self, vocab_size, feature_dim, hidden_dim=512, 
                 time_embed_dim=128, sigma_min=0.001):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.sigma_min = sigma_min
        
        # Initialize CFM with Optimal Transport
        self.cfm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma_min)
        
        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Token condition encoder
        self.token_encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Hidden state condition encoder (for when using hidden states instead of logits)
        # Assume hidden states are 4096-dimensional (LLaMA dimension)
        self.hidden_encoder = nn.Sequential(
            nn.Linear(4096, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Vector field network v_θ(x_t, t, c)
        self.vector_field = nn.Sequential(
            nn.Linear(feature_dim + time_embed_dim + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        
    def encode_condition(self, input_repr):
        """Encode token distribution or hidden states as condition"""
        # Handle different input shapes and types
        if input_repr.dim() == 3:  # (B, seq_len, D)
            # Use last position or pool
            input_repr = input_repr[:, -1, :]  # (B, D)
        
        # Determine if input is logits (vocab_size) or hidden states (hidden_dim)
        if input_repr.size(-1) == self.vocab_size:
            # Input is logits, convert to probabilities
            token_probs = F.softmax(input_repr, dim=-1)
            return self.token_encoder(token_probs)
        else:
            # Input is hidden states, use hidden encoder
            return self.hidden_encoder(input_repr)
    
    def compute_vector_field(self, x_t, t, condition):
        """Compute the vector field v(x_t, t | c)"""
        # Expand time to match batch size
        if t.dim() == 0:
            t = t.view(1, 1).expand(x_t.size(0), 1)
        elif t.dim() == 1:
            t = t.view(-1, 1)
            
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Ensure condition has the same batch size as x_t
        if condition.size(0) != x_t.size(0):
            # If condition has different batch size, expand or repeat to match
            if condition.size(0) == 1:
                condition = condition.expand(x_t.size(0), -1)
            elif x_t.size(0) == 1:
                condition = condition[:1]  # Take first sample
            else:
                raise ValueError(f"Incompatible batch sizes: x_t={x_t.size(0)}, condition={condition.size(0)}")
        
        # Concatenate state, time, and condition
        h = torch.cat([x_t, t_emb, condition], dim=-1)
        
        # Compute vector field
        return self.vector_field(h)
    
    def training_step(self, input_repr, target_features):
        """
        Compute CFM loss for training
        
        Args:
            input_repr: (B, vocab_size), (B, seq_len, vocab_size) for logits, or
                       (B, hidden_dim), (B, seq_len, hidden_dim) for hidden states
            target_features: (B, feature_dim) ground truth features
        """
        batch_size = target_features.size(0)
        device = target_features.device
        
        # Encode condition
        condition = self.encode_condition(input_repr)
        
        # Sample noise (source distribution)
        x_0 = torch.randn_like(target_features)
        x_1 = target_features
        
        # Get flow matching loss using TorchCFM
        t, xt, ut = self.cfm.sample_location_and_conditional_flow(x_0, x_1)
        
        # Predict vector field
        vt = self.compute_vector_field(xt, t, condition)
        
        # Flow matching loss
        loss = F.mse_loss(vt, ut)
        
        return loss
    
    @torch.no_grad()
    def generate_features(self, token_logits, steps=50, method='euler'):
        """
        Generate features from token distribution by solving ODE
        
        Args:
            token_logits: Token logits to condition on
            steps: Number of integration steps
            method: ODE solver ('euler' or 'dopri5')
        """
        batch_size = token_logits.size(0) if token_logits.dim() > 1 else 1
        device = token_logits.device
        
        # Encode condition once
        condition = self.encode_condition(token_logits)
        
        # Start from noise
        x_0 = torch.randn(batch_size, self.feature_dim, device=device)
        
        if method == 'euler':
            # Simple Euler integration
            dt = 1.0 / steps
            x_t = x_0
            
            for step in range(steps):
                t = torch.tensor([step * dt], device=device)
                v_t = self.compute_vector_field(x_t, t, condition)
                x_t = x_t + v_t * dt
                
            return x_t
            
        else:  # dopri5 or other adaptive methods
            def ode_func(t, x):
                # t is a scalar, x is (batch, feature_dim)
                with torch.no_grad():
                    return self.compute_vector_field(x, t, condition)
            
            t_span = torch.linspace(0, 1, steps, device=device)
            solution = odeint(ode_func, x_0, t_span, method='dopri5')
            return solution[-1]  # Return final state