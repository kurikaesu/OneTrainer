import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modules.util.quantization_util import get_unquantized_weight
from modules.module.DoRAMixin import DoRAMixin
from modules.module.LoKrModule import LoKrModule


class DoRALoKrModule(LoKrModule, DoRAMixin):
    """LoKr module with DoRA weight decomposition support."""
    
    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float, **kwargs):
        # Extract DoRA-specific kwargs
        norm_epsilon = kwargs.pop('norm_epsilon', False)
        decompose_output_axis = kwargs.pop('decompose_output_axis', False)
        train_device = kwargs.pop('train_device')
        
        # Set DoRA parameters early to ensure they exist
        self.norm_epsilon = norm_epsilon
        self.decompose_output_axis = decompose_output_axis
        self.train_device = train_device
        
        # Initialize LoKr parent (which calls DoRAMixin.__init__ via MRO)
        super().__init__(prefix, orig_module, rank, alpha)
        
        # Complete DoRA initialization with specific parameters
        if orig_module is not None:
            self.init_dora_params(norm_epsilon, decompose_output_axis, train_device)
            
        # Initialize weights after setup to ensure dora_scale is set
        if orig_module is not None:
            self.initialize_weights()
            
    def initialize_weights(self):
        """Initialize both LoKr and DoRA weights."""
        # Initialize LoKr matrices
        super().initialize_weights()
        
        # Initialize DoRA scale
        if isinstance(self.orig_module, nn.Linear):
            orig_weight = get_unquantized_weight(self.orig_module, torch.float, self.train_device)
        else:
            orig_weight = self.orig_module.weight.detach().float()
            
        self.initialize_dora_scale(orig_weight)
        del orig_weight
    
    def check_initialized(self):
        """Check initialization of both LoKr and DoRA."""
        super().check_initialized()
        assert self.dora_scale is not None, "DoRA scale not initialized"
        
    def forward(self, x, *args, **kwargs):
        """Forward pass with LoKr + DoRA."""
        self.check_initialized()
        
        # Apply dropout
        w1_a_drop = self.dropout(self.w1_a)
        w1_b_drop = self.dropout(self.w1_b)
        w2_a_drop = self.dropout(self.w2_a)
        w2_b_drop = self.dropout(self.w2_b)
        
        # Compute Kronecker products
        # W1 = w1_a ⊗ w1_b
        # W2 = w2_a ⊗ w2_b
        W1 = self.kronecker_product(w1_a_drop, w1_b_drop)
        W2 = self.kronecker_product(w2_a_drop, w2_b_drop)
        
        # Final weight: ΔW = W1 @ W2 * (alpha / rank)
        W = self.make_weight(W1, W2) * (self.alpha / self.rank)
        
        # Get original weight
        if isinstance(self.orig_module, nn.Linear):
            orig_weight = get_unquantized_weight(self.orig_module, torch.float, self.train_device)
        else:
            orig_weight = self.orig_module.weight.detach().float()
        
        # Combine: original + LoKr adaptation
        combined_weight = orig_weight + W
        del orig_weight
        
        # Apply DoRA decomposition
        final_weight = self.apply_dora(combined_weight)
        
        # Forward with DoRA weight
        # Note: DoRA paper applies dropout to input, not intermediate layers
        return self.op(
            self.dropout(x),
            final_weight,
            self.orig_module.bias,
            **self.layer_kwargs
        )


DummyDoRALoKrModule = DoRALoKrModule.make_dummy()
