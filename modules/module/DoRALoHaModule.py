import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modules.util.quantization_util import get_unquantized_weight
from modules.module.LoRAModule import LoHaModule, DoRAMixin


class DoRALoHaModule(LoHaModule, DoRAMixin):
    """LoHa module with DoRA weight decomposition support."""
    
    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float, **kwargs):
        # Extract DoRA-specific kwargs
        norm_epsilon = kwargs.pop('norm_epsilon', False)
        decompose_output_axis = kwargs.pop('decompose_output_axis', False)
        train_device = kwargs.pop('train_device')
        
        # Initialize LoHa parent
        super().__init__(prefix, orig_module, rank, alpha)
        
        # Initialize DoRA
        if orig_module is not None:
            self.init_dora_params(norm_epsilon, decompose_output_axis, train_device)
            
    def initialize_weights(self):
        """Initialize both LoHa and DoRA weights."""
        # Initialize LoHa matrices
        super().initialize_weights()
        
        # Initialize DoRA scale
        if isinstance(self.orig_module, nn.Linear):
            orig_weight = get_unquantized_weight(self.orig_module, torch.float, self.train_device)
        else:
            orig_weight = self.orig_module.weight.detach().float()
            
        self.initialize_dora_scale(orig_weight)
        del orig_weight
        
    def check_initialized(self):
        """Check initialization of both LoHa and DoRA."""
        super().check_initialized()
        assert self.dora_scale is not None, "DoRA scale not initialized"
        
    def forward(self, x, *args, **kwargs):
        """Forward pass with LoHa + DoRA."""
        self.check_initialized()
        
        # Compute LoHa weight (Hadamard product)
        W1 = self.make_weight(
            self.dropout(self.hada_w1_b),
            self.dropout(self.hada_w1_a)
        )
        W2 = self.make_weight(
            self.dropout(self.hada_w2_b),
            self.dropout(self.hada_w2_a)
        )
        loha_weight = (W1 * W2) * (self.alpha / self.rank)
        
        # Get original weight
        if isinstance(self.orig_module, nn.Linear):
            orig_weight = get_unquantized_weight(self.orig_module, torch.float, self.train_device)
        else:
            orig_weight = self.orig_module.weight.detach().float()
        
        # Combine: original + LoHa adaptation
        combined_weight = orig_weight + loha_weight
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
