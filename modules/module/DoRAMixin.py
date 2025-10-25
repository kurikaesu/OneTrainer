import torch
import torch.nn.functional as F
from torch import Tensor, nn

class DoRAMixin:
    """Mixin providing DoRA (Weight-Decomposed Low-Rank Adaptation) functionality.
    
    Can be mixed into any PEFT module to add weight decomposition.
    """
    dora_scale: Tensor | None
    norm_epsilon: bool
    decompose_output_axis: bool
    dora_num_dims: int
    train_device: torch.device

    def __init__(self, *args, **kwargs):
        """Initialize DoRA attributes with safe defaults."""
        # Set default DoRA attributes to ensure they exist
        # even if init_dora_params hasn't been called yet
        if not hasattr(self, 'dora_scale'):
            self.dora_scale = None
        if not hasattr(self, 'norm_epsilon'):
            self.norm_epsilon = False
        if not hasattr(self, 'decompose_output_axis'):
            self.decompose_output_axis = False
        if not hasattr(self, 'train_device'):
            self.train_device = None
        if not hasattr(self, 'dora_num_dims'):
            self.dora_num_dims = 0
        
        # Continue with MRO chain
        super().__init__(*args, **kwargs)

    def init_dora_params(
        self, 
        norm_epsilon: bool = False,
        decompose_output_axis: bool = False,
        train_device: torch.device = None
    ):
        """Initialize DoRA-specific parameters.
        
        Args:
            norm_epsilon: Add epsilon to norm for numerical stability
            decompose_output_axis: Decompose along output axis instead of input
            train_device: Device for training (needed for quantized weights)
        """
        self.dora_scale = None
        self.norm_epsilon = norm_epsilon
        self.decompose_output_axis = decompose_output_axis
        self.train_device = train_device
        
    def initialize_dora_scale(self, orig_weight: Tensor):
        """Initialize the DoRA scale parameter from original weights.
        
        Args:
            orig_weight: The original weight tensor (unquantized, float)
        """
        self.dora_num_dims = orig_weight.dim() - 1
        
        if self.decompose_output_axis:
            scale = torch.norm(
                orig_weight.reshape(orig_weight.shape[0], -1),
                dim=1, keepdim=True
            ).reshape(orig_weight.shape[0], *[1] * self.dora_num_dims)
        else:
            scale = torch.norm(
                orig_weight.transpose(1, 0).reshape(orig_weight.shape[1], -1),
                dim=1, keepdim=True
            ).reshape(orig_weight.shape[1], *[1] * self.dora_num_dims).transpose(1, 0)
        
        self.dora_scale = nn.Parameter(
            scale.to(device=self.orig_module.weight.device)
        )
        
    def apply_dora(self, weight: Tensor) -> Tensor:
        """Apply DoRA decomposition to a weight matrix.
        
        Args:
            weight: Combined weight (original + PEFT adaptation)
            
        Returns:
            Weight after DoRA decomposition
        """
        assert self.dora_scale is not None, "DoRA scale not initialized"
        
        # Compute norm (detached for memory efficiency per paper section 4.3)
        eps = torch.finfo(weight.dtype).eps if self.norm_epsilon else 0.0
        
        if self.decompose_output_axis:
            norm = weight.detach() \
                .reshape(weight.shape[0], -1) \
                .norm(dim=1) \
                .reshape(weight.shape[0], *[1] * self.dora_num_dims) + eps
        else:
            norm = weight.detach() \
                .transpose(0, 1) \
                .reshape(weight.shape[1], -1) \
                .norm(dim=1, keepdim=True) \
                .reshape(weight.shape[1], *[1] * self.dora_num_dims) \
                .transpose(0, 1) + eps
        
        # Apply decomposition: scale * (weight / norm)
        return self.dora_scale * (weight / norm)
