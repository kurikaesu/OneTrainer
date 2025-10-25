import copy
from collections.abc import Mapping
from typing import Any

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import PeftType
from modules.util.ModuleFilter import ModuleFilter

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Conv2d, Linear, Parameter


class LoRAModuleWrapper:
    orig_module: nn.Module
    rank: int
    alpha: float
    module_filters: list[ModuleFilter]

    lora_modules: dict[str, 'PeftBase']

    def __init__(
            self,
            orig_module: nn.Module | None,
            prefix: str,
            config: TrainConfig,
            module_filter: list[str] = None,
    ):
        self.orig_module = orig_module
        self.prefix = prefix
        self.peft_type = config.peft_type
        self.rank = config.lora_rank
        self.alpha = config.lora_alpha

        self.module_filters = [
            ModuleFilter(pattern, use_regex=config.layer_filter_regex)
            for pattern in (module_filter or [])
        ]

        weight_decompose = config.lora_decompose
        if self.peft_type == PeftType.LORA:
            if weight_decompose:
                from modules.module.DoRALoRAModule import DoRALoRAModule
                self.klass = DoRALoRAModule
                from modules.module.DoRALoRAModule import DummyDoRALoRAModule
                self.dummy_klass = DummyDoRALoRAModule
                self.additional_args = [self.rank, self.alpha]
                self.additional_kwargs = {
                    'norm_epsilon': config.lora_decompose_norm_epsilon,
                    'decompose_output_axis': config.lora_decompose_output_axis,
                    'train_device': torch.device(config.train_device),
                }
            else:
                from modules.module.LoRAModule import LoRAModule
                self.klass = LoRAModule
                from modules.module.LoRAModule import DummyLoRAModule
                self.dummy_klass = DummyLoRAModule
                self.additional_args = [self.rank, self.alpha]
                self.additional_kwargs = {}
        elif self.peft_type == PeftType.LOHA:
            from modules.module.LoHaModule import LoHaModule
            self.klass = LoHaModule
            from modules.module.LoHaModule import DummyLoHaModule
            self.dummy_klass = DummyLoHaModule
            self.additional_args = [self.rank, self.alpha]
            self.additional_kwargs = {}
        elif self.peft_type == PeftType.LOKR:
            if weight_decompose:
                from modules.module.DoRALoKrModule import DoRALoKrModule
                self.klass = DoRALoKrModule
                from modules.module.DoRALoKrModule import DummyDoRALoKrModule
                self.dummy_klass = DummyDoRALoKrModule
                self.additional_args = [self.rank, self.alpha]
                self.additional_kwargs = {
                    'norm_epsilon': config.lora_decompose_norm_epsilon,
                    'decompose_output_axis': config.lora_decompose_output_axis,
                    'train_device': torch.device(config.train_device),
                }
            else:
                from modules.module.LoKrModule import LoKrModule
                self.klass = LoKrModule
                from modules.module.LoKrModule import DummyLoKrModule
                self.dummy_klass = DummyLoKrModule
                self.additional_args = [self.rank, self.alpha]
                self.additional_kwargs = {}

        self.lora_modules = self.__create_modules(orig_module, config)

    def __create_modules(self, orig_module: nn.Module | None, config: TrainConfig) -> dict[str, 'PeftBase']:
        if orig_module is None:
            return {}

        lora_modules = {}
        selected = []
        deselected = []
        unsuitable = []

        for name, child_module in orig_module.named_modules():
            if not isinstance(child_module, Linear | Conv2d):
                unsuitable.append(name)
                continue
            if len(self.module_filters) == 0 or any(f.matches(name) for f in self.module_filters):
                lora_modules[name] = self.klass(self.prefix + "." + name, child_module, *self.additional_args, **self.additional_kwargs)
                selected.append(name)
            else:
                deselected.append(name)

        if len(self.module_filters) > 0:
            if config.debug_mode:
                print(f"Selected layers: {selected}")
                print(f"Deselected layers: {deselected}")
                print(f"Unsuitable for LoRA training: {unsuitable}")
            else:
                print(f"Selected layers: {len(selected)}")
                print(f"Deselected layers: {len(deselected)}")
                print("Note: Enable Debug mode to see the full list of layer names")

        unused_filters = [mf for mf in self.module_filters if not mf.was_used()]
        if len(unused_filters) > 0:
            raise ValueError('Custom layer filters: no modules were matched by the custom filter(s)')

        return lora_modules

    def requires_grad_(self, requires_grad: bool):
        for module in self.lora_modules.values():
            module.requires_grad_(requires_grad)

    def parameters(self) -> list[Parameter]:
        parameters = []
        for module in self.lora_modules.values():
            parameters += module.parameters()
        return parameters

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> 'LoRAModuleWrapper':
        for module in self.lora_modules.values():
            module.to(device, dtype)
        return self

    def _check_rank_matches(self, state_dict: dict[str, Tensor]):
        if not state_dict:
            return

        if rank_key := next((k for k in state_dict if k.endswith((".lora_down.weight", ".hada_w1_a", ".w1_a"))), None):
            if (checkpoint_rank := state_dict[rank_key].shape[0]) != self.rank:
                raise ValueError(f"Rank mismatch: checkpoint={checkpoint_rank}, config={self.rank}, please correct in the UI.")

    def load_state_dict(self, state_dict: dict[str, Tensor], strict: bool = True):
        """
        Loads the state dict

        Args:
            state_dict: the state dict
            strict: whether to strictly enforce that the keys in state_dict match the module's parameters
        """
        # create a copy, so the modules can pop states
        state_dict = {k: v for (k, v) in state_dict.items() if k.startswith(self.prefix)}

        self._check_rank_matches(state_dict)

        for module in self.lora_modules.values():
            module.load_state_dict(state_dict, strict=strict)

        # Temporarily re-create the state dict, so we can see what keys were left.
        remaining_names = set(state_dict) - set(self.state_dict())

        # create dummy modules for the remaining keys
        for name in remaining_names:
            if name.endswith(".alpha"):
                prefix = name.removesuffix(".alpha")
                module = self.dummy_klass(prefix, None, *self.additional_args, **self.additional_kwargs)
                module.load_state_dict(state_dict)
                self.lora_modules[prefix] = module

    def state_dict(self) -> dict:
        """
        Returns the state dict
        """
        state_dict = {}

        for module in self.lora_modules.values():
            state_dict |= module.state_dict(prefix=module.prefix)

        return state_dict

    def modules(self) -> list[nn.Module]:
        """
        Returns a list of all modules
        """
        modules = []
        for module in self.lora_modules.values():
            modules += module.modules()

        return modules

    def hook_to_module(self):
        """
        Hooks the LoRA into the module without changing its weights
        """
        for module in self.lora_modules.values():
            module.hook_to_module()

    def remove_hook_from_module(self):
        """
        Removes the LoRA hook from the module without changing its weights
        """
        for module in self.lora_modules.values():
            module.remove_hook_from_module()

    def apply_to_module(self):
        """
        Applys the LoRA to the module, changing its weights
        """
        for module in self.lora_modules.values():
            module.apply_to_module()

    def extract_from_module(self, base_module: nn.Module):
        """
        Creates a LoRA from the difference between the base_module and the orig_module
        """
        for module in self.lora_modules.values():
            module.extract_from_module(base_module)

    def prune(self):
        """
        Removes all dummy modules
        """
        self.lora_modules = {k: v for (k, v) in self.lora_modules.items() if not isinstance(v, self.dummy_klass)}

    def set_dropout(self, dropout_probability: float):
        """
        Sets the dropout probability
        """
        if dropout_probability < 0 or dropout_probability > 1:
            raise ValueError("Dropout probability must be in [0, 1]")
        for module in self.lora_modules.values():
            module.dropout.p = dropout_probability
