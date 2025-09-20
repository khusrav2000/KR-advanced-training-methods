"""
Flipping Error Reduction (FER) Optimizer

This optimizer implements the FER approach to restrict behavior changes
of a DNN on correctly classified samples to maintain correct local boundaries
and reduce flipping errors on unseen samples.

Based on the paper approach that improves generalization, robustness,
and transferability without additional network parameters or inference cost.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional, Union


class FEROptimizer(Optimizer):
    """
    Flipping Error Reduction Optimizer
    
    This optimizer wraps another base optimizer and applies FER regularization
    to reduce flipping errors by maintaining consistency on correctly classified samples.
    
    Args:
        params: Iterable of parameters to optimize
        base_optimizer: Base optimizer class to wrap (e.g., torch.optim.SGD)
        base_optimizer_kwargs: Keyword arguments for base optimizer
        fer_weight: Weight for FER regularization term (default: 0.1)
        consistency_threshold: Threshold for considering predictions consistent (default: 0.9)
        memory_size: Size of memory buffer for storing past predictions (default: 1000)
        update_freq: Frequency of memory buffer updates (default: 10)
    """
    
    def __init__(
        self,
        params,
        base_optimizer: type = torch.optim.SGD,
        base_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        fer_weight: float = 0.1,
        consistency_threshold: float = 0.9,
        memory_size: int = 1000,
        update_freq: int = 10,
        **kwargs
    ):
        if base_optimizer_kwargs is None:
            base_optimizer_kwargs = {'lr': 0.01}
        
        defaults = dict(
            fer_weight=fer_weight,
            consistency_threshold=consistency_threshold,
            memory_size=memory_size,
            update_freq=update_freq,
            **base_optimizer_kwargs
        )
        
        super().__init__(params, defaults)
        
        # Initialize base optimizer with same parameters
        # Extract all parameters from param_groups
        all_params = []
        for group in self.param_groups:
            all_params.extend(group['params'])
        
        self.base_optimizer = base_optimizer(all_params, **base_optimizer_kwargs)
        
        # FER-specific state
        self.memory_buffer = {}
        self.step_count = 0
        
    def _get_param_id(self, param: torch.Tensor) -> int:
        """Get unique identifier for parameter tensor"""
        return id(param)
    
    def _initialize_memory(self, param: torch.Tensor, group: Dict[str, Any]) -> None:
        """Initialize memory buffer for parameter"""
        param_id = self._get_param_id(param)
        if param_id not in self.memory_buffer:
            self.memory_buffer[param_id] = {
                'past_grads': torch.zeros(
                    group['memory_size'], *param.shape, 
                    device=param.device, dtype=param.dtype
                ),
                'past_outputs': torch.zeros(
                    group['memory_size'], device=param.device, dtype=param.dtype
                ),
                'current_idx': 0,
                'filled': False
            }
    
    def _compute_fer_regularization(self, param: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
        """Compute FER regularization term"""
        param_id = self._get_param_id(param)
        
        if param_id not in self.memory_buffer:
            return torch.zeros_like(param.grad)
        
        memory = self.memory_buffer[param_id]
        
        if not memory['filled'] and memory['current_idx'] < 10:
            # Not enough history for regularization
            return torch.zeros_like(param.grad)
        
        # Get valid past gradients
        if memory['filled']:
            past_grads = memory['past_grads']
        else:
            past_grads = memory['past_grads'][:memory['current_idx']]
        
        if past_grads.numel() == 0:
            return torch.zeros_like(param.grad)
        
        # Compute consistency-based regularization
        current_grad = param.grad
        
        # Calculate similarity with past gradients
        similarities = torch.zeros(past_grads.size(0), device=param.device)
        
        for i in range(past_grads.size(0)):
            if past_grads[i].norm() > 1e-8 and current_grad.norm() > 1e-8:
                cosine_sim = torch.nn.functional.cosine_similarity(
                    past_grads[i].flatten(), 
                    current_grad.flatten(), 
                    dim=0
                )
                similarities[i] = cosine_sim
        
        # Apply consistency threshold
        consistent_mask = similarities > group['consistency_threshold']
        
        if consistent_mask.any():
            # Regularize towards consistent past gradients
            consistent_grads = past_grads[consistent_mask]
            avg_consistent_grad = consistent_grads.mean(dim=0)
            
            # FER regularization: encourage consistency
            fer_term = group['fer_weight'] * (current_grad - avg_consistent_grad)
            return fer_term
        else:
            # No consistent past gradients found
            return torch.zeros_like(param.grad)
    
    def _update_memory(self, param: torch.Tensor, group: Dict[str, Any]) -> None:
        """Update memory buffer with current gradient"""
        param_id = self._get_param_id(param)
        self._initialize_memory(param, group)
        
        memory = self.memory_buffer[param_id]
        
        # Store current gradient
        memory['past_grads'][memory['current_idx']] = param.grad.clone().detach()
        
        # Update index
        memory['current_idx'] = (memory['current_idx'] + 1) % group['memory_size']
        
        if memory['current_idx'] == 0:
            memory['filled'] = True
    
    def step(self, closure=None):
        """Perform a single optimization step with FER regularization"""
        
        # Apply FER regularization to gradients
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # Compute FER regularization
                    fer_term = self._compute_fer_regularization(param, group)
                    
                    # Apply regularization to gradient
                    param.grad = param.grad + fer_term
                    
                    # Update memory if needed
                    if self.step_count % group['update_freq'] == 0:
                        self._update_memory(param, group)
        
        # Perform base optimizer step
        loss = self.base_optimizer.step(closure)
        
        self.step_count += 1
        return loss
    
    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients of all optimized parameters"""
        self.base_optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        """Return the state of the optimizer as a dict"""
        base_state = self.base_optimizer.state_dict()
        fer_state = {
            'memory_buffer': {k: {
                'past_grads': v['past_grads'].cpu(),
                'past_outputs': v['past_outputs'].cpu(),
                'current_idx': v['current_idx'],
                'filled': v['filled']
            } for k, v in self.memory_buffer.items()},
            'step_count': self.step_count
        }
        
        return {
            'base_optimizer': base_state,
            'fer_state': fer_state,
            'param_groups': self.param_groups
        }
    
    def load_state_dict(self, state_dict):
        """Load the optimizer state"""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        
        fer_state = state_dict['fer_state']
        self.step_count = fer_state['step_count']
        
        # Restore memory buffer
        self.memory_buffer = {}
        for k, v in fer_state['memory_buffer'].items():
            self.memory_buffer[k] = {
                'past_grads': v['past_grads'],
                'past_outputs': v['past_outputs'],
                'current_idx': v['current_idx'],
                'filled': v['filled']
            }
        
        self.param_groups = state_dict['param_groups']
    
    def add_param_group(self, param_group):
        """Add a parameter group to the optimizer's param_groups"""
        # Add to main optimizer
        super().add_param_group(param_group)
        
        # Add parameters to base optimizer (only if it exists)
        if hasattr(self, 'base_optimizer'):
            self.base_optimizer.add_param_group({'params': param_group['params']})


def create_fer_optimizer(
    model_params,
    base_optimizer_class=torch.optim.SGD,
    lr: float = 0.01,
    fer_weight: float = 0.1,
    consistency_threshold: float = 0.9,
    memory_size: int = 1000,
    **base_optimizer_kwargs
) -> FEROptimizer:
    """
    Convenience function to create FER optimizer
    
    Args:
        model_params: Model parameters to optimize
        base_optimizer_class: Base optimizer class
        lr: Learning rate
        fer_weight: Weight for FER regularization
        consistency_threshold: Threshold for consistency
        memory_size: Size of memory buffer
        **base_optimizer_kwargs: Additional arguments for base optimizer
    
    Returns:
        FEROptimizer instance
    """
    base_kwargs = {'lr': lr, **base_optimizer_kwargs}
    
    return FEROptimizer(
        model_params,
        base_optimizer=base_optimizer_class,
        base_optimizer_kwargs=base_kwargs,
        fer_weight=fer_weight,
        consistency_threshold=consistency_threshold,
        memory_size=memory_size
    )