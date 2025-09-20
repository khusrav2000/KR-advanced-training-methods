"""
Selective Localized Learning (LoCal+SGD)

This optimizer implements LoCal+SGD, which accelerates DNN training by selectively 
combining localized or Hebbian learning within a SGD framework. It reduces computation
by using 1 GEMM operation instead of 2 for selected layers and reduces memory footprint.

Based on the approach that achieves up to 1.5× improvement in training time with 
~0.5% loss in Top-1 classification accuracy.
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import math


class HebbianLearningRule:
    """
    Implements Hebbian learning rule for localized parameter updates
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 decay_factor: float = 0.99,
                 normalization: str = 'none'):
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.normalization = normalization
    
    def update_weights(self, 
                      pre_activations: torch.Tensor, 
                      post_activations: torch.Tensor,
                      current_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply Hebbian weight update: Δw = lr * pre * post^T
        
        Args:
            pre_activations: Pre-synaptic activations (input)
            post_activations: Post-synaptic activations (output)
            current_weights: Current weight matrix
            
        Returns:
            Weight update tensor
        """
        batch_size = pre_activations.size(0)
        
        # Compute Hebbian update: outer product of activations
        if pre_activations.dim() == 2 and post_activations.dim() == 2:
            # Standard case: batch_size x input_dim, batch_size x output_dim
            hebbian_update = torch.mm(post_activations.t(), pre_activations) / batch_size
        else:
            # Handle other dimensionalities
            pre_flat = pre_activations.view(batch_size, -1)
            post_flat = post_activations.view(batch_size, -1)
            hebbian_update = torch.mm(post_flat.t(), pre_flat) / batch_size
        
        # Apply normalization
        if self.normalization == 'l2':
            norm = hebbian_update.norm()
            if norm > 1e-8:
                hebbian_update = hebbian_update / norm
        elif self.normalization == 'layer':
            hebbian_update = hebbian_update / (1e-8 + hebbian_update.abs().max())
        
        # Apply learning rate and decay
        weight_update = self.learning_rate * hebbian_update
        
        return weight_update


class LearningModeSelector:
    """
    Manages the selection between localized and SGD learning modes
    """
    
    def __init__(self, 
                 mode: str = 'dynamic',
                 total_layers: int = 10,
                 transition_schedule: Optional[Callable[[int], int]] = None):
        self.mode = mode
        self.total_layers = total_layers
        self.transition_schedule = transition_schedule
        self.step_count = 0
        
        # Dynamic mode state
        self.transition_layer = 0
        self.performance_history = []
        self.loss_threshold = 0.01
        
    def get_transition_layer(self, current_loss: Optional[float] = None) -> int:
        """
        Determine the boundary between localized and SGD layers
        
        Returns:
            Layer index where transition occurs (layers < index use localized learning)
        """
        if self.mode == 'static':
            if self.transition_schedule is not None:
                return self.transition_schedule(self.step_count)
            else:
                # Default static schedule: gradually increase localized layers
                progress = min(1.0, self.step_count / 1000)
                return int(progress * self.total_layers * 0.7)
        
        elif self.mode == 'dynamic':
            return self._dynamic_transition_selection(current_loss)
        
        else:
            raise ValueError(f"Unknown learning mode selection: {self.mode}")
    
    def _dynamic_transition_selection(self, current_loss: Optional[float]) -> int:
        """Dynamic transition layer selection based on loss dynamics"""
        
        if current_loss is not None:
            self.performance_history.append(current_loss)
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-50:]
        
        # Conservative approach: start with more SGD layers
        if len(self.performance_history) < 10:
            return max(0, self.total_layers - 3)
        
        # Analyze loss trend
        recent_losses = self.performance_history[-10:]
        loss_trend = recent_losses[-1] - recent_losses[0] if len(recent_losses) >= 2 else 0
        
        # If loss is decreasing well, can use more localized layers
        if loss_trend < -self.loss_threshold:
            self.transition_layer = min(self.total_layers - 1, self.transition_layer + 1)
        elif loss_trend > self.loss_threshold:
            # If loss is increasing, use fewer localized layers
            self.transition_layer = max(0, self.transition_layer - 1)
        
        return self.transition_layer
    
    def step(self):
        """Update internal step counter"""
        self.step_count += 1


class LocalizedLearningOptimizer(Optimizer):
    """
    LoCal+SGD Optimizer
    
    Selectively applies localized (Hebbian) learning to some layers while using
    SGD for others, based on a learning mode selection algorithm.
    
    Args:
        params: Iterable of parameters to optimize
        model: The neural network model
        base_optimizer: Base optimizer class for SGD layers (default: SGD)
        base_optimizer_kwargs: Keyword arguments for base optimizer
        hebbian_lr: Learning rate for Hebbian updates (default: 0.001)
        selection_mode: Mode for transition layer selection ('static' or 'dynamic')
        weak_supervision_weight: Weight for weak supervision signal (default: 0.1)
        transition_schedule: Custom schedule for static mode
        memory_efficient: Whether to use memory-efficient mode (default: True)
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        base_optimizer: type = torch.optim.SGD,
        base_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        hebbian_lr: float = 0.001,
        selection_mode: str = 'dynamic',
        weak_supervision_weight: float = 0.1,
        transition_schedule: Optional[Callable[[int], int]] = None,
        memory_efficient: bool = True,
        hebbian_decay: float = 0.99,
        normalization: str = 'layer',
        **kwargs
    ):
        if base_optimizer_kwargs is None:
            base_optimizer_kwargs = {'lr': 0.01, 'momentum': 0.9}
        
        defaults = dict(
            hebbian_lr=hebbian_lr,
            selection_mode=selection_mode,
            weak_supervision_weight=weak_supervision_weight,
            memory_efficient=memory_efficient,
            hebbian_decay=hebbian_decay,
            normalization=normalization,
            **base_optimizer_kwargs
        )
        
        super().__init__(params, defaults)
        
        self.model = model
        self.memory_efficient = memory_efficient
        
        # Initialize base optimizer for SGD layers
        param_groups_copy = []
        for group in self.param_groups:
            group_copy = {k: v for k, v in group.items() 
                         if k not in ['hebbian_lr', 'selection_mode', 'weak_supervision_weight', 
                                    'memory_efficient', 'hebbian_decay', 'normalization']}
            group_copy['params'] = group['params']
            param_groups_copy.append(group_copy)
        
        self.base_optimizer = base_optimizer(param_groups_copy, **base_optimizer_kwargs)
        
        # Get model layers for mode selection
        self.model_layers = list(self.model.modules())
        self.total_layers = len([m for m in self.model_layers if isinstance(m, (nn.Linear, nn.Conv2d))])
        
        # Initialize learning mode selector
        self.mode_selector = LearningModeSelector(
            mode=selection_mode,
            total_layers=self.total_layers,
            transition_schedule=transition_schedule
        )
        
        # Initialize Hebbian learning rule
        self.hebbian_rule = HebbianLearningRule(
            learning_rate=hebbian_lr,
            decay_factor=hebbian_decay,
            normalization=normalization
        )
        
        # State for activation storage
        self.layer_activations = {}
        self.layer_outputs = {}
        self.hooks = []
        self.current_loss = None
        
        # Weak supervision state
        self.global_loss_signal = None
        
    def _register_hooks(self):
        """Register forward hooks to capture activations"""
        self.hooks = []
        
        def make_hook(name, layer_idx):
            def hook(module, input, output):
                if self.memory_efficient:
                    # Only store activations for localized layers
                    transition_layer = self.mode_selector.get_transition_layer(self.current_loss)
                    if layer_idx < transition_layer:
                        self.layer_activations[name] = input[0].detach().clone()
                        self.layer_outputs[name] = output.detach().clone()
                else:
                    # Store all activations
                    self.layer_activations[name] = input[0].detach().clone()
                    self.layer_outputs[name] = output.detach().clone()
            return hook
        
        layer_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handle = module.register_forward_hook(make_hook(name, layer_idx))
                self.hooks.append(handle)
                layer_idx += 1
    
    def _remove_hooks(self):
        """Remove forward hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _apply_hebbian_update(self, module: nn.Module, name: str) -> bool:
        """
        Apply Hebbian learning update to a module
        
        Returns:
            True if update was applied, False otherwise
        """
        if name not in self.layer_activations or name not in self.layer_outputs:
            return False
        
        pre_activations = self.layer_activations[name]
        post_activations = self.layer_outputs[name]
        
        if isinstance(module, nn.Linear):
            # Linear layer update
            weight_update = self.hebbian_rule.update_weights(
                pre_activations, post_activations, module.weight
            )
            
            # Apply weak supervision if available
            if self.global_loss_signal is not None:
                supervision_factor = self.defaults['weak_supervision_weight'] * self.global_loss_signal
                weight_update = weight_update * (1.0 + supervision_factor)
            
            # Update weights
            module.weight.data.add_(weight_update)
            
            # Update bias with simplified rule
            if module.bias is not None:
                bias_update = self.hebbian_rule.learning_rate * post_activations.mean(dim=0)
                if self.global_loss_signal is not None:
                    bias_update = bias_update * (1.0 + supervision_factor)
                module.bias.data.add_(bias_update)
            
            return True
            
        elif isinstance(module, nn.Conv2d):
            # Convolutional layer update (simplified)
            batch_size = pre_activations.size(0)
            
            # Reshape for convolution
            if pre_activations.dim() == 4:  # B, C, H, W
                pre_flat = pre_activations.view(batch_size, -1)
                post_flat = post_activations.view(batch_size, -1)
                
                # Simplified Hebbian update for conv layers
                weight_flat = module.weight.view(-1)
                update_scale = torch.dot(pre_flat.mean(dim=0), post_flat.mean(dim=0)) / pre_flat.size(1)
                
                weight_update = self.hebbian_rule.learning_rate * update_scale
                
                if self.global_loss_signal is not None:
                    supervision_factor = self.defaults['weak_supervision_weight'] * self.global_loss_signal
                    weight_update = weight_update * (1.0 + supervision_factor)
                
                module.weight.data.add_(weight_update)
                
                if module.bias is not None:
                    bias_update = self.hebbian_rule.learning_rate * post_activations.mean()
                    if self.global_loss_signal is not None:
                        bias_update = bias_update * (1.0 + supervision_factor)
                    module.bias.data.add_(bias_update)
                
                return True
        
        return False
    
    def step(self, closure=None, loss=None):
        """
        Perform optimization step with selective localized learning
        
        Args:
            closure: Optional closure for loss computation
            loss: Current loss value for weak supervision and mode selection
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Update current loss for mode selection
        if loss is not None:
            self.current_loss = loss.item() if torch.is_tensor(loss) else loss
            
            # Compute weak supervision signal
            self.global_loss_signal = torch.tanh(torch.tensor(self.current_loss)).item()
        
        # Register hooks to capture activations
        self._register_hooks()
        
        try:
            # Determine transition layer
            transition_layer = self.mode_selector.get_transition_layer(self.current_loss)
            
            # Apply updates based on layer assignment
            layer_idx = 0
            localized_count = 0
            sgd_count = 0
            
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if layer_idx < transition_layer:
                        # Apply localized (Hebbian) learning
                        success = self._apply_hebbian_update(module, name)
                        if success:
                            localized_count += 1
                            
                            # Zero out gradients for localized layers to prevent SGD update
                            if module.weight.grad is not None:
                                module.weight.grad.zero_()
                            if module.bias is not None and module.bias.grad is not None:
                                module.bias.grad.zero_()
                    else:
                        # Will be updated by SGD
                        sgd_count += 1
                    
                    layer_idx += 1
            
            # Apply SGD updates to remaining layers
            if sgd_count > 0:
                self.base_optimizer.step()
            
        finally:
            # Clean up hooks and activations
            self._remove_hooks()
            self.layer_activations.clear()
            self.layer_outputs.clear()
        
        # Update mode selector
        self.mode_selector.step()
        
        return {
            'transition_layer': transition_layer,
            'localized_layers': localized_count,
            'sgd_layers': sgd_count,
            'current_loss': self.current_loss
        }
    
    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients of all optimized parameters"""
        self.base_optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        """Return the state of the optimizer as a dict"""
        base_state = self.base_optimizer.state_dict()
        
        local_state = {
            'mode_selector': {
                'step_count': self.mode_selector.step_count,
                'transition_layer': self.mode_selector.transition_layer,
                'performance_history': self.mode_selector.performance_history[-50:],  # Keep recent history
            },
            'hebbian_rule': {
                'learning_rate': self.hebbian_rule.learning_rate,
                'decay_factor': self.hebbian_rule.decay_factor,
            },
            'current_loss': self.current_loss,
        }
        
        return {
            'base_optimizer': base_state,
            'local_state': local_state,
            'param_groups': self.param_groups
        }
    
    def load_state_dict(self, state_dict):
        """Load the optimizer state"""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        
        local_state = state_dict['local_state']
        
        # Restore mode selector state
        self.mode_selector.step_count = local_state['mode_selector']['step_count']
        self.mode_selector.transition_layer = local_state['mode_selector']['transition_layer']
        self.mode_selector.performance_history = local_state['mode_selector']['performance_history']
        
        # Restore Hebbian rule state
        self.hebbian_rule.learning_rate = local_state['hebbian_rule']['learning_rate']
        self.hebbian_rule.decay_factor = local_state['hebbian_rule']['decay_factor']
        
        self.current_loss = local_state['current_loss']
        self.param_groups = state_dict['param_groups']
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about learning mode usage"""
        return {
            'transition_layer': self.mode_selector.transition_layer,
            'total_layers': self.total_layers,
            'localized_ratio': self.mode_selector.transition_layer / self.total_layers if self.total_layers > 0 else 0,
            'current_loss': self.current_loss,
            'step_count': self.mode_selector.step_count,
            'performance_history_length': len(self.mode_selector.performance_history)
        }


def create_localized_optimizer(
    model: nn.Module,
    base_optimizer_class=torch.optim.SGD,
    lr: float = 0.01,
    hebbian_lr: float = 0.001,
    momentum: float = 0.9,
    selection_mode: str = 'dynamic',
    weak_supervision_weight: float = 0.1,
    memory_efficient: bool = True,
    **base_optimizer_kwargs
) -> LocalizedLearningOptimizer:
    """
    Convenience function to create LoCal+SGD optimizer
    
    Args:
        model: Neural network model
        base_optimizer_class: Base optimizer class for SGD layers
        lr: Learning rate for SGD layers
        hebbian_lr: Learning rate for Hebbian updates
        momentum: Momentum for SGD
        selection_mode: Learning mode selection strategy
        weak_supervision_weight: Weight for weak supervision
        memory_efficient: Whether to use memory-efficient mode
        **base_optimizer_kwargs: Additional arguments for base optimizer
    
    Returns:
        LocalizedLearningOptimizer instance
    """
    base_kwargs = {'lr': lr, 'momentum': momentum, **base_optimizer_kwargs}
    
    return LocalizedLearningOptimizer(
        model.parameters(),
        model=model,
        base_optimizer=base_optimizer_class,
        base_optimizer_kwargs=base_kwargs,
        hebbian_lr=hebbian_lr,
        selection_mode=selection_mode,
        weak_supervision_weight=weak_supervision_weight,
        memory_efficient=memory_efficient
    )


def linear_transition_schedule(max_layers: int, warmup_steps: int = 1000) -> Callable[[int], int]:
    """Create linear transition schedule for static mode"""
    def schedule(step: int) -> int:
        progress = min(1.0, step / warmup_steps)
        return int(progress * max_layers * 0.8)  # Use up to 80% localized layers
    return schedule


def cosine_transition_schedule(max_layers: int, period: int = 2000) -> Callable[[int], int]:
    """Create cosine transition schedule for static mode"""
    def schedule(step: int) -> int:
        cos_factor = 0.5 * (1 + math.cos(math.pi * (step % period) / period))
        return int(cos_factor * max_layers * 0.6)  # Vary between 0 and 60% localized layers
    return schedule