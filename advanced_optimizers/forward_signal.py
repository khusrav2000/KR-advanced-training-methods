"""
Forward Signal Propagation Learning (SigProp)

This module implements SigProp, a learning algorithm that propagates learning signals
and updates neural network parameters via a forward pass only, as an alternative 
to backpropagation. SigProp enables global supervised learning without backward 
connectivity, making it ideal for parallel training and biologically plausible learning.

Based on the approach that eliminates structural and computational constraints
like feedback connectivity, weight transport, or backward pass requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import math


class ForwardSignalModule(nn.Module):
    """
    Forward Signal Propagation Module
    
    Implements the core SigProp mechanism for forward-only learning
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 signal_strength: float = 1.0,
                 local_loss_weight: float = 1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.signal_strength = signal_strength
        self.local_loss_weight = local_loss_weight
        
        # Signal transformation networks
        self.signal_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        # Local prediction head for auxiliary loss
        self.local_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, target_signal: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with signal propagation
        
        Args:
            x: Input tensor
            target_signal: Target signal for learning (optional)
            
        Returns:
            Tuple of (transformed_signal, local_prediction)
        """
        # Transform input to create learning signal
        signal = self.signal_transform(x) * self.signal_strength
        
        # Local prediction for auxiliary supervision
        local_pred = self.local_predictor(x)
        
        return signal, local_pred
    
    def compute_local_loss(self, local_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute local auxiliary loss"""
        if target.dim() > 1:
            target = target.mean(dim=1, keepdim=True)
        
        # Binary cross-entropy for local prediction
        loss = F.binary_cross_entropy(local_pred, target.float())
        return self.local_loss_weight * loss


class SigPropLayer(nn.Module):
    """
    A neural network layer that supports SigProp learning
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 use_bias: bool = True,
                 signal_decay: float = 0.9,
                 learning_signal_dim: Optional[int] = None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.signal_decay = signal_decay
        
        # Main transformation
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * math.sqrt(2.0 / in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if use_bias else None
        
        # Forward signal propagation components
        signal_dim = learning_signal_dim or max(32, min(128, in_features // 4))
        self.signal_module = ForwardSignalModule(
            input_dim=in_features,
            hidden_dim=signal_dim,
            signal_strength=1.0 / math.sqrt(in_features)
        )
        
        # Signal integration for parameter updates
        self.signal_integrator = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        
    def forward(self, x: torch.Tensor, learning_signal: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with signal propagation
        
        Args:
            x: Input tensor
            learning_signal: Incoming learning signal from previous layer
            
        Returns:
            Tuple of (output, outgoing_signal, local_loss)
        """
        batch_size = x.size(0)
        
        # Standard forward pass
        output = F.linear(x, self.weight, self.bias)
        
        # Generate and propagate learning signal
        signal, local_pred = self.signal_module(x, learning_signal)
        
        # Compute local loss for signal-based learning
        # Use output statistics as pseudo-target for unsupervised signal
        pseudo_target = torch.sigmoid(output.mean(dim=1, keepdim=True))
        local_loss = self.signal_module.compute_local_loss(local_pred, pseudo_target)
        
        # Apply signal decay for propagation
        outgoing_signal = signal * self.signal_decay
        
        return output, outgoing_signal, local_loss
    
    def get_signal_gradient(self, learning_signal: torch.Tensor, input_activation: torch.Tensor) -> torch.Tensor:
        """Compute gradient from learning signal for parameter updates"""
        # Transform signal for gradient computation
        signal_grad = torch.outer(learning_signal.mean(dim=0), input_activation.mean(dim=0))
        return signal_grad * self.signal_integrator


class ForwardSignalOptimizer(Optimizer):
    """
    Forward Signal Propagation Optimizer
    
    Implements SigProp learning that updates parameters using forward-propagated
    learning signals instead of backpropagated gradients.
    
    Args:
        params: Iterable of parameters to optimize
        model: The neural network model (must use SigPropLayer)
        lr: Learning rate for signal-based updates (default: 0.01)
        signal_lr: Learning rate for signal propagation (default: 0.001)
        local_loss_weight: Weight for local auxiliary losses (default: 0.1)
        signal_momentum: Momentum for signal updates (default: 0.9)
        use_global_signal: Whether to use global supervision signal (default: True)
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        signal_lr: float = 0.001,
        local_loss_weight: float = 0.1,
        signal_momentum: float = 0.9,
        use_global_signal: bool = True,
        signal_noise_std: float = 0.01,
        **kwargs
    ):
        defaults = dict(
            lr=lr,
            signal_lr=signal_lr,
            local_loss_weight=local_loss_weight,
            signal_momentum=signal_momentum,
            use_global_signal=use_global_signal,
            signal_noise_std=signal_noise_std
        )
        
        super().__init__(params, defaults)
        
        self.model = model
        self.step_count = 0
        
        # Storage for signals and activations
        self.layer_signals = {}
        self.layer_activations = {}
        self.local_losses = []
        
        # Global signal state
        self.global_signal = None
        
    def _register_forward_hooks(self):
        """Register forward hooks to capture intermediate activations and signals"""
        self.hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(module, SigPropLayer):
                    self.layer_activations[name] = input[0].detach()
                    if hasattr(output, '__len__') and len(output) >= 3:
                        # SigPropLayer returns (output, signal, local_loss)
                        self.layer_signals[name] = output[1].detach()
                        if output[2] is not None:
                            self.local_losses.append(output[2])
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, SigPropLayer):
                handle = module.register_forward_hook(make_hook(name))
                self.hooks.append(handle)
    
    def _remove_forward_hooks(self):
        """Remove forward hooks"""
        for hook in getattr(self, 'hooks', []):
            hook.remove()
        self.hooks = []
    
    def _compute_global_signal(self, loss: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        """Compute global learning signal from loss"""
        # Simple gradient-free signal based on loss and output statistics
        batch_size = model_output.size(0)
        output_dim = model_output.view(batch_size, -1).size(1)
        
        # Create signal based on loss magnitude and output distribution
        loss_scalar = loss.item() if loss.dim() == 0 else loss.mean().item()
        loss_signal = torch.tanh(torch.tensor(loss_scalar, device=model_output.device)) * torch.ones(batch_size, 1, device=model_output.device)
        output_stats = torch.std(model_output.view(batch_size, -1), dim=1, keepdim=True)
        
        global_signal = loss_signal * torch.tanh(output_stats)
        
        # Add small amount of noise for exploration
        noise = torch.randn_like(global_signal) * self.defaults['signal_noise_std']
        global_signal = global_signal + noise
        
        return global_signal
    
    def _apply_signal_updates(self, global_signal: torch.Tensor):
        """Apply parameter updates based on propagated signals"""
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # Standard gradient update
                    param.data.add_(param.grad, alpha=-group['lr'])
        
        # Apply signal-based updates to SigPropLayers
        for name, module in self.model.named_modules():
            if isinstance(module, SigPropLayer) and name in self.layer_signals:
                layer_signal = self.layer_signals[name]
                layer_activation = self.layer_activations.get(name)
                
                if layer_activation is not None:
                    # Compute signal-based gradient
                    signal_grad = module.get_signal_gradient(layer_signal, layer_activation)
                    
                    # Apply signal update
                    if hasattr(module, 'weight') and module.weight is not None:
                        signal_update = signal_grad * group['signal_lr']
                        module.weight.data.add_(signal_update)
    
    def step(self, closure=None, model_output=None, loss=None):
        """
        Perform optimization step using forward signal propagation
        
        Args:
            closure: Optional closure for loss computation
            model_output: Model output tensor (required for signal computation)
            loss: Loss tensor (required for global signal)
        """
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Clear previous signals and activations
        self.layer_signals.clear()
        self.layer_activations.clear()
        self.local_losses.clear()
        
        # Register hooks to capture signals
        self._register_forward_hooks()
        
        try:
            # Compute global signal if provided
            if loss is not None and model_output is not None:
                self.global_signal = self._compute_global_signal(loss, model_output)
            
            # Apply signal-based updates
            if self.global_signal is not None:
                self._apply_signal_updates(self.global_signal)
            
            # Apply standard gradient updates if available
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        # Apply momentum if specified
                        param_state = self.state[param]
                        
                        if 'momentum_buffer' not in param_state:
                            param_state['momentum_buffer'] = torch.zeros_like(param.grad)
                        
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['signal_momentum']).add_(param.grad)
                        
                        param.data.add_(buf, alpha=-group['lr'])
        
        finally:
            # Clean up hooks
            self._remove_forward_hooks()
        
        self.step_count += 1
        
        # Return local losses for monitoring
        total_local_loss = sum(self.local_losses) if self.local_losses else torch.tensor(0.0)
        return total_local_loss
    
    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients of all optimized parameters"""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.zero_()
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics about signal propagation"""
        stats = {}
        
        if self.layer_signals:
            for layer_name, signal in self.layer_signals.items():
                stats[f'{layer_name}_signal'] = {
                    'mean': signal.mean().item(),
                    'std': signal.std().item(),
                    'norm': signal.norm().item()
                }
        
        if self.global_signal is not None:
            stats['global_signal'] = {
                'mean': self.global_signal.mean().item(),
                'std': self.global_signal.std().item(),
                'norm': self.global_signal.norm().item()
            }
        
        if self.local_losses:
            local_loss_values = [loss.item() for loss in self.local_losses]
            stats['local_losses'] = {
                'mean': sum(local_loss_values) / len(local_loss_values),
                'total': sum(local_loss_values),
                'count': len(local_loss_values)
            }
        
        return stats


class SigPropNet(nn.Module):
    """
    Example neural network using SigProp layers
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int], 
                 output_dim: int,
                 signal_decay: float = 0.9):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(SigPropLayer(
                prev_dim, 
                hidden_dim,
                signal_decay=signal_decay,
                learning_signal_dim=max(16, hidden_dim // 8)
            ))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(SigPropLayer(prev_dim, output_dim, signal_decay=signal_decay))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SigProp network"""
        return self.layers(x)


def create_sigprop_optimizer(
    model: nn.Module,
    lr: float = 0.01,
    signal_lr: float = 0.001,
    local_loss_weight: float = 0.1,
    signal_momentum: float = 0.9,
    **kwargs
) -> ForwardSignalOptimizer:
    """
    Convenience function to create SigProp optimizer
    
    Args:
        model: Neural network model with SigPropLayers
        lr: Learning rate for parameter updates
        signal_lr: Learning rate for signal propagation
        local_loss_weight: Weight for local auxiliary losses
        signal_momentum: Momentum for signal updates
        **kwargs: Additional optimizer arguments
    
    Returns:
        ForwardSignalOptimizer instance
    """
    return ForwardSignalOptimizer(
        model.parameters(),
        model=model,
        lr=lr,
        signal_lr=signal_lr,
        local_loss_weight=local_loss_weight,
        signal_momentum=signal_momentum,
        **kwargs
    )