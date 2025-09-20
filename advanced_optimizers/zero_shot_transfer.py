"""
Zero-Shot Hyperparameter Transfer (μTransfer)

This module implements μTransfer, which enables hyperparameter tuning on smaller models
and zero-shot transfer to larger models using Maximal Update Parametrization (μP).
This approach allows tuning on small models and transferring HPs to large models
without direct tuning, reducing computational cost significantly.

Based on the approach that achieves better performance than published baselines
with tuning cost equivalent to only a fraction of total pretraining cost.
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import math
import copy


class MuParametrization:
    """
    Implements Maximal Update Parametrization (μP) scaling rules
    
    μP ensures that optimal hyperparameters remain stable as model size changes
    by applying specific scaling rules to different components of the model.
    """
    
    def __init__(self, 
                 base_width: int = 64,
                 target_width: int = 512,
                 base_depth: int = 4,
                 target_depth: int = 12):
        self.base_width = base_width
        self.target_width = target_width
        self.base_depth = base_depth
        self.target_depth = target_depth
        
        # Compute scaling factors
        self.width_scaling = target_width / base_width
        self.depth_scaling = target_depth / base_depth
        
    def get_lr_scaling(self, layer_type: str, is_output_layer: bool = False) -> float:
        """
        Get learning rate scaling factor for different layer types
        
        Args:
            layer_type: Type of layer ('embedding', 'hidden', 'output', 'attention')
            is_output_layer: Whether this is the final output layer
            
        Returns:
            Learning rate scaling factor
        """
        if layer_type == 'embedding':
            # Embedding layers scale with 1/√width
            return 1.0 / math.sqrt(self.width_scaling)
        elif layer_type == 'output' or is_output_layer:
            # Output layers scale with 1/width
            return 1.0 / self.width_scaling
        elif layer_type == 'hidden':
            # Hidden layers keep same learning rate
            return 1.0
        elif layer_type == 'attention':
            # Attention layers scale with 1/√width
            return 1.0 / math.sqrt(self.width_scaling)
        else:
            # Default: no scaling
            return 1.0
    
    def get_init_scaling(self, layer_type: str, is_output_layer: bool = False) -> float:
        """
        Get initialization scaling factor for different layer types
        
        Args:
            layer_type: Type of layer
            is_output_layer: Whether this is the final output layer
            
        Returns:
            Initialization scaling factor
        """
        if layer_type == 'embedding':
            # Embedding layers use standard initialization
            return 1.0
        elif layer_type == 'output' or is_output_layer:
            # Output layers scale with 1/√width
            return 1.0 / math.sqrt(self.width_scaling)
        elif layer_type == 'hidden':
            # Hidden layers scale with 1/√width
            return 1.0 / math.sqrt(self.width_scaling)
        elif layer_type == 'attention':
            # Attention layers use standard initialization
            return 1.0
        else:
            return 1.0
    
    def apply_parametrization(self, model: nn.Module, layer_mapping: Dict[str, str]):
        """
        Apply μP parametrization to a model
        
        Args:
            model: The neural network model
            layer_mapping: Mapping from layer names to layer types
        """
        for name, module in model.named_modules():
            if name in layer_mapping:
                layer_type = layer_mapping[name]
                is_output = 'output' in name.lower() or layer_type == 'output'
                
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Apply initialization scaling
                    init_scale = self.get_init_scaling(layer_type, is_output)
                    with torch.no_grad():
                        module.weight.data *= init_scale
                
                elif isinstance(module, nn.Embedding):
                    # Embedding specific initialization
                    init_scale = self.get_init_scaling('embedding')
                    with torch.no_grad():
                        module.weight.data *= init_scale


class HyperparameterTransfer:
    """
    Manages hyperparameter transfer from base model to target model
    """
    
    def __init__(self):
        self.base_hyperparams = {}
        self.transfer_rules = {}
        
    def register_base_hyperparams(self, 
                                  lr: float,
                                  batch_size: int,
                                  weight_decay: float,
                                  optimizer_kwargs: Dict[str, Any],
                                  model_config: Dict[str, Any]):
        """Register hyperparameters from base model training"""
        self.base_hyperparams = {
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'optimizer_kwargs': optimizer_kwargs,
            'model_config': model_config
        }
    
    def compute_transferred_hyperparams(self, 
                                       target_model_config: Dict[str, Any],
                                       mu_param: MuParametrization) -> Dict[str, Any]:
        """
        Compute transferred hyperparameters for target model
        
        Args:
            target_model_config: Configuration of target model
            mu_param: μP parametrization object
            
        Returns:
            Transferred hyperparameters
        """
        if not self.base_hyperparams:
            raise ValueError("Base hyperparameters not registered")
        
        base_lr = self.base_hyperparams['lr']
        base_batch_size = self.base_hyperparams['batch_size']
        
        # Learning rate scaling based on μP rules
        # For most layers, LR stays the same in μP
        transferred_lr = base_lr
        
        # Batch size scaling (optional, often kept same)
        transferred_batch_size = base_batch_size
        
        # Weight decay typically stays the same
        transferred_weight_decay = self.base_hyperparams['weight_decay']
        
        # Copy optimizer kwargs
        transferred_optimizer_kwargs = copy.deepcopy(self.base_hyperparams['optimizer_kwargs'])
        
        return {
            'lr': transferred_lr,
            'batch_size': transferred_batch_size,
            'weight_decay': transferred_weight_decay,
            'optimizer_kwargs': transferred_optimizer_kwargs,
            'mu_scaling': {
                'width_scaling': mu_param.width_scaling,
                'depth_scaling': mu_param.depth_scaling
            }
        }


class ZeroShotTransferOptimizer(Optimizer):
    """
    Zero-Shot Hyperparameter Transfer Optimizer
    
    This optimizer implements μTransfer by applying different learning rates
    to different layer types according to μP scaling rules, enabling
    zero-shot transfer of hyperparameters from smaller to larger models.
    
    Args:
        params: Iterable of parameters to optimize
        model: The neural network model
        base_optimizer: Base optimizer class
        base_optimizer_kwargs: Base optimizer arguments
        mu_parametrization: μP configuration
        layer_mapping: Mapping from layer names to types
        hyperparameter_transfer: HP transfer configuration
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        base_optimizer: type = torch.optim.AdamW,
        base_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        mu_parametrization: Optional[MuParametrization] = None,
        layer_mapping: Optional[Dict[str, str]] = None,
        hyperparameter_transfer: Optional[HyperparameterTransfer] = None,
        auto_detect_layers: bool = True,
        **kwargs
    ):
        if base_optimizer_kwargs is None:
            base_optimizer_kwargs = {'lr': 1e-3, 'weight_decay': 0.01}
        
        self.model = model
        self.mu_param = mu_parametrization
        self.hp_transfer = hyperparameter_transfer
        self.layer_mapping = layer_mapping or {}
        
        # Auto-detect layer types if not provided
        if auto_detect_layers and not self.layer_mapping:
            self.layer_mapping = self._auto_detect_layer_types()
        
        # Apply μP parametrization to model
        if self.mu_param is not None:
            self.mu_param.apply_parametrization(model, self.layer_mapping)
        
        # Create parameter groups with different learning rates
        param_groups = self._create_parameter_groups(base_optimizer_kwargs)
        
        defaults = dict(**base_optimizer_kwargs)
        super().__init__(param_groups, defaults)
        
        # Initialize base optimizers for each group
        self.base_optimizers = []
        for group in self.param_groups:
            group_params = [group]  # Single group for each optimizer
            optimizer = base_optimizer([group], **base_optimizer_kwargs)
            self.base_optimizers.append(optimizer)
        
        self.step_count = 0
        
    def _auto_detect_layer_types(self) -> Dict[str, str]:
        """Automatically detect layer types from model structure"""
        layer_mapping = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                layer_mapping[name] = 'embedding'
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                # Detect output layers
                if any(keyword in name.lower() for keyword in ['output', 'classifier', 'head', 'final']):
                    layer_mapping[name] = 'output'
                # Detect attention layers
                elif any(keyword in name.lower() for keyword in ['attention', 'attn', 'self_attn']):
                    layer_mapping[name] = 'attention'
                else:
                    layer_mapping[name] = 'hidden'
        
        return layer_mapping
    
    def _create_parameter_groups(self, base_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create parameter groups with layer-specific learning rates"""
        
        # Track which parameters have already been assigned
        assigned_params = set()
        
        # Group parameters by layer type
        param_groups_by_type = {
            'embedding': [],
            'hidden': [],
            'output': [],
            'attention': [],
            'other': []
        }
        
        # First pass: assign parameters from mapped modules
        for name, module in self.model.named_modules():
            if name in self.layer_mapping:
                layer_type = self.layer_mapping[name]
                
                for param_name, param in module.named_parameters(recurse=False):  # Don't recurse
                    if param.requires_grad and param not in assigned_params:
                        param_groups_by_type[layer_type].append(param)
                        assigned_params.add(param)
        
        # Second pass: assign remaining unassigned parameters to 'other'
        for param in self.model.parameters():
            if param.requires_grad and param not in assigned_params:
                param_groups_by_type['other'].append(param)
                assigned_params.add(param)
        
        # Create parameter groups with scaled learning rates
        param_groups = []
        base_lr = base_kwargs.get('lr', 1e-3)
        
        for layer_type, params in param_groups_by_type.items():
            if params:  # Only create group if there are parameters
                # Get learning rate scaling
                if self.mu_param is not None:
                    is_output = layer_type == 'output'
                    lr_scaling = self.mu_param.get_lr_scaling(layer_type, is_output)
                else:
                    lr_scaling = 1.0
                
                group = {
                    'params': params,
                    'lr': base_lr * lr_scaling,
                    'layer_type': layer_type,
                    'lr_scaling': lr_scaling,
                    **{k: v for k, v in base_kwargs.items() if k != 'lr'}
                }
                param_groups.append(group)
        
        return param_groups
    
    def step(self, closure=None):
        """Perform optimization step with layer-specific learning rates"""
        loss = None
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Step each base optimizer
        for optimizer in self.base_optimizers:
            optimizer.step()
        
        self.step_count += 1
        return loss
    
    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients of all optimized parameters"""
        for optimizer in self.base_optimizers:
            optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        """Return the state of the optimizer as a dict"""
        optimizer_states = []
        for optimizer in self.base_optimizers:
            optimizer_states.append(optimizer.state_dict())
        
        return {
            'base_optimizers': optimizer_states,
            'param_groups': self.param_groups,
            'layer_mapping': self.layer_mapping,
            'step_count': self.step_count,
            'mu_param_config': {
                'base_width': self.mu_param.base_width if self.mu_param else None,
                'target_width': self.mu_param.target_width if self.mu_param else None,
                'base_depth': self.mu_param.base_depth if self.mu_param else None,
                'target_depth': self.mu_param.target_depth if self.mu_param else None,
            } if self.mu_param else None
        }
    
    def load_state_dict(self, state_dict):
        """Load the optimizer state"""
        optimizer_states = state_dict['base_optimizers']
        
        for optimizer, state in zip(self.base_optimizers, optimizer_states):
            optimizer.load_state_dict(state)
        
        self.param_groups = state_dict['param_groups']
        self.layer_mapping = state_dict['layer_mapping']
        self.step_count = state_dict['step_count']
    
    def get_layer_statistics(self) -> Dict[str, Any]:
        """Get statistics about layer-wise learning rates"""
        stats = {}
        
        for i, group in enumerate(self.param_groups):
            layer_type = group.get('layer_type', f'group_{i}')
            stats[layer_type] = {
                'lr': group['lr'],
                'lr_scaling': group.get('lr_scaling', 1.0),
                'num_params': sum(p.numel() for p in group['params']),
                'param_groups_idx': i
            }
        
        if self.mu_param:
            stats['mu_parametrization'] = {
                'width_scaling': self.mu_param.width_scaling,
                'depth_scaling': self.mu_param.depth_scaling,
                'base_width': self.mu_param.base_width,
                'target_width': self.mu_param.target_width
            }
        
        return stats


def create_mu_transfer_optimizer(
    model: nn.Module,
    base_model_config: Dict[str, int],
    target_model_config: Dict[str, int],
    base_hyperparams: Dict[str, Any],
    base_optimizer_class=torch.optim.AdamW,
    auto_detect_layers: bool = True,
    layer_mapping: Optional[Dict[str, str]] = None
) -> ZeroShotTransferOptimizer:
    """
    Create μTransfer optimizer with automatic hyperparameter scaling
    
    Args:
        model: Target model to optimize
        base_model_config: Base model configuration {'width': int, 'depth': int}
        target_model_config: Target model configuration {'width': int, 'depth': int}
        base_hyperparams: Hyperparameters tuned on base model
        base_optimizer_class: Base optimizer class
        auto_detect_layers: Whether to auto-detect layer types
        layer_mapping: Manual layer type mapping
    
    Returns:
        ZeroShotTransferOptimizer instance
    """
    
    # Create μP parametrization
    mu_param = MuParametrization(
        base_width=base_model_config['width'],
        target_width=target_model_config['width'],
        base_depth=base_model_config['depth'],
        target_depth=target_model_config['depth']
    )
    
    # Create hyperparameter transfer
    hp_transfer = HyperparameterTransfer()
    hp_transfer.register_base_hyperparams(
        lr=base_hyperparams['lr'],
        batch_size=base_hyperparams.get('batch_size', 32),
        weight_decay=base_hyperparams.get('weight_decay', 0.01),
        optimizer_kwargs={k: v for k, v in base_hyperparams.items() 
                         if k not in ['lr', 'batch_size', 'weight_decay']},
        model_config=base_model_config
    )
    
    # Compute transferred hyperparams
    transferred_hyperparams = hp_transfer.compute_transferred_hyperparams(
        target_model_config, mu_param
    )
    
    # Create optimizer
    return ZeroShotTransferOptimizer(
        model.parameters(),
        model=model,
        base_optimizer=base_optimizer_class,
        base_optimizer_kwargs={
            'lr': transferred_hyperparams['lr'],
            'weight_decay': transferred_hyperparams['weight_decay'],
            **transferred_hyperparams['optimizer_kwargs']
        },
        mu_parametrization=mu_param,
        layer_mapping=layer_mapping,
        hyperparameter_transfer=hp_transfer,
        auto_detect_layers=auto_detect_layers
    )


class MuTransferTrainer:
    """
    Training framework for μTransfer methodology
    """
    
    def __init__(self):
        self.base_results = {}
        self.transfer_results = {}
    
    def tune_base_model(self, 
                       base_model: nn.Module,
                       train_loader,
                       val_loader,
                       hyperparameter_grid: Dict[str, List[Any]],
                       num_epochs: int = 10) -> Dict[str, Any]:
        """
        Tune hyperparameters on base model
        
        Args:
            base_model: Small model for hyperparameter tuning
            train_loader: Training data loader
            val_loader: Validation data loader
            hyperparameter_grid: Grid of hyperparameters to search
            num_epochs: Number of training epochs
            
        Returns:
            Best hyperparameters found
        """
        best_hyperparams = None
        best_score = float('-inf')
        
        # Simple grid search (can be replaced with more sophisticated methods)
        import itertools
        
        param_combinations = list(itertools.product(*hyperparameter_grid.values()))
        param_names = list(hyperparameter_grid.keys())
        
        for combination in param_combinations:
            hyperparams = dict(zip(param_names, combination))
            
            # Create optimizer with current hyperparams
            optimizer = torch.optim.AdamW(base_model.parameters(), **hyperparams)
            
            # Train for a few epochs
            score = self._train_and_evaluate(base_model, optimizer, train_loader, val_loader, num_epochs)
            
            if score > best_score:
                best_score = score
                best_hyperparams = hyperparams
        
        self.base_results = {
            'best_hyperparams': best_hyperparams,
            'best_score': best_score
        }
        
        return best_hyperparams
    
    def _train_and_evaluate(self, model, optimizer, train_loader, val_loader, num_epochs):
        """Simple training and evaluation loop"""
        model.train()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx >= 10:  # Quick training for HP search
                    break
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / total
    
    def transfer_to_large_model(self,
                               large_model: nn.Module,
                               base_model_config: Dict[str, int],
                               target_model_config: Dict[str, int]) -> ZeroShotTransferOptimizer:
        """
        Create optimizer for large model using transferred hyperparameters
        
        Args:
            large_model: Large target model
            base_model_config: Configuration of base model used for tuning
            target_model_config: Configuration of target model
            
        Returns:
            Optimizer with transferred hyperparameters
        """
        if not self.base_results:
            raise ValueError("Must tune base model first")
        
        best_hyperparams = self.base_results['best_hyperparams']
        
        optimizer = create_mu_transfer_optimizer(
            large_model,
            base_model_config,
            target_model_config,
            best_hyperparams
        )
        
        return optimizer