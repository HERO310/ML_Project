"""
Configuration file for ICE-LoRA implementation
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "microsoft/DialoGPT-small"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["c_attn", "c_proj"]

@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    lambda_ice: float = 1.0  # Weight for ICE loss
    max_length: int = 128
    device: str = "cuda"
    seed: int = 42

@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = "zsre"  # zsre or counterfact
    data_path: Optional[str] = None
    num_train_samples: int = 100
    num_eval_samples: int = 50
    context_template: str = "Context: {context}\nQuestion: {question}\nAnswer:"
    
@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    edit_success_threshold: float = 0.80
    locality_threshold: float = 0.90
    portability_threshold: float = 0.70
    generalization_threshold: float = 0.75

class ICELoRAConfig:
    """Main configuration class"""
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        data_config: Optional[DataConfig] = None,
        eval_config: Optional[EvaluationConfig] = None
    ):
        self.model = model_config or ModelConfig()
        self.training = training_config or TrainingConfig()
        self.data = data_config or DataConfig()
        self.eval = eval_config or EvaluationConfig()
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "eval": self.eval.__dict__
        }