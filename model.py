"""
ICE-LoRA Model: Combines In-Context Editing loss with LoRA adaptation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Optional, Tuple

class ICELoRAEditor(nn.Module):
    """
    ICE-LoRA Editor combining In-Context Editing loss with LoRA
    
    The ICE loss: L_ICE = KL(p_θ(x|[c,q]) || p_θ(x|q))
    Total loss: L_total = L_FT + λ × L_ICE
    """
    
    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: list = None,
        lambda_ice: float = 1.0
    ):
        super().__init__()
        
        self.model_name = model_name
        self.lambda_ice = lambda_ice
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Configure LoRA
        if target_modules is None:
            target_modules = ["c_attn", "c_proj"]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.base_model, peft_config)
        
        print(f"Model loaded: {model_name}")
        print(f"Trainable parameters: {self.model.print_trainable_parameters()}")
    
    def forward(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        compute_ice_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing both standard FT loss and ICE loss
        
        Args:
            context_input_ids: Input with context [c, q]
            context_attention_mask: Attention mask for context input
            query_input_ids: Query alone [q]
            query_attention_mask: Attention mask for query
            target_input_ids: Target tokens
            target_attention_mask: Target attention mask
            compute_ice_loss: Whether to compute ICE loss
        
        Returns:
            Dictionary with loss components
        """
        
        # Standard fine-tuning loss (cross-entropy with target)
        context_outputs = self.model(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            labels=target_input_ids
        )
        
        ft_loss = context_outputs.loss
        
        total_loss = ft_loss
        ice_loss = torch.tensor(0.0, device=ft_loss.device)
        
        if compute_ice_loss:
            # Get distribution with context p_θ(x|[c,q])
            with torch.no_grad():
                context_logits = context_outputs.logits
            
            # Get distribution without context p_θ(x|q)
            query_outputs = self.model(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask
            )
            query_logits = query_outputs.logits
            
            # Compute ICE loss: KL divergence
            # KL(p_context || p_query)
            ice_loss = self._compute_ice_loss(
                context_logits,
                query_logits,
                target_attention_mask
            )
            
            total_loss = ft_loss + self.lambda_ice * ice_loss
        
        return {
            'loss': total_loss,
            'ft_loss': ft_loss,
            'ice_loss': ice_loss
        }
    
    def _compute_ice_loss(
        self,
        context_logits: torch.Tensor,
        query_logits: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ICE loss: KL(p_θ(x|[c,q]) || p_θ(x|q))
        
        Args:
            context_logits: Logits with context
            query_logits: Logits without context
            attention_mask: Mask for valid positions
        
        Returns:
            ICE loss value
        """
        # Convert logits to probabilities
        context_probs = F.softmax(context_logits, dim=-1)
        query_log_probs = F.log_softmax(query_logits, dim=-1)
        
        # Compute KL divergence
        kl_div = F.kl_div(
            query_log_probs,
            context_probs,
            reduction='none',
            log_target=False
        )
        
        # Sum over vocabulary dimension
        kl_div = kl_div.sum(dim=-1)
        
        # Apply attention mask and average
        mask = attention_mask.float()
        kl_div = (kl_div * mask).sum() / mask.sum()
        
        return kl_div
    
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text given a prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def save_model(self, save_path: str):
        """Save LoRA weights"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load LoRA weights"""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, load_path)
        print(f"Model loaded from {load_path}")