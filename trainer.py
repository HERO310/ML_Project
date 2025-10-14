"""
Training pipeline for ICE-LoRA
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import json
from typing import Dict, Optional
import numpy as np

class ICELoRATrainer:
    """Trainer for ICE-LoRA model"""
    
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        config,
        output_dir: str = "./outputs"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup device
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.model.model.to(self.device)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Setup scheduler
        total_steps = len(self.train_loader) * config.training.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.training_history = []
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.training.num_epochs} epochs")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Evaluation samples: {len(self.eval_dataset)}")
        print(f"Device: {self.device}")
        
        for epoch in range(self.config.training.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
            print(f"{'='*50}")
            
            # Training phase
            train_metrics = self._train_epoch(epoch)
            
            # Evaluation phase
            eval_metrics = self._eval_epoch(epoch)
            
            # Log metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'eval': eval_metrics
            }
            self.training_history.append(epoch_metrics)
            
            # Save best model
            if eval_metrics['loss'] < self.best_eval_loss:
                self.best_eval_loss = eval_metrics['loss']
                self._save_checkpoint('best_model')
                print(f"✓ New best model saved (eval loss: {self.best_eval_loss:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % 1 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}')
        
        # Save final model
        self._save_checkpoint('final_model')
        
        # Save training history
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n{'='*50}")
        print("Training completed!")
        print(f"Best eval loss: {self.best_eval_loss:.4f}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*50}")
    
    def _train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.model.train()
        
        total_loss = 0.0
        total_ft_loss = 0.0
        total_ice_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                context_input_ids=batch['context_input_ids'],
                context_attention_mask=batch['context_attention_mask'],
                query_input_ids=batch['query_input_ids'],
                query_attention_mask=batch['query_attention_mask'],
                target_input_ids=batch['target_input_ids'],
                target_attention_mask=batch['target_attention_mask'],
                compute_ice_loss=True
            )
            
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(),
                self.config.training.max_grad_norm
            )
            
            # Optimizer step
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item()
            total_ft_loss += outputs['ft_loss'].item()
            total_ice_loss += outputs['ice_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'ft_loss': outputs['ft_loss'].item(),
                'ice_loss': outputs['ice_loss'].item(),
                'lr': self.scheduler.get_last_lr()[0]
            })
        
        return {
            'loss': total_loss / num_batches,
            'ft_loss': total_ft_loss / num_batches,
            'ice_loss': total_ice_loss / num_batches
        }
    
    def _eval_epoch(self, epoch: int) -> Dict:
        """Evaluate for one epoch"""
        self.model.model.eval()
        
        total_loss = 0.0
        total_ft_loss = 0.0
        total_ice_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.eval_loader, desc=f"Evaluating Epoch {epoch + 1}")
            
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    context_input_ids=batch['context_input_ids'],
                    context_attention_mask=batch['context_attention_mask'],
                    query_input_ids=batch['query_input_ids'],
                    query_attention_mask=batch['query_attention_mask'],
                    target_input_ids=batch['target_input_ids'],
                    target_attention_mask=batch['target_attention_mask'],
                    compute_ice_loss=True
                )
                
                # Update metrics
                total_loss += outputs['loss'].item()
                total_ft_loss += outputs['ft_loss'].item()
                total_ice_loss += outputs['ice_loss'].item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': outputs['loss'].item()
                })
        
        return {
            'loss': total_loss / num_batches,
            'ft_loss': total_ft_loss / num_batches,
            'ice_loss': total_ice_loss / num_batches
        }
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        save_path = os.path.join(self.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save LoRA weights
        self.model.save_model(save_path)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config.to_dict()
        }
        
        with open(os.path.join(save_path, 'trainer_state.json'), 'w') as f:
            json.dump(state, f, indent=2)