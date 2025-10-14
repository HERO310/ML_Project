"""
Main training script for ICE-LoRA
"""
import os
import json
import argparse
import random
import numpy as np
import torch

from config import ICELoRAConfig, ModelConfig, TrainingConfig, DataConfig
from dataset import KnowledgeEditDataset, create_sample_dataset
from model import ICELoRAEditor
from trainer import ICELoRATrainer
from evaluator import ICELoRAEvaluator


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    """Main function"""
    
    print("="*60)
    print("ICE-LoRA: Efficient Knowledge Editing")
    print("In-Context Editing with Low-Rank Adaptation")
    print("="*60)
    print()
    
    # Set seed
    set_seed(args.seed)
    
    # Create configuration
    config = ICELoRAConfig(
        model_config=ModelConfig(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        ),
        training_config=TrainingConfig(
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lambda_ice=args.lambda_ice,
            device=args.device,
            seed=args.seed
        ),
        data_config=DataConfig(
            dataset_name=args.dataset,
            data_path=args.data_path,
            num_train_samples=args.num_train_samples,
            num_eval_samples=args.num_eval_samples
        )
    )
    
    print("Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    print()
    
    # Create sample dataset if needed
    if args.create_sample_data:
        print("Creating sample dataset...")
        os.makedirs("./data", exist_ok=True)
        data_path = f"./data/{args.dataset}_sample.json"
        create_sample_dataset(
            data_path,
            dataset_type=args.dataset,
            num_samples=args.num_train_samples + args.num_eval_samples
        )
        config.data.data_path = data_path
        print()
    
    # Initialize model
    print("Initializing ICE-LoRA model...")
    model = ICELoRAEditor(
        model_name=config.model.model_name,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        lambda_ice=config.training.lambda_ice
    )
    print()
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if config.data.data_path is None:
        raise ValueError("Please provide --data_path or use --create_sample_data")
    
    full_dataset = KnowledgeEditDataset(
        data_path=config.data.data_path,
        tokenizer=model.tokenizer,
        max_length=config.training.max_length
    )
    
    # Split dataset
    train_size = min(config.data.num_train_samples, int(0.8 * len(full_dataset)))
    eval_size = min(config.data.num_eval_samples, len(full_dataset) - train_size)
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    eval_dataset = torch.utils.data.Subset(
        full_dataset, 
        range(train_size, train_size + eval_size)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print()
    
    # Initialize trainer
    trainer = ICELoRATrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        output_dir=args.output_dir
    )
    
    # Train model
    if args.do_train:
        trainer.train()
    
    # Evaluate model
    if args.do_eval:
        print("\n" + "="*60)
        print("Running Final Evaluation")
        print("="*60 + "\n")
        
        evaluator = ICELoRAEvaluator(
            model=model,
            tokenizer=model.tokenizer,
            device=trainer.device
        )
        
        # Evaluate on eval dataset
        eval_results = evaluator.evaluate(eval_dataset)
        
        # Generate report
        report = evaluator.generate_report(
            eval_results,
            save_path=os.path.join(args.output_dir, "evaluation_report.txt")
        )
        
        print(report)
        
        # Save results
        with open(os.path.join(args.output_dir, "evaluation_results.json"), 'w') as f:
            json.dump(eval_results, f, indent=2)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICE-LoRA Training")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-small",
                       help="Pre-trained model name")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lambda_ice", type=float, default=1.0,
                       help="Weight for ICE loss")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="zsre",
                       choices=["zsre", "counterfact"],
                       help="Dataset name")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to dataset file")
    parser.add_argument("--create_sample_data", action="store_true",
                       help="Create sample dataset for testing")
    parser.add_argument("--num_train_samples", type=int, default=100,
                       help="Number of training samples")
    parser.add_argument("--num_eval_samples", type=int, default=50,
                       help="Number of evaluation samples")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory")
    
    # Actions
    parser.add_argument("--do_train", action="store_true",
                       help="Run training")
    parser.add_argument("--do_eval", action="store_true",
                       help="Run evaluation")
    
    args = parser.parse_args()
    
    main(args)