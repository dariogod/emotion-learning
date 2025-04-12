#!/usr/bin/env python
# coding: utf-8

"""
ModernBERT Finetuning Script for GLUE Tasks

This script finetunes the ModernBERT model on a specified GLUE benchmark task.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction
)
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Task to keys mapping
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# Task to metrics mapping
task_to_metrics = {
    "cola": {"matthews_correlation": matthews_corrcoef},
    "mnli": {"accuracy": accuracy_score},
    "mrpc": {"accuracy": accuracy_score, "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average='binary')},
    "qnli": {"accuracy": accuracy_score},
    "qqp": {"accuracy": accuracy_score, "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average='binary')},
    "rte": {"accuracy": accuracy_score},
    "sst2": {"accuracy": accuracy_score},
    "stsb": {"pearson": pearsonr, "spearmanr": spearmanr},
    "wnli": {"accuracy": accuracy_score},
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune ModernBERT on a GLUE task"
    )
    
    parser.add_argument(
        "--task", 
        type=str, 
        default="sst2", 
        choices=list(task_to_keys.keys()),
        help="GLUE task name"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="answerdotai/ModernBERT-base",
        help="Pretrained model name or path"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=8e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=8e-6,
        help="Weight decay"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./finetuned_model",
        help="Output directory for model checkpoints and logs"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()

def prepare_datasets(task_name, tokenizer, max_length=128):
    """
    Prepare datasets for a given GLUE task.
    
    Args:
        task_name: Name of the GLUE task
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        
    Returns:
        train_dataset, eval_dataset, test_dataset
    """
    logger.info(f"Loading {task_name} dataset")
    
    # Load dataset from GLUE
    raw_datasets = load_dataset("glue", task_name)
    
    # Get the preprocessing function for the task
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    # Preprocessing function
    def preprocess_function(examples):
        # Handle single and paired sentences
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else 
            (examples[sentence1_key], examples[sentence2_key])
        )
        
        # Tokenize the inputs
        result = tokenizer(*texts, padding=False, truncation=True, max_length=max_length)
        
        # Rename label to labels for the Trainer
        if "label" in examples:
            result["labels"] = examples["label"]
            
        return result
    
    # Apply preprocessing to all splits
    train_dataset = raw_datasets["train"].map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on train dataset",
    )
    
    eval_dataset = raw_datasets["validation"].map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on validation dataset",
    )
    
    # Handle datasets that have a separate validation set for mismatched examples
    if task_name == "mnli":
        eval_dataset = {
            "matched": eval_dataset,
            "mismatched": raw_datasets["validation_mismatched"].map(
                preprocess_function,
                batched=True,
                desc="Running tokenizer on mismatched validation dataset",
            )
        }
    
    # For testing, use validation set if test set doesn't have labels
    if "test" in raw_datasets:
        test_dataset = raw_datasets["test"].map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on test dataset",
        )
    else:
        test_dataset = eval_dataset
    
    return train_dataset, eval_dataset, test_dataset

def compute_metrics(task_name):
    """
    Create a metric function for a given task.
    
    Args:
        task_name: Name of the GLUE task
        
    Returns:
        Function that computes metrics for the task
    """
    metrics_dict = task_to_metrics[task_name]
    
    def compute_metrics_func(eval_pred: EvalPrediction):
        preds, labels = eval_pred
        
        # For regression task (STS-B)
        if task_name == "stsb":
            preds = preds.squeeze()
        else:
            # For classification tasks
            preds = np.argmax(preds, axis=1)
        
        result = {}
        for metric_name, metric_func in metrics_dict.items():
            if metric_name in ["pearson", "spearmanr"]:
                # Correlation metrics return a tuple with the correlation and p-value
                correlation, _ = metric_func(labels, preds)
                result[metric_name] = correlation
            else:
                result[metric_name] = metric_func(labels, preds)
                
        return result
    
    return compute_metrics_func

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging to file
    log_filename = f"{args.output_dir}/{args.task}_finetune.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log arguments
    logger.info(f"Running with arguments: {args}")
    
    # Initialize tokenizer and model
    logger.info(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Get the number of labels for the task
    num_labels = 1 if args.task == "stsb" else 3 if args.task == "mnli" else 2
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="regression" if args.task == "stsb" else "single_label_classification"
    )
    
    # Prepare datasets
    train_dataset, eval_dataset, test_dataset = prepare_datasets(args.task, tokenizer)
    
    # Data collator for padding batches
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.task}_output"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=next(iter(task_to_metrics[args.task].keys())),
        push_to_hub=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if not isinstance(eval_dataset, dict) else eval_dataset["matched"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics(args.task),
    )
    
    # Train the model
    logger.info(f"Starting training for {args.epochs} epochs")
    trainer.train()
    
    # Evaluate the model
    logger.info("Running final evaluation")
    eval_results = trainer.evaluate()
    
    # Log evaluation results
    for metric_name, value in eval_results.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Save the model
    logger.info(f"Saving model to {args.output_dir}/{args.task}_model")
    trainer.save_model(f"{args.output_dir}/{args.task}_model")
    
    # Save evaluation results
    results_file = f"{args.output_dir}/{args.task}_results.json"
    with open(results_file, "w") as f:
        import json
        json.dump(eval_results, f, indent=4)
    logger.info(f"Saved results to {results_file}")
    
    logger.info("Finetuning completed successfully")

if __name__ == "__main__":
    main() 