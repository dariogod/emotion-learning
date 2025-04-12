#!/usr/bin/env python
# coding: utf-8

"""
Emotion-Aware Finetuning Script for MNLI

This script extends the standard GLUE finetuning to incorporate emotional signals
in the training process, either through sample weighting, loss modification, or 
curriculum learning.
"""

import os
import argparse
import logging
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction,
    get_scheduler
)
from datasets import load_dataset, Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class EmotionWeightedMNLIDataset(Dataset):
    """Dataset class that incorporates emotion weights for MNLI samples."""
    
    def __init__(
        self, 
        dataset, 
        tokenizer, 
        sample_weights=None, 
        max_length=128,
        sort_by_emotion=False,
        emotion_key="premise_intensity"
    ):
        """
        Initialize the emotion-weighted dataset.
        
        Args:
            dataset: Hugging Face dataset with MNLI samples
            tokenizer: Tokenizer for encoding text
            sample_weights: Optional sample weights for each instance
            max_length: Maximum sequence length for tokenization
            sort_by_emotion: Whether to sort samples by emotion score (for curriculum learning)
            emotion_key: Key for emotion score to use for sorting 
                         (premise_intensity, premise_valence, premise_arousal, 
                          hypothesis_intensity, hypothesis_valence, hypothesis_arousal,
                          contrast, average_intensity, etc.)
        """
        self.original_dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_weights = sample_weights if sample_weights is not None else np.ones(len(dataset))
        
        # Extract features
        self.premises = dataset["premise"]
        self.hypotheses = dataset["hypothesis"]
        self.labels = dataset["label"]
        
        # If emotion scores are available in the dataset
        self.emotion_scores = dataset.get("emotion_scores", None)
        
        # For curriculum learning, sort by emotion
        if sort_by_emotion and self.emotion_scores is not None:
            # Determine which scores to use based on the emotion_key
            if "_" in emotion_key:
                # Parse keys like "premise_intensity", "hypothesis_valence", "premise_arousal"
                parts = emotion_key.split("_")
                if len(parts) >= 2:
                    target = parts[0]  # premise or hypothesis
                    dimension = parts[1].capitalize()  # Intensity, Valence, Arousal
                    
                    # Map dimension name to the actual key in the data
                    dim_key = dimension
                    if dimension == "Intensity":
                        dim_key = "Emotional Intensity"
                    
                    # Get the values for sorting
                    sort_values = [score.get(target, {}).get(dim_key, 0.5) 
                                  for score in self.emotion_scores]
                else:
                    # Default fallback
                    logger.warning(f"Could not parse emotion key: {emotion_key}, using premise intensity")
                    sort_values = [score.get("premise", {}).get("Emotional Intensity", 0.5) 
                                  for score in self.emotion_scores]
            elif emotion_key == "contrast":
                sort_values = [score.get("contrast", 0.0) 
                              for score in self.emotion_scores]
            elif emotion_key == "average_intensity":
                sort_values = [(score.get("premise", {}).get("Emotional Intensity", 0.5) + 
                               score.get("hypothesis", {}).get("Emotional Intensity", 0.5)) / 2
                              for score in self.emotion_scores]
            else:
                logger.warning(f"Unknown emotion key: {emotion_key}, using premise intensity")
                sort_values = [score.get("premise", {}).get("Emotional Intensity", 0.5) 
                              for score in self.emotion_scores]
            
            # Sort by the selected emotion values
            sorted_indices = np.argsort(sort_values)
            
            self.premises = [self.premises[i] for i in sorted_indices]
            self.hypotheses = [self.hypotheses[i] for i in sorted_indices]
            self.labels = [self.labels[i] for i in sorted_indices]
            self.sample_weights = self.sample_weights[sorted_indices]
            self.emotion_scores = [self.emotion_scores[i] for i in sorted_indices]
            
    def __len__(self):
        return len(self.premises)
    
    def __getitem__(self, idx):
        premise = self.premises[idx]
        hypothesis = self.hypotheses[idx]
        label = self.labels[idx]
        
        # Tokenize the input
        encoding = self.tokenizer(
            premise, 
            hypothesis, 
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Convert to appropriate tensor types and squeeze batch dimension
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
            "weight": torch.tensor(self.sample_weights[idx], dtype=torch.float)
        }
        
        # Add emotion scores if available
        if self.emotion_scores is not None:
            emotion_score = self.emotion_scores[idx]
            
            # Add premise emotion scores
            premise_scores = emotion_score.get("premise", {})
            item["premise_intensity"] = torch.tensor(
                premise_scores.get("Emotional Intensity", 0.5), 
                dtype=torch.float
            )
            item["premise_valence"] = torch.tensor(
                premise_scores.get("Valence", 0.5),
                dtype=torch.float
            )
            item["premise_arousal"] = torch.tensor(
                premise_scores.get("Arousal", 0.5),
                dtype=torch.float
            )
            
            # Add hypothesis emotion scores
            hypothesis_scores = emotion_score.get("hypothesis", {})
            item["hypothesis_intensity"] = torch.tensor(
                hypothesis_scores.get("Emotional Intensity", 0.5), 
                dtype=torch.float
            )
            item["hypothesis_valence"] = torch.tensor(
                hypothesis_scores.get("Valence", 0.5),
                dtype=torch.float
            )
            item["hypothesis_arousal"] = torch.tensor(
                hypothesis_scores.get("Arousal", 0.5),
                dtype=torch.float
            )
            
            # Add contrast
            item["contrast"] = torch.tensor(
                emotion_score.get("contrast", 0.0),
                dtype=torch.float
            )
        
        return item

class EmotionWeightedTrainer(Trainer):
    """Custom trainer that supports sample weighting based on emotion scores."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation that incorporates sample weights.
        
        Args:
            model: The model to train
            inputs: The inputs and targets
            return_outputs: Whether to return outputs along with the loss
            
        Returns:
            Loss value and optionally model outputs
        """
        # Get sample weights
        weights = inputs.pop("weight", None)
        
        # Extract any emotion features (not used in loss but could be used for monitoring)
        premise_intensity = inputs.pop("premise_intensity", None)
        premise_valence = inputs.pop("premise_valence", None)
        premise_arousal = inputs.pop("premise_arousal", None)
        hypothesis_intensity = inputs.pop("hypothesis_intensity", None)
        hypothesis_valence = inputs.pop("hypothesis_valence", None)
        hypothesis_arousal = inputs.pop("hypothesis_arousal", None)
        contrast = inputs.pop("contrast", None)
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute standard loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        labels = inputs.get("labels")
        per_sample_loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        # Apply sample weights if provided
        if weights is not None:
            weighted_loss = (per_sample_loss * weights).mean()
        else:
            weighted_loss = per_sample_loss.mean()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune MNLI with emotion-aware training"
    )
    
    parser.add_argument(
        "--augmented_data", 
        type=str, 
        required=True,
        help="Path to the augmented MNLI dataset created by data_augmentation.py"
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
        default="./emotion_models",
        help="Output directory for model checkpoints and logs"
    )
    
    parser.add_argument(
        "--weighting_strategy", 
        type=str, 
        default="premise_bell_curve",
        help="Strategy for weighting samples based on emotion (e.g., premise_bell_curve, hypothesis_linear, contrast_inverse)"
    )
    
    parser.add_argument(
        "--curriculum", 
        action="store_true",
        help="Whether to use curriculum learning based on emotion intensity"
    )
    
    parser.add_argument(
        "--curriculum_key",
        type=str,
        default="premise_intensity",
        choices=[
            "premise_intensity", "premise_valence", "premise_arousal",
            "hypothesis_intensity", "hypothesis_valence", "hypothesis_arousal",
            "contrast", "average_intensity"
        ],
        help="Which emotion dimension to use for curriculum learning"
    )
    
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=100,
        help="Number of steps between logging training metrics"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()

def prepare_augmented_dataset(augmented_data_path, tokenizer, weighting_strategy="premise_bell_curve", 
                              curriculum=False, curriculum_key="premise_intensity"):
    """
    Prepare datasets from the augmented data file.
    
    Args:
        augmented_data_path: Path to the augmented data file
        tokenizer: Tokenizer for encoding text
        weighting_strategy: Strategy for sample weighting
        curriculum: Whether to use curriculum learning
        curriculum_key: Which emotion dimension to use for curriculum learning
        
    Returns:
        train_dataset, eval_dataset
    """
    logger.info(f"Loading augmented dataset from {augmented_data_path}")
    
    # Load the augmented data
    with open(augmented_data_path, "r") as f:
        augmented_data = json.load(f)
    
    # Extract components
    original_data = augmented_data["original_data"]
    emotion_scores = augmented_data["emotion_scores"]
    
    # Get sample weights based on strategy
    if weighting_strategy in augmented_data["sample_weights"]:
        sample_weights = augmented_data["sample_weights"][weighting_strategy]
    else:
        logger.warning(f"Weighting strategy {weighting_strategy} not found, using uniform weights")
        sample_weights = augmented_data["sample_weights"]["uniform"]
    
    # Convert to HF Dataset
    train_data_dict = {
        "premise": original_data["premise"],
        "hypothesis": original_data["hypothesis"],
        "label": original_data["label"],
        "emotion_scores": emotion_scores
    }
    train_dataset = HFDataset.from_dict(train_data_dict)
    
    # Create custom dataset with emotion weighting
    train_dataset = EmotionWeightedMNLIDataset(
        train_dataset,
        tokenizer,
        sample_weights=np.array(sample_weights),
        sort_by_emotion=curriculum,
        emotion_key=curriculum_key
    )
    
    # Load validation dataset from GLUE
    logger.info("Loading validation dataset from GLUE")
    validation_dataset = load_dataset("glue", "mnli", split="validation_matched")
    
    # Process validation dataset
    validation_dataset = EmotionWeightedMNLIDataset(
        validation_dataset,
        tokenizer,
        sample_weights=None  # No weighting for validation
    )
    
    return train_dataset, validation_dataset

def compute_metrics(preds, labels):
    """
    Compute evaluation metrics for the MNLI task.
    
    Args:
        preds: Model predictions
        labels: Ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    preds = np.argmax(preds, axis=1)
    
    # Compute accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Compute F1 scores for each class (multi-class F1)
    f1_per_class = f1_score(labels, preds, average=None)
    
    metrics = {
        "accuracy": accuracy,
        "f1_contradiction": f1_per_class[0],
        "f1_entailment": f1_per_class[1],
        "f1_neutral": f1_per_class[2],
        "f1_macro": f1_score(labels, preds, average="macro")
    }
    
    return metrics

def train_custom(args, train_dataset, eval_dataset, model, tokenizer):
    """
    Custom training loop for more control and tracking of metrics.
    
    Args:
        args: Command-line arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        model: The model to train
        tokenizer: Tokenizer for the model
        
    Returns:
        Trained model and training history
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Calculate number of training steps
    num_update_steps_per_epoch = len(train_loader)
    num_training_steps = args.epochs * num_update_steps_per_epoch
    
    # Set up learning rate scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize training history
    history = {
        "train_loss": [],
        "eval_loss": [],
        "eval_metrics": [],
        "learning_rates": []
    }
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        
        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            # Get loss and apply sample weighting
            loss = outputs.loss
            if "weight" in batch:
                per_sample_loss = F.cross_entropy(
                    outputs.logits, batch["labels"], reduction="none"
                )
                loss = (per_sample_loss * batch["weight"]).mean()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update metrics
            total_train_loss += loss.item()
            current_lr = lr_scheduler.get_last_lr()[0]
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            # Log metrics at specified intervals
            if global_step % args.logging_steps == 0:
                history["train_loss"].append(loss.item())
                history["learning_rates"].append(current_lr)
                
            global_step += 1
        
        # Compute average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} average train loss: {avg_train_loss:.4f}")
        
        # Evaluation
        logger.info("Running evaluation")
        model.eval()
        eval_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Filter out emotion features to avoid errors
                emotion_keys = ["premise_intensity", "premise_valence", 
                                "premise_arousal", "hypothesis_intensity", 
                                "hypothesis_valence", "hypothesis_arousal", 
                                "contrast", "weight"]
                batch = {k: v.to(device) for k, v in batch.items() 
                         if k not in emotion_keys}
                
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs.loss
                eval_loss += loss.item()
                
                logits = outputs.logits
                preds = logits.detach().cpu().numpy()
                labels = batch["labels"].detach().cpu().numpy()
                
                all_preds.append(preds)
                all_labels.append(labels)
        
        # Compute average evaluation loss
        avg_eval_loss = eval_loss / len(eval_loader)
        
        # Compute evaluation metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_preds, all_labels)
        
        # Update history
        history["eval_loss"].append(avg_eval_loss)
        history["eval_metrics"].append(metrics)
        
        # Log evaluation results
        logger.info(f"Epoch {epoch+1} evaluation - Loss: {avg_eval_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Save training history
        with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
    
    return model, history

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging to file
    log_filename = f"{args.output_dir}/training.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log arguments
    logger.info(f"Running with arguments: {args}")
    
    # Initialize tokenizer and model
    logger.info(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # MNLI has 3 labels: contradiction (0), entailment (1), neutral (2)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3
    )
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_augmented_dataset(
        args.augmented_data, 
        tokenizer,
        weighting_strategy=args.weighting_strategy,
        curriculum=args.curriculum,
        curriculum_key=args.curriculum_key
    )
    
    # Train the model
    model, history = train_custom(args, train_dataset, eval_dataset, model, tokenizer)
    
    # Save the final model
    logger.info(f"Saving final model to {args.output_dir}/final_model")
    model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main() 