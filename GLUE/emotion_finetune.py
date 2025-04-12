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
import random

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
    """Custom trainer that incorporates emotion features into the loss function."""
    
    def __init__(self, *args, emotion_loss_weight=0.5, emotion_features_to_use=None, **kwargs):
        """
        Initialize the emotion-weighted trainer.
        
        Args:
            emotion_loss_weight: Weight for the emotion-based loss term (0.0 to 1.0)
            emotion_features_to_use: List of emotion features to incorporate in loss
        """
        super().__init__(*args, **kwargs)
        self.emotion_loss_weight = emotion_loss_weight
        
        # Default emotion features to use if not specified
        self.emotion_features = emotion_features_to_use or [
            "premise_intensity", "hypothesis_intensity", "contrast"
        ]
        logger.info(f"Using emotion features in loss: {self.emotion_features}")
        logger.info(f"Emotion loss weight: {self.emotion_loss_weight}")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation that incorporates emotion features.
        
        Args:
            model: The model to train
            inputs: The inputs and targets
            return_outputs: Whether to return outputs along with the loss
            
        Returns:
            Loss value and optionally model outputs
        """
        # Get sample weights
        weights = inputs.pop("weight", None)
        
        # Extract emotion features before passing inputs to model
        emotion_features = {}
        for feature in self.emotion_features:
            if feature in inputs:
                emotion_features[feature] = inputs.pop(feature)
        
        # Also extract any other emotion features not used in loss (to avoid errors)
        unused_features = [
            "premise_valence", "premise_arousal",
            "hypothesis_valence", "hypothesis_arousal"
        ]
        for feature in unused_features:
            if feature in inputs:
                inputs.pop(feature)
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute standard cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        labels = inputs.get("labels")
        per_sample_loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        # Apply sample weights if provided
        if weights is not None:
            weighted_loss = (per_sample_loss * weights).mean()
        else:
            weighted_loss = per_sample_loss.mean()
        
        # Calculate emotion-based loss component if any emotion features were provided
        emotion_loss = 0.0
        if emotion_features and self.emotion_loss_weight > 0:
            # Get predicted class probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Higher emotion intensity should increase loss for contradictions (label 0)
            # and decrease loss for entailments (label 1)
            if "premise_intensity" in emotion_features:
                # Scale emotion intensity to be between 0 and 1
                premise_intensity = emotion_features["premise_intensity"]
                # For contradictions (label 0): higher intensity -> higher loss
                # For entailments (label 1): higher intensity -> lower loss
                intensity_loss = torch.zeros_like(per_sample_loss)
                contradiction_mask = (labels == 0)
                entailment_mask = (labels == 1)
                
                # Contradiction: penalize low confidence when intensity is high
                if contradiction_mask.any():
                    intensity_loss[contradiction_mask] = premise_intensity[contradiction_mask] * (1 - probs[contradiction_mask, 0])
                
                # Entailment: penalize low confidence when intensity is high
                if entailment_mask.any():
                    intensity_loss[entailment_mask] = premise_intensity[entailment_mask] * (1 - probs[entailment_mask, 1])
                
                emotion_loss += intensity_loss.mean()
            
            # Similar approach for hypothesis intensity
            if "hypothesis_intensity" in emotion_features:
                hypothesis_intensity = emotion_features["hypothesis_intensity"]
                intensity_loss = torch.zeros_like(per_sample_loss)
                contradiction_mask = (labels == 0)
                entailment_mask = (labels == 1)
                
                if contradiction_mask.any():
                    intensity_loss[contradiction_mask] = hypothesis_intensity[contradiction_mask] * (1 - probs[contradiction_mask, 0])
                
                if entailment_mask.any():
                    intensity_loss[entailment_mask] = hypothesis_intensity[entailment_mask] * (1 - probs[entailment_mask, 1])
                
                emotion_loss += intensity_loss.mean()
            
            # Contrast is the emotional difference between premise and hypothesis
            # Higher contrast should increase the probability of contradiction
            if "contrast" in emotion_features:
                contrast = emotion_features["contrast"]
                contrast_loss = torch.zeros_like(per_sample_loss)
                contradiction_mask = (labels == 0)
                entailment_mask = (labels == 1)
                
                # For contradictions: penalize low contradiction confidence when contrast is high
                if contradiction_mask.any():
                    contrast_loss[contradiction_mask] = contrast[contradiction_mask] * (1 - probs[contradiction_mask, 0])
                
                # For entailments: penalize high contradiction confidence when contrast is high
                if entailment_mask.any():
                    contrast_loss[entailment_mask] = contrast[entailment_mask] * probs[entailment_mask, 0]
                
                emotion_loss += contrast_loss.mean()
        
        # Combine the standard loss with the emotion-based loss
        final_loss = (1 - self.emotion_loss_weight) * weighted_loss
        if emotion_loss > 0:
            final_loss += self.emotion_loss_weight * emotion_loss
        
        return (final_loss, outputs) if return_outputs else final_loss

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune MNLI with emotion-aware training"
    )
    
    parser.add_argument(
        "--augmented_data", 
        type=str, 
        default="train_set.json",
        help="Path to the augmented MNLI dataset created by data_augmentation.py (default: train_set.json)"
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
        "--use_emotion",
        action="store_true",
        help="Whether to use emotion features in training (if False, performs standard finetuning)"
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
        "--emotion_loss_weight",
        type=float,
        default=0.5,
        help="Weight for the emotion-based loss term (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--emotion_features",
        type=str,
        nargs="+",
        default=["premise_intensity", "hypothesis_intensity", "contrast"],
        help="Emotion features to use in the loss function"
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
        augmented_data_path: Path to the augmented MNLI dataset created by data_augmentation.py
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
    
    # Load test dataset from test_set.json
    logger.info("Loading test dataset from data/test_set.json")
    test_set_path = "data/test_set.json"
    
    try:
        with open(test_set_path, "r") as f:
            test_data = json.load(f)
        
        # Convert test data to the format expected by EmotionWeightedMNLIDataset
        test_premises = []
        test_hypotheses = []
        test_labels = []
        test_emotion_scores = []
        
        for item in test_data:
            test_premises.append(item["premise"]["text"])
            test_hypotheses.append(item["hypothesis"]["text"])
            test_labels.append(item["label"])
            
            # Create emotion_scores in the same format as training data
            emotion_score = {
                "premise": {
                    "Emotional Intensity": item["premise"]["emotion_info"]["intensity"],
                    "Valence": item["premise"]["emotion_info"]["valence"],
                    "Arousal": item["premise"]["emotion_info"]["arousal"]
                },
                "hypothesis": {
                    "Emotional Intensity": item["hypothesis"]["emotion_info"]["intensity"],
                    "Valence": item["hypothesis"]["emotion_info"]["valence"],
                    "Arousal": item["hypothesis"]["emotion_info"]["arousal"]
                },
                "contrast": abs(item["premise"]["emotion_info"]["intensity"] - 
                               item["hypothesis"]["emotion_info"]["intensity"])
            }
            test_emotion_scores.append(emotion_score)
        
        test_data_dict = {
            "premise": test_premises,
            "hypothesis": test_hypotheses,
            "label": test_labels,
            "emotion_scores": test_emotion_scores
        }
        
        test_dataset = HFDataset.from_dict(test_data_dict)
        
        # Create custom dataset for evaluation
        validation_dataset = EmotionWeightedMNLIDataset(
            test_dataset,
            tokenizer,
            sample_weights=None  # No weighting for validation
        )
        
        logger.info(f"Test dataset loaded with {len(validation_dataset)} examples")
        
    except Exception as e:
        logger.error(f"Error loading test dataset: {e}")
        logger.info("Falling back to random sample from MNLI validation dataset")
        
        # Fallback to MNLI validation dataset
        validation_dataset = load_dataset("glue", "mnli", split="validation_matched")
        
        # Take 200 random samples from the validation dataset
        logger.info("Selecting 200 random samples for evaluation")
        total_samples = len(validation_dataset)
        random_indices = np.random.choice(total_samples, size=200, replace=False)
        validation_dataset = validation_dataset.select(random_indices)
        
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

def evaluate_on_test_set(model, tokenizer, emotion_loss_weight=0.0):
    """
    Perform a final evaluation on the test set.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        emotion_loss_weight: Whether to use emotion features in loss
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Running final evaluation on test set...")
    
    # Load test dataset
    test_set_path = "data/test_set.json"
    
    try:
        with open(test_set_path, "r") as f:
            test_data = json.load(f)
        
        # Convert test data to the format expected by EmotionWeightedMNLIDataset
        test_premises = []
        test_hypotheses = []
        test_labels = []
        test_emotion_scores = []
        
        for item in test_data:
            test_premises.append(item["premise"]["text"])
            test_hypotheses.append(item["hypothesis"]["text"])
            test_labels.append(item["label"])
            
            # Create emotion_scores in the same format as training data
            emotion_score = {
                "premise": {
                    "Emotional Intensity": item["premise"]["emotion_info"]["intensity"],
                    "Valence": item["premise"]["emotion_info"]["valence"],
                    "Arousal": item["premise"]["emotion_info"]["arousal"]
                },
                "hypothesis": {
                    "Emotional Intensity": item["hypothesis"]["emotion_info"]["intensity"],
                    "Valence": item["hypothesis"]["emotion_info"]["valence"],
                    "Arousal": item["hypothesis"]["emotion_info"]["arousal"]
                },
                "contrast": abs(item["premise"]["emotion_info"]["intensity"] - 
                               item["hypothesis"]["emotion_info"]["intensity"])
            }
            test_emotion_scores.append(emotion_score)
        
        test_data_dict = {
            "premise": test_premises,
            "hypothesis": test_hypotheses,
            "label": test_labels,
            "emotion_scores": test_emotion_scores
        }
        
        test_dataset = HFDataset.from_dict(test_data_dict)
        
        # Create custom dataset for evaluation
        test_dataset = EmotionWeightedMNLIDataset(
            test_dataset,
            tokenizer,
            sample_weights=None  # No weighting for testing
        )
        
        logger.info(f"Test dataset loaded with {len(test_dataset)} examples")
        
        # Set up evaluation args
        eval_args = TrainingArguments(
            output_dir="./eval_output",
            per_device_eval_batch_size=32,
            remove_unused_columns=False,
        )
        
        # Set up the emotion-aware trainer for evaluation only
        trainer = EmotionWeightedTrainer(
            model=model,
            args=eval_args,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(p.predictions, p.label_ids),
            emotion_loss_weight=emotion_loss_weight
        )
        
        # Run evaluation
        metrics = trainer.evaluate()
        
        # Convert to more readable format
        final_metrics = {
            "accuracy": metrics["eval_accuracy"],
            "f1_macro": metrics["eval_f1_macro"],
            "f1_contradiction": metrics.get("eval_f1_contradiction", 0.0),
            "f1_entailment": metrics.get("eval_f1_entailment", 0.0),
            "f1_neutral": metrics.get("eval_f1_neutral", 0.0),
        }
        
        # Log results
        logger.info("Final test metrics:")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Also log per-emotion-level metrics if possible
        # Split test data by emotion intensity levels
        high_intensity_indices = []
        medium_intensity_indices = []
        low_intensity_indices = []
        
        for i, item in enumerate(test_data):
            intensity = item["premise"]["emotion_info"]["intensity"]
            if intensity >= 0.7:
                high_intensity_indices.append(i)
            elif intensity >= 0.4:
                medium_intensity_indices.append(i)
            else:
                low_intensity_indices.append(i)
        
        # Log distribution
        logger.info(f"Test examples by intensity: high={len(high_intensity_indices)}, "
                   f"medium={len(medium_intensity_indices)}, low={len(low_intensity_indices)}")
        
        # Calculate per-intensity metrics
        all_preds = trainer.predict(test_dataset).predictions
        all_preds = np.argmax(all_preds, axis=1)
        all_labels = np.array(test_labels)
        
        if len(high_intensity_indices) > 0:
            high_preds = all_preds[high_intensity_indices]
            high_labels = all_labels[high_intensity_indices]
            high_acc = accuracy_score(high_labels, high_preds)
            logger.info(f"  High intensity accuracy: {high_acc:.4f}")
        
        if len(medium_intensity_indices) > 0:
            med_preds = all_preds[medium_intensity_indices]
            med_labels = all_labels[medium_intensity_indices]
            med_acc = accuracy_score(med_labels, med_preds)
            logger.info(f"  Medium intensity accuracy: {med_acc:.4f}")
        
        if len(low_intensity_indices) > 0:
            low_preds = all_preds[low_intensity_indices]
            low_labels = all_labels[low_intensity_indices]
            low_acc = accuracy_score(low_labels, low_preds)
            logger.info(f"  Low intensity accuracy: {low_acc:.4f}")
        
        return final_metrics
        
    except Exception as e:
        logger.error(f"Error during final evaluation: {e}")
        return {"error": str(e)}

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging to file
    log_filename = f"{args.output_dir}/training.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log arguments
    logger.info(f"Running with arguments: {args}")
    logger.info(f"Using emotion features: {args.use_emotion}")
    
    # Initialize tokenizer and model
    logger.info(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # MNLI has 3 labels: contradiction (0), entailment (1), neutral (2)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3
    )
    
    # Hardcoded to use train_set.json
    train_data_path = "train_set.json"
    logger.info(f"Using hardcoded training data file: {train_data_path}")
    
    # Prepare datasets - if not using emotion features, ignore emotion weightings
    use_weighting = args.use_emotion and args.weighting_strategy != "uniform"
    use_curriculum = args.use_emotion and args.curriculum
    
    train_dataset, eval_dataset = prepare_augmented_dataset(
        train_data_path, 
        tokenizer,
        weighting_strategy="uniform" if not use_weighting else args.weighting_strategy,
        curriculum=use_curriculum,
        curriculum_key=args.curriculum_key
    )
    
    # Configure training arguments with deterministic behavior
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=args.seed,
        data_seed=args.seed,
        # Force deterministic behavior
        dataloader_drop_last=False,
        dataloader_num_workers=0,  # Use main process for data loading
    )
    
    # Setup trainer - use standard Trainer if not using emotion features
    if args.use_emotion:
        logger.info("Using emotion-aware trainer with emotion-based loss")
        trainer = EmotionWeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(p.predictions, p.label_ids),
            # Emotion loss parameters
            emotion_loss_weight=args.emotion_loss_weight,
            emotion_features_to_use=args.emotion_features
        )
    else:
        logger.info("Using standard trainer without emotion features")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(p.predictions, p.label_ids)
        )
    
    # Train the model
    trainer_type = "emotion-aware" if args.use_emotion else "standard"
    logger.info(f"Starting training with {trainer_type} trainer")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving final model to {args.output_dir}/final_model")
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    # Run final evaluation on test set
    logger.info("Running final evaluation on test set")
    final_metrics = evaluate_on_test_set(
        model=model, 
        tokenizer=tokenizer,
        emotion_loss_weight=args.emotion_loss_weight if args.use_emotion else 0.0
    )
    
    # Save final metrics
    with open(os.path.join(args.output_dir, "final_test_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info("Training and evaluation completed successfully")

def set_seed(seed):
    """
    Set random seeds for full reproducibility.
    
    Args:
        seed: The seed to use
    """
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # Set deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables that might affect reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed} for full reproducibility")

if __name__ == "__main__":
    main() 