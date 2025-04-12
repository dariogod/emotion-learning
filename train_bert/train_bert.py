import argparse
import json
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score


class BertWithEmotion(torch.nn.Module):
    """BERT model that incorporates emotion intensity to weight loss during training.
    
    This model wraps a base BERT model and modifies the loss calculation to account
    for emotion intensity values.
    """
    def __init__(self, base_model: BertForSequenceClassification):
        super().__init__()
        self.bert = base_model
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        intensity: Optional[torch.Tensor] = None
    ) -> Any:
        """Forward pass with optional emotion-weighted loss calculation.
        
        Args:
            input_ids: Token IDs to process
            attention_mask: Attention mask for tokens
            labels: Optional target labels for loss calculation
            intensity: Optional emotion intensity values for weighting loss
            
        Returns:
            Model outputs including loss if labels are provided
        """
        # Forward pass through BERT
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            return_dict=True
        )
        
        # If we're training (labels are provided), modify the loss using intensity
        if labels is not None:
            if intensity is None:
                raise ValueError("Intensity must be provided when labels are provided")
            
            # Calculate loss manually
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.bert.config.num_labels), labels.view(-1))
            
            # Apply intensity weighting
            weighted_loss = (loss * intensity).mean()
            outputs.loss = weighted_loss
            
        return outputs


class EmotionMNLI(Dataset):
    """Dataset for NLI tasks with emotion intensity information.
    
    Handles the processing of NLI examples with optional emotion intensity.
    """
    def __init__(
        self, 
        data: List[Dict[str, Any]], 
        tokenizer: BertTokenizer, 
        include_emotion: bool
    ):
        """Initialize the dataset.
        
        Args:
            data: List of NLI examples with emotion information
            tokenizer: BERT tokenizer
            include_emotion: Whether to include emotion intensity in samples
        """
        self.tokenizer = tokenizer
        self.data = data
        self.include_emotion = include_emotion

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from the dataset.
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Dictionary containing the processed example
        """
        item = self.data[idx]
        premise = item['premise']['text']
        hypothesis = item['hypothesis']['text']
        label = item['label']

        # Tokenize input
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        )

        # Base inputs
        sample = {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        # Optionally add emotion intensity
        if self.include_emotion:
            prem_intensity = item['premise']['emotion_info']['intensity']
            hyp_intensity = item['hypothesis']['emotion_info']['intensity']
            avg_intensity = (prem_intensity + hyp_intensity) / 2
            sample['intensity'] = torch.tensor(avg_intensity, dtype=torch.float)

        return sample


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute evaluation metrics from prediction outputs.
    
    Args:
        eval_pred: Tuple containing model predictions and labels
        
    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def load_datasets(train_file: str, test_file: str) -> Tuple[List[Dict], List[Dict]]:
    """Load training and testing datasets from files.
    
    Args:
        train_file: Path to training data JSON file
        test_file: Path to test data JSON file
        
    Returns:
        Tuple of (train_data, test_data)
    """
    with open(train_file) as f:
        train_data = json.load(f)
    with open(test_file) as f:
        test_data = json.load(f)
    return train_data, test_data


def create_model(include_emotion: bool) -> torch.nn.Module:
    """Create and configure the model based on whether emotion should be included.
    
    Args:
        include_emotion: Whether to use emotion intensity for weighting loss
        
    Returns:
        Configured model
    """
    base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    
    if include_emotion:
        return BertWithEmotion(base_model)
    else:
        return base_model


def create_training_args(include_emotion: bool) -> TrainingArguments:
    """Create training arguments with appropriate configuration.
    
    Args:
        include_emotion: Whether emotion intensity is being used
        
    Returns:
        Configured TrainingArguments
    """
    output_dir = "outputs/bert-with-emotion_2" if include_emotion else "outputs/bert-without-emotion_2"
    
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        eval_strategy="steps",
        eval_steps=100,
        logging_dir="logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
    )


def main():
    """Main entry point for training BERT models with or without emotion weighting."""
    parser = argparse.ArgumentParser(description="Train BERT models for NLI with optional emotion weighting")
    parser.add_argument("--include_emotion", action="store_true", help="Use emotion intensity to weight loss")
    parser.add_argument("--train_file", type=str, default="train.json", help="Path to training dataset JSON")
    parser.add_argument("--test_file", type=str, default="test.json", help="Path to test dataset JSON")
    args = parser.parse_args()

    # Load data
    train_data, test_data = load_datasets(args.train_file, args.test_file)

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = create_model(args.include_emotion)

    # Create datasets
    train_dataset = EmotionMNLI(train_data, tokenizer, include_emotion=args.include_emotion)
    eval_dataset = EmotionMNLI(test_data, tokenizer, include_emotion=args.include_emotion)
    
    # Configure training
    training_args = create_training_args(args.include_emotion)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
