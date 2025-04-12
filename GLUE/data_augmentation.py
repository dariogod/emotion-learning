#!/usr/bin/env python
# coding: utf-8

"""
Data Augmentation Script for MNLI with Emotion Signals

This script uses the Anthropic Claude API to augment MNLI dataset samples with emotional 
dimensions (intensity, valence, and arousal), which can then be used in emotion-aware training.
"""

import os
import argparse
import logging
import time
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
import anthropic
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Maximum retries for API calls
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Augment MNLI dataset with emotion scores using Claude API"
    )
    
    parser.add_argument(
        "--api_key",
        type=str,
        help="Anthropic API key (or set ANTHROPIC_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-sonnet-20240229",
        help="Claude model to use for emotion scoring"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./augmented_data",
        help="Output directory for augmented datasets"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to process (None for all)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for API calls to reduce number of requests"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()

def setup_claude_client(api_key: Optional[str] = None) -> anthropic.Anthropic:
    """Set up the Anthropic client with the provided API key."""
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Anthropic API key must be provided via --api_key or ANTHROPIC_API_KEY environment variable")
    
    client = anthropic.Anthropic(api_key=key)
    logger.info("Claude client configured")
    return client

def get_emotion_score_batch(client: anthropic.Anthropic, texts: List[Dict[str, str]], model: str) -> List[Dict[str, Dict[str, float]]]:
    """
    Get emotion scores for a batch of texts using Claude API.
    
    Args:
        client: Anthropic client
        texts: List of dictionaries with 'premise' and 'hypothesis' keys
        model: Claude model to use
        
    Returns:
        List of dictionaries with separate emotion scores for premise and hypothesis
    """
    results = []
    
    for i, text in enumerate(texts):
        premise = text['premise']
        hypothesis = text['hypothesis']
        
        system_prompt = "You are an expert emotion analyst. Analyze the emotional content of text and provide numerical scores."
        
        user_prompt = (
            f"Analyze the emotional content in the following premise and hypothesis SEPARATELY:\n\n"
            f"Premise: {premise}\n\nHypothesis: {hypothesis}\n\n"
            f"For EACH text (premise and hypothesis), rate on a scale from 0.0 to 1.0 for each of these dimensions:\n"
            f"1. Emotional Intensity: How strong is the emotional content?\n"
            f"2. Valence: How positive (1.0) or negative (0.0) is the emotional content?\n"
            f"3. Arousal: How exciting/stimulating (1.0) or calming/soothing (0.0) is the content?\n\n"
            f"Provide your answer as a JSON object with nested objects for premise and hypothesis, each containing the three scores. Format: "
            f"{{'premise': {{'Emotional Intensity': float, 'Valence': float, 'Arousal': float}}, "
            f"'hypothesis': {{'Emotional Intensity': float, 'Valence': float, 'Arousal': float}}}}"
        )
        
        for attempt in range(MAX_RETRIES):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0  # Keep deterministic
                )
                
                # Parse the JSON response
                try:
                    # Extract JSON from the response content
                    content = response.content[0].text
                    
                    # Try to find JSON in the response (Claude sometimes adds extra text)
                    import re
                    json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(1)
                        scores = json.loads(json_str)
                    else:
                        # Fallback to trying to parse the entire response
                        scores = json.loads(content)
                    
                    # Calculate emotional contrast
                    premise_intensity = scores.get("premise", {}).get("Emotional Intensity", 0.5)
                    premise_valence = scores.get("premise", {}).get("Valence", 0.5)
                    premise_arousal = scores.get("premise", {}).get("Arousal", 0.5)
                    
                    hypo_intensity = scores.get("hypothesis", {}).get("Emotional Intensity", 0.5)
                    hypo_valence = scores.get("hypothesis", {}).get("Valence", 0.5)
                    hypo_arousal = scores.get("hypothesis", {}).get("Arousal", 0.5)
                    
                    # Calculate contrast as average of differences across all dimensions
                    intensity_diff = abs(premise_intensity - hypo_intensity)
                    valence_diff = abs(premise_valence - hypo_valence)
                    arousal_diff = abs(premise_arousal - hypo_arousal)
                    
                    emotional_contrast = (intensity_diff + valence_diff + arousal_diff) / 3
                    
                    # Add derived contrast score
                    scores["contrast"] = emotional_contrast
                    
                    results.append(scores)
                    break
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"Failed to parse JSON response for item {i}, attempt {attempt+1}: {str(e)}")
                    if attempt == MAX_RETRIES - 1:
                        # If this is the last attempt, add default values
                        results.append({
                            "premise": {
                                "Emotional Intensity": 0.5,
                                "Valence": 0.5,
                                "Arousal": 0.5
                            },
                            "hypothesis": {
                                "Emotional Intensity": 0.5,
                                "Valence": 0.5,
                                "Arousal": 0.5
                            },
                            "contrast": 0.0
                        })
                    else:
                        time.sleep(RETRY_DELAY)
            
            except Exception as e:
                logger.warning(f"API call failed for item {i}, attempt {attempt+1}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    # If this is the last attempt, add default values
                    results.append({
                        "premise": {
                            "Emotional Intensity": 0.5,
                            "Valence": 0.5,
                            "Arousal": 0.5
                        },
                        "hypothesis": {
                            "Emotional Intensity": 0.5,
                            "Valence": 0.5,
                            "Arousal": 0.5
                        },
                        "contrast": 0.0
                    })
                else:
                    time.sleep(RETRY_DELAY)
    
    return results

def calculate_weights(emotion_scores: List[Dict[str, Any]], strategy: str = "bell_curve", 
                      target: str = "premise", dimension: str = "Emotional Intensity") -> np.ndarray:
    """
    Calculate sample weights based on emotion scores using different strategies.
    
    Args:
        emotion_scores: List of dictionaries with emotion scores
        strategy: Strategy to use for weight calculation (bell_curve, linear, etc.)
        target: Which scores to use for weighting (premise, hypothesis, contrast, average)
        dimension: Which emotion dimension to use (Emotional Intensity, Valence, Arousal)
        
    Returns:
        Array of weights for each sample
    """
    # Extract target values based on dimension
    if target == "premise":
        # Use premise dimension
        values = np.array([score.get("premise", {}).get(dimension, 0.5) 
                         for score in emotion_scores])
    elif target == "hypothesis":
        # Use hypothesis dimension
        values = np.array([score.get("hypothesis", {}).get(dimension, 0.5) 
                         for score in emotion_scores])
    elif target == "contrast":
        # Use emotional contrast
        if dimension != "Emotional Intensity":
            logger.warning(f"Contrast doesn't have specific dimensions; using overall contrast score")
        values = np.array([score.get("contrast", 0.0) for score in emotion_scores])
    elif target == "average":
        # Use average of premise and hypothesis
        values = np.array([(score.get("premise", {}).get(dimension, 0.5) + 
                          score.get("hypothesis", {}).get(dimension, 0.5)) / 2
                         for score in emotion_scores])
    else:
        raise ValueError(f"Unknown target: {target}")
    
    if strategy == "bell_curve":
        # Bell curve weighting: prioritize samples with moderate values
        # Samples with values around 0.5 get highest weight
        weights = 1.0 - 2.0 * np.abs(values - 0.5)
        weights = weights + 0.2  # Ensure minimum weight of 0.2
        
    elif strategy == "linear":
        # Linear weighting: higher values get higher weight
        weights = values
        
    elif strategy == "inverse":
        # Inverse weighting: lower values get higher weight
        weights = 1.0 - values
        
    else:
        # Default: uniform weighting
        weights = np.ones_like(values)
    
    # Normalize weights to sum to the number of samples (to maintain overall loss scale)
    weights = weights * (len(weights) / weights.sum())
    
    return weights

def augment_dataset(args, client):
    """Augment the MNLI dataset with emotion scores."""
    # Load MNLI dataset
    logger.info("Loading MNLI dataset")
    dataset = load_dataset("glue", "mnli")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process training split
    logger.info("Processing training split")
    train_data = dataset["train"]
    
    # Limit sample size if specified
    if args.sample_size is not None:
        indices = np.random.RandomState(args.seed).choice(
            len(train_data), 
            min(args.sample_size, len(train_data)), 
            replace=False
        )
        train_data = train_data.select(indices)
    
    # Prepare batches
    batch_indices = list(range(0, len(train_data), args.batch_size))
    
    # Process each batch
    all_emotion_scores = []
    
    for start_idx in tqdm(batch_indices, desc="Processing batches"):
        end_idx = min(start_idx + args.batch_size, len(train_data))
        batch = train_data[start_idx:end_idx]
        
        # Extract text pairs
        text_pairs = [
            {"premise": premise, "hypothesis": hypothesis}
            for premise, hypothesis in zip(batch["premise"], batch["hypothesis"])
        ]
        
        # Get emotion scores for batch
        batch_scores = get_emotion_score_batch(client, text_pairs, args.model)
        all_emotion_scores.extend(batch_scores)
        
        # Save intermediate results every 50 batches
        if (start_idx // args.batch_size) % 50 == 0 and start_idx > 0:
            interim_save_path = os.path.join(args.output_dir, f"interim_results_{start_idx}.json")
            with open(interim_save_path, "w") as f:
                json.dump(all_emotion_scores[:end_idx], f, indent=2)
            logger.info(f"Saved interim results to {interim_save_path}")
    
    # Calculate sample weights using different strategies and targets
    logger.info("Calculating sample weights using different strategies")
    
    # Define weighting targets, dimensions, and strategies
    targets = ["premise", "hypothesis", "contrast", "average"]
    dimensions = ["Emotional Intensity", "Valence", "Arousal"]
    strategies = ["bell_curve", "linear", "inverse", "uniform"]
    
    # Calculate weights for all combinations
    sample_weights = {}
    
    # First add uniform weights
    sample_weights["uniform"] = np.ones(len(all_emotion_scores)).tolist()
    
    # Then add other combinations
    for target in targets:
        for dimension in dimensions:
            # Skip dimension for contrast since it's a composite score
            if target == "contrast" and dimension != "Emotional Intensity":
                continue
                
            for strategy in strategies:
                if strategy == "uniform":
                    continue  # Already added uniform weights
                
                try:
                    weights = calculate_weights(
                        all_emotion_scores, 
                        strategy=strategy, 
                        target=target, 
                        dimension=dimension
                    )
                    
                    # Create key in the format target_dimension_strategy
                    # Use shortened dimension name for cleaner keys
                    dim_short = dimension.split()[0].lower()  # intensity, valence, arousal
                    key = f"{target}_{dim_short}_{strategy}"
                    sample_weights[key] = weights.tolist()
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate weights for {target}_{dimension}_{strategy}: {str(e)}")
    
    # Create augmented dataset
    augmented_data = {
        "original_data": train_data,
        "emotion_scores": all_emotion_scores,
        "sample_weights": sample_weights
    }
    
    # Save augmented dataset
    output_path = os.path.join(args.output_dir, "mnli_augmented.json")
    with open(output_path, "w") as f:
        json.dump(augmented_data, f, indent=2)
    logger.info(f"Saved augmented dataset to {output_path}")
    
    # Create summary statistics
    summary = {
        "num_samples": len(train_data),
        "premise_emotional_intensity": {
            "mean": np.mean([score.get("premise", {}).get("Emotional Intensity", 0.5) for score in all_emotion_scores]),
            "median": np.median([score.get("premise", {}).get("Emotional Intensity", 0.5) for score in all_emotion_scores]),
            "std": np.std([score.get("premise", {}).get("Emotional Intensity", 0.5) for score in all_emotion_scores])
        },
        "premise_valence": {
            "mean": np.mean([score.get("premise", {}).get("Valence", 0.5) for score in all_emotion_scores]),
            "median": np.median([score.get("premise", {}).get("Valence", 0.5) for score in all_emotion_scores]),
            "std": np.std([score.get("premise", {}).get("Valence", 0.5) for score in all_emotion_scores])
        },
        "premise_arousal": {
            "mean": np.mean([score.get("premise", {}).get("Arousal", 0.5) for score in all_emotion_scores]),
            "median": np.median([score.get("premise", {}).get("Arousal", 0.5) for score in all_emotion_scores]),
            "std": np.std([score.get("premise", {}).get("Arousal", 0.5) for score in all_emotion_scores])
        },
        "hypothesis_emotional_intensity": {
            "mean": np.mean([score.get("hypothesis", {}).get("Emotional Intensity", 0.5) for score in all_emotion_scores]),
            "median": np.median([score.get("hypothesis", {}).get("Emotional Intensity", 0.5) for score in all_emotion_scores]),
            "std": np.std([score.get("hypothesis", {}).get("Emotional Intensity", 0.5) for score in all_emotion_scores])
        },
        "hypothesis_valence": {
            "mean": np.mean([score.get("hypothesis", {}).get("Valence", 0.5) for score in all_emotion_scores]),
            "median": np.median([score.get("hypothesis", {}).get("Valence", 0.5) for score in all_emotion_scores]),
            "std": np.std([score.get("hypothesis", {}).get("Valence", 0.5) for score in all_emotion_scores])
        },
        "hypothesis_arousal": {
            "mean": np.mean([score.get("hypothesis", {}).get("Arousal", 0.5) for score in all_emotion_scores]),
            "median": np.median([score.get("hypothesis", {}).get("Arousal", 0.5) for score in all_emotion_scores]),
            "std": np.std([score.get("hypothesis", {}).get("Arousal", 0.5) for score in all_emotion_scores])
        },
        "emotional_contrast": {
            "mean": np.mean([score.get("contrast", 0.0) for score in all_emotion_scores]),
            "median": np.median([score.get("contrast", 0.0) for score in all_emotion_scores]),
            "std": np.std([score.get("contrast", 0.0) for score in all_emotion_scores])
        }
    }
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "emotion_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary statistics to {summary_path}")

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Set up Claude client
    client = setup_claude_client(args.api_key)
    
    # Augment dataset
    augment_dataset(args, client)
    
    logger.info("Data augmentation completed successfully")

if __name__ == "__main__":
    main() 