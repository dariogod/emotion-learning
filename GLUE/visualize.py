#!/usr/bin/env python
# coding: utf-8

"""
Visualization Script for Emotion-Aware Finetuning Results

This script reads metrics JSON files from multiple training runs and 
creates comparison plots to visualize the effect of emotion-aware training.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import glob

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create plots from emotion-aware finetuning metrics files"
    )
    
    parser.add_argument(
        "--metrics_files", 
        type=str,
        nargs="+",
        help="Paths to metrics JSON files to visualize"
    )
    
    parser.add_argument(
        "--metrics_dir", 
        type=str,
        default=None,
        help="Directory containing metrics files (alternative to specifying individual files)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./plots",
        help="Directory to save the plots"
    )
    
    parser.add_argument(
        "--compare_mode",
        type=str,
        choices=["emotion_vs_standard", "all"],
        default="emotion_vs_standard",
        help="How to compare the metrics: 'emotion_vs_standard' (group by emotion/standard) or 'all' (individual runs)"
    )
    
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Apply moving average smoothing with the specified window size"
    )
    
    return parser.parse_args()

def load_metrics_files(metrics_files: List[str] = None, metrics_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load metrics from JSON files.
    
    Args:
        metrics_files: List of file paths to metrics JSON files
        metrics_dir: Directory containing metrics files
    
    Returns:
        List of loaded metrics data dictionaries
    """
    metrics_data = []
    
    # If metrics_dir is provided, find all metrics files in that directory
    if metrics_dir and os.path.isdir(metrics_dir):
        pattern = os.path.join(metrics_dir, "*metrics.json")
        file_paths = glob.glob(pattern)
        print(f"Found {len(file_paths)} metrics files in directory {metrics_dir}")
    else:
        file_paths = metrics_files if metrics_files else []
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Add the filename to the data for reference
                    data['file_path'] = file_path
                    metrics_data.append(data)
                    print(f"Loaded metrics from {file_path}")
            except Exception as e:
                print(f"Error loading metrics from {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    if not metrics_data:
        print("No metrics files were loaded. Please check your file paths.")
    
    return metrics_data

def apply_smoothing(values: List[float], window_size: int) -> List[float]:
    """
    Apply moving average smoothing to a list of values.
    
    Args:
        values: List of values to smooth
        window_size: Size of the moving average window
    
    Returns:
        Smoothed list of values
    """
    if window_size <= 1:
        return values
    
    smoothed = []
    for i in range(len(values)):
        # Calculate window boundaries
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        # Calculate moving average
        window_values = values[start:end]
        smoothed.append(sum(window_values) / len(window_values))
    
    return smoothed

def plot_accuracy_comparison(metrics_data: List[Dict[str, Any]], output_dir: str, 
                            compare_mode: str = "emotion_vs_standard", smooth: int = 0):
    """
    Create a plot comparing accuracy across different runs.
    
    Args:
        metrics_data: List of loaded metrics data
        output_dir: Directory to save the plot
        compare_mode: How to compare the metrics ('emotion_vs_standard' or 'all')
        smooth: Window size for moving average smoothing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    if compare_mode == "emotion_vs_standard":
        # Group runs by emotion/standard
        emotion_runs = [data for data in metrics_data if data['run_info'].get('emotion_used', False)]
        standard_runs = [data for data in metrics_data if not data['run_info'].get('emotion_used', False)]
        
        # Plot emotion runs
        emotion_accuracies = []
        for run in emotion_runs:
            epochs = [m.get('epoch', i+1) for i, m in enumerate(run['metrics'])]
            accuracies = [m.get('eval_accuracy', 0) for m in run['metrics']]
            
            if smooth > 0:
                accuracies = apply_smoothing(accuracies, smooth)
            
            label = f"Emotion Run ({os.path.basename(run['file_path'])})"
            plt.plot(epochs, accuracies, 'r-', alpha=0.3)
            emotion_accuracies.append(accuracies)
        
        # Plot standard runs
        standard_accuracies = []
        for run in standard_runs:
            epochs = [m.get('epoch', i+1) for i, m in enumerate(run['metrics'])]
            accuracies = [m.get('eval_accuracy', 0) for m in run['metrics']]
            
            if smooth > 0:
                accuracies = apply_smoothing(accuracies, smooth)
            
            label = f"Standard Run ({os.path.basename(run['file_path'])})"
            plt.plot(epochs, accuracies, 'b-', alpha=0.3)
            standard_accuracies.append(accuracies)
        
        # Plot averages if we have multiple runs in each category
        if emotion_accuracies:
            # Find the minimum length to align arrays
            min_length = min(len(acc) for acc in emotion_accuracies)
            aligned_accuracies = [acc[:min_length] for acc in emotion_accuracies]
            
            # Calculate average
            if aligned_accuracies:
                avg_emotion = np.mean(aligned_accuracies, axis=0)
                epochs = list(range(1, min_length + 1))
                plt.plot(epochs, avg_emotion, 'r-', linewidth=3, label="Emotion Average")
        
        if standard_accuracies:
            # Find the minimum length to align arrays
            min_length = min(len(acc) for acc in standard_accuracies)
            aligned_accuracies = [acc[:min_length] for acc in standard_accuracies]
            
            # Calculate average
            if aligned_accuracies:
                avg_standard = np.mean(aligned_accuracies, axis=0)
                epochs = list(range(1, min_length + 1))
                plt.plot(epochs, avg_standard, 'b-', linewidth=3, label="Standard Average")
    
    else:  # compare_mode == "all"
        # Plot each run individually
        for data in metrics_data:
            epochs = [m.get('epoch', i+1) for i, m in enumerate(data['metrics'])]
            accuracies = [m.get('eval_accuracy', 0) for m in data['metrics']]
            
            if smooth > 0:
                accuracies = apply_smoothing(accuracies, smooth)
            
            # Determine label and color
            is_emotion = data['run_info'].get('emotion_used', False)
            color = 'r' if is_emotion else 'b'
            label = f"{'Emotion' if is_emotion else 'Standard'} ({os.path.basename(data['file_path'])})"
            
            plt.plot(epochs, accuracies, f'{color}-o', label=label)
    
    plt.title('Accuracy Comparison: Emotion-Aware vs Standard Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(plot_path)
    print(f"Accuracy comparison plot saved to {plot_path}")
    
    plt.close()

def plot_f1_comparison(metrics_data: List[Dict[str, Any]], output_dir: str, 
                      compare_mode: str = "emotion_vs_standard", smooth: int = 0):
    """
    Create a plot comparing F1 scores across different runs.
    
    Args:
        metrics_data: List of loaded metrics data
        output_dir: Directory to save the plot
        compare_mode: How to compare the metrics ('emotion_vs_standard' or 'all')
        smooth: Window size for moving average smoothing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    if compare_mode == "emotion_vs_standard":
        # Group runs by emotion/standard
        emotion_runs = [data for data in metrics_data if data['run_info'].get('emotion_used', False)]
        standard_runs = [data for data in metrics_data if not data['run_info'].get('emotion_used', False)]
        
        # Plot emotion runs
        emotion_f1s = []
        for run in emotion_runs:
            epochs = [m.get('epoch', i+1) for i, m in enumerate(run['metrics'])]
            f1_scores = [m.get('eval_f1_macro', 0) for m in run['metrics']]
            
            if smooth > 0:
                f1_scores = apply_smoothing(f1_scores, smooth)
            
            plt.plot(epochs, f1_scores, 'r-', alpha=0.3)
            emotion_f1s.append(f1_scores)
        
        # Plot standard runs
        standard_f1s = []
        for run in standard_runs:
            epochs = [m.get('epoch', i+1) for i, m in enumerate(run['metrics'])]
            f1_scores = [m.get('eval_f1_macro', 0) for m in run['metrics']]
            
            if smooth > 0:
                f1_scores = apply_smoothing(f1_scores, smooth)
            
            plt.plot(epochs, f1_scores, 'b-', alpha=0.3)
            standard_f1s.append(f1_scores)
        
        # Plot averages if we have multiple runs in each category
        if emotion_f1s:
            # Find the minimum length to align arrays
            min_length = min(len(f1) for f1 in emotion_f1s)
            aligned_f1s = [f1[:min_length] for f1 in emotion_f1s]
            
            # Calculate average
            if aligned_f1s:
                avg_emotion = np.mean(aligned_f1s, axis=0)
                epochs = list(range(1, min_length + 1))
                plt.plot(epochs, avg_emotion, 'r-', linewidth=3, label="Emotion Average")
        
        if standard_f1s:
            # Find the minimum length to align arrays
            min_length = min(len(f1) for f1 in standard_f1s)
            aligned_f1s = [f1[:min_length] for f1 in standard_f1s]
            
            # Calculate average
            if aligned_f1s:
                avg_standard = np.mean(aligned_f1s, axis=0)
                epochs = list(range(1, min_length + 1))
                plt.plot(epochs, avg_standard, 'b-', linewidth=3, label="Standard Average")
    
    else:  # compare_mode == "all"
        # Plot each run individually
        for data in metrics_data:
            epochs = [m.get('epoch', i+1) for i, m in enumerate(data['metrics'])]
            f1_scores = [m.get('eval_f1_macro', 0) for m in data['metrics']]
            
            if smooth > 0:
                f1_scores = apply_smoothing(f1_scores, smooth)
            
            # Determine label and color
            is_emotion = data['run_info'].get('emotion_used', False)
            color = 'r' if is_emotion else 'b'
            label = f"{'Emotion' if is_emotion else 'Standard'} ({os.path.basename(data['file_path'])})"
            
            plt.plot(epochs, f1_scores, f'{color}-o', label=label)
    
    plt.title('F1 Score Comparison: Emotion-Aware vs Standard Training')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Macro Score')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'f1_comparison.png')
    plt.savefig(plot_path)
    print(f"F1 comparison plot saved to {plot_path}")
    
    plt.close()

def plot_per_class_f1(metrics_data: List[Dict[str, Any]], output_dir: str, 
                     compare_mode: str = "emotion_vs_standard", smooth: int = 0):
    """
    Create plots comparing F1 scores per class across different runs.
    
    Args:
        metrics_data: List of loaded metrics data
        output_dir: Directory to save the plots
        compare_mode: How to compare the metrics ('emotion_vs_standard' or 'all')
        smooth: Window size for moving average smoothing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have per-class F1 scores in all metrics
    class_keys = ["eval_f1_contradiction", "eval_f1_entailment", "eval_f1_neutral"]
    all_have_class_f1 = all(
        all(key in m for key in class_keys) 
        for data in metrics_data 
        for m in data['metrics']
    )
    
    if not all_have_class_f1:
        print("Not all metrics contain per-class F1 scores. Skipping per-class F1 plot.")
        return
    
    classes = ["Contradiction", "Entailment", "Neutral"]
    metrics_keys = class_keys
    
    for cls, key in zip(classes, metrics_keys):
        plt.figure(figsize=(12, 8))
        
        if compare_mode == "emotion_vs_standard":
            # Group runs by emotion/standard
            emotion_runs = [data for data in metrics_data if data['run_info'].get('emotion_used', False)]
            standard_runs = [data for data in metrics_data if not data['run_info'].get('emotion_used', False)]
            
            # Plot emotion runs
            emotion_f1s = []
            for run in emotion_runs:
                epochs = [m.get('epoch', i+1) for i, m in enumerate(run['metrics'])]
                f1_scores = [m.get(key, 0) for m in run['metrics']]
                
                if smooth > 0:
                    f1_scores = apply_smoothing(f1_scores, smooth)
                
                plt.plot(epochs, f1_scores, 'r-', alpha=0.3)
                emotion_f1s.append(f1_scores)
            
            # Plot standard runs
            standard_f1s = []
            for run in standard_runs:
                epochs = [m.get('epoch', i+1) for i, m in enumerate(run['metrics'])]
                f1_scores = [m.get(key, 0) for m in run['metrics']]
                
                if smooth > 0:
                    f1_scores = apply_smoothing(f1_scores, smooth)
                
                plt.plot(epochs, f1_scores, 'b-', alpha=0.3)
                standard_f1s.append(f1_scores)
            
            # Plot averages if we have multiple runs in each category
            if emotion_f1s:
                # Find the minimum length to align arrays
                min_length = min(len(f1) for f1 in emotion_f1s)
                aligned_f1s = [f1[:min_length] for f1 in emotion_f1s]
                
                # Calculate average
                if aligned_f1s:
                    avg_emotion = np.mean(aligned_f1s, axis=0)
                    epochs = list(range(1, min_length + 1))
                    plt.plot(epochs, avg_emotion, 'r-', linewidth=3, label=f"Emotion Average")
            
            if standard_f1s:
                # Find the minimum length to align arrays
                min_length = min(len(f1) for f1 in standard_f1s)
                aligned_f1s = [f1[:min_length] for f1 in standard_f1s]
                
                # Calculate average
                if aligned_f1s:
                    avg_standard = np.mean(aligned_f1s, axis=0)
                    epochs = list(range(1, min_length + 1))
                    plt.plot(epochs, avg_standard, 'b-', linewidth=3, label=f"Standard Average")
        
        else:  # compare_mode == "all"
            # Plot each run individually
            for data in metrics_data:
                epochs = [m.get('epoch', i+1) for i, m in enumerate(data['metrics'])]
                f1_scores = [m.get(key, 0) for m in data['metrics']]
                
                if smooth > 0:
                    f1_scores = apply_smoothing(f1_scores, smooth)
                
                # Determine label and color
                is_emotion = data['run_info'].get('emotion_used', False)
                color = 'r' if is_emotion else 'b'
                label = f"{'Emotion' if is_emotion else 'Standard'} ({os.path.basename(data['file_path'])})"
                
                plt.plot(epochs, f1_scores, f'{color}-o', label=label)
        
        plt.title(f'F1 Score for {cls} Class: Emotion-Aware vs Standard Training')
        plt.xlabel('Epoch')
        plt.ylabel(f'F1 Score ({cls})')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        plot_path = os.path.join(output_dir, f'f1_{cls.lower()}_comparison.png')
        plt.savefig(plot_path)
        print(f"F1 score for {cls} class comparison plot saved to {plot_path}")
        
        plt.close()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics files
    metrics_data = load_metrics_files(args.metrics_files, args.metrics_dir)
    
    if not metrics_data:
        print("No metrics data to visualize. Exiting.")
        return
    
    # Create plots
    plot_accuracy_comparison(metrics_data, args.output_dir, args.compare_mode, args.smooth)
    plot_f1_comparison(metrics_data, args.output_dir, args.compare_mode, args.smooth)
    plot_per_class_f1(metrics_data, args.output_dir, args.compare_mode, args.smooth)
    
    print(f"All plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
