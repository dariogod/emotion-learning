#!/usr/bin/env python
# coding: utf-8

"""
Visualization Script for Emotion-Aware NLI Training

This script creates visualizations to compare the performance of different 
emotion-aware training approaches for MNLI.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize and compare emotion-aware NLI training results"
    )
    
    parser.add_argument(
        "--experiment_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Directories containing experiment results"
    )
    
    parser.add_argument(
        "--experiment_names",
        type=str,
        nargs="+",
        help="Names to use for experiments in visualizations"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Output directory for visualizations"
    )
    
    parser.add_argument(
        "--baseline_dir",
        type=str,
        help="Directory for baseline (non-emotion-aware) experiment"
    )
    
    return parser.parse_args()

def load_training_history(experiment_dir: str) -> Dict[str, Any]:
    """
    Load training history from an experiment directory.
    
    Args:
        experiment_dir: Directory containing experiment results
        
    Returns:
        Dictionary with training history
    """
    history_path = os.path.join(experiment_dir, "training_history.json")
    
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history not found at {history_path}")
    
    with open(history_path, "r") as f:
        history = json.load(f)
    
    return history

def plot_learning_curves(experiment_dirs: List[str], experiment_names: List[str], output_dir: str):
    """
    Plot learning curves for multiple experiments.
    
    Args:
        experiment_dirs: List of directories containing experiment results
        experiment_names: Names to use for experiments in visualizations
        output_dir: Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))
    
    # Load and plot training history for each experiment
    for i, (exp_dir, exp_name) in enumerate(zip(experiment_dirs, experiment_names)):
        try:
            history = load_training_history(exp_dir)
            
            # Plot training loss
            plt.subplot(2, 2, 1)
            plt.plot(history["train_loss"], label=exp_name, marker="o", markersize=4, linestyle="-", alpha=0.7)
            plt.title("Training Loss", fontsize=14)
            plt.xlabel("Steps", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Plot evaluation loss
            plt.subplot(2, 2, 2)
            plt.plot(history["eval_loss"], label=exp_name, marker="s", markersize=6, linestyle="-", alpha=0.7)
            plt.title("Evaluation Loss", fontsize=14)
            plt.xlabel("Epochs", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Plot evaluation accuracy
            plt.subplot(2, 2, 3)
            accuracy_values = [metrics["accuracy"] for metrics in history["eval_metrics"]]
            plt.plot(accuracy_values, label=exp_name, marker="^", markersize=6, linestyle="-", alpha=0.7)
            plt.title("Evaluation Accuracy", fontsize=14)
            plt.xlabel("Epochs", fontsize=12)
            plt.ylabel("Accuracy", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Plot F1 macro
            plt.subplot(2, 2, 4)
            f1_values = [metrics["f1_macro"] for metrics in history["eval_metrics"]]
            plt.plot(f1_values, label=exp_name, marker="d", markersize=6, linestyle="-", alpha=0.7)
            plt.title("Evaluation F1 Macro", fontsize=14)
            plt.xlabel("Epochs", fontsize=12)
            plt.ylabel("F1 Score", fontsize=12)
            plt.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error loading history for {exp_name}: {str(e)}")
    
    # Add legends and adjust layout
    for i in range(1, 5):
        plt.subplot(2, 2, i)
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "learning_curves.pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"Saved learning curves to {output_dir}/learning_curves.png")

def plot_final_performance_comparison(experiment_dirs: List[str], experiment_names: List[str], output_dir: str):
    """
    Plot final performance metrics for multiple experiments.
    
    Args:
        experiment_dirs: List of directories containing experiment results
        experiment_names: Names to use for experiments in visualizations
        output_dir: Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 10))
    
    # Metrics to compare
    metric_names = ["accuracy", "f1_contradiction", "f1_entailment", "f1_neutral", "f1_macro"]
    pretty_names = ["Accuracy", "F1 Contradiction", "F1 Entailment", "F1 Neutral", "F1 Macro"]
    
    # Collect final metrics for each experiment
    final_metrics = {name: {} for name in experiment_names}
    
    for i, (exp_dir, exp_name) in enumerate(zip(experiment_dirs, experiment_names)):
        try:
            history = load_training_history(exp_dir)
            
            # Get metrics from the last epoch
            last_metrics = history["eval_metrics"][-1]
            
            for metric in metric_names:
                final_metrics[exp_name][metric] = last_metrics[metric]
                
        except Exception as e:
            print(f"Error loading metrics for {exp_name}: {str(e)}")
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(final_metrics).T
    
    # Plot bar charts for each metric
    for i, (metric, pretty_name) in enumerate(zip(metric_names, pretty_names)):
        plt.subplot(3, 2, i+1)
        sns.barplot(x=metrics_df.index, y=metrics_df[metric], palette="viridis", alpha=0.8)
        plt.title(pretty_name, fontsize=14)
        plt.xlabel("")
        plt.ylabel("Score", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        for j, val in enumerate(metrics_df[metric]):
            plt.text(j, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_performance.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "final_performance.pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"Saved final performance comparison to {output_dir}/final_performance.png")
    
    # Save metrics as CSV for reference
    metrics_df.to_csv(os.path.join(output_dir, "final_metrics.csv"))
    print(f"Saved metrics to {output_dir}/final_metrics.csv")

def plot_learning_rate_vs_metric(experiment_dirs: List[str], experiment_names: List[str], output_dir: str):
    """
    Plot learning rate vs metrics for each experiment.
    
    Args:
        experiment_dirs: List of directories containing experiment results
        experiment_names: Names to use for experiments in visualizations
        output_dir: Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 6))
    
    for i, (exp_dir, exp_name) in enumerate(zip(experiment_dirs, experiment_names)):
        try:
            history = load_training_history(exp_dir)
            
            # Extract learning rates and training loss
            learning_rates = history["learning_rates"]
            train_loss = history["train_loss"]
            
            # Plot learning rate vs training loss
            plt.subplot(1, 2, 1)
            plt.plot(learning_rates, train_loss, label=exp_name, alpha=0.7, marker="o", markersize=4)
            plt.xscale("log")
            plt.title("Learning Rate vs Training Loss", fontsize=14)
            plt.xlabel("Learning Rate", fontsize=12)
            plt.ylabel("Training Loss", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # If we have enough data points, analyze convergence rate
            if len(train_loss) > 10:
                # Calculate smoothed loss differences (convergence rate)
                window_size = min(10, len(train_loss) // 5)
                smoothed_loss = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
                loss_diffs = np.diff(smoothed_loss)
                
                plt.subplot(1, 2, 2)
                plt.plot(loss_diffs, label=exp_name, alpha=0.7)
                plt.title("Loss Change Rate (Convergence Speed)", fontsize=14)
                plt.xlabel("Steps", fontsize=12)
                plt.ylabel("Change in Loss (smoothed)", fontsize=12)
                plt.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error processing learning rate analysis for {exp_name}: {str(e)}")
    
    # Add legends and adjust layout
    for i in range(1, 3):
        plt.subplot(1, 2, i)
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_rate_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved learning rate analysis to {output_dir}/learning_rate_analysis.png")

def analyze_emotional_impact(experiment_dirs: List[str], experiment_names: List[str], output_dir: str):
    """
    Analyze impact of different emotion-aware approaches if emotion score summaries are available.
    
    Args:
        experiment_dirs: List of directories containing experiment results
        experiment_names: Names to use for experiments in visualizations
        output_dir: Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for emotion summary data
    for exp_dir, exp_name in zip(experiment_dirs, experiment_names):
        summary_path = os.path.join(os.path.dirname(exp_dir), "emotion_summary.json")
        
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                emotion_summary = json.load(f)
                
            # Create summary visualization
            plt.figure(figsize=(16, 12))
            
            # Define the statistics to plot and their positions in the grid
            plot_data = [
                {"key": "premise_emotional_intensity", "title": "Premise Emotional Intensity", "pos": 1, "color": "skyblue"},
                {"key": "premise_valence", "title": "Premise Valence", "pos": 2, "color": "lightgreen"},
                {"key": "hypothesis_emotional_intensity", "title": "Hypothesis Emotional Intensity", "pos": 3, "color": "salmon"},
                {"key": "hypothesis_valence", "title": "Hypothesis Valence", "pos": 4, "color": "plum"},
                {"key": "emotional_contrast", "title": "Emotional Contrast", "pos": 5, "color": "orange"}
            ]
            
            # Plot each emotion statistic
            for plot_info in plot_data:
                if plot_info["key"] in emotion_summary:
                    data = emotion_summary[plot_info["key"]]
                    plt.subplot(2, 3, plot_info["pos"])
                    plt.bar(["Mean", "Median", "Std Dev"], 
                            [data["mean"], data["median"], data["std"]], 
                            color=plot_info["color"], alpha=0.7)
                    plt.title(plot_info["title"], fontsize=14)
                    plt.ylabel("Value", fontsize=12)
                    plt.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for i, val in enumerate([data["mean"], data["median"], data["std"]]):
                        plt.text(i, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=10)
            
            # Create a correlation heatmap if we have multiple emotion dimensions
            if len(plot_data) > 2:
                plt.subplot(2, 3, 6)
                
                # Extract mean values to show correlation
                correlation_data = {}
                for plot_info in plot_data:
                    if plot_info["key"] in emotion_summary:
                        correlation_data[plot_info["title"]] = emotion_summary[plot_info["key"]]["mean"]
                
                # Convert to pandas Series for visualization
                correlation_df = pd.Series(correlation_data).reset_index()
                correlation_df.columns = ["Emotion Dimension", "Mean Value"]
                
                # Plot as horizontal bar chart
                sns.barplot(x="Mean Value", y="Emotion Dimension", data=correlation_df, 
                           palette="viridis", orient="h")
                plt.title("Mean Values Comparison", fontsize=14)
                plt.xlabel("Mean Value", fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Add value labels
                for i, val in enumerate(correlation_df["Mean Value"]):
                    plt.text(val + 0.01, i, f"{val:.4f}", va="center", fontsize=10)
            
            plt.suptitle(f"Emotion Statistics for Dataset", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(output_dir, "emotion_statistics.png"), dpi=300, bbox_inches="tight")
            plt.close()
            
            print(f"Saved emotion statistics to {output_dir}/emotion_statistics.png")
            
            # Create a second visualization comparing premise vs hypothesis emotions
            plt.figure(figsize=(14, 6))
            
            # Compare premise vs hypothesis intensity
            plt.subplot(1, 2, 1)
            data = [
                emotion_summary["premise_emotional_intensity"]["mean"],
                emotion_summary["hypothesis_emotional_intensity"]["mean"]
            ]
            plt.bar(["Premise", "Hypothesis"], data, color=["skyblue", "salmon"], alpha=0.7)
            plt.title("Average Emotional Intensity: Premise vs Hypothesis", fontsize=14)
            plt.ylabel("Mean Intensity", fontsize=12)
            plt.grid(True, alpha=0.3)
            for i, val in enumerate(data):
                plt.text(i, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=10)
            
            # Compare premise vs hypothesis valence
            plt.subplot(1, 2, 2)
            data = [
                emotion_summary["premise_valence"]["mean"],
                emotion_summary["hypothesis_valence"]["mean"]
            ]
            plt.bar(["Premise", "Hypothesis"], data, color=["lightgreen", "plum"], alpha=0.7)
            plt.title("Average Emotional Valence: Premise vs Hypothesis", fontsize=14)
            plt.ylabel("Mean Valence", fontsize=12)
            plt.grid(True, alpha=0.3)
            for i, val in enumerate(data):
                plt.text(i, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "premise_vs_hypothesis.png"), dpi=300, bbox_inches="tight")
            plt.close()
            
            print(f"Saved premise vs hypothesis comparison to {output_dir}/premise_vs_hypothesis.png")
            break

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Ensure experiment_names are provided or generate them
    if args.experiment_names is None or len(args.experiment_names) != len(args.experiment_dirs):
        args.experiment_names = [os.path.basename(d) for d in args.experiment_dirs]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot learning curves
    plot_learning_curves(args.experiment_dirs, args.experiment_names, args.output_dir)
    
    # Plot final performance comparison
    plot_final_performance_comparison(args.experiment_dirs, args.experiment_names, args.output_dir)
    
    # Plot learning rate vs metric
    plot_learning_rate_vs_metric(args.experiment_dirs, args.experiment_names, args.output_dir)
    
    # Analyze emotional impact if data is available
    analyze_emotional_impact(args.experiment_dirs, args.experiment_names, args.output_dir)
    
    print("Visualization completed successfully")

if __name__ == "__main__":
    main() 