"""
Evaluation script for HRI30 Action Recognition
Implements comprehensive evaluation metrics as described in the paper
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, top_k_accuracy_score
)
from typing import Dict, List, Tuple, Optional, Any
import json
from tqdm import tqdm
import argparse
from pathlib import Path

from config import get_config_for_device, DataConfig
from models import create_model
from data_loader import HRI30DataModule


class HRI30Evaluator:
    """
    Comprehensive evaluator for HRI30 Action Recognition
    Implements evaluation metrics mentioned in the paper
    """
    
    def __init__(
        self,
        model_path: str,
        data_config: DataConfig,
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.data_config = data_config
        self.device = device
        
        # Load model
        self.model = self._load_model()
        
        # Results storage
        self.results = {
            'predictions': [],
            'targets': [],
            'class_names': data_config.action_classes,
            'probabilities': []
        }
    
    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint"""
        print(f"Loading model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Recreate model from saved config
        from config import ModelConfig
        model_config_dict = checkpoint['model_config']
        model_config = ModelConfig(**model_config_dict)
        
        model = create_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Best validation accuracy from training: {checkpoint.get('best_val_accuracy', 'N/A'):.2f}%")
        
        return model
    
    def evaluate_split(self, split_id: int = 1) -> Dict[str, Any]:
        """
        Evaluate on specific train/test split
        HRI30 has 3 splits as mentioned in Table II
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING ON SPLIT {split_id}")
        print(f"{'='*60}")
        
        # Create data module for this split
        from config import ModelConfig
        model_config = ModelConfig()  # Default config
        data_module = HRI30DataModule(self.data_config, model_config, split_id=split_id)
        
        # Get test dataloader
        test_loader = data_module.get_val_dataloader()
        
        print(f"Test dataset size: {len(test_loader.dataset)}")
        
        # Reset results
        self.results = {
            'predictions': [],
            'targets': [],
            'class_names': self.data_config.action_classes,
            'probabilities': []
        }
        
        # Evaluate
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Evaluating Split {split_id}")
            
            for videos, labels in pbar:
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(videos)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get predictions
                _, predictions = torch.max(outputs, 1)
                
                # Store results
                self.results['predictions'].extend(predictions.cpu().numpy())
                self.results['targets'].extend(labels.cpu().numpy())
                self.results['probabilities'].extend(probabilities.cpu().numpy())
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        print(f"\nSplit {split_id} Results:")
        print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
        print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        
        return metrics
    
    def evaluate_all_splits(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate on all 3 splits as done in the paper"""
        all_results = {}
        
        for split_id in range(1, 4):
            split_results = self.evaluate_split(split_id)
            all_results[f'split_{split_id}'] = split_results
        
        # Compute average metrics across splits
        avg_results = self._compute_average_metrics(all_results)
        all_results['average'] = avg_results
        
        self._print_comparison_table(all_results)
        
        return all_results
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics"""
        predictions = np.array(self.results['predictions'])
        targets = np.array(self.results['targets'])
        probabilities = np.array(self.results['probabilities'])
        
        if len(predictions) == 0:
            return {}
        
        # Top-1 Accuracy (primary metric from paper)
        top1_accuracy = accuracy_score(targets, predictions) * 100
        
        # Top-5 Accuracy (secondary metric from paper)
        # For top-k accuracy, we need the probability scores
        try:
            top5_accuracy = top_k_accuracy_score(targets, probabilities, k=5) * 100
        except:
            # Fallback if sklearn version doesn't support top_k_accuracy_score
            top5_accuracy = self._compute_top_k_accuracy_manual(probabilities, targets, k=5) * 100
        
        # Precision, Recall, F1-Score
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Weighted averages
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        # Per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(self.data_config.action_classes):
            class_metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        # Category-wise analysis
        category_metrics = self._compute_category_metrics(predictions, targets)
        
        return {
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'class_metrics': class_metrics,
            'category_metrics': category_metrics,
            'confusion_matrix': confusion_matrix(targets, predictions).tolist(),
            'num_samples': len(predictions)
        }
    
    def _compute_top_k_accuracy_manual(self, probabilities: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
        """Manual computation of top-k accuracy"""
        correct = 0
        for prob, target in zip(probabilities, targets):
            top_k_preds = np.argsort(prob)[-k:]
            if target in top_k_preds:
                correct += 1
        return correct / len(targets)
    
    def _compute_category_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each action category
        Categories: Human-Object Interaction, Body-Motion Only, Human-Robot Collaboration
        """
        category_metrics = {}
        
        for category, class_names in self.data_config.action_categories.items():
            # Get indices for this category
            category_indices = [self.data_config.class_to_idx[cls] for cls in class_names if cls in self.data_config.class_to_idx]
            
            if not category_indices:
                continue
            
            # Filter predictions and targets for this category
            category_mask = np.isin(targets, category_indices)
            if not np.any(category_mask):
                continue
            
            category_predictions = predictions[category_mask]
            category_targets = targets[category_mask]
            
            # Compute accuracy for this category
            accuracy = accuracy_score(category_targets, category_predictions) * 100
            
            # Compute F1 score
            try:
                _, _, f1, _ = precision_recall_fscore_support(
                    category_targets, category_predictions, average='macro', zero_division=0
                )
                f1_score = f1
            except:
                f1_score = 0.0
            
            category_metrics[category] = {
                'accuracy': accuracy,
                'f1_score': f1_score,
                'num_samples': int(np.sum(category_mask))
            }
        
        return category_metrics
    
    def _compute_average_metrics(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Compute average metrics across all splits"""
        metrics_to_average = ['top1_accuracy', 'top5_accuracy', 'macro_f1', 'weighted_f1']
        
        avg_metrics = {}
        for metric in metrics_to_average:
            values = [all_results[f'split_{i}'][metric] for i in range(1, 4) if f'split_{i}' in all_results]
            if values:
                avg_metrics[metric] = np.mean(values)
                avg_metrics[f'{metric}_std'] = np.std(values)
        
        return avg_metrics
    
    def _print_comparison_table(self, all_results: Dict[str, Dict[str, Any]]):
        """Print comparison table similar to Table III in the paper"""
        print(f"\n{'='*80}")
        print("COMPARISON WITH PAPER RESULTS (Table III)")
        print(f"{'='*80}")
        
        # Paper results for comparison (SlowOnly with Kinetics-400 pretraining)
        paper_results = {
            'split_1': {'top1_accuracy': 86.55, 'top5_accuracy': 99.76},
            'split_2': {'top1_accuracy': 83.49, 'top5_accuracy': 99.84},
            'split_3': {'top1_accuracy': 82.43, 'top5_accuracy': 99.90}
        }
        
        print(f"{'Split':<10} {'Top-1 Acc (Paper)':<20} {'Top-1 Acc (Ours)':<20} {'Top-5 Acc (Paper)':<20} {'Top-5 Acc (Ours)':<20}")
        print("-" * 90)
        
        for split_id in range(1, 4):
            split_key = f'split_{split_id}'
            if split_key in all_results:
                our_top1 = all_results[split_key]['top1_accuracy']
                our_top5 = all_results[split_key]['top5_accuracy']
                paper_top1 = paper_results[split_key]['top1_accuracy']
                paper_top5 = paper_results[split_key]['top5_accuracy']
                
                print(f"{split_id:<10} {paper_top1:<20.2f} {our_top1:<20.2f} {paper_top5:<20.2f} {our_top5:<20.2f}")
        
        # Average results
        if 'average' in all_results:
            avg = all_results['average']
            paper_avg_top1 = np.mean([paper_results[f'split_{i}']['top1_accuracy'] for i in range(1, 4)])
            paper_avg_top5 = np.mean([paper_results[f'split_{i}']['top5_accuracy'] for i in range(1, 4)])
            
            print("-" * 90)
            print(f"{'Average':<10} {paper_avg_top1:<20.2f} {avg['top1_accuracy']:<20.2f} {paper_avg_top5:<20.2f} {avg['top5_accuracy']:<20.2f}")
    
    def create_visualizations(self, save_dir: str):
        """Create comprehensive visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        predictions = np.array(self.results['predictions'])
        targets = np.array(self.results['targets'])
        
        if len(predictions) == 0:
            print("No data to visualize")
            return
        
        # 1. Confusion Matrix
        plt.figure(figsize=(20, 16))
        cm = confusion_matrix(targets, predictions)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=False,
            cmap='Blues',
            xticklabels=self.data_config.action_classes,
            yticklabels=self.data_config.action_classes
        )
        plt.title('Confusion Matrix (Normalized)', fontsize=16)
        plt.xlabel('Predicted Class', fontsize=14)
        plt.ylabel('True Class', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-class accuracy
        class_accuracies = []
        for i, class_name in enumerate(self.data_config.action_classes):
            class_mask = (targets == i)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(targets[class_mask], predictions[class_mask]) * 100
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        plt.figure(figsize=(25, 8))
        bars = plt.bar(range(len(self.data_config.action_classes)), class_accuracies)
        plt.title('Per-Class Accuracy', fontsize=16)
        plt.xlabel('Action Class', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.xticks(range(len(self.data_config.action_classes)), self.data_config.action_classes, rotation=45, ha='right')
        
        # Color bars by category
        category_colors = {'Human_Object_Interaction': 'red', 'Body_Motion_Only': 'blue', 'Human_Robot_Collaboration': 'green'}
        for i, bar in enumerate(bars):
            class_name = self.data_config.action_classes[i]
            for category, classes in self.data_config.action_categories.items():
                if class_name in classes:
                    bar.set_color(category_colors.get(category, 'gray'))
                    break
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=category.replace('_', ' ')) 
                          for category, color in category_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Category-wise performance
        metrics = self._compute_metrics()
        category_metrics = metrics['category_metrics']
        
        if category_metrics:
            categories = list(category_metrics.keys())
            accuracies = [category_metrics[cat]['accuracy'] for cat in categories]
            f1_scores = [category_metrics[cat]['f1_score'] * 100 for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', alpha=0.8)
            ax.bar(x + width/2, f1_scores, width, label='F1-Score (%)', alpha=0.8)
            
            ax.set_xlabel('Action Category')
            ax.set_ylabel('Score (%)')
            ax.set_title('Performance by Action Category')
            ax.set_xticks(x)
            ax.set_xticklabels([cat.replace('_', ' ') for cat in categories])
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'category_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {save_dir}")
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """Save evaluation results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = self._convert_to_serializable(results)
        
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {save_path}")
    
    def _convert_to_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description='Evaluate HRI30 Action Recognition Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--split_id', type=int, default=0, choices=[0, 1, 2, 3],
                       help='Split to evaluate (0 for all splits, 1-3 for specific split)')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config_for_device(args.device)
    data_config = config['data']
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create evaluator
    evaluator = HRI30Evaluator(args.model_path, data_config, device)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Evaluate
    if args.split_id == 0:
        # Evaluate all splits
        results = evaluator.evaluate_all_splits()
        evaluator.save_results(results, os.path.join(args.save_dir, 'evaluation_all_splits.json'))
    else:
        # Evaluate specific split
        results = evaluator.evaluate_split(args.split_id)
        evaluator.save_results(results, os.path.join(args.save_dir, f'evaluation_split_{args.split_id}.json'))
    
    # Create visualizations
    evaluator.create_visualizations(args.save_dir)
    
    print(f"\nEvaluation completed! Results saved in {args.save_dir}")


if __name__ == "__main__":
    main()