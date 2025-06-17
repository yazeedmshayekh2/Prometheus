#!/usr/bin/env python3
"""
Fine-tuning script for FAQ embedding model using sentence-transformers
Specialized for insurance domain FAQ retrieval with Qwen-generated training data
"""

import os
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from collections import defaultdict

from sentence_transformers import SentenceTransformer, losses, evaluation
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report

# Optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available. Evaluation plots will be skipped.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('faq_finetuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FAQEmbeddingFineTuner:
    def __init__(self, 
                 base_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 output_dir: str = "finetuned_faq_model",
                 max_seq_length: int = 512):
        """
        Initialize the FAQ embedding fine-tuner
        
        Args:
            base_model_name: Base sentence-transformers model to fine-tune
            output_dir: Directory to save the fine-tuned model
            max_seq_length: Maximum sequence length for the model
        """
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_seq_length = max_seq_length
        
        # Initialize model
        logger.info(f"Loading base model: {base_model_name}")
        self.model = SentenceTransformer(base_model_name)
        self.model.max_seq_length = max_seq_length
        
        # Training parameters
        self.batch_size = 16
        self.num_epochs = 4
        self.learning_rate = 2e-5
        self.warmup_steps_ratio = 0.1
        
        # Dataset statistics
        self.dataset_stats = {}
        
        logger.info(f"Model loaded successfully. Max sequence length: {max_seq_length}")

    def load_training_data(self, data_dir: str) -> tuple:
        """
        Load training, validation, and test data with enhanced statistics
        
        Args:
            data_dir: Directory containing the training data files
            
        Returns:
            Tuple of (train_examples, val_examples, test_examples)
        """
        data_path = Path(data_dir)
        
        # Load dataset statistics if available
        stats_file = data_path / "dataset_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.dataset_stats = json.load(f)
                logger.info(f"Dataset generated using: {self.dataset_stats.get('generation_method', 'unknown')}")
                logger.info(f"Model used for generation: {self.dataset_stats.get('model_used', 'unknown')}")
        
        # Load datasets
        train_examples = self._load_examples(data_path / "train.json")
        val_examples = self._load_examples(data_path / "val.json")
        test_examples = self._load_examples(data_path / "test.json")
        
        # Analyze dataset composition
        self._analyze_dataset_composition(train_examples, val_examples, test_examples)
        
        logger.info(f"Loaded {len(train_examples)} training examples")
        logger.info(f"Loaded {len(val_examples)} validation examples")
        logger.info(f"Loaded {len(test_examples)} test examples")
        
        return train_examples, val_examples, test_examples

    def _load_examples(self, file_path: Path) -> List[InputExample]:
        """Load examples from JSON file with enhanced validation"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        skipped_examples = 0
        
        for i, item in enumerate(data):
            # Validate the example
            if not item.get('text1') or not item.get('text2'):
                skipped_examples += 1
                continue
                
            # Ensure similarity score is valid
            similarity_score = float(item.get('similarity_score', 0.5))
            if similarity_score < 0 or similarity_score > 1:
                similarity_score = max(0, min(1, similarity_score))  # Clamp to [0,1]
            
            examples.append(InputExample(
                texts=[item['text1'], item['text2']],
                label=similarity_score,
                guid=f"example_{i}"
            ))
        
        if skipped_examples > 0:
            logger.warning(f"Skipped {skipped_examples} invalid examples from {file_path.name}")
        
        return examples

    def _analyze_dataset_composition(self, train_examples: List[InputExample], 
                                   val_examples: List[InputExample], 
                                   test_examples: List[InputExample]):
        """Analyze the composition of the dataset"""
        
        all_examples = train_examples + val_examples + test_examples
        
        # Analyze similarity score distribution
        scores = [example.label for example in all_examples]
        score_stats = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'high_similarity': sum(1 for s in scores if s > 0.8),
            'medium_similarity': sum(1 for s in scores if 0.4 <= s <= 0.8),
            'low_similarity': sum(1 for s in scores if s < 0.4)
        }
        
        logger.info("Dataset Composition Analysis:")
        logger.info(f"  Similarity Score Distribution:")
        logger.info(f"    Mean: {score_stats['mean']:.3f} ± {score_stats['std']:.3f}")
        logger.info(f"    Range: [{score_stats['min']:.3f}, {score_stats['max']:.3f}]")
        logger.info(f"    High similarity (>0.8): {score_stats['high_similarity']} ({score_stats['high_similarity']/len(scores)*100:.1f}%)")
        logger.info(f"    Medium similarity (0.4-0.8): {score_stats['medium_similarity']} ({score_stats['medium_similarity']/len(scores)*100:.1f}%)")
        logger.info(f"    Low similarity (<0.4): {score_stats['low_similarity']} ({score_stats['low_similarity']/len(scores)*100:.1f}%)")
        
        # Analyze text lengths
        text1_lengths = [len(example.texts[0].split()) for example in all_examples]
        text2_lengths = [len(example.texts[1].split()) for example in all_examples]
        
        logger.info(f"  Text Length Analysis:")
        logger.info(f"    Text1 avg length: {np.mean(text1_lengths):.1f} words")
        logger.info(f"    Text2 avg length: {np.mean(text2_lengths):.1f} words")
        
        # Detect language distribution (simple heuristic)
        arabic_count = sum(1 for example in all_examples 
                          if self._contains_arabic(example.texts[0]) or self._contains_arabic(example.texts[1]))
        
        logger.info(f"  Language Distribution:")
        logger.info(f"    Pairs with Arabic: {arabic_count} ({arabic_count/len(all_examples)*100:.1f}%)")
        logger.info(f"    English-only pairs: {len(all_examples) - arabic_count} ({(len(all_examples) - arabic_count)/len(all_examples)*100:.1f}%)")

    def _contains_arabic(self, text: str) -> bool:
        """Simple check if text contains Arabic characters"""
        return any('\u0600' <= char <= '\u06FF' for char in text)

    def create_evaluator(self, val_examples: List[InputExample]) -> EmbeddingSimilarityEvaluator:
        """Create evaluator for monitoring training progress"""
        
        # Convert examples to the format expected by EmbeddingSimilarityEvaluator
        sentences1 = [example.texts[0] for example in val_examples]
        sentences2 = [example.texts[1] for example in val_examples]
        scores = [example.label for example in val_examples]
        
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=sentences1,
            sentences2=sentences2,
            scores=scores,
            main_similarity=SimilarityFunction.COSINE,
            name="faq_validation"
        )
        
        return evaluator

    def create_advanced_loss(self, train_examples: List[InputExample]):
        """
        Create an advanced loss function based on dataset characteristics
        """
        
        # Analyze the score distribution to choose the best loss function
        scores = [example.label for example in train_examples]
        high_similarity_ratio = sum(1 for s in scores if s > 0.8) / len(scores)
        
        logger.info(f"High similarity ratio: {high_similarity_ratio:.3f}")
        
        if high_similarity_ratio > 0.6:
            # Many high-similarity pairs - use CosineSimilarityLoss
            logger.info("Using CosineSimilarityLoss (many high-similarity pairs)")
            return losses.CosineSimilarityLoss(self.model)
        elif high_similarity_ratio < 0.3:
            # Many negative pairs - use ContrastiveLoss
            logger.info("Using ContrastiveLoss (many negative pairs)")
            return losses.ContrastiveLoss(self.model)
        else:
            # Balanced dataset - use CosineSimilarityLoss with margin
            logger.info("Using CosineSimilarityLoss (balanced dataset)")
            return losses.CosineSimilarityLoss(self.model)

    def train(self, train_examples: List[InputExample], val_examples: List[InputExample]):
        """
        Fine-tune the model on FAQ data with enhanced training
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples
        """
        
        # Create DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.batch_size)
        
        # Create advanced loss function
        train_loss = self.create_advanced_loss(train_examples)
        
        # Calculate warmup steps
        num_train_steps = len(train_dataloader) * self.num_epochs
        warmup_steps = int(num_train_steps * self.warmup_steps_ratio)
        
        # Create evaluator
        evaluator = self.create_evaluator(val_examples)
        
        # Training arguments
        training_args = {
            'train_objectives': [(train_dataloader, train_loss)],
            'evaluator': evaluator,
            'epochs': self.num_epochs,
            'warmup_steps': warmup_steps,
            'optimizer_params': {'lr': self.learning_rate},
            'evaluation_steps': max(1, len(train_dataloader) // 4),  # Evaluate 4 times per epoch
            'output_path': str(self.output_dir),
            'save_best_model': True,
            'show_progress_bar': True,
            'checkpoint_path': str(self.output_dir / "checkpoints"),
            'checkpoint_save_steps': len(train_dataloader) // 2,  # Save checkpoint twice per epoch
            'checkpoint_save_total_limit': 3
        }
        
        logger.info("Starting fine-tuning...")
        logger.info(f"Training parameters:")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Epochs: {self.num_epochs}")
        logger.info(f"  - Learning rate: {self.learning_rate}")
        logger.info(f"  - Warmup steps: {warmup_steps}")
        logger.info(f"  - Total training steps: {num_train_steps}")
        logger.info(f"  - Loss function: {type(train_loss).__name__}")
        
        # Start training
        self.model.fit(**training_args)
        
        logger.info("Fine-tuning completed!")
        logger.info(f"Model saved to: {self.output_dir}")

    def evaluate_model(self, test_examples: List[InputExample]) -> Dict[str, float]:
        """
        Comprehensive evaluation of the fine-tuned model
        
        Args:
            test_examples: Test examples
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating fine-tuned model...")
        
        # Create evaluator for test data
        evaluator = self.create_evaluator(test_examples)
        
        # Evaluate
        eval_result = evaluator(self.model)
        # Extract the main score from the evaluation result
        if isinstance(eval_result, dict):
            spearman_score = eval_result.get('spearman_cosine', eval_result.get('cosine_spearman', 0.0))
        else:
            spearman_score = eval_result
        
        # Enhanced evaluation with more metrics
        sentences1 = [example.texts[0] for example in test_examples]
        sentences2 = [example.texts[1] for example in test_examples]
        true_scores = [example.label for example in test_examples]
        
        # Get embeddings
        logger.info("Generating embeddings for evaluation...")
        embeddings1 = self.model.encode(sentences1, show_progress_bar=True)
        embeddings2 = self.model.encode(sentences2, show_progress_bar=True)
        
        # Calculate cosine similarities
        predicted_scores = [
            cosine_similarity([emb1], [emb2])[0][0] 
            for emb1, emb2 in zip(embeddings1, embeddings2)
        ]
        
        # Calculate various correlation metrics
        try:
            pearson_correlation = np.corrcoef(true_scores, predicted_scores)[0, 1]
            if np.isnan(pearson_correlation):
                pearson_correlation = 0.0
        except Exception as e:
            logger.warning(f"Could not calculate Pearson correlation: {e}")
            pearson_correlation = 0.0
        
        # Calculate accuracy for different similarity thresholds
        high_sim_accuracy = self._calculate_threshold_accuracy(true_scores, predicted_scores, 0.8)
        medium_sim_accuracy = self._calculate_threshold_accuracy(true_scores, predicted_scores, 0.5)
        low_sim_accuracy = self._calculate_threshold_accuracy(true_scores, predicted_scores, 0.3)
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(np.array(true_scores) - np.array(predicted_scores)))
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((np.array(true_scores) - np.array(predicted_scores))**2))
        
        # Language-specific evaluation
        english_metrics = self._evaluate_by_language(test_examples, predicted_scores, is_arabic=False)
        arabic_metrics = self._evaluate_by_language(test_examples, predicted_scores, is_arabic=True)
        
        metrics = {
            'spearman_correlation': float(spearman_score),
            'pearson_correlation': float(pearson_correlation),
            'high_similarity_accuracy': high_sim_accuracy,
            'medium_similarity_accuracy': medium_sim_accuracy,
            'low_similarity_accuracy': low_sim_accuracy,
            'mean_absolute_error': float(mae),
            'root_mean_square_error': float(rmse),
            'test_samples': len(test_examples),
            'english_correlation': english_metrics['correlation'],
            'arabic_correlation': arabic_metrics['correlation'],
            'english_samples': english_metrics['count'],
            'arabic_samples': arabic_metrics['count']
        }
        
        # Save evaluation results
        with open(self.output_dir / "evaluation_results.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create evaluation plots
        self._create_evaluation_plots(true_scores, predicted_scores)
        
        logger.info("Evaluation results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
        
        return metrics

    def _calculate_threshold_accuracy(self, true_scores: List[float], 
                                    predicted_scores: List[float], 
                                    threshold: float) -> float:
        """Calculate accuracy for a specific similarity threshold"""
        
        correct = 0
        total = 0
        
        for true_score, pred_score in zip(true_scores, predicted_scores):
            if true_score >= threshold or true_score <= (1 - threshold):
                total += 1
                if (true_score >= threshold and pred_score >= threshold) or \
                   (true_score <= (1 - threshold) and pred_score <= (1 - threshold)):
                    correct += 1
        
        return correct / total if total > 0 else 0.0

    def _evaluate_by_language(self, test_examples: List[InputExample], 
                            predicted_scores: List[float], 
                            is_arabic: bool) -> Dict[str, Any]:
        """Evaluate performance by language"""
        
        language_examples = []
        language_predictions = []
        language_true_scores = []
        
        for i, example in enumerate(test_examples):
            has_arabic = (self._contains_arabic(example.texts[0]) or 
                         self._contains_arabic(example.texts[1]))
            
            if (is_arabic and has_arabic) or (not is_arabic and not has_arabic):
                language_examples.append(example)
                language_predictions.append(predicted_scores[i])
                language_true_scores.append(example.label)
        
        if len(language_true_scores) > 1:
            try:
                correlation = np.corrcoef(language_true_scores, language_predictions)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except Exception:
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            'correlation': float(correlation),
            'count': len(language_examples)
        }

    def _create_evaluation_plots(self, true_scores: List[float], predicted_scores: List[float]):
        """Create evaluation plots"""
        
        if not VISUALIZATION_AVAILABLE:
            logger.info("Visualization libraries not available. Skipping plots.")
            return
        
        try:
            # Create scatter plot
            plt.figure(figsize=(12, 5))
            
            # Scatter plot
            plt.subplot(1, 2, 1)
            plt.scatter(true_scores, predicted_scores, alpha=0.6)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel('True Similarity Scores')
            plt.ylabel('Predicted Similarity Scores')
            plt.title('True vs Predicted Similarity Scores')
            plt.grid(True, alpha=0.3)
            
            # Distribution plot
            plt.subplot(1, 2, 2)
            plt.hist(true_scores, alpha=0.5, label='True Scores', bins=20)
            plt.hist(predicted_scores, alpha=0.5, label='Predicted Scores', bins=20)
            plt.xlabel('Similarity Scores')
            plt.ylabel('Frequency')
            plt.title('Score Distribution Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "evaluation_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Evaluation plots saved to: {self.output_dir / 'evaluation_plots.png'}")
            
        except Exception as e:
            logger.warning(f"Could not create evaluation plots: {e}")

    def compare_with_base_model(self, test_examples: List[InputExample]) -> Dict[str, Any]:
        """
        Compare fine-tuned model with base model
        
        Args:
            test_examples: Test examples for comparison
            
        Returns:
            Comparison results
        """
        logger.info("Comparing with base model...")
        
        # Load base model
        base_model = SentenceTransformer(self.base_model_name)
        
        # Sample test data for comparison (use more samples if available)
        sample_size = min(100, len(test_examples))
        sample_examples = test_examples[:sample_size]
        
        sentences1 = [example.texts[0] for example in sample_examples]
        sentences2 = [example.texts[1] for example in sample_examples]
        true_scores = [example.label for example in sample_examples]
        
        # Get embeddings from both models
        logger.info("Generating base model embeddings...")
        base_embeddings1 = base_model.encode(sentences1, show_progress_bar=True)
        base_embeddings2 = base_model.encode(sentences2, show_progress_bar=True)
        
        logger.info("Generating fine-tuned model embeddings...")
        finetuned_embeddings1 = self.model.encode(sentences1, show_progress_bar=True)
        finetuned_embeddings2 = self.model.encode(sentences2, show_progress_bar=True)
        
        # Calculate similarities
        base_similarities = [
            cosine_similarity([emb1], [emb2])[0][0]
            for emb1, emb2 in zip(base_embeddings1, base_embeddings2)
        ]
        
        finetuned_similarities = [
            cosine_similarity([emb1], [emb2])[0][0]
            for emb1, emb2 in zip(finetuned_embeddings1, finetuned_embeddings2)
        ]
        
        # Calculate correlations
        try:
            base_correlation = np.corrcoef(true_scores, base_similarities)[0, 1]
            if np.isnan(base_correlation):
                base_correlation = 0.0
        except Exception:
            base_correlation = 0.0
            
        try:
            finetuned_correlation = np.corrcoef(true_scores, finetuned_similarities)[0, 1]
            if np.isnan(finetuned_correlation):
                finetuned_correlation = 0.0
        except Exception:
            finetuned_correlation = 0.0
        
        # Calculate MAE for both models
        base_mae = np.mean(np.abs(np.array(true_scores) - np.array(base_similarities)))
        finetuned_mae = np.mean(np.abs(np.array(true_scores) - np.array(finetuned_similarities)))
        
        comparison = {
            'base_model_correlation': float(base_correlation),
            'finetuned_model_correlation': float(finetuned_correlation),
            'correlation_improvement': float(finetuned_correlation - base_correlation),
            'correlation_improvement_percentage': float(((finetuned_correlation - base_correlation) / abs(base_correlation)) * 100) if base_correlation != 0 else 0.0,
            'base_model_mae': float(base_mae),
            'finetuned_model_mae': float(finetuned_mae),
            'mae_improvement': float(base_mae - finetuned_mae),
            'mae_improvement_percentage': float(((base_mae - finetuned_mae) / base_mae) * 100) if base_mae != 0 else 0.0,
            'sample_size': sample_size
        }
        
        # Save comparison results
        with open(self.output_dir / "model_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info("Model comparison results:")
        for metric, value in comparison.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
        
        return comparison

    def save_training_config(self, train_examples: List[InputExample], val_examples: List[InputExample]):
        """Save training configuration and metadata"""
        config = {
            'base_model': self.base_model_name,
            'training_timestamp': datetime.now().isoformat(),
            'model_parameters': {
                'max_seq_length': self.max_seq_length,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'warmup_steps_ratio': self.warmup_steps_ratio
            },
            'dataset_info': {
                'train_samples': len(train_examples),
                'val_samples': len(val_examples),
                'dataset_stats': self.dataset_stats,
                'training_data_created': datetime.now().isoformat()
            },
            'hardware_info': {
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            }
        }
        
        with open(self.output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training configuration saved to: {self.output_dir / 'training_config.json'}")

def main():
    """Main function to run the fine-tuning process"""
    
    # Configuration
    BASE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    DATA_DIR = "faq_training_data"
    OUTPUT_DIR = "finetuned_faq_model_v1"
    
    # Alternative models you can try:
    # BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"  # English only, better performance
    # BASE_MODEL = "intfloat/multilingual-e5-base"  # Good multilingual model
    # BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Smaller, faster
    
    # Initialize fine-tuner
    finetuner = FAQEmbeddingFineTuner(
        base_model_name=BASE_MODEL,
        output_dir=OUTPUT_DIR,
        max_seq_length=512
    )
    
    # Check if training data exists
    if not Path(DATA_DIR).exists():
        logger.error(f"Training data directory {DATA_DIR} not found!")
        logger.error("Please run create_faq_training_dataset.py first to generate training data")
        return
    
    # Load training data
    try:
        train_examples, val_examples, test_examples = finetuner.load_training_data(DATA_DIR)
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return
    
    # Save training configuration
    finetuner.save_training_config(train_examples, val_examples)
    
    # Start fine-tuning
    try:
        finetuner.train(train_examples, val_examples)
        
        # Evaluate the model
        metrics = finetuner.evaluate_model(test_examples)
        
        # Compare with base model
        comparison = finetuner.compare_with_base_model(test_examples)
        
        logger.info("\n" + "="*60)
        logger.info("FINE-TUNING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Fine-tuned model saved to: {OUTPUT_DIR}")
        logger.info(f"Correlation improvement: {comparison['correlation_improvement']:.4f}")
        logger.info(f"Correlation improvement %: {comparison['correlation_improvement_percentage']:.2f}%")
        logger.info(f"MAE improvement: {comparison['mae_improvement']:.4f}")
        logger.info(f"MAE improvement %: {comparison['mae_improvement_percentage']:.2f}%")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise

if __name__ == "__main__":
    main() 