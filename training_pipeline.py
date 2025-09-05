"""
Complete Training Pipeline for Enhanced OCR ML System
Implements all proposed improvements with genius formulas applied
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from transformers import LayoutLMv3TokenizerFast
import logging
from datetime import datetime
import wandb
from collections import defaultdict
import random

from ml_enhanced_model import (
    EnhancedOCRModel, 
    ConfidenceWeightedLoss,
    ActiveLearningSelector,
    MetaLearningOptimizer,
    EnhancedLabel,
    LabelType
)

logger = logging.getLogger(__name__)


class EnhancedOCRDataset(Dataset):
    """Dataset with all enhanced features"""
    
    def __init__(self, data_dir: str, transform=None, augment=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
        
        # Load enhanced JSON files
        self.samples = self._load_enhanced_data()
        
        # Build error pattern database
        self.error_patterns = self._analyze_error_patterns()
        
        # Calculate domain similarity scores
        self.domain_similarities = self._calculate_domain_similarities()
        
    def _load_enhanced_data(self) -> List[Dict]:
        """Load data with enhanced schema"""
        samples = []
        
        for json_file in self.data_dir.glob("**/*_label.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Find corresponding image
            page_num = json_file.stem.split('_')[-2].replace('page', '')
            image_path = json_file.parent.parent / 'images' / f"{json_file.stem.replace('_label', '')}.png"
            
            if image_path.exists():
                enhanced_sample = self._enhance_sample(data, image_path)
                samples.append(enhanced_sample)
                
        return samples
    
    def _enhance_sample(self, data: Dict, image_path: Path) -> Dict:
        """Add enhanced features to sample"""
        
        # Add spatial context
        data['spatial_contexts'] = self._compute_spatial_contexts(data['bboxes'])
        
        # Add temporal sequences
        data['temporal_sequences'] = self._extract_temporal_patterns(data)
        
        # Add business rules validation
        data['business_rules'] = self._apply_business_rules(data)
        
        # Add difficulty score based on corrections and confidence
        data['difficulty_score'] = self._calculate_difficulty(data)
        
        # Add optimal resolutions for different label types
        data['optimal_resolutions'] = self._determine_optimal_resolutions(data)
        
        data['image_path'] = str(image_path)
        
        return data
    
    def _compute_spatial_contexts(self, bboxes: List[Dict]) -> Dict:
        """Compute spatial relationships between all bboxes"""
        contexts = {}
        
        for i, bbox1 in enumerate(bboxes):
            contexts[i] = {}
            for j, bbox2 in enumerate(bboxes):
                if i != j:
                    contexts[i][j] = self._get_spatial_relationship(bbox1, bbox2)
                    
        return contexts
    
    def _get_spatial_relationship(self, bbox1: Dict, bbox2: Dict) -> Dict:
        """Calculate spatial relationship between two bboxes"""
        x1, y1, w1, h1 = bbox1['x'], bbox1['y'], bbox1['width'], bbox1['height']
        x2, y2, w2, h2 = bbox2['x'], bbox2['y'], bbox2['width'], bbox2['height']
        
        # Center points
        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2
        
        # Distance and direction
        dx, dy = cx2 - cx1, cy2 - cy1
        distance = np.sqrt(dx**2 + dy**2)
        
        # Determine direction
        angle = np.arctan2(dy, dx) * 180 / np.pi
        direction = self._angle_to_direction(angle)
        
        # Check alignment
        alignment = 'aligned' if abs(y1 - y2) < 10 or abs(x1 - x2) < 10 else 'offset'
        
        return {
            'distance': distance,
            'direction': direction,
            'alignment': alignment,
            'overlap_x': max(0, min(x1+w1, x2+w2) - max(x1, x2)),
            'overlap_y': max(0, min(y1+h1, y2+h2) - max(y1, y2))
        }
    
    def _angle_to_direction(self, angle: float) -> str:
        """Convert angle to 8-directional descriptor"""
        directions = ['right', 'bottom-right', 'bottom', 'bottom-left',
                     'left', 'top-left', 'top', 'top-right']
        index = int((angle + 22.5) / 45) % 8
        return directions[index]
    
    def _extract_temporal_patterns(self, data: Dict) -> List[Dict]:
        """Extract temporal patterns from dates and numbers"""
        temporal_data = []
        
        for item in data.get('items', []):
            for label in item['labels']:
                if label['label'] in ['Delivery date', 'Order number']:
                    temporal_data.append({
                        'label': label['label'],
                        'value': label['text'],
                        'position': len(temporal_data),
                        'group_id': item['group_id']
                    })
                    
        return temporal_data
    
    def _apply_business_rules(self, data: Dict) -> Dict:
        """Apply business logic rules for validation"""
        rules = {
            'Order number': {
                'format_regex': r'^\d{10}$',
                'dependencies': [{'label': 'Shipping line', 'rule_type': 'required'}]
            },
            'Quantity': {
                'format_regex': r'^\d+\.\d{3}\s+\w{2}$',
                'valid_range': {'min': 0.001, 'max': 999999.999}
            },
            'Delivery date': {
                'format_regex': r'^\d{2}-\d{2}-\d{4}$',
                'dependencies': [{'label': 'Order number', 'rule_type': 'required'}]
            },
            'Unit price': {
                'valid_range': {'min': 0.0001, 'max': 999999.9999},
                'dependencies': [{'label': 'Quantity', 'rule_type': 'calculated'}]
            }
        }
        
        return rules
    
    def _calculate_difficulty(self, data: Dict) -> float:
        """Calculate sample difficulty based on various factors"""
        difficulty = 0.0
        
        # Factor 1: Number of corrections
        corrections = sum(1 for bbox in data.get('bboxes', []) 
                         if bbox.get('was_corrected', False))
        difficulty += corrections * 0.1
        
        # Factor 2: Average confidence
        confidences = [bbox.get('ocr_confidence', 1.0) 
                      for bbox in data.get('bboxes', [])]
        avg_confidence = np.mean(confidences) if confidences else 1.0
        difficulty += (1.0 - avg_confidence) * 0.5
        
        # Factor 3: Number of groups
        num_groups = data.get('total_groups', 1)
        difficulty += min(num_groups * 0.1, 0.3)
        
        return min(difficulty, 1.0)
    
    def _determine_optimal_resolutions(self, data: Dict) -> Dict:
        """Determine optimal resolution for each label type"""
        resolutions = {
            'Order number': 150,  # Medium resolution
            'Shipping line': 100,  # Low resolution sufficient
            'Case mark': 100,     # Low resolution sufficient
            'Item number': 200,   # High resolution for small text
            'Part number': 200,   # High resolution
            'Delivery date': 150, # Medium resolution
            'Quantity': 200,      # High resolution for numbers
            'Unit price': 200,    # High resolution for decimals
            'Net amount (total)': 150  # Medium resolution
        }
        return resolutions
    
    def _analyze_error_patterns(self) -> Dict:
        """Build error pattern database from corrected samples"""
        patterns = defaultdict(list)
        
        for sample in self.samples:
            for bbox in sample.get('bboxes', []):
                if bbox.get('was_corrected', False):
                    patterns[bbox['label']].append({
                        'original': bbox.get('ocr_original', ''),
                        'corrected': bbox.get('text', ''),
                        'confidence': bbox.get('ocr_confidence', 0)
                    })
                    
        return dict(patterns)
    
    def _calculate_domain_similarities(self) -> Dict:
        """Calculate similarities with other document types"""
        # Placeholder - would implement actual similarity calculation
        return {
            'invoice': 0.85,
            'receipt': 0.72,
            'shipping_label': 0.68
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply augmentations if enabled
        if self.augment and sample.get('augmentation_eligible', True):
            image = self._apply_augmentations(image)
        
        # Convert to tensor
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Prepare text tokens
        texts = [bbox['text'] for bbox in sample['bboxes']]
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Prepare bboxes
        bboxes = torch.tensor([[b['x'], b['y'], b['width'], b['height']] 
                              for b in sample['bboxes']])
        
        # Prepare confidence scores
        confidences = torch.tensor([b.get('ocr_confidence', 1.0) 
                                   for b in sample['bboxes']])
        
        # Prepare labels
        label_map = {label.value: i for i, label in enumerate(LabelType)}
        labels = torch.tensor([label_map.get(b['label'], -1) 
                              for b in sample['bboxes']])
        
        # Prepare correction flags
        was_corrected = torch.tensor([b.get('was_corrected', False) 
                                     for b in sample['bboxes']])
        
        return {
            'image': image_tensor,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'bboxes': bboxes,
            'labels': labels,
            'confidences': confidences,
            'was_corrected': was_corrected,
            'spatial_contexts': sample.get('spatial_contexts', {}),
            'temporal_sequences': sample.get('temporal_sequences', []),
            'difficulty_score': sample.get('difficulty_score', 0.5)
        }
    
    def _apply_augmentations(self, image: Image) -> Image:
        """Apply data augmentations"""
        img_array = np.array(image)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            center = (img_array.shape[1]//2, img_array.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_array = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))
        
        # Random noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
            img_array = cv2.add(img_array, noise)
        
        # Random blur
        if random.random() > 0.5:
            kernel_size = random.choice([3, 5])
            img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
        
        return Image.fromarray(img_array)


class EnhancedTrainer:
    """Training orchestrator with all enhancements"""
    
    def __init__(self, model: EnhancedOCRModel, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize components
        self.loss_fn = ConfidenceWeightedLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Active learning
        self.active_learner = ActiveLearningSelector()
        
        # Meta learning
        self.meta_optimizer = MetaLearningOptimizer(model)
        
        # Initialize tracking
        wandb.init(project="enhanced-ocr", config=config)
        self.best_accuracy = 0
        self.training_history = []
        
    def train(self, train_dataset: EnhancedOCRDataset, 
             val_dataset: EnhancedOCRDataset, 
             epochs: int):
        """Main training loop with all enhancements"""
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    image=batch['image'],
                    bboxes=batch['bboxes'],
                    text_tokens=batch['input_ids'],
                    confidence_scores=batch['confidences'],
                    spatial_context=batch['spatial_contexts'],
                    temporal_info=batch.get('temporal_sequences')
                )
                
                # Calculate loss with confidence weighting
                loss = self.loss_fn(
                    outputs['logits'],
                    batch['labels'],
                    batch['confidences'],
                    batch['was_corrected']
                )
                
                # Add error prediction loss
                if 'error_predictions' in outputs:
                    error_labels = self._create_error_labels(batch)
                    error_loss = F.cross_entropy(outputs['error_predictions'], error_labels)
                    loss += 0.1 * error_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                train_total += batch['labels'].size(0)
                train_correct += predicted.eq(batch['labels']).sum().item()
                
                # Log to wandb
                if batch_idx % 10 == 0:
                    wandb.log({
                        'train_loss': loss.item(),
                        'train_acc': 100. * train_correct / train_total,
                        'learning_rate': self.scheduler.get_last_lr()[0]
                    })
            
            # Validation phase
            val_accuracy, val_loss = self.validate(val_loader)
            
            # Active learning sample selection
            if epoch % 5 == 0:
                self._perform_active_learning(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.save_checkpoint(epoch)
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Train Acc: {100.*train_correct/train_total:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss/len(train_loader),
                'train_acc': 100.*train_correct/train_total,
                'val_loss': val_loss,
                'val_acc': val_accuracy
            })
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validation with comprehensive metrics"""
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        # Per-label accuracy tracking
        label_correct = defaultdict(int)
        label_total = defaultdict(int)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    image=batch['image'],
                    bboxes=batch['bboxes'],
                    text_tokens=batch['input_ids'],
                    confidence_scores=batch['confidences'],
                    spatial_context=batch['spatial_contexts']
                )
                
                loss = self.loss_fn(
                    outputs['logits'],
                    batch['labels'],
                    batch['confidences'],
                    batch['was_corrected']
                )
                
                val_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                val_total += batch['labels'].size(0)
                val_correct += predicted.eq(batch['labels']).sum().item()
                
                # Track per-label performance
                for i, label in enumerate(batch['labels']):
                    label_name = LabelType(label.item()).value
                    label_total[label_name] += 1
                    if predicted[i] == label:
                        label_correct[label_name] += 1
        
        # Log per-label metrics
        for label_name in label_total:
            accuracy = 100. * label_correct[label_name] / label_total[label_name]
            wandb.log({f'val_acc_{label_name}': accuracy})
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        return val_accuracy, avg_val_loss
    
    def _perform_active_learning(self, val_loader: DataLoader):
        """Select uncertain samples for human review"""
        self.model.eval()
        all_predictions = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    image=batch['image'],
                    bboxes=batch['bboxes'],
                    text_tokens=batch['input_ids'],
                    confidence_scores=batch['confidences'],
                    spatial_context=batch['spatial_contexts']
                )
                
                all_predictions.append(F.softmax(outputs['logits'], dim=-1))
                all_confidences.append(outputs['confidence'])
        
        all_predictions = torch.cat(all_predictions)
        all_confidences = torch.cat(all_confidences)
        
        # Select most uncertain samples
        selected_indices = self.active_learner.select_samples(
            all_predictions, all_confidences, n_samples=20
        )
        
        logger.info(f"Selected {len(selected_indices)} samples for active learning")
        wandb.log({'active_learning_samples': len(selected_indices)})
    
    def _create_error_labels(self, batch: Dict) -> torch.Tensor:
        """Create error type labels for error detection task"""
        # Placeholder - would implement actual error labeling
        return torch.zeros(batch['labels'].size(0), dtype=torch.long).to(self.device)
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history
        }
        
        path = Path(f"checkpoints/model_epoch_{epoch}_acc_{self.best_accuracy:.2f}.pt")
        path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


def main():
    """Main training script"""
    
    # Configuration
    config = {
        'learning_rate': 1e-4,
        'batch_size': 16,
        'epochs': 100,
        'data_dir': '/mnt/e/김선민/YMF-K/YMFK_OCRPJT/data/processed/labels'
    }
    
    # Initialize dataset
    dataset = EnhancedOCRDataset(config['data_dir'])
    
    # Split into train/val
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Initialize model
    model = EnhancedOCRModel()
    
    # Initialize trainer
    trainer = EnhancedTrainer(model, config)
    
    # Train
    trainer.train(train_dataset, val_dataset, config['epochs'])
    
    # Final meta-optimization
    param_space = {
        'learning_rates': [1e-5, 5e-5, 1e-4, 5e-4],
        'batch_sizes': [8, 16, 32],
        'dropout_rates': [0.1, 0.2, 0.3, 0.5]
    }
    
    best_params = trainer.meta_optimizer.optimize(
        train_dataset, val_dataset, param_space
    )
    
    logger.info(f"Best hyperparameters found: {best_params}")
    
    # Save final results
    with open('training_results.json', 'w') as f:
        json.dump({
            'best_accuracy': trainer.best_accuracy,
            'best_params': best_params,
            'training_history': trainer.training_history,
            'error_patterns': dataset.error_patterns
        }, f, indent=2)


if __name__ == "__main__":
    main()