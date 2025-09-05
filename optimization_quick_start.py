#!/usr/bin/env python3
"""
Quick optimization script for improving model performance
Implements Priority 1 recommendations for immediate improvement
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime

class QuickOptimizer:
    """Quick optimization implementations for immediate performance improvement"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger('QuickOptimizer')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare training data"""
        features = []
        labels = []
        
        labels_dir = self.data_dir / 'labels_v2'
        if not labels_dir.exists():
            labels_dir = self.data_dir / 'labels'
        
        for label_file in labels_dir.glob('*.json'):
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            entities = data.get('entities', data.get('words', []))
            for entity in entities:
                # Extract features
                feature_vector = self._extract_features(entity)
                if feature_vector is not None:
                    features.append(feature_vector)
                    
                    # Extract label
                    label = self._extract_label(entity)
                    labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def _extract_features(self, entity: Dict) -> np.ndarray:
        """Extract feature vector from entity"""
        try:
            # Basic features
            bbox = entity.get('bbox', {})
            text = entity.get('text', '')
            if isinstance(text, dict):
                text = text.get('value', '')
            
            features = [
                bbox.get('x', 0) / 1000,  # Normalize
                bbox.get('y', 0) / 1000,
                bbox.get('width', 100) / 1000,
                bbox.get('height', 30) / 1000,
                len(text),
                1 if text.isdigit() else 0,
                1 if text.isalpha() else 0,
                1 if '-' in text else 0,
                text.count(' '),
                1 if text.isupper() else 0
            ]
            
            return np.array(features)
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return None
    
    def _extract_label(self, entity: Dict) -> str:
        """Extract label from entity"""
        label = entity.get('label', entity.get('predicted_label', 'unknown'))
        if isinstance(label, dict):
            label = label.get('primary', 'unknown')
        return label
    
    def apply_stratified_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict:
        """Apply stratified cross-validation"""
        self.logger.info(f"Applying {n_splits}-fold stratified cross-validation")
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_results = {
            'fold_scores': [],
            'label_distribution': {},
            'total_samples': len(y)
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
            # Check label distribution
            train_labels = y[train_idx]
            val_labels = y[val_idx]
            
            fold_info = {
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_labels': dict(zip(*np.unique(train_labels, return_counts=True))),
                'val_labels': dict(zip(*np.unique(val_labels, return_counts=True)))
            }
            
            cv_results['fold_scores'].append(fold_info)
            self.logger.info(f"Fold {fold}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        # Overall label distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        cv_results['label_distribution'] = dict(zip(unique_labels, counts.tolist()))
        
        return cv_results
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset using SMOTE"""
        self.logger.info("Balancing dataset with SMOTE")
        
        # Check class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        
        if min_samples < 2:
            self.logger.warning(f"Minimum class samples: {min_samples}. Skipping SMOTE.")
            return X, y
        
        # Encode labels for SMOTE
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Apply SMOTE
        k_neighbors = min(5, min_samples - 1)
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        
        try:
            X_balanced, y_balanced_encoded = smote.fit_resample(X, y_encoded)
            y_balanced = le.inverse_transform(y_balanced_encoded)
            
            self.logger.info(f"Original samples: {len(y)}, Balanced samples: {len(y_balanced)}")
            
            # Log new distribution
            new_unique, new_counts = np.unique(y_balanced, return_counts=True)
            for label, count in zip(new_unique, new_counts):
                original_count = counts[unique_labels == label][0] if label in unique_labels else 0
                self.logger.info(f"  {label}: {original_count} -> {count}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            self.logger.error(f"SMOTE failed: {e}")
            return X, y
    
    def add_regularization_params(self) -> Dict[str, Any]:
        """Get regularization parameters for models"""
        params = {
            'random_forest': {
                'max_depth': 10,  # Limit depth
                'min_samples_split': 5,  # Increase min samples
                'min_samples_leaf': 2,
                'max_features': 'sqrt',  # Use subset of features
                'n_estimators': 100,
                'random_state': 42
            },
            'xgboost': {
                'max_depth': 6,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 1.0,  # L2 regularization
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42
            },
            'lightgbm': {
                'max_depth': 8,
                'num_leaves': 31,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42
            }
        }
        
        return params
    
    def run_optimization(self) -> Dict:
        """Run all quick optimizations"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'optimizations': []
        }
        
        try:
            # Load data
            self.logger.info("Loading training data...")
            X, y = self.load_training_data()
            
            results['data_stats'] = {
                'total_samples': len(y),
                'total_features': X.shape[1] if len(X) > 0 else 0,
                'unique_labels': len(np.unique(y))
            }
            
            # Apply stratified CV
            self.logger.info("\n1. Stratified Cross-Validation")
            cv_results = self.apply_stratified_cv(X, y)
            results['optimizations'].append({
                'name': 'Stratified CV',
                'status': 'completed',
                'results': cv_results
            })
            
            # Balance dataset
            self.logger.info("\n2. Dataset Balancing")
            X_balanced, y_balanced = self.balance_dataset(X, y)
            results['optimizations'].append({
                'name': 'SMOTE Balancing',
                'status': 'completed',
                'results': {
                    'original_size': len(y),
                    'balanced_size': len(y_balanced),
                    'improvement': f"{((len(y_balanced) - len(y)) / len(y) * 100):.1f}%"
                }
            })
            
            # Get regularization params
            self.logger.info("\n3. Regularization Parameters")
            reg_params = self.add_regularization_params()
            results['optimizations'].append({
                'name': 'Regularization',
                'status': 'completed',
                'parameters': reg_params
            })
            
            # Save results
            output_file = self.data_dir / 'reports' / f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"\nOptimization complete! Results saved to {output_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            results['error'] = str(e)
            return results


def main():
    """Main execution"""
    import sys
    import os
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Run optimization
    data_dir = project_root / 'data' / 'processed'
    optimizer = QuickOptimizer(data_dir)
    
    print("=" * 60)
    print("OCR Learning System - Quick Optimization")
    print("=" * 60)
    
    results = optimizer.run_optimization()
    
    print("\n" + "=" * 60)
    print("Optimization Summary")
    print("=" * 60)
    
    for opt in results.get('optimizations', []):
        status = "✅" if opt['status'] == 'completed' else "❌"
        print(f"{status} {opt['name']}")
    
    print("\nNext Steps:")
    print("1. Update HybridOCRLabeler with regularization parameters")
    print("2. Retrain with balanced dataset")
    print("3. Implement stratified CV in training pipeline")
    print("4. Monitor validation performance")


if __name__ == "__main__":
    main()