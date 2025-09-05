"""
Advanced Data Preprocessing Module for Enhanced OCR System
Implements intelligent data transformation and augmentation
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class IntelligentPreprocessor:
    """Advanced preprocessing with pattern recognition and self-improvement"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.pattern_cache = {}
        self.preprocessing_stats = {
            'total_processed': 0,
            'corrections_applied': 0,
            'patterns_learned': 0,
            'augmentations_applied': 0
        }
        
    def preprocess_batch(self, json_files: List[Path], 
                        parallel: bool = True) -> List[Dict]:
        """Preprocess multiple files with parallel processing"""
        
        if parallel:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(self.preprocess_single, f): f 
                          for f in json_files}
                
                results = []
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing file: {e}")
                        
            return results
        else:
            return [self.preprocess_single(f) for f in tqdm(json_files)]
    
    def preprocess_single(self, json_file: Path) -> Dict:
        """Comprehensive preprocessing for single document"""
        
        # Load enhanced JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Find corresponding image
        image_path = self._find_image_path(json_file)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        image = self._preprocess_image(image_path)
        
        # Enhance bounding boxes
        enhanced_bboxes = self._enhance_bboxes(data['bboxes'], image.shape)
        
        # Extract advanced features
        features = self._extract_advanced_features(enhanced_bboxes, image)
        
        # Apply pattern-based corrections
        corrected_data = self._apply_pattern_corrections(data, features)
        
        # Generate synthetic context
        synthetic_context = self._generate_synthetic_context(corrected_data)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(corrected_data, image)
        
        # Update statistics
        self.preprocessing_stats['total_processed'] += 1
        
        return {
            'original_data': data,
            'enhanced_bboxes': enhanced_bboxes,
            'features': features,
            'corrected_data': corrected_data,
            'synthetic_context': synthetic_context,
            'quality_metrics': quality_metrics,
            'image_tensor': torch.tensor(image).permute(2, 0, 1),
            'metadata': {
                'file_path': str(json_file),
                'image_path': str(image_path),
                'preprocessing_timestamp': datetime.now().isoformat()
            }
        }
    
    def _find_image_path(self, json_file: Path) -> Path:
        """Find corresponding image file"""
        base_name = json_file.stem.replace('_label', '')
        image_dir = json_file.parent.parent / 'images'
        return image_dir / f"{base_name}.png"
    
    def _preprocess_image(self, image_path: Path) -> np.ndarray:
        """Advanced image preprocessing"""
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Denoise
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Adaptive histogram equalization
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        image = cv2.merge([l, a, b])
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)
        
        # Perspective correction (if needed)
        image = self._correct_perspective(image)
        
        return image
    
    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Correct document perspective using edge detection"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (assumed to be document)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate to quadrilateral
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) == 4:
                # Apply perspective transform
                pts = approx.reshape(4, 2)
                rect = self._order_points(pts)
                
                (tl, tr, br, bl) = rect
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")
                
                M = cv2.getPerspectiveTransform(rect, dst)
                image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return image
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points for perspective transform"""
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _enhance_bboxes(self, bboxes: List[Dict], image_shape: Tuple) -> List[Dict]:
        """Enhance bounding boxes with additional features"""
        
        enhanced = []
        img_height, img_width = image_shape[:2]
        
        for bbox in bboxes:
            enhanced_bbox = bbox.copy()
            
            # Normalize coordinates
            enhanced_bbox['normalized_bbox'] = [
                bbox['x'] / img_width,
                bbox['y'] / img_height,
                bbox['width'] / img_width,
                bbox['height'] / img_height
            ]
            
            # Calculate center point
            enhanced_bbox['center'] = [
                bbox['x'] + bbox['width'] / 2,
                bbox['y'] + bbox['height'] / 2
            ]
            
            # Calculate aspect ratio
            enhanced_bbox['aspect_ratio'] = bbox['width'] / max(bbox['height'], 1)
            
            # Calculate area
            enhanced_bbox['area'] = bbox['width'] * bbox['height']
            
            # Relative position in document
            enhanced_bbox['relative_position'] = {
                'quadrant': self._get_quadrant(enhanced_bbox['center'], image_shape),
                'vertical_zone': self._get_vertical_zone(bbox['y'], img_height),
                'horizontal_zone': self._get_horizontal_zone(bbox['x'], img_width)
            }
            
            enhanced.append(enhanced_bbox)
        
        return enhanced
    
    def _get_quadrant(self, center: List[float], shape: Tuple) -> int:
        """Determine which quadrant the bbox center is in"""
        h, w = shape[:2]
        x, y = center
        
        if x < w/2 and y < h/2:
            return 1  # Top-left
        elif x >= w/2 and y < h/2:
            return 2  # Top-right
        elif x < w/2 and y >= h/2:
            return 3  # Bottom-left
        else:
            return 4  # Bottom-right
    
    def _get_vertical_zone(self, y: float, height: float) -> str:
        """Determine vertical zone (header/body/footer)"""
        relative_y = y / height
        if relative_y < 0.2:
            return 'header'
        elif relative_y > 0.8:
            return 'footer'
        else:
            return 'body'
    
    def _get_horizontal_zone(self, x: float, width: float) -> str:
        """Determine horizontal zone"""
        relative_x = x / width
        if relative_x < 0.33:
            return 'left'
        elif relative_x > 0.67:
            return 'right'
        else:
            return 'center'
    
    def _extract_advanced_features(self, bboxes: List[Dict], 
                                  image: np.ndarray) -> Dict:
        """Extract advanced features for ML"""
        
        features = {
            'layout_features': self._extract_layout_features(bboxes),
            'text_density': self._calculate_text_density(bboxes, image.shape),
            'alignment_features': self._extract_alignment_features(bboxes),
            'clustering_features': self._extract_clustering_features(bboxes),
            'statistical_features': self._extract_statistical_features(bboxes)
        }
        
        return features
    
    def _extract_layout_features(self, bboxes: List[Dict]) -> Dict:
        """Extract document layout features"""
        
        if not bboxes:
            return {}
        
        # Group by vertical position
        rows = self._group_into_rows(bboxes)
        
        # Group by horizontal position
        columns = self._group_into_columns(bboxes)
        
        return {
            'num_rows': len(rows),
            'num_columns': len(columns),
            'avg_row_height': np.mean([self._get_row_height(r) for r in rows]),
            'avg_column_width': np.mean([self._get_column_width(c) for c in columns]),
            'layout_regularity': self._calculate_layout_regularity(rows, columns)
        }
    
    def _group_into_rows(self, bboxes: List[Dict], threshold: float = 10) -> List[List[Dict]]:
        """Group bboxes into rows based on y-coordinate"""
        sorted_bboxes = sorted(bboxes, key=lambda x: x['y'])
        rows = []
        current_row = [sorted_bboxes[0]]
        
        for bbox in sorted_bboxes[1:]:
            if abs(bbox['y'] - current_row[-1]['y']) < threshold:
                current_row.append(bbox)
            else:
                rows.append(current_row)
                current_row = [bbox]
        
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def _group_into_columns(self, bboxes: List[Dict], threshold: float = 10) -> List[List[Dict]]:
        """Group bboxes into columns based on x-coordinate"""
        sorted_bboxes = sorted(bboxes, key=lambda x: x['x'])
        columns = []
        current_column = [sorted_bboxes[0]]
        
        for bbox in sorted_bboxes[1:]:
            if abs(bbox['x'] - current_column[-1]['x']) < threshold:
                current_column.append(bbox)
            else:
                columns.append(current_column)
                current_column = [bbox]
        
        if current_column:
            columns.append(current_column)
        
        return columns
    
    def _get_row_height(self, row: List[Dict]) -> float:
        """Calculate average height of bboxes in a row"""
        return np.mean([bbox['height'] for bbox in row])
    
    def _get_column_width(self, column: List[Dict]) -> float:
        """Calculate average width of bboxes in a column"""
        return np.mean([bbox['width'] for bbox in column])
    
    def _calculate_layout_regularity(self, rows: List, columns: List) -> float:
        """Calculate how regular/grid-like the layout is"""
        if not rows or not columns:
            return 0.0
        
        # Calculate variance in row heights
        row_heights = [self._get_row_height(r) for r in rows]
        row_variance = np.var(row_heights) if len(row_heights) > 1 else 0
        
        # Calculate variance in column widths
        col_widths = [self._get_column_width(c) for c in columns]
        col_variance = np.var(col_widths) if len(col_widths) > 1 else 0
        
        # Lower variance means more regular
        regularity = 1.0 / (1.0 + row_variance + col_variance)
        
        return regularity
    
    def _calculate_text_density(self, bboxes: List[Dict], 
                               image_shape: Tuple) -> float:
        """Calculate text density in the document"""
        if not bboxes:
            return 0.0
        
        total_text_area = sum(bbox['width'] * bbox['height'] for bbox in bboxes)
        total_image_area = image_shape[0] * image_shape[1]
        
        return total_text_area / total_image_area
    
    def _extract_alignment_features(self, bboxes: List[Dict]) -> Dict:
        """Extract alignment-based features"""
        
        if len(bboxes) < 2:
            return {'alignment_score': 0.0}
        
        # Check horizontal alignment
        x_coords = [bbox['x'] for bbox in bboxes]
        x_alignment = self._calculate_alignment_score(x_coords)
        
        # Check vertical alignment
        y_coords = [bbox['y'] for bbox in bboxes]
        y_alignment = self._calculate_alignment_score(y_coords)
        
        return {
            'horizontal_alignment': x_alignment,
            'vertical_alignment': y_alignment,
            'overall_alignment': (x_alignment + y_alignment) / 2
        }
    
    def _calculate_alignment_score(self, coords: List[float]) -> float:
        """Calculate how well aligned coordinates are"""
        if len(coords) < 2:
            return 0.0
        
        # Find clusters of aligned coordinates
        sorted_coords = sorted(coords)
        clusters = []
        current_cluster = [sorted_coords[0]]
        
        for coord in sorted_coords[1:]:
            if abs(coord - current_cluster[-1]) < 5:  # 5 pixel threshold
                current_cluster.append(coord)
            else:
                clusters.append(current_cluster)
                current_cluster = [coord]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        # Score based on cluster sizes
        max_cluster_size = max(len(c) for c in clusters)
        alignment_score = max_cluster_size / len(coords)
        
        return alignment_score
    
    def _extract_clustering_features(self, bboxes: List[Dict]) -> Dict:
        """Extract clustering-based features"""
        
        if len(bboxes) < 2:
            return {'num_clusters': 1, 'cluster_cohesion': 1.0}
        
        # Extract centroids
        centroids = np.array([bbox['center'] for bbox in bboxes 
                             if 'center' in bbox])
        
        if len(centroids) < 2:
            return {'num_clusters': 1, 'cluster_cohesion': 1.0}
        
        # Perform clustering (simple distance-based)
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=50, min_samples=2).fit(centroids)
        
        num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        # Calculate cluster cohesion
        cohesion = self._calculate_cluster_cohesion(centroids, clustering.labels_)
        
        return {
            'num_clusters': num_clusters,
            'cluster_cohesion': cohesion,
            'outlier_ratio': np.sum(clustering.labels_ == -1) / len(clustering.labels_)
        }
    
    def _calculate_cluster_cohesion(self, points: np.ndarray, 
                                   labels: np.ndarray) -> float:
        """Calculate how cohesive clusters are"""
        
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            return 0.0
        
        cohesions = []
        for label in unique_labels:
            cluster_points = points[labels == label]
            if len(cluster_points) > 1:
                # Calculate average distance within cluster
                distances = []
                for i in range(len(cluster_points)):
                    for j in range(i+1, len(cluster_points)):
                        dist = np.linalg.norm(cluster_points[i] - cluster_points[j])
                        distances.append(dist)
                
                avg_dist = np.mean(distances) if distances else 0
                cohesions.append(1.0 / (1.0 + avg_dist))
        
        return np.mean(cohesions) if cohesions else 0.0
    
    def _extract_statistical_features(self, bboxes: List[Dict]) -> Dict:
        """Extract statistical features from bboxes"""
        
        if not bboxes:
            return {}
        
        widths = [bbox['width'] for bbox in bboxes]
        heights = [bbox['height'] for bbox in bboxes]
        areas = [bbox['width'] * bbox['height'] for bbox in bboxes]
        confidences = [bbox.get('ocr_confidence', 1.0) for bbox in bboxes]
        
        return {
            'width_stats': {
                'mean': np.mean(widths),
                'std': np.std(widths),
                'min': np.min(widths),
                'max': np.max(widths)
            },
            'height_stats': {
                'mean': np.mean(heights),
                'std': np.std(heights),
                'min': np.min(heights),
                'max': np.max(heights)
            },
            'area_stats': {
                'mean': np.mean(areas),
                'std': np.std(areas),
                'min': np.min(areas),
                'max': np.max(areas)
            },
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        }
    
    def _apply_pattern_corrections(self, data: Dict, features: Dict) -> Dict:
        """Apply learned patterns to correct data"""
        
        corrected = data.copy()
        
        # Learn patterns from high-confidence samples
        patterns = self._learn_patterns(data)
        
        # Apply corrections based on patterns
        for pattern_type, pattern_data in patterns.items():
            if pattern_type == 'date_format':
                corrected = self._correct_date_formats(corrected, pattern_data)
            elif pattern_type == 'number_format':
                corrected = self._correct_number_formats(corrected, pattern_data)
            elif pattern_type == 'label_position':
                corrected = self._correct_label_positions(corrected, pattern_data)
        
        self.preprocessing_stats['corrections_applied'] += 1
        
        return corrected
    
    def _learn_patterns(self, data: Dict) -> Dict:
        """Learn patterns from high-confidence data"""
        
        patterns = {}
        
        # Learn date patterns
        date_pattern = self._learn_date_pattern(data)
        if date_pattern:
            patterns['date_format'] = date_pattern
        
        # Learn number patterns
        number_pattern = self._learn_number_pattern(data)
        if number_pattern:
            patterns['number_format'] = number_pattern
        
        # Learn position patterns
        position_pattern = self._learn_position_pattern(data)
        if position_pattern:
            patterns['label_position'] = position_pattern
        
        self.preprocessing_stats['patterns_learned'] += len(patterns)
        
        return patterns
    
    def _learn_date_pattern(self, data: Dict) -> Optional[str]:
        """Learn common date format pattern"""
        
        date_labels = []
        for bbox in data.get('bboxes', []):
            if bbox.get('label') == 'Delivery date' and bbox.get('ocr_confidence', 0) > 0.9:
                date_labels.append(bbox['text'])
        
        if not date_labels:
            return None
        
        # Detect common pattern
        patterns = {
            r'\d{2}-\d{2}-\d{4}': 'MM-DD-YYYY',
            r'\d{4}-\d{2}-\d{2}': 'YYYY-MM-DD',
            r'\d{2}/\d{2}/\d{4}': 'MM/DD/YYYY'
        }
        
        for pattern, format_name in patterns.items():
            if all(re.match(pattern, date) for date in date_labels):
                return format_name
        
        return None
    
    def _learn_number_pattern(self, data: Dict) -> Optional[Dict]:
        """Learn number formatting patterns"""
        
        patterns = {}
        
        # Learn quantity pattern
        quantities = [bbox['text'] for bbox in data.get('bboxes', [])
                     if bbox.get('label') == 'Quantity' and bbox.get('ocr_confidence', 0) > 0.9]
        
        if quantities:
            # Check decimal places
            decimal_places = []
            for qty in quantities:
                match = re.search(r'\.(\d+)', qty)
                if match:
                    decimal_places.append(len(match.group(1)))
            
            if decimal_places:
                patterns['quantity_decimals'] = max(set(decimal_places), key=decimal_places.count)
        
        return patterns if patterns else None
    
    def _learn_position_pattern(self, data: Dict) -> Optional[Dict]:
        """Learn typical positions for each label type"""
        
        position_patterns = {}
        
        label_positions = defaultdict(list)
        for bbox in data.get('bboxes', []):
            if bbox.get('ocr_confidence', 0) > 0.9:
                label_positions[bbox['label']].append({
                    'x': bbox['x'],
                    'y': bbox['y'],
                    'quadrant': bbox.get('relative_position', {}).get('quadrant')
                })
        
        for label, positions in label_positions.items():
            if positions:
                position_patterns[label] = {
                    'avg_x': np.mean([p['x'] for p in positions]),
                    'avg_y': np.mean([p['y'] for p in positions]),
                    'typical_quadrant': max(set(p.get('quadrant', 0) for p in positions),
                                           key=lambda x: [p.get('quadrant', 0) for p in positions].count(x))
                }
        
        return position_patterns if position_patterns else None
    
    def _correct_date_formats(self, data: Dict, pattern: str) -> Dict:
        """Correct date formats based on learned pattern"""
        
        for bbox in data.get('bboxes', []):
            if bbox.get('label') == 'Delivery date' and bbox.get('ocr_confidence', 0) < 0.8:
                # Attempt to parse and reformat
                original = bbox['text']
                corrected = self._standardize_date(original, pattern)
                if corrected != original:
                    bbox['text'] = corrected
                    bbox['was_corrected'] = True
                    bbox['correction_method'] = 'pattern_based'
        
        return data
    
    def _standardize_date(self, date_str: str, target_format: str) -> str:
        """Standardize date to target format"""
        
        # Try to parse date
        from dateutil import parser
        try:
            parsed_date = parser.parse(date_str, fuzzy=True)
            
            if target_format == 'MM-DD-YYYY':
                return parsed_date.strftime('%m-%d-%Y')
            elif target_format == 'YYYY-MM-DD':
                return parsed_date.strftime('%Y-%m-%d')
            elif target_format == 'MM/DD/YYYY':
                return parsed_date.strftime('%m/%d/%Y')
        except:
            pass
        
        return date_str
    
    def _correct_number_formats(self, data: Dict, pattern: Dict) -> Dict:
        """Correct number formats based on learned pattern"""
        
        for bbox in data.get('bboxes', []):
            if bbox.get('label') == 'Quantity' and 'quantity_decimals' in pattern:
                original = bbox['text']
                # Extract number part
                match = re.search(r'([\d.]+)', original)
                if match:
                    number = float(match.group(1))
                    decimal_places = pattern['quantity_decimals']
                    formatted = f"{number:.{decimal_places}f}"
                    
                    # Preserve unit if present
                    unit_match = re.search(r'\s+(\w+)$', original)
                    if unit_match:
                        formatted += f" {unit_match.group(1)}"
                    
                    if formatted != original:
                        bbox['text'] = formatted
                        bbox['was_corrected'] = True
                        bbox['correction_method'] = 'pattern_based'
        
        return data
    
    def _correct_label_positions(self, data: Dict, pattern: Dict) -> Dict:
        """Flag labels that are in unexpected positions"""
        
        for bbox in data.get('bboxes', []):
            label = bbox.get('label')
            if label in pattern:
                expected = pattern[label]
                
                # Check if position is significantly different
                x_diff = abs(bbox['x'] - expected['avg_x'])
                y_diff = abs(bbox['y'] - expected['avg_y'])
                
                if x_diff > 100 or y_diff > 100:
                    bbox['position_anomaly'] = True
                    bbox['expected_position'] = expected
        
        return data
    
    def _generate_synthetic_context(self, data: Dict) -> Dict:
        """Generate synthetic contextual information"""
        
        synthetic = {
            'document_structure': self._infer_document_structure(data),
            'implicit_relationships': self._infer_relationships(data),
            'missing_labels': self._predict_missing_labels(data),
            'confidence_adjustments': self._calculate_confidence_adjustments(data)
        }
        
        return synthetic
    
    def _infer_document_structure(self, data: Dict) -> Dict:
        """Infer high-level document structure"""
        
        structure = {
            'has_header': False,
            'has_footer': False,
            'has_table': False,
            'num_sections': 1
        }
        
        # Check for header (top 20% of document)
        header_labels = [bbox for bbox in data.get('bboxes', [])
                        if bbox.get('relative_position', {}).get('vertical_zone') == 'header']
        structure['has_header'] = len(header_labels) > 0
        
        # Check for footer
        footer_labels = [bbox for bbox in data.get('bboxes', [])
                        if bbox.get('relative_position', {}).get('vertical_zone') == 'footer']
        structure['has_footer'] = len(footer_labels) > 0
        
        # Check for table structure (aligned items)
        if 'Item number' in [bbox['label'] for bbox in data.get('bboxes', [])]:
            structure['has_table'] = True
        
        return structure
    
    def _infer_relationships(self, data: Dict) -> List[Dict]:
        """Infer implicit relationships between labels"""
        
        relationships = []
        
        # Find item groups
        item_groups = defaultdict(list)
        for bbox in data.get('bboxes', []):
            if bbox.get('group_id') and bbox['group_id'] != '-':
                item_groups[bbox['group_id']].append(bbox)
        
        # Infer relationships within groups
        for group_id, group_bboxes in item_groups.items():
            labels_in_group = [bbox['label'] for bbox in group_bboxes]
            
            # Quantity-Price relationship
            if 'Quantity' in labels_in_group and 'Unit price' in labels_in_group:
                relationships.append({
                    'type': 'calculation',
                    'source': ['Quantity', 'Unit price'],
                    'target': 'Line total',
                    'group_id': group_id
                })
        
        return relationships
    
    def _predict_missing_labels(self, data: Dict) -> List[str]:
        """Predict which labels might be missing"""
        
        expected_labels = set(LabelType.__members__.values())
        found_labels = set(bbox['label'] for bbox in data.get('bboxes', []))
        
        missing = list(expected_labels - found_labels)
        
        # Prioritize based on importance
        priority_order = ['Order number', 'Net amount (total)', 'Delivery date']
        missing.sort(key=lambda x: priority_order.index(x) if x in priority_order else 999)
        
        return missing
    
    def _calculate_confidence_adjustments(self, data: Dict) -> Dict:
        """Calculate confidence adjustments based on context"""
        
        adjustments = {}
        
        for i, bbox in enumerate(data.get('bboxes', [])):
            label = bbox['label']
            original_confidence = bbox.get('ocr_confidence', 1.0)
            
            # Adjust based on position consistency
            if bbox.get('position_anomaly'):
                adjustments[i] = original_confidence * 0.8
            else:
                adjustments[i] = min(original_confidence * 1.1, 1.0)
            
            # Adjust based on format validation
            if label == 'Order number' and re.match(r'^\d{10}$', bbox['text']):
                adjustments[i] = min(adjustments[i] * 1.2, 1.0)
        
        return adjustments
    
    def _calculate_quality_metrics(self, data: Dict, image: np.ndarray) -> Dict:
        """Calculate comprehensive quality metrics"""
        
        metrics = {
            'overall_confidence': np.mean([bbox.get('ocr_confidence', 0) 
                                          for bbox in data.get('bboxes', [])]),
            'correction_rate': sum(1 for bbox in data.get('bboxes', []) 
                                  if bbox.get('was_corrected', False)) / max(len(data.get('bboxes', [])), 1),
            'completeness': len(data.get('bboxes', [])) / 10,  # Assuming 10 expected labels
            'image_quality': self._assess_image_quality(image),
            'layout_quality': self._assess_layout_quality(data)
        }
        
        metrics['overall_quality'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess image quality"""
        
        # Calculate sharpness (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 1000, 1.0)
        
        # Calculate contrast
        contrast = gray.std() / 128
        
        # Calculate brightness
        brightness = gray.mean() / 255
        
        # Combined score
        quality = (sharpness + contrast + abs(brightness - 0.5)) / 3
        
        return min(quality, 1.0)
    
    def _assess_layout_quality(self, data: Dict) -> float:
        """Assess document layout quality"""
        
        features = data.get('features', {})
        layout_features = features.get('layout_features', {})
        
        regularity = layout_features.get('layout_regularity', 0)
        alignment = features.get('alignment_features', {}).get('overall_alignment', 0)
        
        return (regularity + alignment) / 2
    
    def save_preprocessing_stats(self, output_path: Path):
        """Save preprocessing statistics"""
        
        stats_df = pd.DataFrame([self.preprocessing_stats])
        stats_df['timestamp'] = datetime.now()
        
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            stats_df = pd.concat([existing_df, stats_df], ignore_index=True)
        
        stats_df.to_csv(output_path, index=False)
        logger.info(f"Saved preprocessing stats to {output_path}")


def main():
    """Main preprocessing pipeline"""
    
    config = {
        'input_dir': Path('/mnt/e/김선민/YMF-K/YMFK_OCRPJT/data/processed/labels'),
        'output_dir': Path('/mnt/e/김선민/YMF-K/YMFK_OCRPJT/data/preprocessed'),
        'stats_file': Path('/mnt/e/김선민/YMF-K/YMFK_OCRPJT/preprocessing_stats.csv')
    }
    
    # Create output directory
    config['output_dir'].mkdir(exist_ok=True, parents=True)
    
    # Initialize preprocessor
    preprocessor = IntelligentPreprocessor(config)
    
    # Get all JSON files
    json_files = list(config['input_dir'].glob('**/*.json'))
    logger.info(f"Found {len(json_files)} files to process")
    
    # Process files
    results = preprocessor.preprocess_batch(json_files, parallel=True)
    
    # Save preprocessed data
    for i, result in enumerate(results):
        output_file = config['output_dir'] / f"preprocessed_{i:04d}.pt"
        torch.save(result, output_file)
    
    logger.info(f"Saved {len(results)} preprocessed files")
    
    # Save statistics
    preprocessor.save_preprocessing_stats(config['stats_file'])
    
    print(f"Preprocessing complete!")
    print(f"Total processed: {preprocessor.preprocessing_stats['total_processed']}")
    print(f"Corrections applied: {preprocessor.preprocessing_stats['corrections_applied']}")
    print(f"Patterns learned: {preprocessor.preprocessing_stats['patterns_learned']}")


if __name__ == "__main__":
    main()