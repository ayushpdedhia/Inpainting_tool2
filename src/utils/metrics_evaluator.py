# src/utils/metrics_evaluator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, Union, List, Any
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging
from torchvision.models import inception_v3
from scipy import linalg
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class MetricsEvaluator:
    """Evaluates image inpainting quality using multiple metrics"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.inception_model = None  # Lazy loading
        self.metrics_history = []
        
    def calculate_all_metrics(self, 
                            original: np.ndarray,
                            inpainted: np.ndarray,
                            mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate all available metrics
        
        Args:
            original: Original image [H, W, C]
            inpainted: Inpainted image [H, W, C]
            mask: Binary mask [H, W] where 0=inpainted region, 1=original
            
        Returns:
            Dictionary of metric names and values
        """
        try:
            metrics = {}
            
            # Basic image quality metrics
            metrics['psnr'] = self.calculate_psnr(original, inpainted)
            metrics['ssim'] = self.calculate_ssim(original, inpainted)
            metrics['l1_error'] = self.calculate_l1_error(original, inpainted)
            metrics['l2_error'] = self.calculate_l2_error(original, inpainted)
            
            # Masked region metrics
            metrics['masked_psnr'] = self.calculate_masked_psnr(original, inpainted, mask)
            metrics['masked_ssim'] = self.calculate_masked_ssim(original, inpainted, mask)
            
            # Edge coherence
            metrics['edge_coherence'] = self.calculate_edge_coherence(original, inpainted, mask)
            
            # Color consistency
            metrics['color_consistency'] = self.calculate_color_consistency(original, inpainted, mask)
            
            # Texture similarity
            metrics['texture_similarity'] = self.calculate_texture_similarity(original, inpainted, mask)
            
            # Log metrics
            self._log_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def calculate_psnr(self, original: np.ndarray, inpainted: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        return float(psnr(original, inpainted))

    def calculate_ssim(self, original: np.ndarray, inpainted: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        return float(ssim(original, inpainted, channel_axis=2))

    def calculate_l1_error(self, original: np.ndarray, inpainted: np.ndarray) -> float:
        """Calculate L1 (Manhattan) distance"""
        return float(np.mean(np.abs(original - inpainted)))

    def calculate_l2_error(self, original: np.ndarray, inpainted: np.ndarray) -> float:
        """Calculate L2 (Euclidean) distance"""
        return float(np.sqrt(np.mean((original - inpainted) ** 2)))

    def calculate_masked_psnr(self, original: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> float:
        """Calculate PSNR only in the inpainted region"""
        inpaint_region = (mask == 0)
        if not np.any(inpaint_region):
            return 0.0
        return float(psnr(
            original[inpaint_region],
            inpainted[inpaint_region]
        ))

    def calculate_masked_ssim(self, original: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> float:
        """Calculate SSIM only in the inpainted region"""
        # Create masked versions of the images
        masked_original = original.copy()
        masked_inpainted = inpainted.copy()
        
        # Only compare the inpainted regions
        inpaint_region = (mask == 0)
        if not np.any(inpaint_region):
            return 0.0
            
        return float(ssim(
            masked_original,
            masked_inpainted,
            channel_axis=2,
            mask=inpaint_region
        ))

    def calculate_edge_coherence(self, original: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate edge coherence between original and inpainted images
        Uses Sobel edge detection and compares edge responses
        """
        try:
            # Convert to grayscale
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY)
            
            # Calculate Sobel edges
            sobelx = cv2.Sobel(original_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(original_gray, cv2.CV_64F, 0, 1, ksize=3)
            orig_edges = np.sqrt(sobelx**2 + sobely**2)
            
            sobelx = cv2.Sobel(inpainted_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(inpainted_gray, cv2.CV_64F, 0, 1, ksize=3)
            inpaint_edges = np.sqrt(sobelx**2 + sobely**2)
            
            # Compare edges in transition regions (dilated mask boundary)
            kernel = np.ones((5,5), np.uint8)
            mask_boundary = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) - mask.astype(np.uint8)
            
            if not np.any(mask_boundary):
                return 1.0  # Perfect coherence if no boundary
                
            edge_diff = np.abs(orig_edges - inpaint_edges)
            coherence = 1.0 - np.mean(edge_diff[mask_boundary == 1])
            
            return float(coherence)
            
        except Exception as e:
            logger.error(f"Error calculating edge coherence: {str(e)}")
            return 0.0
        
    def calculate_color_consistency(self, original: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> float:
            """
            Calculate color consistency between original and inpainted regions
            Compares color histograms at boundary regions
            """
            try:
                # Create boundary region for analysis
                kernel = np.ones((7,7), np.uint8)
                dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
                eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=2)
                boundary = dilated - eroded
                
                if not np.any(boundary):
                    return 1.0
                    
                # Calculate color histograms in boundary regions
                orig_hist = cv2.calcHist([original], [0,1,2], boundary.astype(np.uint8),
                                    [8,8,8], [0,256, 0,256, 0,256])
                inpaint_hist = cv2.calcHist([inpainted], [0,1,2], boundary.astype(np.uint8),
                                        [8,8,8], [0,256, 0,256, 0,256])
                                        
                # Normalize histograms
                orig_hist = cv2.normalize(orig_hist, orig_hist).flatten()
                inpaint_hist = cv2.normalize(inpaint_hist, inpaint_hist).flatten()
                
                # Compare histograms using correlation
                consistency = cv2.compareHist(orig_hist, inpaint_hist, cv2.HISTCMP_CORREL)
                
                return float(consistency)
                
            except Exception as e:
                logger.error(f"Error calculating color consistency: {str(e)}")
                return 0.0

    def calculate_texture_similarity(self, original: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate texture similarity using Gray-Level Co-occurrence Matrix (GLCM)
        """
        try:
            from skimage.feature import graycomatrix, graycomatrix_properties
            
            # Convert to grayscale
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            inpaint_gray = cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY)
            
            # Calculate GLCM for both images
            orig_glcm = graycomatrix(orig_gray, distances=[1], angles=[0], levels=256,
                                symmetric=True, normed=True)
            inpaint_glcm = graycomatrix(inpaint_gray, distances=[1], angles=[0], levels=256,
                                    symmetric=True, normed=True)
                                    
            # Calculate GLCM properties
            properties = ['contrast', 'homogeneity', 'energy', 'correlation']
            similarity = 0.0
            
            for prop in properties:
                orig_feat = graycomatrix_properties(orig_glcm, prop)[0,0]
                inpaint_feat = graycomatrix_properties(inpaint_glcm, prop)[0,0]
                similarity += 1.0 - abs(orig_feat - inpaint_feat)
                
            return float(similarity / len(properties))
            
        except Exception as e:
            logger.error(f"Error calculating texture similarity: {str(e)}")
            return 0.0

    def calculate_fid(self, real_images: List[np.ndarray], generated_images: List[np.ndarray]) -> float:
        """
        Calculate Fréchet Inception Distance between real and generated images
        """
        try:
            if self.inception_model is None:
                self.inception_model = inception_v3(pretrained=True, transform_input=False)
                self.inception_model.fc = nn.Identity()  # Remove classification layer
                self.inception_model = self.inception_model.to(self.device)
                self.inception_model.eval()

            def get_features(images):
                features = []
                with torch.no_grad():
                    for img in images:
                        # Preprocess image
                        img = torch.from_numpy(img).permute(2, 0, 1).float()
                        img = F.interpolate(img.unsqueeze(0), size=(299, 299),
                                        mode='bilinear', align_corners=False)
                        img = img.to(self.device)
                        features.append(self.inception_model(img).cpu().numpy())
                return np.concatenate(features)

            # Get features
            real_features = get_features(real_images)
            gen_features = get_features(generated_images)

            # Calculate mean and covariance
            mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
            mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

            # Calculate FID
            ssdiff = np.sum((mu1 - mu2) ** 2)
            covmean = linalg.sqrtm(sigma1.dot(sigma2))
            
            if np.iscomplexobj(covmean):
                covmean = covmean.real

            fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
            
            return float(fid)

        except Exception as e:
            logger.error(f"Error calculating FID: {str(e)}")
            return float('inf')

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics and save to history"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics['timestamp'] = timestamp
        self.metrics_history.append(metrics)
        
        # Save to file
        save_dir = 'metrics_logs'
        os.makedirs(save_dir, exist_ok=True)
        
        log_file = os.path.join(save_dir, 'metrics_history.json')
        try:
            with open(log_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    def visualize_metrics(self) -> Dict[str, Any]:
        """
        Generate visualization data for Streamlit
        Returns dict with plot data
        """
        if not self.metrics_history:
            return {}
            
        metrics_data = {
            'timestamps': [],
            'psnr': [],
            'ssim': [],
            'edge_coherence': [],
            'color_consistency': [],
            'texture_similarity': []
        }
        
        for entry in self.metrics_history:
            metrics_data['timestamps'].append(entry['timestamp'])
            for metric in metrics_data.keys():
                if metric != 'timestamps':
                    metrics_data[metric].append(entry.get(metric, 0))
                    
        return metrics_data