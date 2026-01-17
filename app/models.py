import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
import os
from concurrent.futures import ThreadPoolExecutor
from .config import config

class ImageProcessor:
    """Preprocesses images for the model using GPU if available."""
    TARGET_SIZE = 448
    PADDING_COLOR = 255 
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ImageProcessor using device: {self.device}")

    def _preprocess_cpu(self, img: Image.Image) -> torch.Tensor:
        """
        Preprocess a single PIL image on CPU.
        Returns: Tensor (3, 448, 448) normalized and ready for model
        """
        w, h = img.size
        aspect_ratio = w / h

        if aspect_ratio > 1:
            new_w = self.TARGET_SIZE
            new_h = int(self.TARGET_SIZE / aspect_ratio)
        else:
            new_h = self.TARGET_SIZE
            new_w = int(self.TARGET_SIZE * aspect_ratio)

        # Resize
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        
        # Convert to tensor [0, 1]
        tensor = transforms.functional.to_tensor(img)

        # Pad
        padding_left = (self.TARGET_SIZE - new_w) // 2
        padding_top = (self.TARGET_SIZE - new_h) // 2
        padding_right = self.TARGET_SIZE - new_w - padding_left
        padding_bottom = self.TARGET_SIZE - new_h - padding_top
        
        tensor = transforms.functional.pad(tensor, (padding_left, padding_top, padding_right, padding_bottom), fill=1.0)
        
        # RGB -> BGR
        tensor = tensor[[2, 1, 0], :, :]

        # Normalize
        tensor = transforms.functional.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        return tensor

    def load_images_parallel(self, filepaths: list[str]) -> list[tuple[int, torch.Tensor, tuple[int, int]]]:
        """
        Load AND Preprocess images sequentially on CPU.
        Returns: List of (index, processed_tensor, (w, h))
        """
        results = []
        
        for idx, fpath in enumerate(filepaths):
            try:
                img = Image.open(fpath).convert('RGB')
                w, h = img.size
                # Perform full preprocessing here
                tensor = self._preprocess_cpu(img)
                results.append((idx, tensor, (w, h)))
            except Exception as e:
                # logger.warning(f"Error loading {fpath}: {e}")
                pass
        
        return results

    def process_tensors_to_features(self, loaded_data: list[tuple[int, torch.Tensor, tuple[int, int]]]) -> tuple[np.ndarray, list[int]]:
        """
        Take loaded CPU tensors (already preprocessed), move to GPU (batched), and extract features.
        """
        if not loaded_data:
            return np.array([]), []

        valid_indices = []
        tensors = []
        
        for idx, cpu_tensor, _ in loaded_data:
            tensors.append(cpu_tensor)
            valid_indices.append(idx)
            
        if not tensors:
            return np.array([]), []
            
        # Stack and Move to GPU in one go
        batch_tensor = torch.stack(tensors).to(self.device)
        
        # Extract features
        features = feature_extractor.extract(batch_tensor)
        return features, valid_indices

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        """
        Preprocess a single PIL image (for API search).
        Returns: Tensor (1, 3, 448, 448)
        """
        # Reuse CPU preprocessing logic
        tensor = self._preprocess_cpu(img)
        
        # Add batch dimension: (3, H, W) -> (1, 3, H, W)
        return tensor.unsqueeze(0)

    # Legacy method kept for compatibility if needed, but wrapper around new methods
    def preprocess_batch(self, filepaths: list[str]) -> tuple[torch.Tensor, list[int]]:
        pass


class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model {config.MODEL_REPO} on {self.device}...")
        
        self.model = timm.create_model(f"hf-hub:{config.MODEL_REPO}", pretrained=True).eval()
        self.model = self.model.to(self.device)
        
        # Save head for tags if needed later
        self.original_head = self.model.head
        
        # Extract weights for tag mapping
        head_state = self.model.head.state_dict()
        self.tag_feature = head_state["weight"].cpu().numpy().astype(np.float32) # Check dtype
        self.tag_feature_bias = head_state["bias"].cpu().numpy().astype(np.float32)
        
        # Move original head to device for probability calc
        self.original_head = self.original_head.to(self.device)

        self.model.head = nn.Identity()
        
    def extract(self, img_tensor: torch.Tensor) -> np.ndarray:
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor)
            return features.cpu().numpy()

    def extract_tag_feature(self, index: int) -> np.ndarray:
        # Replicate logic from prototype
        # baias_mult = 1 + (np.arctan(self.tag_feature_bias[index])+np.pi/2)/np.pi
        # normalized_feature = self.tag_feature[index] / np.linalg.norm(self.tag_feature[index])
        # return normalized_feature * baias_mult
        
        bias_val = self.tag_feature_bias[index]
        weight_vec = self.tag_feature[index]
        
        bias_mult = 1 + (np.arctan(bias_val) + np.pi/2) / np.pi
        norm = np.linalg.norm(weight_vec)
        if norm == 0: norm = 1e-6
        normalized_feature = weight_vec / norm
        
        return normalized_feature * bias_mult

    def get_tag_probabilities(self, feature: np.ndarray) -> np.ndarray:
        # feature: (1, dim) or (N, dim)
        tensor = torch.from_numpy(feature).to(self.device).float()
        with torch.no_grad():
            logits = self.original_head(tensor)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

model_processor = ImageProcessor()
feature_extractor = FeatureExtractor()
