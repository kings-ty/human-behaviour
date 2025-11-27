"""
HRI30 Dataset Loader and Preprocessor
Implements data loading pipeline based on the HRI30 paper specifications
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
import json
from pathlib import Path
import random
from configs.config import DataConfig, ModelConfig


class VideoProcessor:
    """Video processing utilities for HRI30 dataset"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.transform = self._get_transforms()
    
    def _get_transforms(self) -> A.Compose:
        """Get image transformations"""
        transforms = [
            A.Resize(height=self.config.input_resolution[0], 
                    width=self.config.input_resolution[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        return A.Compose(transforms)
    
    def _get_train_transforms(self) -> A.Compose:
        """Get training augmentations (random flip mentioned in paper)"""
        transforms = [
            A.Resize(height=self.config.input_resolution[0],
                    width=self.config.input_resolution[1]),
            A.HorizontalFlip(p=0.5),  # From paper
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        return A.Compose(transforms)
    
    def load_video(self, video_path: str, target_frames: int = 8) -> torch.Tensor:
        """
        Load and process video according to paper specifications
        - Sample every 4th frame (from Section IV-A)
        - Resize to 256x256 (from Section IV-A)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames every 4 frames as mentioned in paper
        indices = list(range(0, frame_count, self.config.sample_rate))
        
        # If we have too many frames, sample uniformly
        if len(indices) > target_frames:
            step = len(indices) // target_frames
            indices = indices[::step][:target_frames]
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        # Pad or truncate to target_frames
        while len(frames) < target_frames:
            frames.append(frames[-1] if frames else np.zeros_like(frame))
        frames = frames[:target_frames]
        
        return np.array(frames)
    
    def process_frames(self, frames: np.ndarray, training: bool = False) -> torch.Tensor:
        """Process frames with transformations"""
        transform = self._get_train_transforms() if training else self.transform
        
        processed_frames = []
        for frame in frames:
            transformed = transform(image=frame)
            processed_frames.append(transformed['image'])
        
        # Stack frames: (T, C, H, W)
        video_tensor = torch.stack(processed_frames)
        
        # Rearrange to (C, T, H, W) for 3D CNN
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        return video_tensor


class HRI30Dataset(Dataset):
    """
    HRI30 Dataset implementation
    Based on the dataset structure described in the paper
    """
    
    def __init__(
        self, 
        data_config: DataConfig,
        split: str = "train",
        split_id: int = 1,
        transform: Optional[A.Compose] = None
    ):
        self.config = data_config
        self.split = split
        self.split_id = split_id
        self.processor = VideoProcessor(data_config)
        
        # Parse video files with naming convention: v_L_gG_cC.avi
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split} split {split_id}")
    
    def _parse_filename(self, filename: str) -> Dict[str, Any]:
        """
        Parse HRI30 filename format: v_L_gG_cC.avi
        L = action class label, G = group, C = clip number
        """
        basename = os.path.splitext(filename)[0]
        parts = basename.split('_')
        
        if len(parts) >= 4 and parts[0] == 'v':
            try:
                class_label = int(parts[1])
                group = int(parts[2][1:])  # Remove 'g' prefix
                clip = int(parts[3][1:])   # Remove 'c' prefix
                
                return {
                    'class_label': class_label,
                    'group': group,
                    'clip': clip,
                    'filename': filename
                }
            except ValueError:
                pass
        
        return None
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load video samples based on split"""
        samples = []
        
        # Look for video files in train_set directory
        video_dir = os.path.join(self.config.data_root, self.config.train_dir)
        
        if not os.path.exists(video_dir):
            print(f"Warning: {video_dir} does not exist. Looking for video files in root directory...")
            video_dir = self.config.data_root
        
        # Find all video files
        video_patterns = ["*.avi", "*.mp4", "*.mov"]
        video_files = []
        
        for pattern in video_patterns:
            video_files.extend(glob.glob(os.path.join(video_dir, pattern)))
            video_files.extend(glob.glob(os.path.join(video_dir, "**", pattern), recursive=True))
        
        print(f"Found {len(video_files)} video files")
        
        # Parse filenames and create samples
        for video_file in video_files:
            filename = os.path.basename(video_file)
            parsed = self._parse_filename(filename)
            
            if parsed is not None:
                sample = {
                    'video_path': video_file,
                    'class_idx': parsed['class_label'] % self.config.num_classes,  # Ensure valid class
                    'class_name': self.config.action_classes[parsed['class_label'] % self.config.num_classes],
                    'group': parsed['group'],
                    'clip': parsed['clip']
                }
                samples.append(sample)
            else:
                # If filename doesn't match pattern, try to infer from directory structure
                rel_path = os.path.relpath(video_file, self.config.data_root)
                path_parts = rel_path.split(os.sep)
                
                # Try to find class name in path
                for part in path_parts:
                    if part in self.config.action_classes:
                        sample = {
                            'video_path': video_file,
                            'class_idx': self.config.action_classes.index(part),
                            'class_name': part,
                            'group': 0,
                            'clip': 0
                        }
                        samples.append(sample)
                        break
        
        # Apply train/test split based on paper specifications
        total_samples = len(samples)
        if total_samples == 0:
            print("Warning: No valid samples found!")
            return []
        
        # Random split based on split_id
        random.seed(42 + self.split_id)  # Ensure reproducibility
        random.shuffle(samples)
        
        if self.split == "train":
            split_ratio = self.config.train_split_ratios[self.split_id - 1]
        else:
            split_ratio = self.config.test_split_ratios[self.split_id - 1]
        
        if self.split == "train":
            samples = samples[:int(total_samples * split_ratio)]
        else:
            samples = samples[int(total_samples * self.config.train_split_ratios[self.split_id - 1]):]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        try:
            # Load video frames
            frames = self.processor.load_video(sample['video_path'])
            
            # Process frames
            video_tensor = self.processor.process_frames(
                frames, training=(self.split == "train")
            )
            
            return video_tensor, sample['class_idx']
        
        except Exception as e:
            print(f"Error loading {sample['video_path']}: {e}")
            # Return dummy data
            dummy_video = torch.zeros(3, 8, *self.config.input_resolution)
            return dummy_video, sample['class_idx']


class HRI30DataModule:
    """Data module for easy dataset management"""
    
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        split_id: int = 1
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.split_id = split_id
    
    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        dataset = HRI30Dataset(
            self.data_config,
            split="train",
            split_id=self.split_id
        )
        
        return DataLoader(
            dataset,
            batch_size=self.model_config.batch_size,
            shuffle=True,
            num_workers=self.model_config.num_workers,
            pin_memory=self.model_config.pin_memory,
            drop_last=True
        )
    
    def get_val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        dataset = HRI30Dataset(
            self.data_config,
            split="test", 
            split_id=self.split_id
        )
        
        return DataLoader(
            dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=self.model_config.num_workers,
            pin_memory=self.model_config.pin_memory
        )
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Calculate class weights for imbalanced datasets"""
        train_dataset = HRI30Dataset(self.data_config, split="train", split_id=self.split_id)
        
        if len(train_dataset) == 0:
            return None
        
        class_counts = torch.zeros(self.data_config.num_classes)
        for _, class_idx in train_dataset:
            class_counts[class_idx] += 1
        
        # Avoid division by zero
        class_counts = torch.clamp(class_counts, min=1)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        
        return class_weights


def create_sample_data_structure():
    """Create sample data structure for testing"""
    base_dir = "/home/ty/human-bahviour"
    
    # Create directories
    os.makedirs(os.path.join(base_dir, "train_set"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "annotations"), exist_ok=True)
    
    # Create sample annotation file
    sample_annotations = {
        "classes": DataConfig().action_classes,
        "splits": {
            "split1": {"train": 2100, "test": 840},
            "split2": {"train": 2310, "test": 630}, 
            "split3": {"train": 1890, "test": 1050}
        },
        "total_clips": 2940,
        "clips_per_class": 98
    }
    
    with open(os.path.join(base_dir, "annotations", "dataset_info.json"), 'w') as f:
        json.dump(sample_annotations, f, indent=2)
    
    print(f"Created sample data structure in {base_dir}")


if __name__ == "__main__":
    # Test the data loader
    from config import get_config_for_device
    
    config = get_config_for_device()
    data_config = config['data']
    model_config = config['model']
    
    # Create sample structure if needed
    if not os.path.exists(os.path.join(data_config.data_root, "train_set")):
        create_sample_data_structure()
    
    # Test dataset
    try:
        data_module = HRI30DataModule(data_config, model_config)
        train_loader = data_module.get_train_dataloader()
        
        print(f"Dataset loaded successfully!")
        print(f"Train dataset size: {len(train_loader.dataset)}")
        
        # Test one batch
        for batch_idx, (videos, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Video shape: {videos.shape}, Labels shape: {labels.shape}")
            print(f"Label examples: {labels[:5]}")
            break
            
    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("Make sure to add your video files to the train_set directory")