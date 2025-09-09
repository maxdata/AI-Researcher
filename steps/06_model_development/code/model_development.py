#!/usr/bin/env python3
"""
Model Development Module for AI-Researcher Pipeline
Implements ML models, training pipelines, and testing infrastructure.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeveloper:
    """Handles complete ML project implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_implementation_plan(self) -> Dict[str, Any]:
        """Load the implementation plan"""
        with open("inputs/implementation_plan.json") as f:
            plan = json.load(f)
        return plan
    
    def generate_project_structure(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete ML project structure"""
        project_name = plan["architecture"]["model_name"]
        
        structure = {
            "project_name": project_name,
            "directories": [
                "src/models/",
                "src/data/", 
                "src/training/",
                "src/evaluation/",
                "src/utils/",
                "configs/",
                "experiments/",
                "notebooks/",
                "tests/",
                "data/raw/",
                "data/processed/",
                "models/saved/",
                "results/"
            ],
            "files": {
                "requirements.txt": self._generate_requirements(plan),
                "setup.py": self._generate_setup_py(project_name),
                "README.md": self._generate_readme(plan),
                "src/models/base_model.py": self._generate_base_model(plan),
                "src/models/custom_model.py": self._generate_custom_model(plan),
                "src/data/dataset.py": self._generate_dataset_code(plan),
                "src/training/trainer.py": self._generate_trainer_code(plan),
                "src/evaluation/evaluator.py": self._generate_evaluator_code(plan),
                "configs/config.yaml": self._generate_config(plan),
                "train.py": self._generate_train_script(plan),
                "evaluate.py": self._generate_eval_script(plan)
            }
        }
        
        return structure
    
    def create_project_files(self, structure: Dict[str, Any]) -> str:
        """Create the actual project files"""
        project_dir = Path("outputs/project")
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories
        for dir_path in structure["directories"]:
            (project_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create files
        for file_path, content in structure["files"].items():
            file_full_path = project_dir / file_path
            file_full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_full_path, "w") as f:
                f.write(content)
        
        logger.info(f"Created project structure at {project_dir}")
        return str(project_dir)
    
    def _generate_requirements(self, plan: Dict[str, Any]) -> str:
        """Generate requirements.txt content"""
        framework = plan["architecture"]["implementation_framework"]
        
        base_requirements = [
            "numpy>=1.21.0",
            "scipy>=1.7.0", 
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "tqdm>=4.62.0",
            "tensorboard>=2.7.0",
            "wandb>=0.12.0",
            "hydra-core>=1.1.0",
            "omegaconf>=2.1.0"
        ]
        
        if framework == "PyTorch":
            base_requirements.extend([
                "torch>=1.12.0",
                "torchvision>=0.13.0",
                "pytorch-lightning>=1.7.0"
            ])
        elif framework == "TensorFlow":
            base_requirements.extend([
                "tensorflow>=2.9.0",
                "keras>=2.9.0"
            ])
        
        # Add category-specific requirements
        category = plan.get("research_idea", {}).get("category", "general")
        if "gnn" in category or any("graph" in comp.lower() for comp in plan["architecture"]["core_components"]):
            base_requirements.append("torch-geometric>=2.1.0")
        
        return "\n".join(base_requirements)
    
    def _generate_base_model(self, plan: Dict[str, Any]) -> str:
        """Generate base model implementation"""
        framework = plan["architecture"]["implementation_framework"]
        
        if framework == "PyTorch":
            return '''import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    """Base model implementation"""
    
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        
    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented")
    
    def loss_function(self, outputs, targets):
        """Default loss function"""
        return F.mse_loss(outputs, targets)
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return optimizer, scheduler
'''
        else:
            return '''import tensorflow as tf
from tensorflow import keras

class BaseModel(keras.Model):
    """Base model implementation"""
    
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
    
    def call(self, inputs, training=False):
        raise NotImplementedError("Call method must be implemented")
    
    def compute_loss(self, outputs, targets):
        """Default loss function"""
        return tf.keras.losses.mse(outputs, targets)
'''
    
    def _generate_custom_model(self, plan: Dict[str, Any]) -> str:
        """Generate custom model based on research idea"""
        idea_title = plan["research_idea"]["title"]
        components = plan["architecture"]["core_components"]
        
        return f'''"""
Custom Model Implementation: {idea_title}

This module implements the novel approach described in the research idea.
Core components: {", ".join(components)}
"""

import torch
import torch.nn as nn
from .base_model import BaseModel

class CustomModel(BaseModel):
    """
    Implementation of: {idea_title}
    
    Architecture components:
    {chr(10).join(f"    - {comp}" for comp in components)}
    """
    
    def __init__(self, config):
        super(CustomModel, self).__init__(config)
        
        # Initialize components based on research idea
        self.input_dim = config.get('input_dim', 784)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.output_dim = config.get('output_dim', 10)
        
        # Core model components
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # Novel components would be implemented here
        # TODO: Implement specific novel components from research idea
        
    def forward(self, x):
        """Forward pass implementation"""
        encoded = self.encoder(x)
        
        # Apply novel processing here
        processed = self._apply_novel_components(encoded)
        
        decoded = self.decoder(processed)
        return decoded
    
    def _apply_novel_components(self, x):
        """Apply novel components from research idea"""
        # Placeholder for novel component implementation
        # This would be customized based on the specific research idea
        return x
    
    def loss_function(self, outputs, targets):
        """Custom loss function for the research idea"""
        # Base reconstruction loss
        recon_loss = super().loss_function(outputs, targets)
        
        # Add any novel loss terms here
        novel_loss = 0.0  # Placeholder for novel loss components
        
        total_loss = recon_loss + novel_loss
        return total_loss
'''
    
    def _generate_dataset_code(self, plan: Dict[str, Any]) -> str:
        """Generate dataset handling code"""
        datasets = plan.get("datasets", [])
        dataset_names = [d["name"] for d in datasets]
        
        return f'''"""
Dataset handling for {", ".join(dataset_names)}
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np

class CustomDataset(Dataset):
    """Custom dataset implementation"""
    
    def __init__(self, data_path, transform=None, split='train'):
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split
        
        # Load data
        self.data, self.labels = self._load_data()
    
    def _load_data(self):
        """Load dataset from path"""
        # Placeholder implementation
        # This would be customized based on selected datasets
        data = torch.randn(1000, 784)  # Example data
        labels = torch.randint(0, 10, (1000,))  # Example labels
        return data, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

def create_dataloaders(config):
    """Create train/val/test dataloaders"""
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create datasets
    train_dataset = CustomDataset(
        config.data_path, 
        transform=transform, 
        split='train'
    )
    
    val_dataset = CustomDataset(
        config.data_path,
        transform=transform,
        split='val'
    )
    
    test_dataset = CustomDataset(
        config.data_path,
        transform=transform, 
        split='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    return train_loader, val_loader, test_loader
'''
    
    def _generate_trainer_code(self, plan: Dict[str, Any]) -> str:
        """Generate training pipeline code"""
        return '''"""
Training pipeline implementation
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Trainer:
    """Model trainer with comprehensive logging and checkpointing"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = model.configure_optimizers()
        
        # Setup logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.model.loss_function(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if batch_idx % self.config.log_interval == 0:
                step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Loss/Train', loss.item(), step)
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.model.loss_function(output, target)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        
        # Log validation loss
        self.writer.add_scalar('Loss/Validation', avg_loss, self.current_epoch)
        
        return avg_loss
    
    def save_checkpoint(self, filepath, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.parent / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def train(self, train_loader, val_loader, num_epochs):
        """Complete training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate epoch
            val_loss = self.validate_epoch(val_loader)
            
            # Step scheduler
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pth'
            self.save_checkpoint(checkpoint_path, is_best)
            
            logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        self.writer.close()
        logger.info("Training completed!")
'''
    
    def _generate_readme(self, plan: Dict[str, Any]) -> str:
        """Generate README.md"""
        title = plan["research_idea"]["title"]
        description = plan["research_idea"]["description"]
        
        return f'''# {title}

## Description
{description}

## Architecture
{plan["architecture"]["architecture_overview"]}

### Core Components
{chr(10).join(f"- {comp}" for comp in plan["architecture"]["core_components"])}

### Novel Components  
{chr(10).join(f"- {comp}" for comp in plan["architecture"].get("novel_components", []))}

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py --config configs/config.yaml
```

### Evaluation
```bash
python evaluate.py --model_path models/saved/best_model.pth
```

## Datasets
{chr(10).join(f"- **{d['name']}**: {d['purpose']}" for d in plan.get("datasets", []))}

## Results
Results will be saved in the `results/` directory.

## Project Structure
```
src/
├── models/          # Model implementations
├── data/           # Dataset handling
├── training/       # Training pipeline
├── evaluation/     # Evaluation metrics
└── utils/          # Utility functions
```

## Timeline
{plan["timeline"]["total_duration"]} estimated completion time.

## Requirements
- Python 3.8+
- {plan["architecture"]["implementation_framework"]}
- CUDA-compatible GPU recommended
'''
    
    def _generate_setup_py(self, project_name: str) -> str:
        """Generate setup.py"""
        return f'''from setuptools import setup, find_packages

setup(
    name="{project_name}",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    author="AI Researcher",
    description="Implementation of {project_name}",
    python_requires=">=3.8",
)
'''
    
    def _generate_config(self, plan: Dict[str, Any]) -> str:
        """Generate configuration file"""
        return '''# Model Configuration
model:
  name: "custom_model"
  input_dim: 784
  hidden_dim: 256
  output_dim: 10

# Training Configuration  
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  num_workers: 4
  
# Data Configuration
data:
  data_path: "data/"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

# Logging Configuration
logging:
  log_dir: "experiments/logs"
  checkpoint_dir: "models/saved"
  log_interval: 100

# Hardware Configuration
device: "cuda"
seed: 42
'''
    
    def _generate_train_script(self, plan: Dict[str, Any]) -> str:
        """Generate main training script"""
        return '''#!/usr/bin/env python3
"""
Main training script
"""

import argparse
import yaml
import torch
import logging
from pathlib import Path
from omegaconf import OmegaConf

from src.models.custom_model import CustomModel
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer

def main(config_path):
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    torch.manual_seed(config.seed)
    
    # Create model
    model = CustomModel(config.model)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config.data)
    
    # Create trainer
    trainer = Trainer(model, config.training)
    
    # Start training
    trainer.train(train_loader, val_loader, config.training.num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    args = parser.parse_args()
    
    main(args.config)
'''
    
    def _generate_eval_script(self, plan: Dict[str, Any]) -> str:
        """Generate evaluation script"""  
        return '''#!/usr/bin/env python3
"""
Model evaluation script
"""

import argparse
import torch
import logging
from pathlib import Path
from src.models.custom_model import CustomModel
from src.evaluation.evaluator import Evaluator

def main(model_path, data_path):
    logger = logging.getLogger(__name__)
    
    # Load checkpoint
    checkpoint = torch.load(model_path)
    
    # Create model and load weights
    model = CustomModel(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = Evaluator(model)
    
    # Run evaluation
    results = evaluator.evaluate(data_path)
    
    logger.info("Evaluation Results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to saved model")
    parser.add_argument("--data_path", required=True, help="Path to test data")
    args = parser.parse_args()
    
    main(args.model_path, args.data_path)
'''
    
    def run_development(self) -> Dict[str, Any]:
        """Run complete model development"""
        logger.info("Starting model development...")
        
        # Load implementation plan
        plan = self.load_implementation_plan()
        
        # Generate project structure
        structure = self.generate_project_structure(plan)
        
        # Create project files
        project_path = self.create_project_files(structure)
        
        # Generate development summary
        summary = {
            "project_path": project_path,
            "project_name": structure["project_name"],
            "files_created": len(structure["files"]),
            "directories_created": len(structure["directories"]),
            "implementation_status": "complete",
            "next_steps": [
                "Install dependencies: pip install -r requirements.txt",
                "Prepare datasets",
                "Configure training parameters",
                "Run training: python train.py",
                "Evaluate results: python evaluate.py"
            ]
        }
        
        logger.info("Model development completed successfully")
        return summary

def main():
    """Main entry point for model development"""
    try:
        config = {}  # Default config
        
        developer = ModelDeveloper(config)
        results = developer.run_development()
        
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save development summary
        with open(output_dir / "development_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("Model development completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Model development failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())