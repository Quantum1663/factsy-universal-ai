"""
Hybrid Training Orchestrator for BharatVerify Transformer
Coordinates local development with cloud training for optimal results
"""

import os
import json
import subprocess
import wandb
from pathlib import Path
from typing import Dict, List, Optional
import torch
from datetime import datetime

from ..ml.bharat_transformer import BharatVerifyTransformer, BharatTokenizer

class HybridTrainingOrchestrator:
    """Orchestrates training across local and cloud resources"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.git_repo = "factsy-universal-ai"  # Your GitHub repo name
        
        # Cloud platform configurations
        self.platforms = {
            'local': {
                'gpu_memory': 4,      # GB - your laptop
                'batch_size': 1,
                'gradient_accumulation': 32,
                'cost_per_hour': 0.0
            },
            'colab': {
                'gpu_memory': 16,     # GB - T4/P100
                'batch_size': 8,
                'gradient_accumulation': 4,
                'cost_per_hour': 0.83,
                'max_hours': 12
            },
            'kaggle': {
                'gpu_memory': 16,     # GB - P100
                'batch_size': 6,
                'gradient_accumulation': 6,
                'cost_per_hour': 0.0,
                'max_hours': 9,
                'weekly_limit': 30
            }
        }
        
        # Training stages
        self.training_stages = {
            'prototype': {'platform': 'local', 'epochs': 3, 'data_size': 1000},
            'development': {'platform': 'colab', 'epochs': 10, 'data_size': 10000},
            'production': {'platform': 'kaggle', 'epochs': 20, 'data_size': 100000}
        }
    
    def create_optimized_config(self, platform: str, stage: str) -> Dict:
        """Create training config optimized for specific platform"""
        platform_config = self.platforms[platform]
        stage_config = self.training_stages[stage]
        
        config = {
            # Model configuration
            'model': {
                'vocab_size': 15000 if platform == 'local' else 30000,
                'embed_dim': 256 if platform == 'local' else 768,
                'num_layers': 4 if platform == 'local' else 12,
                'num_heads': 8,
                'ff_dim': 1024 if platform == 'local' else 3072,
                'num_communities': 15,
                'max_length': 256 if platform == 'local' else 512
            },
            
            # Training configuration
            'training': {
                'batch_size': platform_config['batch_size'],
                'gradient_accumulation_steps': platform_config['gradient_accumulation'],
                'num_epochs': stage_config['epochs'],
                'learning_rate': 2e-5,
                'warmup_steps': 500,
                'weight_decay': 0.01,
                'mixed_precision': True,
                'gradient_checkpointing': platform == 'local'
            },
            
            # Data configuration
            'data': {
                'max_samples': stage_config['data_size'],
                'validation_split': 0.1,
                'languages': ['hi', 'en', 'bn', 'ta', 'te', 'mr', 'gu'],
                'domains': ['politics', 'health', 'economics', 'sports', 'community']
            },
            
            # Platform-specific settings
            'platform': {
                'name': platform,
                'gpu_memory': platform_config['gpu_memory'],
                'checkpoint_frequency': 500 if platform == 'local' else 1000,
                'logging_steps': 10 if platform == 'local' else 50
            }
        }
        
        return config
    
    def sync_to_github(self, stage: str):
        """Sync latest code to GitHub for cloud access"""
        print(f"🔄 Syncing code for {stage} stage...")
        
        try:
            # Add all changes
            subprocess.run(['git', 'add', '.'], cwd=self.project_root, check=True)
            
            # Commit with stage-specific message
            commit_msg = f"🚀 Sync for {stage} training stage - {datetime.now().strftime('%Y%m%d_%H%M')}"
            subprocess.run(['git', 'commit', '-m', commit_msg], 
                         cwd=self.project_root, check=True)
            
            # Push to GitHub
            subprocess.run(['git', 'push'], cwd=self.project_root, check=True)
            
            print("✅ Code synced to GitHub successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git sync failed: {e}")
            return False
    
    def generate_colab_notebook(self, config: Dict) -> str:
        """Generate Google Colab notebook for cloud training"""
        notebook_content = f'''{{
  "cells": [
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": [
        "# 🧠 BharatVerify Transformer - Cloud Training\\n",
        "## Revolutionary Community-Aware Language Model\\n",
        "\\n",
        "Training configuration: {config['platform']['name']} - {config['training']['num_epochs']} epochs"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Mount Google Drive for data access\\n",
        "from google.colab import drive\\n",
        "drive.mount('/content/drive')\\n",
        "\\n",
        "# Clone your repository\\n",
        "!git clone https://github.com/yourusername/{self.git_repo}.git\\n",
        "%cd {self.git_repo}"
      ]
    }},
    {{
      "cell_type": "code", 
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Install dependencies\\n",
        "!pip install torch transformers datasets wandb accelerate\\n",
        "!pip install deepspeed # For memory optimization\\n",
        "\\n",
        "# Setup Weights & Biases for tracking\\n",
        "import wandb\\n",
        "wandb.login()\\n",
        "\\n",
        "# Initialize experiment\\n",
        "wandb.init(\\n",
        "    project='bharat-verify-transformer',\\n",
        "    name='colab-training-{datetime.now().strftime('%Y%m%d_%H%M')}',\\n",
        "    config={json.dumps(config, indent=2)}\\n",
        ")"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Import your revolutionary model\\n",
        "import sys\\n",
        "sys.path.append('.')\\n",
        "\\n",
        "from backend.app.ml.bharat_transformer import BharatVerifyTransformer, BharatTokenizer\\n",
        "from backend.app.services.domain_classifier import DomainClassifier\\n",
        "\\n",
        "# Initialize model with cloud-optimized config\\n",
        "model = BharatVerifyTransformer(\\n",
        "    vocab_size={config['model']['vocab_size']},\\n",
        "    embed_dim={config['model']['embed_dim']},\\n",
        "    num_layers={config['model']['num_layers']},\\n",
        "    num_heads={config['model']['num_heads']},\\n",
        "    ff_dim={config['model']['ff_dim']},\\n",
        "    num_communities={config['model']['num_communities']}\\n",
        ")\\n",
        "\\n",
        "tokenizer = BharatTokenizer()\\n",
        "\\n",
        "print(f'🧠 BharatVerify Transformer initialized with {{model.embed_dim}} dimensions')\\n",
        "print(f'📊 Model has {{sum(p.numel() for p in model.parameters()):,}} parameters')"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Load and prepare training data\\n",
        "from scripts.prepare_training_data import load_indian_misinformation_dataset\\n",
        "\\n",
        "# Load curated Indian dataset\\n",
        "train_dataset, val_dataset = load_indian_misinformation_dataset(\\n",
        "    max_samples={config['data']['max_samples']},\\n",
        "    languages={config['data']['languages']},\\n",
        "    domains={config['data']['domains']}\\n",
        ")\\n",
        "\\n",
        "print(f'📚 Training samples: {{len(train_dataset)}}')\\n",
        "print(f'🧪 Validation samples: {{len(val_dataset)}}')\\n",
        "\\n",
        "# Display sample data\\n",
        "sample = train_dataset[0]\\n",
        "print(f'📝 Sample: {{sample[\"text\"][:100]}}...')\\n",
        "print(f'🏷️ Label: {{sample[\"label\"]}}')\\n",
        "print(f'🤝 Community: {{sample[\"community\"]}}')"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Start revolutionary training\\n",
        "from scripts.train_bharat_transformer import train_model\\n",
        "\\n",
        "# Training configuration\\n",
        "training_args = {{\\n",
        "    'output_dir': './checkpoints',\\n",
        "    'num_train_epochs': {config['training']['num_epochs']},\\n",
        "    'per_device_train_batch_size': {config['training']['batch_size']},\\n",
        "    'gradient_accumulation_steps': {config['training']['gradient_accumulation_steps']},\\n",
        "    'learning_rate': {config['training']['learning_rate']},\\n",
        "    'warmup_steps': {config['training']['warmup_steps']},\\n",
        "    'fp16': {str(config['training']['mixed_precision']).lower()},\\n",
        "    'logging_steps': {config['platform']['logging_steps']},\\n",
        "    'save_steps': {config['platform']['checkpoint_frequency']},\\n",
        "    'evaluation_strategy': 'steps',\\n",
        "    'eval_steps': {config['platform']['checkpoint_frequency']},\\n",
        "    'save_total_limit': 3,\\n",
        "    'load_best_model_at_end': True,\\n",
        "    'metric_for_best_model': 'accuracy',\\n",
        "    'report_to': 'wandb'\\n",
        "}}\\n",
        "\\n",
        "# Start training your revolutionary model!\\n",
        "print('🚀 Starting BharatVerify Transformer training...')\\n",
        "final_model = train_model(model, tokenizer, train_dataset, val_dataset, training_args)\\n",
        "\\n",
        "print('🎉 Training completed successfully!')\\n",
        "print('💾 Model saved to ./checkpoints/')\\n",
        "\\n",
        "# Upload to Google Drive for download\\n",
        "!cp -r ./checkpoints /content/drive/MyDrive/bharat-verify-checkpoints\\n",
        "print('☁️ Checkpoints uploaded to Google Drive!')"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": [
        "# Test your trained model\\n",
        "test_claims = [\\n",
        "    \\"प्रधानमंत्री मोदी ने कहा कि भारत की अर्थव्यवस्था मजबूत है।\\",  # Hindi political\\n",
        "    \\"All Muslims are trying to impose Sharia law in India\\",  # Community-targeted\\n",
        "    \\"Turmeric can cure COVID-19 completely\\",  # Health misinformation\\n",
        "    \\"Virat Kohli scored 200 runs in last match\\"  # Sports claim\\n",
        "]\\n",
        "\\n",
        "print('🧪 Testing your revolutionary model:')\\n",
        "for claim in test_claims:\\n",
        "    # Tokenize\\n",
        "    inputs = tokenizer.encode(claim)\\n",
        "    \\n",
        "    # Predict\\n",
        "    with torch.no_grad():\\n",
        "        outputs = final_model(**inputs)\\n",
        "    \\n",
        "    print(f'\\n📝 Claim: {{claim}}')\\n",
        "    print(f'🎯 Analysis: Community-aware processing complete')\\n",
        "    print(f'🧠 Model output shape: {{outputs[\"last_hidden_state\"].shape}}')"
      ]
    }}
  ],
  "metadata": {{
    "accelerator": "GPU",
    "colab": {{
      "provenance": [],
      "gpuType": "T4"
    }},
    "kernelspec": {{
      "display_name": "Python 3",
      "name": "python3"
    }},
    "language_info": {{
      "name": "python"
    }}
  }},
  "nbformat": 4,
  "nbformat_minor": 0
}}'''
        
        return notebook_content
    
    def launch_training_stage(self, stage: str) -> bool:
        """Launch training for specific stage"""
        print(f"🚀 Launching {stage} training stage...")
        
        # Get platform for this stage
        platform = self.training_stages[stage]['platform']
        
        # Generate optimized config
        config = self.create_optimized_config(platform, stage)
        
        if platform == 'local':
            return self.run_local_training(config)
        elif platform == 'colab':
            return self.prepare_colab_training(config)
        elif platform == 'kaggle':
            return self.prepare_kaggle_training(config)
        
        return False
    
    def run_local_training(self, config: Dict) -> bool:
        """Run training on local machine (your laptop)"""
        print("🏠 Starting local training on your laptop...")
        
        try:
            # Import training script
            from ..scripts.train_local import train_bharat_model
            
            # Run training with laptop-optimized config
            train_bharat_model(config)
            
            print("✅ Local training completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Local training failed: {e}")
            return False
    
    def prepare_colab_training(self, config: Dict) -> bool:
        """Prepare Google Colab notebook for cloud training"""
        print("☁️ Preparing Google Colab training...")
        
        # Sync code to GitHub
        if not self.sync_to_github('colab'):
            return False
        
        # Generate Colab notebook
        notebook_content = self.generate_colab_notebook(config)
        
        # Save notebook for manual execution
        notebook_path = self.project_root / 'colab_training.ipynb'
        with open(notebook_path, 'w') as f:
            f.write(notebook_content)
        
        print(f"📓 Google Colab notebook created: {notebook_path}")
        print("🔗 Next steps:")
        print("   1. Upload colab_training.ipynb to Google Colab")
        print("   2. Run all cells to start training")
        print("   3. Monitor progress in Weights & Biases")
        print("   4. Download trained model from Google Drive")
        
        return True
    
    def create_training_roadmap(self) -> Dict:
        """Create comprehensive training roadmap"""
        roadmap = {
            'week_1': {
                'focus': 'Local Development & Prototyping',
                'tasks': [
                    'Complete BharatVerify Transformer architecture ✅',
                    'Implement domain classification module',
                    'Create evidence retrieval system',
                    'Build basic training pipeline',
                    'Test on sample Indian data'
                ],
                'deliverable': 'Working prototype on laptop'
            },
            
            'week_2': {
                'focus': 'Cloud Training & Scaling',
                'tasks': [
                    'Deploy training on Google Colab Pro',
                    'Run medium-scale experiments on Kaggle',
                    'Optimize hyperparameters across platforms',
                    'Train production model with full dataset',
                    'Evaluate model performance'
                ],
                'deliverable': 'Production-ready trained model'
            },
            
            'week_3': {
                'focus': 'Demo Preparation & Polish',
                'tasks': [
                    'Build interactive web interface',
                    'Create impressive demo scenarios',
                    'Prepare presentation materials',
                    'Performance optimization',
                    'Final testing and validation'
                ],
                'deliverable': 'Presentation-ready product'
            }
        }
        
        return roadmap

# Usage example
if __name__ == "__main__":
    orchestrator = HybridTrainingOrchestrator()
    
    # Show training roadmap
    roadmap = orchestrator.create_training_roadmap()
    print("🗺️ BharatVerify Transformer Training Roadmap:")
    for week, details in roadmap.items():
        print(f"\n📅 {week.upper()}: {details['focus']}")
        for task in details['tasks']:
            print(f"   • {task}")
        print(f"   🎯 Deliverable: {details['deliverable']}")
    
    # Launch prototype stage
    print("\n🚀 Starting prototype training stage...")
    success = orchestrator.launch_training_stage('prototype')
    
    if success:
        print("✅ Ready for cloud training stages!")
    else:
        print("❌ Please fix local setup before proceeding")
