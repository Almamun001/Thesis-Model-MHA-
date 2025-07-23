# Autonomous Data Augmentation and Quality Control Agent
# =====================================================
# 
# Self-contained Colab cell for generating synthetic mental health text
# Run this entire cell in Google Colab with T4 GPU enabled

# Install required packages
import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print(f"✅ Installed {package}")
    except Exception as e:
        print(f"❌ Failed to install {package}: {e}")

# Alternative pip install method for robust package management
def pip_install(package):
    """Alternative pip install function for package management."""
    os.system(f"pip install -q {package}")

# Install dependencies
required_packages = [
    "transformers==4.35.0",
    "sentence-transformers==2.2.2", 
    "torch",
    "tqdm",
    "pandas",
    "numpy", 
    "scikit-learn"
]

print("📦 Installing required packages...")
for package in required_packages:
    install_package(package)

print("\n🚀 Packages installed! Starting data augmentation agent...")

# Main augmentation code
import os
import re
import csv
import random
import warnings
from typing import List, Dict, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.nn.functional import cosine_similarity
import transformers

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

# ML Libraries
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    set_seed,
    pipeline
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from difflib import SequenceMatcher

# Set seeds for reproducibility
RANDOM_SEED = 42
set_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class ColabDataAugmentationAgent:
    """
    Colab-optimized autonomous agent for generating synthetic mental health text.
    """
    
    def __init__(self):
        """Initialize with Colab-optimized settings."""
        self.model_name = "openai-community/gpt2"
        self.batch_size = 8  # Smaller for Colab memory constraints
        self.perplexity_threshold = 50.0
        self.similarity_threshold = 0.9
        self.relevance_min = 0.3
        self.relevance_max = 0.8
        self.min_words = 10
        self.max_words = 50
        
        # Target samples per class (reduced for demo)
        self.target_per_class = 100  # Change to 6500 for full run
        
        # Class information
        self.classes = {
            'Anxiety': {'existing': 50, 'target': self.target_per_class},
            'Bipolar': {'existing': 40, 'target': self.target_per_class},
            'Stress': {'existing': 35, 'target': self.target_per_class},
            'Personality disorder': {'existing': 20, 'target': self.target_per_class}
        }
        
        print("🚀 Initializing Colab Data Augmentation Agent...")
        self._load_models()
        
        # Storage
        self.raw_synthetic_data = []
        self.validated_synthetic_data = []
        self.existing_statements = set()
        self.class_centroids = {}
        
    def _load_models(self):
        """Load models optimized for Colab."""
        print(f"📥 Loading GPT-2: {self.model_name}")
        
        # Load GPT-2
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Setup generation pipeline
        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )
        
        # Load sentence transformer
        print("📥 Loading sentence transformer...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        gpu_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        print(f"✅ Models loaded on {gpu_info}")
    
    def _create_class_prompts(self):
        """Create mental health class prompts."""
        return {
            'Anxiety': [
                "I feel worried and nervous about",
                "My anxiety makes me feel like",
                "I can't stop thinking about",
                "The constant worry is making me",
                "I feel overwhelmed and anxious when",
                "My heart races whenever I think about",
                "I have trouble sleeping because I worry about",
                "I feel tense and on edge about"
            ],
            'Bipolar': [
                "My mood swings make me feel",
                "Sometimes I feel extremely high and then",
                "The ups and downs of my emotions",
                "During my manic episodes I feel",
                "When I'm depressed I experience",
                "My energy levels fluctuate and I",
                "The mood changes affect my ability to",
                "I struggle with intense mood shifts that"
            ],
            'Stress': [
                "I feel overwhelmed by the pressure of",
                "The stress is making it difficult to",
                "I can't cope with all the demands",
                "I feel burned out and exhausted from",
                "The constant pressure makes me feel",
                "I'm struggling to manage all the stress from",
                "I feel like I'm at my breaking point with",
                "The workload is causing me to feel"
            ],
            'Personality disorder': [
                "I have difficulty maintaining relationships because",
                "My sense of self is unstable and",
                "I struggle with intense emotions that",
                "My behavior patterns tend to",
                "I have trouble regulating my emotions when",
                "My relationships are affected by my tendency to",  
                "I find it challenging to maintain consistent",
                "My emotional responses are often"
            ]
        }
        
    def _clean_text(self, text):
        """Clean generated text."""
        cleaned = re.sub(r'\s+', ' ', text.strip())
        sentences = cleaned.split('.')
        if len(sentences) > 1 and sentences[-1].strip() == '':
            cleaned = '.'.join(sentences[:-1]) + '.'
        return cleaned
    
    def _count_words(self, text):
        """Count words in text."""
        return len(text.split())
    
    def _calculate_perplexity(self, text):
        """Calculate GPT-2 perplexity."""
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            return perplexity
        except:
            return float('inf')
    
    def _is_duplicate(self, text, existing_texts):
        """Check for duplicates using fuzzy matching."""
        text_lower = text.lower()
        if text_lower in existing_texts:
            return True
            
        for existing in existing_texts:
            similarity = SequenceMatcher(None, text_lower, existing).ratio()
            if similarity >= self.similarity_threshold:
                return True
        return False
    
    def _calculate_class_centroids(self, existing_data):
        """Calculate class centroids for relevance scoring."""
        print("🎯 Calculating class centroids...")
        for class_name, statements in existing_data.items():
            if statements:
                embeddings = self.sentence_model.encode(statements)
                centroid = np.mean(embeddings, axis=0)
                self.class_centroids[class_name] = centroid
                print(f"   ✅ {class_name}: {len(statements)} statements")
    
    def _calculate_relevance_score(self, text, class_name):
        """Calculate relevance to class centroid."""
        if class_name not in self.class_centroids:
            return 0.5
            
        text_embedding = self.sentence_model.encode([text])
        centroid = self.class_centroids[class_name].reshape(1, -1)
        similarity = sklearn_cosine_similarity(text_embedding, centroid)[0][0]
        return float(similarity)
    
    def _validate_sample(self, text, class_name):
        """Apply all quality validation filters."""
        validation_results = {}
        
        # Length check
        word_count = self._count_words(text)
        length_valid = self.min_words <= word_count <= self.max_words
        validation_results['word_count'] = word_count
        validation_results['length_valid'] = length_valid
        
        if not length_valid:
            return False, validation_results
            
        # Perplexity check
        perplexity = self._calculate_perplexity(text)
        perplexity_valid = perplexity <= self.perplexity_threshold
        validation_results['perplexity'] = perplexity
        validation_results['perplexity_valid'] = perplexity_valid
        
        if not perplexity_valid:
            return False, validation_results
            
        # Duplication check
        is_duplicate = self._is_duplicate(text, self.existing_statements)
        validation_results['is_duplicate'] = is_duplicate
        
        if is_duplicate:
            return False, validation_results
            
        # Relevance check
        relevance_score = self._calculate_relevance_score(text, class_name)
        relevance_valid = self.relevance_min <= relevance_score <= self.relevance_max
        validation_results['relevance_score'] = relevance_score
        validation_results['relevance_valid'] = relevance_valid
        
        return relevance_valid, validation_results
    
    def generate_batch(self, class_name, prompts, batch_size):
        """Generate batch of texts for class."""
        generated_texts = []
        selected_prompts = random.choices(prompts, k=batch_size)
        
        for prompt in selected_prompts:
            try:
                generated = self.generator(
                    prompt,
                    max_length=80,
                    min_length=20,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    truncation=True
                )[0]['generated_text']
                
                generated_part = generated[len(prompt):].strip()
                cleaned_text = self._clean_text(generated_part)
                
                if cleaned_text:
                    generated_texts.append(cleaned_text)
                    
            except Exception as e:
                continue
                
        return generated_texts
    
    def augment_class(self, class_name, existing_data):
        """Generate and validate synthetic data for class."""
        print(f"\n🎯 Augmenting class: {class_name}")
        
        existing_count = self.classes[class_name]['existing']
        target_count = self.classes[class_name]['target']
        needed_samples = target_count - existing_count
        
        print(f"   📊 Existing: {existing_count} | Target: {target_count} | Needed: {needed_samples}")
        
        if needed_samples <= 0:
            return {'generated': 0, 'validated': 0}
        
        class_prompts = self._create_class_prompts()[class_name]
        
        if existing_data:
            for text in existing_data:
                self.existing_statements.add(text.lower())
        
        validated_count = 0
        generated_count = 0
        max_attempts = needed_samples * 2
        
        pbar = tqdm(total=needed_samples, desc=f"Generating {class_name}", unit="samples")
        
        while validated_count < needed_samples and generated_count < max_attempts:
            current_batch_size = min(self.batch_size, needed_samples - validated_count)
            batch_texts = self.generate_batch(class_name, class_prompts, current_batch_size)
            generated_count += len(batch_texts)
            
            for text in batch_texts:
                is_valid, validation_info = self._validate_sample(text, class_name)
                
                # Store raw data
                raw_sample = {
                    'statement': text,
                    'status': class_name,
                    'generated': True,
                    'validated': is_valid,
                    **validation_info
                }
                self.raw_synthetic_data.append(raw_sample)
                
                if is_valid:
                    validated_sample = {
                        'statement': text,
                        'status': class_name
                    }
                    self.validated_synthetic_data.append(validated_sample)
                    self.existing_statements.add(text.lower())
                    validated_count += 1
                    pbar.update(1)
                    
                    if validated_count >= needed_samples:
                        break
        
        pbar.close()
        success_rate = (validated_count/generated_count*100) if generated_count > 0 else 0
        print(f"   ✅ Generated: {generated_count} | Validated: {validated_count} | Success: {success_rate:.1f}%")
        
        return {'generated': generated_count, 'validated': validated_count}
    
    def load_sample_data(self):
        """Load sample existing data for demo."""
        sample_data = {
            'Anxiety': [
                "I feel worried and nervous about upcoming events and deadlines",
                "My anxiety makes me feel restless and unable to focus properly",
                "I can't stop thinking about worst case scenarios all the time",
                "The constant worry is making me exhausted and overwhelmed daily",
                "I feel overwhelmed and anxious when facing new social situations"
            ],
            'Bipolar': [
                "My mood swings make me feel unstable and unpredictable constantly",
                "Sometimes I feel extremely energetic and then crash completely afterwards",
                "The ups and downs of my emotions are exhausting for everyone",
                "During manic episodes I feel invincible but it never lasts long",
                "When depressed I experience complete hopelessness and despair every day"
            ],
            'Stress': [
                "I feel overwhelmed by the pressure of daily responsibilities and expectations",
                "The stress is making it difficult to concentrate on simple tasks",
                "I can't cope with all the demands placed on me right now",
                "I feel burned out and exhausted from constant pressure at work",
                "The workload is causing me to feel physically ill and drained"
            ],
            'Personality disorder': [
                "I have difficulty maintaining stable relationships with others around me",
                "My sense of self changes depending on who I'm interacting with",
                "I struggle with intense emotions that feel completely uncontrollable most days",
                "My behavior patterns tend to push people away from me consistently",
                "I have trouble regulating emotions when stressed or under pressure"
            ]
        }
        
        print("📚 Loading sample existing data...")
        for class_name, statements in sample_data.items():
            print(f"   📊 {class_name}: {len(statements)} sample statements")
            
        return sample_data
    
    def save_results(self):
        """Save results to CSV files."""
        print(f"\n💾 Saving results...")
        
        # Save raw synthetic data
        if self.raw_synthetic_data:
            raw_df = pd.DataFrame(self.raw_synthetic_data)
            raw_df.to_csv("raw_synthetic.csv", index=False)
            print(f"   📄 Raw data: raw_synthetic.csv ({len(raw_df)} samples)")
        
        # Save validated synthetic data  
        if self.validated_synthetic_data:
            validated_df = pd.DataFrame(self.validated_synthetic_data)
            validated_df.to_csv("validated_synthetic.csv", index=False)
            print(f"   📄 Validated data: validated_synthetic.csv ({len(validated_df)} samples)")
            
            # Show distribution
            print(f"\n📊 Final validated distribution:")
            distribution = validated_df['status'].value_counts()
            for class_name, count in distribution.items():
                print(f"   {class_name}: {count} samples")
                
        return raw_df if self.raw_synthetic_data else None, validated_df if self.validated_synthetic_data else None
        
    def run_augmentation(self):
        """Main execution function."""
        print("🚀 Starting Data Augmentation Process")
        print("=" * 50)
        
        # Load sample data
        existing_data = self.load_sample_data()
        self._calculate_class_centroids(existing_data)
        
        # Process each class
        total_stats = {'generated': 0, 'validated': 0}
        
        for class_name in self.classes.keys():
            class_existing = existing_data.get(class_name, [])
            stats = self.augment_class(class_name, class_existing)
            total_stats['generated'] += stats['generated']
            total_stats['validated'] += stats['validated']
        
        # Save results
        raw_df, validated_df = self.save_results()
        
        # Final summary
        print(f"\n🎉 Augmentation Complete!")
        print(f"   📊 Total generated: {total_stats['generated']}")
        print(f"   ✅ Total validated: {total_stats['validated']}")
        if total_stats['generated'] > 0:
            success_rate = total_stats['validated']/total_stats['generated']*100
            print(f"   📈 Overall success rate: {success_rate:.1f}%")
        
        return raw_df, validated_df

# Run the augmentation agent
print("🤖 Autonomous Data Augmentation and Quality Control Agent")
print("=========================================================")

# Initialize and run
agent = ColabDataAugmentationAgent()
raw_data, validated_data = agent.run_augmentation()

# Display sample results
if validated_data is not None and len(validated_data) > 0:
    print("\n📝 Sample Generated Statements:")
    print("-" * 40)
    for class_name in agent.classes.keys():
        class_samples = validated_data[validated_data['status'] == class_name].head(2)
        if len(class_samples) > 0:
            print(f"\n{class_name}:")
            for _, row in class_samples.iterrows():
                print(f"  • {row['statement']}")

print("\n✨ Process completed successfully!")
print("Files saved: raw_synthetic.csv, validated_synthetic.csv")