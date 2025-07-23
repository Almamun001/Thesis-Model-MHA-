#!/usr/bin/env python3
"""
Autonomous Data Augmentation and Quality Control Agent
======================================================

This script generates and validates synthetic text to balance four minority classes 
in a mental health dataset using GPT-2 and multiple quality control filters.

Classes to augment:
- Anxiety (3,838 existing → 6,500 target)
- Bipolar (2,777 existing → 6,500 target)  
- Stress (2,587 existing → 6,500 target)
- Personality disorder (1,077 existing → 6,500 target)

Requirements:
- Generate 10-50 word synthetic statements
- Use HuggingFace GPT-2 with batched generation
- Apply quality filters: perplexity, duplication, relevance, length
- Show progress bars and ensure reproducibility
- Output raw and validated CSV files
"""

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

class DataAugmentationAgent:
    """
    Autonomous agent for generating and validating synthetic mental health text data.
    """
    
    def __init__(self, 
                 model_name: str = "openai-community/gpt2",
                 batch_size: int = 16,
                 perplexity_threshold: float = 50.0,
                 similarity_threshold: float = 0.9,
                 relevance_min: float = 0.3,
                 relevance_max: float = 0.8,
                 min_words: int = 10,
                 max_words: int = 50):
        """
        Initialize the data augmentation agent.
        
        Args:
            model_name: HuggingFace model identifier
            batch_size: Number of samples to generate per batch
            perplexity_threshold: Maximum allowed perplexity score
            similarity_threshold: Minimum similarity for duplicate detection
            relevance_min: Minimum relevance score to class centroid
            relevance_max: Maximum relevance score to class centroid
            min_words: Minimum word count for generated text
            max_words: Maximum word count for generated text
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.perplexity_threshold = perplexity_threshold
        self.similarity_threshold = similarity_threshold  
        self.relevance_min = relevance_min
        self.relevance_max = relevance_max
        self.min_words = min_words
        self.max_words = max_words
        
        # Class information
        self.classes = {
            'Anxiety': {'existing': 3838, 'target': 6500},
            'Bipolar': {'existing': 2777, 'target': 6500},
            'Stress': {'existing': 2587, 'target': 6500},
            'Personality disorder': {'existing': 1077, 'target': 6500}
        }
        
        # Initialize models
        print("🚀 Initializing Data Augmentation Agent...")
        self._load_models()
        
        # Storage for generated data
        self.raw_synthetic_data = []
        self.validated_synthetic_data = []
        self.existing_statements = set()
        self.class_centroids = {}
        
    def _load_models(self):
        """Load GPT-2 and sentence transformer models."""
        print(f"📥 Loading GPT-2 model: {self.model_name}")
        
        # Load GPT-2 for generation
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set up generation pipeline
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Load sentence transformer for embeddings
        print("📥 Loading sentence transformer for relevance scoring...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"✅ Models loaded successfully on {device}")
    
    def _create_class_prompts(self) -> Dict[str, List[str]]:
        """Create contextual prompts for each mental health class."""
        prompts = {
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
        return prompts
        
    def _clean_generated_text(self, text: str) -> str:
        """Clean and validate generated text length."""
        # Remove extra whitespaces and newlines
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove incomplete sentences at the end
        sentences = cleaned.split('.')
        if len(sentences) > 1 and sentences[-1].strip() == '':
            cleaned = '.'.join(sentences[:-1]) + '.'
        
        return cleaned
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _calculate_perplexity(self, text: str) -> float:
        """Calculate GPT-2 perplexity for generated text."""
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                
            return perplexity
        except Exception as e:
            print(f"⚠️  Error calculating perplexity: {e}")
            return float('inf')
    
    def _is_duplicate(self, text: str, existing_texts: Set[str]) -> bool:
        """Check if text is a duplicate or near-duplicate."""
        text_lower = text.lower()
        
        # Exact match check
        if text_lower in existing_texts:
            return True
            
        # Fuzzy similarity check
        for existing in existing_texts:
            similarity = SequenceMatcher(None, text_lower, existing).ratio()
            if similarity >= self.similarity_threshold:
                return True
                
        return False
    
    def _calculate_class_centroids(self, existing_data: Dict[str, List[str]]):
        """Calculate embedding centroids for each class from existing data."""
        print("🎯 Calculating class centroids for relevance scoring...")
        
        for class_name, statements in existing_data.items():
            if statements:  # Only if we have existing data
                embeddings = self.sentence_model.encode(statements)
                centroid = np.mean(embeddings, axis=0)
                self.class_centroids[class_name] = centroid
                print(f"   ✅ {class_name}: centroid computed from {len(statements)} statements")
    
    def _calculate_relevance_score(self, text: str, class_name: str) -> float:
        """Calculate relevance score of text to class centroid."""
        if class_name not in self.class_centroids:
            return 0.5  # Default neutral score if no centroid available
            
        text_embedding = self.sentence_model.encode([text])
        centroid = self.class_centroids[class_name].reshape(1, -1)
        
        similarity = sklearn_cosine_similarity(text_embedding, centroid)[0][0]
        return float(similarity)
    
    def _validate_sample(self, text: str, class_name: str) -> Tuple[bool, Dict[str, any]]:
        """Apply all quality validation filters to a sample."""
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
        
        if not relevance_valid:
            return False, validation_results
            
        return True, validation_results
    
    def generate_batch(self, class_name: str, prompts: List[str], batch_size: int) -> List[str]:
        """Generate a batch of synthetic text for a specific class."""
        generated_texts = []
        
        # Randomly select prompts for this batch
        selected_prompts = random.choices(prompts, k=batch_size)
        
        for prompt in selected_prompts:
            try:
                # Generate text with GPT-2
                generated = self.generator(
                    prompt,
                    max_length=100,  # Rough token limit for 10-50 words
                    min_length=20,   # Minimum tokens 
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    truncation=True
                )[0]['generated_text']
                
                # Extract only the generated part (remove prompt)
                generated_part = generated[len(prompt):].strip()
                
                # Clean the generated text
                cleaned_text = self._clean_generated_text(generated_part)
                
                if cleaned_text:
                    generated_texts.append(cleaned_text)
                    
            except Exception as e:
                print(f"⚠️  Error generating text: {e}")
                continue
                
        return generated_texts
    
    def augment_class(self, class_name: str, existing_data: List[str] = None) -> Dict[str, int]:
        """Generate and validate synthetic data for a specific class."""
        print(f"\n🎯 Augmenting class: {class_name}")
        
        # Calculate how many samples we need
        existing_count = self.classes[class_name]['existing']
        target_count = self.classes[class_name]['target']
        needed_samples = target_count - existing_count
        
        print(f"   📊 Existing: {existing_count:,} | Target: {target_count:,} | Needed: {needed_samples:,}")
        
        if needed_samples <= 0:
            print(f"   ✅ Class {class_name} already balanced!")
            return {'generated': 0, 'validated': 0}
        
        # Get prompts for this class
        class_prompts = self._create_class_prompts()[class_name]
        
        # Add existing data to duplication check
        if existing_data:
            for text in existing_data:
                self.existing_statements.add(text.lower())
        
        # Generation loop with progress tracking
        validated_count = 0
        generated_count = 0
        attempts = 0
        max_attempts = needed_samples * 3  # Allow for filtering
        
        # Progress bar for generation
        pbar = tqdm(total=needed_samples, desc=f"Generating {class_name}", 
                   unit="samples", leave=True)
        
        while validated_count < needed_samples and attempts < max_attempts:
            # Calculate batch size (don't exceed remaining need)
            current_batch_size = min(self.batch_size, needed_samples - validated_count)
            
            # Generate batch
            batch_texts = self.generate_batch(class_name, class_prompts, current_batch_size)
            generated_count += len(batch_texts)
            
            # Validate each text in the batch
            for text in batch_texts:
                attempts += 1
                
                # Apply quality validation
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
                
                # Store validated data
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
        
        print(f"   ✅ Generated: {generated_count:,} | Validated: {validated_count:,} | Success rate: {validated_count/generated_count*100:.1f}%")
        
        return {'generated': generated_count, 'validated': validated_count}
    
    def load_existing_data(self, file_path: str = None) -> Dict[str, List[str]]:
        """Load existing dataset to calculate class centroids and avoid duplicates."""
        existing_data = defaultdict(list)
        
        # For demo purposes, create sample existing data
        # In practice, this would load from the actual dataset
        sample_data = {
            'Anxiety': [
                "I feel worried and nervous about upcoming events",
                "My anxiety makes me feel restless and unable to focus",
                "I can't stop thinking about worst case scenarios",
                "The constant worry is making me exhausted",
                "I feel overwhelmed and anxious when facing new situations"
            ],
            'Bipolar': [
                "My mood swings make me feel unstable and unpredictable",
                "Sometimes I feel extremely energetic and then crash completely",
                "The ups and downs of my emotions are exhausting",
                "During manic episodes I feel invincible but it doesn't last",
                "When depressed I experience complete hopelessness"
            ],
            'Stress': [
                "I feel overwhelmed by the pressure of daily responsibilities",
                "The stress is making it difficult to concentrate on anything",
                "I can't cope with all the demands placed on me",
                "I feel burned out and exhausted from constant pressure",
                "The workload is causing me to feel physically ill"
            ],
            'Personality disorder': [
                "I have difficulty maintaining stable relationships with others",
                "My sense of self changes depending on who I'm with",
                "I struggle with intense emotions that feel uncontrollable",
                "My behavior patterns tend to push people away",
                "I have trouble regulating emotions when stressed"
            ]
        }
        
        print("📚 Loading existing data for centroid calculation...")
        for class_name, statements in sample_data.items():
            existing_data[class_name] = statements
            print(f"   📊 {class_name}: {len(statements)} sample statements")
            
        return dict(existing_data)
    
    def save_results(self, raw_filename: str = "raw_synthetic.csv", 
                    validated_filename: str = "validated_synthetic.csv"):
        """Save generated results to CSV files."""
        print(f"\n💾 Saving results...")
        
        # Save raw synthetic data
        if self.raw_synthetic_data:
            raw_df = pd.DataFrame(self.raw_synthetic_data)
            raw_df.to_csv(raw_filename, index=False)
            print(f"   📄 Raw data saved: {raw_filename} ({len(raw_df):,} samples)")
        
        # Save validated synthetic data  
        if self.validated_synthetic_data:
            validated_df = pd.DataFrame(self.validated_synthetic_data)
            validated_df.to_csv(validated_filename, index=False)
            print(f"   📄 Validated data saved: {validated_filename} ({len(validated_df):,} samples)")
            
            # Show final distribution
            print(f"\n📊 Final validated distribution:")
            distribution = validated_df['status'].value_counts()
            for class_name, count in distribution.items():
                print(f"   {class_name}: {count:,} samples")
        
    def run_augmentation(self):
        """Main execution function to run the complete augmentation process."""
        print("🚀 Starting Data Augmentation Process")
        print("=" * 50)
        
        # Load existing data for centroids and duplication check
        existing_data = self.load_existing_data()
        self._calculate_class_centroids(existing_data)
        
        # Process each class
        total_stats = {'generated': 0, 'validated': 0}
        
        for class_name in self.classes.keys():
            class_existing = existing_data.get(class_name, [])
            stats = self.augment_class(class_name, class_existing)
            total_stats['generated'] += stats['generated']
            total_stats['validated'] += stats['validated']
        
        # Save results
        self.save_results()
        
        # Final summary
        print(f"\n🎉 Augmentation Complete!")
        print(f"   📊 Total generated: {total_stats['generated']:,}")
        print(f"   ✅ Total validated: {total_stats['validated']:,}")
        print(f"   📈 Overall success rate: {total_stats['validated']/total_stats['generated']*100:.1f}%")
        
        return total_stats


def main():
    """Main function to run the data augmentation agent."""
    print("🤖 Autonomous Data Augmentation and Quality Control Agent")
    print("=========================================================")
    
    # Initialize the agent
    agent = DataAugmentationAgent(
        model_name="openai-community/gpt2",
        batch_size=16,
        perplexity_threshold=50.0,
        similarity_threshold=0.9,
        relevance_min=0.3,
        relevance_max=0.8,
        min_words=10,
        max_words=50
    )
    
    # Run the augmentation process
    results = agent.run_augmentation()
    
    print("\n✨ Process completed successfully!")
    return results


if __name__ == "__main__":
    main()