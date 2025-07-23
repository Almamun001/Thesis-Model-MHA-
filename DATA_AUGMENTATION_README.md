# Data Augmentation and Quality Control Agent

## Overview

This repository contains an autonomous data augmentation and quality control agent designed to generate synthetic text data for balancing minority classes in mental health datasets. The agent uses GPT-2 to generate contextually relevant text and applies multiple quality validation filters to ensure high-quality synthetic data.

## Target Classes

The agent is configured to augment four minority classes in mental health data:

- **Anxiety** (3,838 existing → 6,500 target samples)
- **Bipolar** (2,777 existing → 6,500 target samples) 
- **Stress** (2,587 existing → 6,500 target samples)
- **Personality disorder** (1,077 existing → 6,500 target samples)

## Features

### Generation Requirements ✅
- Uses HuggingFace "openai-community/gpt2" model with Transformers library
- Batched generation (configurable batch size, default 16) for efficiency
- Live progress bars with `tqdm` for generation and validation
- Fixed random seeds for reproducibility
- Automatic whitespace cleaning and word count enforcement (10-50 words)
- CSV output with `statement` and `status` columns

### Quality Validation Filters ✅
1. **Perplexity Filter**: Removes samples with GPT-2 perplexity above threshold (default: 50)
2. **Duplication Check**: Removes exact and near-duplicates using fuzzy string matching
3. **Relevance Scoring**: Uses sentence embeddings to ensure relevance to target class centroids (0.3-0.8 range)
4. **Length Check**: Enforces 10-50 word constraint on final output
5. **Balance Check**: Automatically generates additional batches until target sample count is reached

## Files

- `data_augmentation_agent.py` - Main implementation with full functionality
- `colab_data_augmentation.py` - Colab-optimized version with package installation
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Usage

### Method 1: Python Script

```bash
# Install dependencies
pip install -r requirements.txt

# Run the agent
python data_augmentation_agent.py
```

### Method 2: Google Colab (Recommended)

1. Upload `colab_data_augmentation.py` to your Colab environment
2. Enable GPU runtime (Runtime → Change runtime type → GPU → T4)
3. Run the entire cell - it will automatically install dependencies and execute

### Method 3: Jupyter Notebook Integration

```python
# Import and initialize the agent
from data_augmentation_agent import DataAugmentationAgent

# Create agent with custom settings
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

# Run augmentation process
results = agent.run_augmentation()
```

## Configuration Options

### DataAugmentationAgent Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | "openai-community/gpt2" | HuggingFace model identifier |
| `batch_size` | 16 | Samples per generation batch |
| `perplexity_threshold` | 50.0 | Maximum allowed perplexity score |
| `similarity_threshold` | 0.9 | Duplicate detection threshold |
| `relevance_min` | 0.3 | Minimum relevance to class centroid |
| `relevance_max` | 0.8 | Maximum relevance to class centroid |
| `min_words` | 10 | Minimum word count |
| `max_words` | 50 | Maximum word count |

## Output Files

### 1. raw_synthetic.csv
Contains all generated samples with validation metrics:
- `statement`: Generated text
- `status`: Target class label
- `generated`: Always True for synthetic data
- `validated`: Boolean indicating if sample passed all filters
- `word_count`: Number of words in statement
- `length_valid`: Boolean for length validation
- `perplexity`: GPT-2 perplexity score
- `perplexity_valid`: Boolean for perplexity validation
- `is_duplicate`: Boolean for duplication check
- `relevance_score`: Cosine similarity to class centroid
- `relevance_valid`: Boolean for relevance validation

### 2. validated_synthetic.csv
Contains only validated samples ready for training:
- `statement`: Generated text that passed all quality filters
- `status`: Target class label

## Quality Control Process

```
Generated Text
      ↓
Length Check (10-50 words)
      ↓
Perplexity Filter (≤50)
      ↓
Duplication Check (fuzzy matching)
      ↓
Relevance Scoring (0.3-0.8 vs class centroid)
      ↓
Validated Sample
```

## Expected Performance

- **Generation Speed**: ~100-200 samples per minute on T4 GPU
- **Validation Rate**: ~30-50% of generated samples pass all filters
- **Total Time**: ~2-3 hours for full dataset (26,000 samples)
- **Memory Usage**: ~2-4GB GPU memory

## Class-Specific Prompts

The agent uses contextually appropriate prompts for each mental health class:

### Anxiety Prompts
- "I feel worried and nervous about..."
- "My anxiety makes me feel like..."
- "I can't stop thinking about..."

### Bipolar Prompts  
- "My mood swings make me feel..."
- "Sometimes I feel extremely high and then..."
- "The ups and downs of my emotions..."

### Stress Prompts
- "I feel overwhelmed by the pressure of..."
- "The stress is making it difficult to..."
- "I can't cope with all the demands..."

### Personality Disorder Prompts
- "I have difficulty maintaining relationships because..."
- "My sense of self is unstable and..."
- "I struggle with intense emotions that..."

## Reproducibility

The agent ensures reproducibility through:
- Fixed random seeds (RANDOM_SEED = 42)
- Deterministic model initialization
- Consistent generation parameters
- Version-pinned dependencies

## Error Handling

The agent includes robust error handling for:
- Model loading failures
- Generation timeouts
- Validation errors
- File I/O issues
- Memory constraints

## Integration with Existing Pipeline

The validated synthetic data can be directly integrated with the existing MentalBERT classification pipeline:

```python
# Load existing data
original_df = pd.read_csv("Mental Health Dataset (Main).csv")

# Load synthetic data
synthetic_df = pd.read_csv("validated_synthetic.csv")

# Combine datasets
balanced_df = pd.concat([original_df, synthetic_df], ignore_index=True)

# Proceed with existing preprocessing pipeline
```

## Monitoring and Validation

### Real-time Monitoring
- Progress bars show generation and validation progress
- Success rates displayed per class
- Memory usage tracking
- Error logging

### Post-generation Analysis
- Class distribution verification  
- Quality metric statistics
- Duplicate detection summary
- Relevance score distributions

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch_size parameter
2. **Model Download Fails**: Check internet connection and HuggingFace access
3. **Low Success Rate**: Adjust validation thresholds
4. **Generation Too Slow**: Ensure GPU is enabled and available

### Performance Optimization

1. **Use GPU**: Enable CUDA for 10x faster generation
2. **Adjust Batch Size**: Increase for better GPU utilization
3. **Tune Thresholds**: Relax validation criteria if needed
4. **Cache Models**: Download models once and reuse

## Citation

If you use this data augmentation agent in your research, please cite:

```bibtex
@misc{mental_health_augmentation,
  title={Autonomous Data Augmentation and Quality Control Agent for Mental Health Classification},
  author={Thesis Research Project},
  year={2024},
  note={Implementation for mental health text classification using GPT-2}
}
```

## License

This project is part of academic research on mental health classification using machine learning techniques.