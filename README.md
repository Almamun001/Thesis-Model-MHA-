# Mental Health Analysis with MentalBERT and Data Augmentation

This repository contains a comprehensive mental health classification system using MentalBERT, BiLSTM, and CNN architectures, enhanced with an autonomous data augmentation agent for balanced dataset generation.

## Components

### 1. MentalBERT Classification Pipeline
- **File**: `MentalBERT_BiLSTM_CNN_(Documented).ipynb`
- **Description**: Complete pipeline for mental health text classification using MentalBERT embeddings with BiLSTM and CNN layers
- **Features**: Data preprocessing, model training, evaluation, and visualization

### 2. Autonomous Data Augmentation Agent
- **Files**: 
  - `data_augmentation_agent.py` - Main implementation
  - `colab_data_augmentation.py` - Colab-optimized version
  - `DATA_AUGMENTATION_README.md` - Detailed documentation
- **Description**: GPT-2 based synthetic text generation with quality validation filters
- **Target**: Balance 4 minority classes (Anxiety, Bipolar, Stress, Personality disorder) to 6,500 samples each

### 3. Integration and Testing
- **Files**:
  - `integration_example.py` - Shows how to integrate augmented data with existing pipeline
  - `test_implementation.py` - Validation tests for the augmentation agent
  - `requirements.txt` - Python dependencies

## Quick Start

### Option 1: Run Data Augmentation in Google Colab
1. Open Google Colab and enable GPU (T4 recommended)
2. Upload `colab_data_augmentation.py`
3. Run the entire cell - it will install dependencies and execute

### Option 2: Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run data augmentation
python data_augmentation_agent.py

# Test implementation
python test_implementation.py

# View integration example
python integration_example.py
```

## Mental Health Classes

The system handles four mental health categories:
- **Anxiety**: Worry, nervousness, restlessness, fear-based symptoms
- **Bipolar**: Mood swings, manic/depressive episodes, energy fluctuations  
- **Stress**: Overwhelm, pressure, burnout, coping difficulties
- **Personality disorder**: Relationship issues, emotional dysregulation, behavioral patterns

## Data Augmentation Features

✅ **GPT-2 Text Generation**: Uses "openai-community/gpt2" for contextual text generation  
✅ **Quality Validation**: Perplexity filtering, duplication check, relevance scoring, length validation  
✅ **Batch Processing**: Efficient generation with progress tracking  
✅ **Reproducible**: Fixed seeds for consistent results  
✅ **CSV Output**: Raw and validated datasets ready for training  

## Integration with Existing Pipeline

The augmented data integrates seamlessly with the existing MentalBERT pipeline:
1. Generate synthetic samples using the augmentation agent
2. Combine with original dataset using `integration_example.py`
3. Feed balanced dataset into existing MentalBERT training pipeline
4. Evaluate improved model performance on balanced data

## File Structure
```
├── MentalBERT_BiLSTM_CNN_(Documented).ipynb  # Main classification pipeline
├── data_augmentation_agent.py                # Full augmentation implementation  
├── colab_data_augmentation.py               # Colab-optimized version
├── integration_example.py                   # Integration demonstration
├── test_implementation.py                   # Validation tests
├── requirements.txt                         # Dependencies
├── DATA_AUGMENTATION_README.md             # Detailed augmentation docs
└── README.md                               # This file
```

## Expected Results

With the balanced dataset:
- **Improved minority class performance**: Better recall for underrepresented classes
- **Reduced bias**: More balanced predictions across all mental health categories  
- **Enhanced generalization**: Better performance on real-world diverse data
- **Robust evaluation**: More reliable metrics with balanced test sets

## Citation

If you use this work in your research, please cite:
```bibtex
@misc{mental_health_classification_2024,
  title={Mental Health Classification with MentalBERT and Autonomous Data Augmentation},
  author={Thesis Research Project},
  year={2024},
  note={Implementation combining MentalBERT classification with GPT-2 data augmentation}
}
```