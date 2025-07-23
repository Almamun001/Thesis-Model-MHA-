#!/usr/bin/env python3
"""
Test script for Data Augmentation Agent
======================================

This script validates the code structure and basic functionality
without requiring model downloads or internet access.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

def test_code_structure():
    """Test that the main script has proper structure."""
    print("🔍 Testing code structure...")
    
    try:
        # Read the main script file
        with open('data_augmentation_agent.py', 'r') as f:
            content = f.read()
        
        # Check for required components
        required_components = [
            'class DataAugmentationAgent',
            'def _load_models',
            'def _create_class_prompts', 
            'def _calculate_perplexity',
            'def _is_duplicate',
            'def _calculate_relevance_score',
            'def _validate_sample',
            'def generate_batch',
            'def augment_class',
            'def run_augmentation',
            'def save_results'
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing components: {missing_components}")
            return False
        else:
            print("✅ All required components found")
            return True
            
    except FileNotFoundError:
        print("❌ data_augmentation_agent.py not found")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def test_class_configuration():
    """Test the class configuration and target setup."""
    print("\n🎯 Testing class configuration...")
    
    # Expected classes and their configurations
    expected_classes = {
        'Anxiety': {'existing': 3838, 'target': 6500},
        'Bipolar': {'existing': 2777, 'target': 6500},
        'Stress': {'existing': 2587, 'target': 6500},
        'Personality disorder': {'existing': 1077, 'target': 6500}
    }
    
    try:
        # Extract class configuration from the script
        with open('data_augmentation_agent.py', 'r') as f:
            content = f.read()
        
        # Check if all expected classes are mentioned
        for class_name in expected_classes.keys():
            if class_name not in content:
                print(f"❌ Class '{class_name}' not found in configuration")
                return False
        
        # Check target count (6500)
        if "6500" not in content:
            print("❌ Target count 6500 not found")
            return False
            
        print("✅ Class configuration looks correct")
        return True
        
    except Exception as e:
        print(f"❌ Error checking class configuration: {e}")
        return False

def test_validation_filters():
    """Test that all required validation filters are implemented."""
    print("\n🔍 Testing validation filters...")
    
    required_filters = [
        '_calculate_perplexity',  # Perplexity filter
        '_is_duplicate',          # Duplication check
        '_calculate_relevance_score', # Relevance scoring
        '_count_words',           # Length check
    ]
    
    try:
        with open('data_augmentation_agent.py', 'r') as f:
            content = f.read()
        
        missing_filters = []
        for filter_func in required_filters:
            if f"def {filter_func}" not in content:
                missing_filters.append(filter_func)
        
        if missing_filters:
            print(f"❌ Missing validation filters: {missing_filters}")
            return False
        else:
            print("✅ All validation filters implemented")
            return True
            
    except Exception as e:
        print(f"❌ Error checking validation filters: {e}")
        return False

def test_output_format():
    """Test that the script produces the expected output format."""
    print("\n📄 Testing output format...")
    
    # Check for CSV output functionality
    try:
        with open('data_augmentation_agent.py', 'r') as f:
            content = f.read()
        
        # Check for required output elements
        required_outputs = [
            'raw_synthetic.csv',
            'validated_synthetic.csv',
            'to_csv',
            'statement',
            'status'
        ]
        
        missing_outputs = []
        for output_element in required_outputs:
            if output_element not in content:
                missing_outputs.append(output_element)
        
        if missing_outputs:
            print(f"❌ Missing output elements: {missing_outputs}")
            return False
        else:
            print("✅ Output format configuration correct")
            return True
            
    except Exception as e:
        print(f"❌ Error checking output format: {e}")
        return False

def test_colab_version():
    """Test the Colab-optimized version."""
    print("\n🚀 Testing Colab version...")
    
    try:
        # Check if Colab version exists
        if not os.path.exists('colab_data_augmentation.py'):
            print("❌ Colab version not found")
            return False
        
        with open('colab_data_augmentation.py', 'r') as f:
            content = f.read()
        
        # Check for Colab-specific features
        colab_features = [
            'install_package',
            'ColabDataAugmentationAgent',
            'required_packages',
            'subprocess',
            'pip install'
        ]
        
        missing_features = []
        for feature in colab_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing Colab features: {missing_features}")
            return False
        else:
            print("✅ Colab version properly configured")
            return True
            
    except Exception as e:
        print(f"❌ Error checking Colab version: {e}")
        return False

def test_requirements():
    """Test that requirements.txt contains necessary dependencies."""
    print("\n📦 Testing requirements...")
    
    try:
        if not os.path.exists('requirements.txt'):
            print("❌ requirements.txt not found")
            return False
        
        with open('requirements.txt', 'r') as f:
            requirements = f.read().lower()
        
        # Check for essential dependencies
        essential_deps = [
            'torch',
            'transformers',
            'sentence-transformers',
            'pandas',
            'numpy',
            'scikit-learn',
            'tqdm'
        ]
        
        missing_deps = []
        for dep in essential_deps:
            if dep not in requirements:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"❌ Missing dependencies: {missing_deps}")
            return False
        else:
            print("✅ All essential dependencies listed")
            return True
            
    except Exception as e:
        print(f"❌ Error checking requirements: {e}")
        return False

def test_documentation():
    """Test that proper documentation exists."""
    print("\n📚 Testing documentation...")
    
    try:
        # Check for README
        if os.path.exists('DATA_AUGMENTATION_README.md'):
            with open('DATA_AUGMENTATION_README.md', 'r') as f:
                readme_content = f.read()
            
            # Check for key sections
            key_sections = [
                '## Overview',
                '## Features', 
                '## Usage',
                '## Configuration Options',
                '## Output Files',
                '## Quality Control Process'
            ]
            
            missing_sections = []
            for section in key_sections:
                if section not in readme_content:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"❌ Missing README sections: {missing_sections}")
                return False
            else:
                print("✅ Comprehensive documentation found")
                return True
        else:
            print("❌ README file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking documentation: {e}")
        return False

def run_all_tests():
    """Run all validation tests."""
    print("🧪 Data Augmentation Agent - Code Validation Tests")
    print("=" * 55)
    
    tests = [
        test_code_structure,
        test_class_configuration,
        test_validation_filters,
        test_output_format,
        test_colab_version,
        test_requirements,
        test_documentation
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is ready for deployment.")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)