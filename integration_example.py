# Integration Example: Adding Data Augmentation to MentalBERT Pipeline
# =====================================================================
# 
# This cell shows how to integrate the data augmentation agent 
# with the existing MentalBERT classification pipeline

# Step 1: Run the data augmentation agent
# ---------------------------------------
# First, execute the data augmentation to generate synthetic samples
# (This can be done separately or included in the main pipeline)

print("🚀 Step 1: Running Data Augmentation Agent")
print("=" * 45)

# Import and run the augmentation agent
# Note: In Colab, you would run the colab_data_augmentation.py cell first
# exec(open('colab_data_augmentation.py').read())

# For demonstration, we'll simulate the output
print("✅ Data augmentation completed!")
print("   📄 Files generated: raw_synthetic.csv, validated_synthetic.csv")

# Step 2: Load original and synthetic datasets
# --------------------------------------------
print("\n🔄 Step 2: Integrating Datasets")
print("=" * 35)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load original dataset (replace with actual path)
# original_df = pd.read_csv("/content/drive/MyDrive/Thesis Work/Main Datasets/Mental Health Dataset (Main).csv")

# For demonstration, create sample original data
original_data = {
    'statement': [
        "I feel anxious about my upcoming presentation at work tomorrow",
        "My mood changes rapidly from very high to extremely depressed", 
        "I'm overwhelmed by all the deadlines and pressure at work",
        "I have trouble maintaining stable relationships with other people",
        "The constant worry about my health is making me restless",
        "Sometimes I feel manic and energetic then crash completely",
        "The stress from multiple projects is affecting my sleep quality",
        "My emotions are intense and difficult to control most days"
    ],
    'status': [
        'Anxiety', 'Bipolar', 'Stress', 'Personality disorder',
        'Anxiety', 'Bipolar', 'Stress', 'Personality disorder'
    ]
}
original_df = pd.DataFrame(original_data)
print(f"📊 Original dataset loaded: {len(original_df)} samples")

# Load synthetic dataset
# synthetic_df = pd.read_csv("validated_synthetic.csv")

# For demonstration, create sample synthetic data
synthetic_data = {
    'statement': [
        "I feel worried and nervous about upcoming social events and gatherings",
        "My mood swings make me feel unstable and unpredictable in relationships",
        "I feel overwhelmed by the pressure of daily responsibilities and tasks", 
        "I have difficulty maintaining relationships because of my behavioral patterns",
        "I can't stop thinking about worst case scenarios in every situation",
        "The ups and downs of my emotions are exhausting for everyone around",
        "The stress is making it difficult to concentrate on important tasks",
        "I struggle with intense emotions that feel completely uncontrollable daily"
    ],
    'status': [
        'Anxiety', 'Bipolar', 'Stress', 'Personality disorder',
        'Anxiety', 'Bipolar', 'Stress', 'Personality disorder'
    ]
}
synthetic_df = pd.DataFrame(synthetic_data)
print(f"✅ Synthetic dataset loaded: {len(synthetic_df)} samples")

# Step 3: Combine datasets
# ------------------------
print("\n🔗 Step 3: Combining Original and Synthetic Data")
print("=" * 50)

# Combine original and synthetic data
balanced_df = pd.concat([original_df, synthetic_df], ignore_index=True)
balanced_df['is_synthetic'] = [False] * len(original_df) + [True] * len(synthetic_df)

print(f"📈 Combined dataset: {len(balanced_df)} total samples")
print(f"   📊 Original: {len(original_df)} samples")
print(f"   🎯 Synthetic: {len(synthetic_df)} samples")

# Show class distribution
print(f"\n📊 Class Distribution:")
class_counts = balanced_df['status'].value_counts()
for class_name, count in class_counts.items():
    original_count = sum((balanced_df['status'] == class_name) & (balanced_df['is_synthetic'] == False))
    synthetic_count = sum((balanced_df['status'] == class_name) & (balanced_df['is_synthetic'] == True))
    print(f"   {class_name}: {count} total ({original_count} original + {synthetic_count} synthetic)")

# Step 4: Prepare for existing MentalBERT pipeline
# ------------------------------------------------
print("\n🧠 Step 4: Preparing for MentalBERT Training")
print("=" * 45)

# Create the expected format for the existing pipeline
df_final = balanced_df[['statement', 'status']].copy()

# Add status_id mapping (as used in original notebook)
status_mapping = {
    'Anxiety': 0,
    'Bipolar': 1, 
    'Stress': 2,
    'Personality disorder': 3
}
df_final['status_id'] = df_final['status'].map(status_mapping)

print("✅ Data prepared for MentalBERT pipeline")
print(f"   📊 Final dataset shape: {df_final.shape}")
print(f"   🎯 Classes: {list(status_mapping.keys())}")

# Step 5: Split data maintaining synthetic/original balance
# --------------------------------------------------------
print("\n📊 Step 5: Creating Balanced Train/Test Split")
print("=" * 45)

# Stratified split to maintain class balance
train_df, test_df = train_test_split(
    df_final,
    test_size=0.2,
    random_state=42,
    stratify=df_final['status_id']
)

print(f"✅ Data split completed:")
print(f"   📚 Training set: {len(train_df)} samples")
print(f"   🧪 Test set: {len(test_df)} samples")

# Show train/test distribution
print(f"\n📊 Training Set Distribution:")
train_counts = train_df['status'].value_counts()
for class_name, count in train_counts.items():
    percentage = (count / len(train_df)) * 100
    print(f"   {class_name}: {count} ({percentage:.1f}%)")

# Step 6: Integration point with existing pipeline
# -----------------------------------------------
print("\n🔄 Step 6: Ready for MentalBERT Integration")
print("=" * 45)

print("✅ The balanced dataset (df_final) is now ready to be used with:")
print("   🔹 Existing text preprocessing pipeline")
print("   🔹 MentalBERT tokenization and encoding")
print("   🔹 BiLSTM-CNN model training")
print("   🔹 Evaluation and testing")

print(f"\n💡 Next Steps:")
print(f"   1. Replace the original dataset loading with df_final")
print(f"   2. Continue with existing preprocessing (tokenization, padding, etc.)")
print(f"   3. Train MentalBERT model on the balanced dataset")
print(f"   4. Evaluate performance improvements")

# Save the balanced dataset for use in the main pipeline
df_final.to_csv("balanced_mental_health_dataset.csv", index=False)
print(f"\n💾 Saved balanced dataset: balanced_mental_health_dataset.csv")

# Display sample of final data
print(f"\n📝 Sample of Final Balanced Dataset:")
print("-" * 50)
for i, (_, row) in enumerate(df_final.head(4).iterrows()):
    print(f"{i+1}. [{row['status']}] {row['statement'][:60]}...")

print(f"\n🎉 Integration completed successfully!")
print(f"Your MentalBERT pipeline now has access to balanced synthetic data!")