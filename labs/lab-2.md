# Lab 2: Transformations and Tokenization

In this lab, you will learn how to transform datasets using map, filter, and other operations. You'll also learn tokenization strategies essential for preparing text data for transformer models.

## Learning Objectives

By the end of this lab, you will be able to:

- Apply filtering, mapping, and selection operations on datasets
- Optimize dataset processing with batching
- Implement tokenization for NLP models
- Handle different sequence lengths through padding and truncation

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install datasets transformers
```

## Key Concepts

- **map()**: Transform each example in the dataset
- **filter()**: Keep only examples matching a condition
- **select()**: Choose specific rows by index
- **shuffle()**: Randomize dataset order
- **train_test_split()**: Split data into training and test sets
- **Tokenization**: Converting text to numerical token IDs
- **Padding**: Making all sequences the same length
- **Truncation**: Cutting sequences to a maximum length

## Lab Exercises

### Exercise 1: Dataset Transformations

Navigate to the [examples/transform](../examples/transform/) directory and run the transformation examples.

1. Run the transformation script:

```bash
cd examples/transform
python README.md  # This file is actually a Python script
```

2. Study the key transformations:

**Map Transformation** - Add new features to each example:
```python
def add_features(example):
    example['status_length'] = len(example['status'])
    example['mentions_closed'] = 'closed' in example['status'].lower()
    return example

mapped_dataset = dataset.map(add_features)
```

**Filter Transformation** - Keep only matching examples:
```python
# Keep only examples where parks are open
open_parks = dataset.filter(lambda x: x['Blankets_Creek'] == 1)
```

**Select Transformation** - Get specific rows:
```python
# Select specific indices
selected = dataset.select([0, 2, 4, 6])
```

### Exercise 2: Train-Test Split

Practice splitting datasets for training and evaluation:

```python
from datasets import Dataset

# Create a train/test split
split_datasets = dataset.train_test_split(
    test_size=0.2,  # 20% for testing
    seed=42,        # For reproducibility
    shuffle=True    # Shuffle before splitting
)

print(f"Train: {len(split_datasets['train'])} examples")
print(f"Test: {len(split_datasets['test'])} examples")
```

### Exercise 3: Chaining Transformations

Create a processing pipeline by chaining multiple operations:

```python
processed = (
    dataset
    .filter(lambda x: 'blankets' in x['status'].lower())
    .map(lambda x: {
        'short_status': x['status'][:40],
        'is_open': x['Blankets_Creek'] == 1
    })
    .shuffle(seed=42)
)
```

### Exercise 4: Tokenization with Padding

Navigate to the [examples/tokenize](../examples/tokenize/) directory.

1. Study [padding.py](../examples/tokenize/padding.py):

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sentences = [
    "All trails are closed due to wet conditions",
    "Blankets Creek is open, Rope Mill is closed",
    "Check website for updates"
]

# Tokenize with padding for equal lengths
padded = tokenizer(sentences, padding=True)
print(f"Padded lengths: {[len(ids) for ids in padded['input_ids']]}")
```

2. Run the padding example:

```bash
python padding.py
```

### Exercise 5: Tokenization with Truncation

1. Study [truncation.py](../examples/tokenize/truncation.py):

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Truncate long sequences
truncated = tokenizer(
    long_text,
    truncation=True,
    max_length=128
)
```

2. Experiment with different max_length values

### Exercise 6: Batch Processing

For large datasets, batch processing improves performance:

```python
def batch_process(batch):
    """Process examples in batches for efficiency"""
    batch['word_count'] = [len(s.split()) for s in batch['status']]
    return batch

# Process in batches
batched_dataset = dataset.map(
    batch_process,
    batched=True,
    batch_size=100
)
```

### Exercise 7: Combined Tokenization for Training

Combine tokenization with label encoding for model training:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_and_label(examples):
    tokenized = tokenizer(
        examples['status'],
        truncation=True,
        padding=True
    )
    tokenized['labels'] = examples['Blankets_Creek']
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_label, batched=True)
```

## Challenge

1. Load a dataset with text data
2. Create a transformation pipeline that:
   - Filters for examples meeting certain criteria
   - Adds a new feature (e.g., text length, word count)
   - Tokenizes the text with padding and truncation
   - Splits into train/test sets
3. Verify the output structure is ready for model training

## Summary

In this lab, you learned how to:
- Transform datasets using map, filter, select, and shuffle
- Split datasets into training and test sets
- Tokenize text using Hugging Face tokenizers
- Apply padding and truncation for consistent sequence lengths
- Use batch processing for efficiency

## Next Steps

Continue to [Lab 3: Custom Datasets and Data Augmentation](./lab-3.md) to learn how to create your own datasets and handle data imbalance.
