# Lab 1: Loading and Exploring Datasets

In this lab, you will learn how to load datasets from various sources using the Hugging Face Datasets library. You'll work with CSV, JSON, and Parquet files, and explore dataset structure and metadata.

## Learning Objectives

By the end of this lab, you will be able to:

- Install and configure the Hugging Face Datasets library
- Load datasets from multiple sources (Hugging Face Hub, local files)
- Understand dataset structure and metadata
- Work with different file formats (CSV, JSON, Parquet)

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install datasets pandas
```

## Key Concepts

- **Datasets Library**: A flexible tool for loading and managing data in various formats
- **DatasetDict**: A dictionary-like structure containing multiple dataset splits (train, test, validation)
- **load_dataset**: The primary function for loading datasets from files or the Hugging Face Hub

## Lab Exercises

### Exercise 1: Loading from CSV

Navigate to the [examples/loading](../examples/loading/) directory. You'll find sample data files in different formats.

1. Open a Python interpreter or create a new Python file
2. Load the CSV file using the Datasets library:

```python
from datasets import load_dataset

# Load from CSV
dataset = load_dataset('csv', data_files='status.csv')
print(dataset)
```

3. Explore the dataset structure:

```python
# Access the train split
train_data = dataset['train']

# View the number of rows and columns
print(f"Rows: {len(train_data)}")
print(f"Columns: {train_data.column_names}")

# View the first example
print(train_data[0])
```

### Exercise 2: Loading from JSON

1. Load the JSON file:

```python
# Load from JSON
json_dataset = load_dataset('json', data_files='status.csv.json')
print(json_dataset)
```

2. Compare the structure with the CSV dataset

### Exercise 3: Loading from Parquet

1. Load the Parquet file:

```python
# Load from Parquet
parquet_dataset = load_dataset('parquet', data_files='status.parquet')
print(parquet_dataset)
```

2. Parquet is a columnar format - discuss when you might prefer it over CSV

### Exercise 4: Loading from Hugging Face Hub

1. Load a dataset directly from the Hugging Face Hub:

```python
# Load a public dataset
imdb = load_dataset('imdb')
print(imdb)

# View the splits
print(f"Train size: {len(imdb['train'])}")
print(f"Test size: {len(imdb['test'])}")
```

2. Explore the dataset features:

```python
# View feature information
print(imdb['train'].features)
```

### Exercise 5: Dataset Exploration

Using any of the loaded datasets, practice these exploration techniques:

```python
# Iterate through examples
for i, example in enumerate(dataset['train']):
    if i >= 3:
        break
    print(example)

# Access specific columns
statuses = dataset['train']['status']
print(f"First 3 statuses: {statuses[:3]}")

# Get dataset info
print(dataset['train'].info)
```

## Challenge

1. Find a dataset on the [Hugging Face Hub](https://huggingface.co/datasets) related to text classification
2. Load it using `load_dataset`
3. Explore its structure, features, and splits
4. Write a summary of what the dataset contains

## Summary

In this lab, you learned how to:
- Load datasets from CSV, JSON, and Parquet files
- Load datasets from the Hugging Face Hub
- Explore dataset structure using column names, features, and indexing
- Understand the DatasetDict structure for train/test splits

## Next Steps

Continue to [Lab 2: Transformations and Tokenization](./lab-2.md) to learn how to transform and prepare your data for model training.
