# Lab 3: Custom Datasets and Data Augmentation

In this lab, you will learn how to create custom datasets, generate synthetic data through augmentation, and handle imbalanced datasets - common challenges in real-world machine learning projects.

## Learning Objectives

By the end of this lab, you will be able to:

- Create datasets from custom data sources
- Generate synthetic data through paraphrasing
- Identify and quantify data imbalance
- Apply techniques to address imbalanced distributions

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install datasets pandas transformers torch
```

## Key Concepts

- **Data Augmentation**: Artificially increasing dataset size and diversity
- **Paraphrasing**: Generating alternative phrasings of the same content
- **Class Imbalance**: Unequal distribution of labels in training data
- **Oversampling**: Duplicating minority class examples
- **Undersampling**: Reducing majority class examples

## Lab Exercises

### Exercise 1: Creating Datasets from Custom Data

Learn to create Hugging Face datasets from your own data:

```python
from datasets import Dataset
import pandas as pd

# From a Python dictionary
data = {
    'status': [
        'All trails are open',
        'Park closed due to weather',
        'Blankets Creek is open'
    ],
    'label': [1, 0, 1]
}
dataset = Dataset.from_dict(data)
print(dataset)

# From a Pandas DataFrame
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)
print(dataset)

# From a list of dictionaries
examples = [
    {'status': 'Trails open', 'label': 1},
    {'status': 'Closed for maintenance', 'label': 0}
]
dataset = Dataset.from_list(examples)
print(dataset)
```

### Exercise 2: Data Augmentation with Paraphrasing

Navigate to the [examples/augment](../examples/augment/) directory.

1. Study [augment.py](../examples/augment/augment.py) which uses the Pegasus model for paraphrasing:

```python
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = "tuner007/pegasus_paraphrase"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer([text], truncation=True, padding='longest', return_tensors="pt")

# Generate multiple paraphrases
for i in range(5):
    paraphrased = model.generate(
        **inputs,
        num_return_sequences=1,
        do_sample=True,
        num_beams=10,
        temperature=0.1,
        top_p=0.95,
        top_k=50
    )
    result = tokenizer.decode(paraphrased[0], skip_special_tokens=True)
    print(f"Paraphrased: {result}")
```

2. Run the augmentation example:

```bash
python augment.py
```

3. Observe how the same meaning is expressed in different ways

### Exercise 3: Using Local LLM APIs for Augmentation

Study [local.py](../examples/augment/local.py) for using local LLM APIs (OpenAI-compatible):

```python
# This approach works with local LLM servers
# that expose an OpenAI-compatible API

import openai

client = openai.Client(base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="local-model",
    messages=[{
        "role": "user",
        "content": f"Paraphrase this: {text}"
    }]
)
```

### Exercise 4: Checking for Data Imbalance

Navigate to the [examples/imbalance](../examples/imbalance/) directory.

1. Study [check-imbalance.py](../examples/imbalance/check-imbalance.py):

```python
import pandas as pd

df = pd.read_csv('imbalanced.csv')

# Check class distribution
print("Class distribution:")
print(df['Blankets_Creek'].value_counts())

print("\nRatios:")
print(df['Blankets_Creek'].value_counts(normalize=True))

# Detect imbalance
open_count = df['Blankets_Creek'].sum()
closed_count = len(df) - open_count

if open_count < closed_count / 2:
    print("Need OVERSAMPLING for open trails")
elif closed_count < open_count / 2:
    print("Need OVERSAMPLING for closed trails")
else:
    print("Dataset is balanced")
```

2. Run the imbalance check:

```bash
python check-imbalance.py
```

### Exercise 5: Addressing Imbalance

Practice techniques for handling imbalanced data:

**Oversampling the minority class:**
```python
from datasets import Dataset, concatenate_datasets

# Separate by class
open_examples = dataset.filter(lambda x: x['label'] == 1)
closed_examples = dataset.filter(lambda x: x['label'] == 0)

# Oversample minority class
if len(open_examples) < len(closed_examples):
    # Duplicate open examples
    factor = len(closed_examples) // len(open_examples)
    oversampled = concatenate_datasets([open_examples] * factor)
    balanced = concatenate_datasets([oversampled, closed_examples])
```

**Undersampling the majority class:**
```python
# Undersample majority class
if len(closed_examples) > len(open_examples):
    undersampled = closed_examples.shuffle(seed=42).select(
        range(len(open_examples))
    )
    balanced = concatenate_datasets([open_examples, undersampled])
```

### Exercise 6: Augmentation for Balancing

Combine augmentation with balancing:

```python
def augment_minority_class(dataset, minority_label, target_count):
    """Augment minority class to reach target count"""
    minority = dataset.filter(lambda x: x['label'] == minority_label)

    augmented_examples = []
    while len(minority) + len(augmented_examples) < target_count:
        for example in minority:
            # Generate paraphrase
            paraphrased = generate_paraphrase(example['status'])
            augmented_examples.append({
                'status': paraphrased,
                'label': minority_label
            })
            if len(minority) + len(augmented_examples) >= target_count:
                break

    return concatenate_datasets([
        minority,
        Dataset.from_list(augmented_examples)
    ])
```

## Challenge

1. Create a dataset from a custom source (web scraping, API, or manual collection)
2. Analyze the class distribution
3. If imbalanced, apply one of these techniques:
   - Oversampling with augmentation
   - Undersampling
   - A combination approach
4. Verify the balanced dataset is ready for training

## Summary

In this lab, you learned how to:
- Create datasets from dictionaries, DataFrames, and lists
- Generate synthetic data using paraphrasing models
- Detect and quantify class imbalance
- Apply oversampling and undersampling techniques
- Combine augmentation with balancing strategies

## Next Steps

Continue to [Lab 4: Training with Trainer API](./lab-4.md) to learn how to train your first model using the prepared dataset.
