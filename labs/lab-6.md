# Lab 6: Publishing Models to Hugging Face Hub

In this lab, you will learn how to publish your trained models to the Hugging Face Hub, making them accessible to others. You'll also learn about model cards, authentication, and automated publishing.

## Learning Objectives

By the end of this lab, you will be able to:

- Authenticate with the Hugging Face Hub
- Create and publish models to the Hub
- Generate comprehensive model cards
- Automate publishing during training
- Load and use models from the Hub

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install transformers datasets huggingface_hub
```

You'll also need a Hugging Face account. Create one at [huggingface.co](https://huggingface.co/join).

## Key Concepts

- **Hugging Face Hub**: Central platform for sharing models and datasets
- **Model Card**: Documentation describing a model's purpose, training, and usage
- **Access Token**: Authentication credential for Hub operations
- **push_to_hub**: Training argument to automatically upload models

## Lab Exercises

### Exercise 1: Authentication Setup

1. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Give it a name (e.g., "fine-tuning-course")
   - Select "Write" permissions
   - Copy the token

2. Login using the Hugging Face CLI:

```bash
huggingface-cli login
# Paste your token when prompted
```

Or login programmatically:

```python
from huggingface_hub import login

login(token="your_token_here")
# Or set HF_TOKEN environment variable
```

3. Verify authentication:

```bash
huggingface-cli whoami
```

### Exercise 2: Manual Model Publishing

After training a model, publish it manually:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load your trained model
model = AutoModelForSequenceClassification.from_pretrained("./trail_classifier")
tokenizer = AutoTokenizer.from_pretrained("./trail_classifier")

# Push to Hub
model.push_to_hub("your-username/trail-status-classifier")
tokenizer.push_to_hub("your-username/trail-status-classifier")
```

### Exercise 3: Automatic Publishing During Training

Navigate to the [examples/publishing](../examples/publishing/) directory.

Study [classifier.py](../examples/publishing/classifier.py):

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./trail_classifier",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    eval_strategy="epoch",
    push_to_hub=True,  # Enable automatic publishing
    hub_model_id="your-username/trail-status-classifier",  # Optional: custom name
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Train and automatically push to Hub
trainer.train()
trainer.push_to_hub()  # Final push with model card
```

### Exercise 4: Creating Model Cards

Model cards are automatically generated but can be customized:

```python
from huggingface_hub import ModelCard

card_content = """
---
license: mit
language: en
tags:
  - text-classification
  - trail-status
  - fine-tuned
datasets:
  - custom
metrics:
  - accuracy
---

# Trail Status Classifier

This model classifies trail status messages as OPEN or CLOSED.

## Model Description

Fine-tuned BERT model for binary classification of trail status updates.

## Training Data

Trained on ~100 trail status messages from Blankets Creek and Rope Mill parks.

## Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="your-username/trail-status-classifier")
result = classifier("All trails are open today!")
print(result)
```

## Training Procedure

- Base model: bert-base-uncased
- Epochs: 3
- Batch size: 8
- Learning rate: 2e-5

## Evaluation Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.95 |
| F1 Score | 0.94 |
"""

card = ModelCard(card_content)
card.push_to_hub("your-username/trail-status-classifier")
```

### Exercise 5: Loading Models from the Hub

Study [inference.py](../examples/publishing/inference.py):

```python
from transformers import pipeline

# Load directly from Hub
classifier = pipeline(
    "text-classification",
    model="your-username/trail-status-classifier"
)

# Make predictions
results = classifier([
    "All trails are open!",
    "Park closed due to weather",
    "Blankets Creek is open, Rope Mill closed"
])

for text, result in zip(texts, results):
    print(f"{text} -> {result['label']} ({result['score']:.2%})")
```

### Exercise 6: Version Management

Track model versions using Git-based versioning:

```python
# Push with a specific commit message
trainer.push_to_hub(commit_message="Improved accuracy with more training data")

# Load a specific version
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "your-username/trail-status-classifier",
    revision="v1.0"  # or commit hash
)
```

### Exercise 7: Private Models

Create private models for sensitive projects:

```python
from huggingface_hub import create_repo

# Create a private repository
create_repo(
    "your-username/private-classifier",
    private=True
)

# Push to private repo
model.push_to_hub("your-username/private-classifier", private=True)
```

### Exercise 8: Using the Inference API

Once published, use the free Inference API:

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/your-username/trail-status-classifier"
headers = {"Authorization": "Bearer YOUR_TOKEN"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

result = query({"inputs": "All trails are open today!"})
print(result)
```

Or use the Python client:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="YOUR_TOKEN")
result = client.text_classification(
    "All trails are open!",
    model="your-username/trail-status-classifier"
)
print(result)
```

## Challenge

1. Train a text classification model on the trail status data
2. Publish it to the Hugging Face Hub with:
   - A descriptive model card
   - Proper tags and metadata
   - Training metrics
3. Create a simple script that loads your model from the Hub and makes predictions
4. Share the model URL and test it with classmates

## Summary

In this lab, you learned how to:
- Authenticate with the Hugging Face Hub using tokens
- Publish models manually and automatically during training
- Create comprehensive model cards with metadata
- Load and use models directly from the Hub
- Manage model versions and private repositories
- Use the Inference API for serverless predictions

## Course Complete!

Congratulations on completing the Fine-tuning with Hugging Face course! You now have the skills to:

1. Load and transform datasets from various sources
2. Tokenize text for transformer models
3. Handle data augmentation and imbalance
4. Train models using the Trainer API
5. Implement advanced training techniques
6. Publish and share your models

## Next Steps

- Explore other model architectures on the [Hugging Face Hub](https://huggingface.co/models)
- Try fine-tuning for different tasks (NER, question answering, summarization)
- Learn about [PEFT](https://huggingface.co/docs/peft) for efficient fine-tuning
- Explore [Accelerate](https://huggingface.co/docs/accelerate) for distributed training
- Check out [GitHub Actions automation](https://docs.github.com/en/actions) for CI/CD pipelines
