# Lab 5: Advanced Training and Callbacks

In this lab, you will learn advanced training techniques including custom metrics, optimizer selection, mixed precision training, callbacks for early stopping, custom logging, and debugging strategies.

## Learning Objectives

By the end of this lab, you will be able to:

- Implement custom metrics for model evaluation
- Select appropriate optimizers and learning rates
- Apply mixed precision training for efficiency
- Use callbacks for early stopping and logging
- Debug training issues and detect overfitting

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install transformers datasets pandas torch numpy scikit-learn
```

## Key Concepts

- **Metrics**: Quantitative measures for evaluating model performance
- **AdamW Optimizer**: Recommended optimizer for transformer models
- **Learning Rate**: Controls how much the model updates during training
- **Mixed Precision (FP16)**: Using half-precision floats for memory efficiency
- **Callbacks**: Functions executed at specific points during training
- **Early Stopping**: Halting training when performance stops improving
- **Overfitting**: When a model memorizes training data but fails to generalize

## Lab Exercises

### Exercise 1: Custom Metrics

Navigate to the [examples/custom](../examples/custom/) directory.

Add custom metrics to evaluate your model:

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(eval_pred):
    """Calculate multiple metrics from predictions"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)
```

### Exercise 2: Optimizer and Learning Rate

Study [learning_rate.py](../examples/custom/learning_rate.py):

```python
training_args = TrainingArguments(
    output_dir="./trail_classifier",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    optim="adamw_torch",      # AdamW optimizer (recommended)
    learning_rate=2e-5,       # 0.00002 - typical for fine-tuning
)
```

**Learning Rate Guidelines:**
- `1e-5` to `5e-5`: Typical range for fine-tuning
- Lower values = slower but more stable training
- Higher values = faster but may overshoot optimal weights

### Exercise 3: Mixed Precision Training

Study [fp16.py](../examples/custom/fp16.py):

```python
training_args = TrainingArguments(
    output_dir="./trail_classifier",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    eval_strategy="epoch",
    optim="adamw_torch",
    learning_rate=2e-5,
    fp16=True,  # Enable mixed precision
)
```

**Benefits of FP16:**
- Reduces memory usage by ~50%
- Enables training larger models
- Often faster on modern GPUs
- Minimal accuracy impact

Run the comparison:
```bash
python fp16.py
```

### Exercise 4: Early Stopping Callback

Navigate to the [examples/callback](../examples/callback/) directory.

Study [classifier.py](../examples/callback/classifier.py):

```python
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./trail_classifier",
    num_train_epochs=10,
    eval_strategy="epoch",
    metric_for_best_model="accuracy",  # Track this metric
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

**How Early Stopping Works:**
1. Monitors the specified metric after each epoch
2. Tracks the best value seen so far
3. Stops training if no improvement for N epochs (patience)
4. Loads the best model checkpoint

### Exercise 5: Custom Logging

Study [classifier-logging.py](../examples/callback/classifier-logging.py):

```python
from transformers import TrainerCallback
import json

class LoggingCallback(TrainerCallback):
    def __init__(self, log_file="training_log.txt"):
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file, "a") as f:
                f.write(json.dumps({
                    "step": state.global_step,
                    "epoch": state.epoch,
                    **logs
                }) + "\n")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    callbacks=[LoggingCallback("my_training_log.txt")],
)
```

Run with logging:
```bash
python classifier-logging.py
cat training_log.txt
```

### Exercise 6: Debugging Training

Study [classifier-debug.py](../examples/callback/classifier-debug.py):

```python
class DebugCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            train_acc = metrics.get("train_accuracy", 0)
            eval_acc = metrics.get("eval_accuracy", 0)
            gap = train_acc - eval_acc

            print(f"\n=== Debug Info ===")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Eval Accuracy: {eval_acc:.4f}")
            print(f"Gap: {gap:.4f}")

            if gap > 0.05:
                print("WARNING: Moderate overfitting detected!")
            elif train_acc > 0.99:
                print("WARNING: Suspiciously high train accuracy!")
            else:
                print("OK: Good generalization")
```

**Overfitting Indicators:**
- Large gap between train and eval accuracy (>5%)
- Perfect training accuracy (100%)
- Eval accuracy decreasing while train accuracy increases

### Exercise 7: Combining Techniques

Create an optimized training configuration:

```python
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./optimized_classifier",
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    optim="adamw_torch",
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        LoggingCallback("training_log.json"),
        DebugCallback(),
    ],
)
```

## Challenge

1. Train a model with early stopping and custom metrics
2. Compare training with and without FP16:
   - Memory usage
   - Training speed
   - Final accuracy
3. Create a custom callback that:
   - Logs metrics to a file
   - Prints warnings for overfitting
   - Saves the best model automatically
4. Experiment with different learning rates (1e-5, 2e-5, 5e-5) and document the impact

## Summary

In this lab, you learned how to:
- Implement custom metrics (accuracy, precision, recall, F1)
- Configure the AdamW optimizer with appropriate learning rates
- Enable mixed precision training for efficiency
- Use early stopping to prevent overfitting
- Create custom callbacks for logging and debugging
- Detect and diagnose training issues

## Next Steps

Continue to [Lab 6: Publishing Models](./lab-6.md) to learn how to share your trained models on the Hugging Face Hub.
