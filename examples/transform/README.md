from datasets import Dataset, DatasetDict, load_dataset
import io

# Create your dataset from the CSV string
csv_content = """status,Blankets_Creek,Rope_Mill
all trails are closed due to wet conditions,0,0
all trails are closed due to freeze/thaw conditions,0,0
all trails are closed at blankets creek and rope mill due to freeze/thaw soil conditions,0,0
all trails are open yippie!!,1,1
blankets creek is closed rope mill is open,0,1
blankets creek is closed  rope mill is open ,0,1
all trails are closed due to rainy conditions,0,0
all trails are open blankets creek will be closed for a race on saturday 9/5 until 4pm,1,1
all trails are open at blankets creek and rope mill parks blankets creek will be closed sunday 12/19/2021 until noon for a private event,1,1"""

# Save to a temporary file to demonstrate load_dataset
with open('temp_trail_data.csv', 'w') as f:
    f.write(csv_content)

print("=== 1. LOADING DATASET ===")
# Load dataset from CSV file
dataset = load_dataset('csv', data_files='temp_trail_data.csv')['train']

print(f"Dataset has {len(dataset)} rows and {len(dataset.column_names)} columns")
print("Dataset structure:", dataset)
print("\nFirst 3 examples:")
for i in range(3):
    print(f"  {dataset[i]}")

# 1. map() - Transform the data
print("\n=== 2. map() TRANSFORMATION ===")
def add_features(example):
    """Add new features to each example"""
    # Calculate status length
    example['status_length'] = len(example['status'])
    
    # Add combined status
    if example['Blankets_Creek'] == 1 and example['Rope_Mill'] == 1:
        example['combined_status'] = 'both_open'
    elif example['Blankets_Creek'] == 0 and example['Rope_Mill'] == 0:
        example['combined_status'] = 'both_closed'
    else:
        example['combined_status'] = 'mixed'
    
    # Add urgency flag (status contains 'closed')
    example['mentions_closed'] = 'closed' in example['status'].lower()
    
    return example

# Apply map transformation
mapped_dataset = dataset.map(add_features)
print("\nAfter map() - added new columns:")
print(f"Columns: {mapped_dataset.column_names}")
print("\nFirst example with new features:")
print(mapped_dataset[0])

# 2. filter() - Keep only specific rows
print("\n=== 3. filter() TRANSFORMATION ===")
# Filter 1: Only open parks
open_parks_dataset = dataset.filter(lambda x: x['Blankets_Creek'] == 1 or x['Rope_Mill'] == 1)
print(f"Filter 1 - Parks that are open (any park): {len(open_parks_dataset)} rows")
for i in range(min(3, len(open_parks_dataset))):
    print(f"  {open_parks_dataset[i]['status'][:50]}...")

# Filter 2: Only statuses mentioning 'closed'
closed_mentions_dataset = dataset.filter(lambda x: 'closed' in x['status'].lower())
print(f"\nFilter 2 - Statuses mentioning 'closed': {len(closed_mentions_dataset)} rows")

# Filter 3: Long status messages (more than 50 chars)
long_status_dataset = mapped_dataset.filter(lambda x: x['status_length'] > 50)
print(f"Filter 3 - Long status messages (>50 chars): {len(long_status_dataset)} rows")

# 3. select() - Get specific rows by index
print("\n=== 4. select() TRANSFORMATION ===")
# Select specific indices
selected_indices = [0, 2, 4, 6]
selected_dataset = dataset.select(selected_indices)
print(f"Selected rows with indices {selected_indices}:")
for i, example in enumerate(selected_dataset):
    print(f"  Row {selected_indices[i]}: Status: {example['status'][:40]}...")

# Select a range
range_dataset = dataset.select(range(3, 7))
print(f"\nSelected rows 3-6:")
for i in range(len(range_dataset)):
    print(f"  Row {i+3}: {range_dataset[i]['status'][:40]}...")

# 4. shuffle() - Randomize the order
print("\n=== 5. shuffle() TRANSFORMATION ===")
# Shuffle with a seed for reproducibility
shuffled_dataset = dataset.shuffle(seed=42)
print("Original order (first 3):")
for i in range(3):
    print(f"  {dataset[i]['status'][:30]}...")
    
print("\nShuffled order (first 3):")
for i in range(3):
    print(f"  {shuffled_dataset[i]['status'][:30]}...")

# 5. train_test_split() - Split the dataset
print("\n=== 6. train_test_split() TRANSFORMATION ===")
# Create a larger dataset to demonstrate splitting (simulating your 80 rows)
# First, let's duplicate our current data
from itertools import cycle

# Create a list of examples
examples = list(dataset)
# Create 80 examples (9 * 9 = 81, close enough)
expanded_examples = []
for i in range(9):  # We'll have 81 examples
    for example in examples:
        # Create a modified copy to avoid reference issues
        new_example = example.copy()
        expanded_examples.append(new_example)

# Create dataset from list
expanded_dataset = Dataset.from_list(expanded_examples[:80])  # Exactly 80
print(f"Expanded dataset: {len(expanded_dataset)} rows")

# Perform train-test split
split_datasets = expanded_dataset.train_test_split(
    test_size=0.2,  # 20% test
    seed=42,        # For reproducibility
    shuffle=True    # Shuffle before splitting
)

print(f"\nTrain set: {len(split_datasets['train'])} rows")
print(f"Test set: {len(split_datasets['test'])} rows")

print("\nFirst train example:")
print(f"  Status: {split_datasets['train'][0]['status'][:40]}...")
print(f"  Blankets_Creek: {split_datasets['train'][0]['Blankets_Creek']}, Rope_Mill: {split_datasets['train'][0]['Rope_Mill']}")

print("\nFirst test example:")
print(f"  Status: {split_datasets['test'][0]['status'][:40]}...")
print(f"  Blankets_Creek: {split_datasets['test'][0]['Blankets_Creek']}, Rope_Mill: {split_datasets['test'][0]['Rope_Mill']}")

# 6. Chaining transformations (real pipeline)
print("\n=== 7. CHAINING TRANSFORMATIONS ===")
# Create a processing pipeline
processed_dataset = (
    dataset
    .filter(lambda x: 'blankets' in x['status'].lower())  # Filter for mentions of blankets
    .map(lambda x: {
        'short_status': x['status'][:40] + '...' if len(x['status']) > 40 else x['status'],
        'both_open': x['Blankets_Creek'] == 1 and x['Rope_Mill'] == 1
    })  # Add new columns
    .shuffle(seed=123)  # Shuffle
)

print("Chained transformations result:")
print(f"  Original dataset: {len(dataset)} rows")
print(f"  After filtering for 'blankets': {len(processed_dataset)} rows")
print("\nProcessed examples:")
for i in range(min(3, len(processed_dataset))):
    print(f"  {processed_dataset[i]}")

# 7. Working with DatasetDict (multiple splits)
print("\n=== 8. DATASETDICT EXAMPLE ===")
# Create a DatasetDict with multiple splits
dataset_dict = DatasetDict({
    'original': dataset,
    'open_only': dataset.filter(lambda x: x['Blankets_Creek'] == 1 or x['Rope_Mill'] == 1),
    'closed_only': dataset.filter(lambda x: x['Blankets_Creek'] == 0 and x['Rope_Mill'] == 0)
})

print("DatasetDict structure:")
for split_name, split_dataset in dataset_dict.items():
    print(f"  {split_name}: {len(split_dataset)} rows")

# 8. Batch processing with map (more efficient for large datasets)
print("\n=== 9. BATCH PROCESSING ===")
def batch_process(batch):
    """Process examples in batches for efficiency"""
    batch['word_count'] = [len(s.split()) for s in batch['status']]
    batch['has_exclamation'] = ['!' in s for s in batch['status']]
    return batch

# Process in batches of 2 (for demonstration)
batched_dataset = dataset.map(batch_process, batched=True, batch_size=2)
print("After batch processing:")
print(f"  New columns: {[col for col in batched_dataset.column_names if col not in dataset.column_names]}")
print("\nFirst example with batch features:")
print(batched_dataset[0])

# Clean up
import os
os.remove('temp_trail_data.csv')
