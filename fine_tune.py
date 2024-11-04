from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np
import torch

# Define paths
model_path = "G:/AI/leadify-ai/model/business_model"
results_path = "G:/AI/leadify-ai/results"
dataset_path = "G:/AI/leadify-ai/data/improved_business_dataset.json"

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# If using GPU, print some information about it
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load the dataset
dataset = load_dataset('json', data_files=dataset_path)

# Define label list and create a mapping
label_list = ["O", "B-BUSINESS", "I-BUSINESS", "B-CITY", "I-CITY"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Load pre-trained model and tokenizer
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=128,  # You can adjust this value based on your data
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Define metrics
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir=results_path,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    dataloader_pin_memory=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"].select(range(1000)),  # Using a subset of train data for evaluation
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(model_path)
print(f"Model saved to {model_path}")

# Test the model
test_sentences = [
    ["Find", "property", "management", "companies", "in", "London"],
    ["I", "need", "contractor", "services", "in", "New", "York"],
    ["Who", "are", "the", "best", "recruitment", "agencies", "in", "Chicago"],
    ["Give", "me", "contact", "details", "for", "software", "development", "firms", "in", "San", "Francisco"],
    ["Looking", "for", "digital", "marketing", "experts", "in", "Berlin"]
]

# Test the model
model.to(device)
model.eval()

for tokens in test_sentences:
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=2)
    predicted_tokens = [id2label[t.item()] for t in predictions[0]]
    
    print("Input:", " ".join(tokens))
    print("Predictions:", predicted_tokens)
    print("Extracted entities:")
    for token, label in zip(tokens, predicted_tokens):
        if label != "O":
            print(f"  {token}: {label}")
    print()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
