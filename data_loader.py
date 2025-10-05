# data_loader.py
# Simple helper to load AlpaCare dataset

from datasets import load_dataset
from sklearn.model_selection import train_test_split

def get_alpacare_dataset(test_size=0.1, seed=42):
    """
    Load and split the AlpaCare dataset.
    
    Returns:
        train_data, test_data (datasets.Dataset)
    """
    dataset = load_dataset("lavita/AlpaCare-MedInstruct-52k")["train"]
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]

def add_disclaimer(example):
    """
    Add instruction-response + educational disclaimer to each example
    """
    text = (
        f"### Instruction:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}\n\n"
        "Disclaimer: This is educational only — consult a qualified clinician."
    )
    return {"text": text}

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """
    Tokenizes dataset for causal LM training
    """
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    
    return dataset.map(tokenize_fn, batched=True)
