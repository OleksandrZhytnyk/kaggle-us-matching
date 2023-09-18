from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler


def create_data_loaders(train_df, tokenizer, batch_size, max_length):
    X_train, X_valid, y_train, y_valid = train_test_split(train_df[["input", "target"]], train_df["score"],
                                                          test_size=0.20, stratify=train_df["score"], random_state=42)

    train_encodings = tokenizer(X_train.values.tolist(), padding="max_length", truncation=True, max_length=max_length,
                                return_tensors='pt')
    valid_encodings = tokenizer(X_valid.values.tolist(), padding="max_length", truncation=True, max_length=max_length,
                                return_tensors='pt')

    train_labels = torch.tensor(y_train.values, dtype=torch.float)
    val_labels = torch.tensor(y_valid.values, dtype=torch.float)

    train_data = TensorDataset(train_encodings['input_ids'], train_encodings["attention_mask"], train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(valid_encodings['input_ids'], valid_encodings["attention_mask"], val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader


def create_test_data_loaders(test_df, tokenizer, max_length):
    test_encodings = tokenizer(test_df[['input', 'target']].values.tolist(), padding="max_length", truncation=True,
                               max_length=max_length, return_tensors='pt')

    test_data = TensorDataset(test_encodings['input_ids'], test_encodings["attention_mask"])
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler)

    return test_dataloader
