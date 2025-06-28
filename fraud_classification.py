import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

# Define the dataset directory
DATASET_DIR = os.getenv("DATASET_DIR")

# Load the data
print("Loading data...")
train_identity = pd.read_csv(f'{DATASET_DIR}/train_identity.csv')
train_transaction = pd.read_csv(f'{DATASET_DIR}/train_transaction.csv')
test_identity = pd.read_csv(f'{DATASET_DIR}/test_identity.csv')
test_transaction = pd.read_csv(f'{DATASET_DIR}/test_transaction.csv')
sample_submission = pd.read_csv(f'{DATASET_DIR}/sample_submission.csv')
print("Data loaded.")

# Merge the data
print("Merging data...")
train_df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test_df = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
del train_identity, train_transaction, test_identity, test_transaction
print("Data merged.")

# Preprocessing
print("Preprocessing data...")
# Fill missing values
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].fillna('Unknown')
    else:
        train_df[col] = train_df[col].fillna(-999)

for col in test_df.columns:
    if test_df[col].dtype == 'object':
        test_df[col] = test_df[col].fillna('Unknown')
    else:
        test_df[col] = test_df[col].fillna(-999)

# Label Encoding for categorical features
for col in tqdm(train_df.columns):
    if train_df[col].dtype == 'object' and col in test_df.columns:
        le = LabelEncoder()
        le.fit(list(train_df[col].astype(str).values) + list(test_df[col].astype(str).values))
        train_df[col] = le.transform(list(train_df[col].astype(str).values))
        test_df[col] = le.transform(list(test_df[col].astype(str).values))

# Combine all features into a single text feature for the transformer
def combine_features(df):
    text_features = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        text_features.append(' '.join([str(x) for x in row.values]))
    return text_features

train_texts = combine_features(train_df.drop(['isFraud', 'TransactionID'], axis=1))
test_texts = combine_features(test_df.drop('TransactionID', axis=1))
train_labels = train_df['isFraud'].values

# Split the training data for validation
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

class FraudDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FraudDataset(train_encodings, train_labels)
val_dataset = FraudDataset(val_encodings, val_labels)
# For the test dataset, we don't have labels, so we'll create a dummy label array
test_dataset = FraudDataset(test_encodings, [0] * len(test_texts))


# Model Training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    # Validation
    model.eval()
    val_loss = 0
    for batch in tqdm(val_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader)}")

print("Training finished.")

# Predictions
print("Making predictions...")
predictions = []
model.eval()
for batch in tqdm(test_loader):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())

# Create submission file
submission = sample_submission.copy()
submission['isFraud'] = predictions
submission.to_csv('submission.csv', index=False)

print("Submission file created.")
