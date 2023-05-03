import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import youtokentome as yttm
import preprocess_transformers as pt

def create_model_and_run(data_path, output_path, vocab_size, max_seq_len, embedding_dim, dataset_directory, min_ratio):
  tokenizer = pt.create_tokenizer(data_path, output_path, vocab_size)
  tsv_file = compose_fns(tokenizer, max_seq_len, embedding_dim, dataset_directory, min_ratio)
  return tsv_file

def load_model_and_run(tokenizer_path, max_seq_len, embedding_dim, dataset_directory, min_ratio):
  tokenizer = pt.load_tokenizer(tokenizer_path)
  tsv_file = compose_fns(tokenizer, max_seq_len, embedding_dim, dataset_directory, min_ratio)
  return tsv_file

def compose_fns(tokenizer, max_seq_len, embedding_dim, dataset_directory, min_ratio):
  vocab_size = tokenizer.vocab_size()
  generator_fn = lambda: pt.get_sample(tokenizer, dataset_directory, max_seq_len, min_ratio, PAD_ID=0)
  out_name = f"data_{vocab_size}_{max_seq_len}.tsv"
  pt.write_dataset_to_tsv(out_name, generator_fn)
  return out_name

from torch.utils.data import DataLoader, Dataset, random_split


class MyDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                # Convert the string data to numeric data as necessary
                data_row = (torch.tensor([int(x) for x in line[0].split(" ")]), int(line[1]))
                self.data.append(data_row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


model_path = "raw_dataset/Talmud/model"
embedding_dim = 500
max_seq_length = 128
min_ratio = 0.05
# tsv_file = load_model_and_run(model_path, max_seq_length, embedding_dim, "./raw_dataset/Talmud/organized", min_ratio)
tsv_file = "data_500_128.tsv"

model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=max_seq_length, nhead=2, dtype=torch.float32), num_layers=1)

dataset = MyDataset(tsv_file)

train_data, val_data, test_data = random_split(dataset, [0.7, 0.15, 0.15])

def collate_fn(batch):
    # Separate the data and labels
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Stack the data and labels
    data = torch.stack(data, dim=0)
    labels = torch.tensor(labels)

    return data, labels


# create the data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Put your model in training mode
model.train()
print(model)
# Train for some number of epochs
for epoch in range(10):
    running_loss = 0.0
    
    # Iterate over the batches in the data loader
    for i, data in enumerate(train_loader):
        # Get the inputs and labels
        inputs, labels = data
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished training')

# attention_mask = ~(input_ids != PAD_ID).unsqueeze(1).unsqueeze(2)

# out = model(input_ids.unsqueeze(1) )

# print(out, out.shape)