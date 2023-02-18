import RNAFoldingNet
import numpy as np
import pandas as pd
import torch

# RNA sequence length
N = 30

# Read the training data
df = pd.read_csv("data_with_folded.csv")

# Convert RNA sequence to one-hot encoding
nucleotides = ['A', 'C', 'G', 'U']
def one_hot_nucl(seq):
    x = np.zeros((len(seq), 4))
    for i, nuc in enumerate(seq):
        x[i, nucleotides.index(nuc)] = 1
    return x

# Convert the RNA sequence to a one-hot encoding
train_data = np.array([one_hot_nucl(seq) for seq in df['seq']])
train_data = train_data.reshape(train_data.shape[0], -1)

# Convert the folded sequence to a one-hot encoding
hbonds = ['.', '(', ')']
def one_hot_hbond(seq):
    x = np.zeros((len(seq), 3))
    for i, hbond in enumerate(seq):
        x[i, hbonds.index(hbond)] = 1
    return x

train_labels = np.array([one_hot_hbond(seq) for seq in df['folded_seq']])
train_labels = train_labels.reshape(train_labels.shape[0], -1)

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2)

# Create the network
input_size = 4*N
hidden_size = 128
output_size = 3*N

net = RNAFoldingNet(input_size, hidden_size, output_size)

# Create the optimizer
import torch.optim as optim
import torch.nn as nn

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the network
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0, len(train_data), batch_size):
        inputs = train_data[i:i+batch_size]
        labels = train_labels[i:i+batch_size]

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Epoch %d, loss: %.3f' % (epoch+1, running_loss/len(train_data)))

# Test the network with the validation set using hamming distance
def hamming_distance(s1, s2):
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

avg_hamming_distance = 0
total = 0
with torch.no_grad():
    for i in range(len(val_data)):
        inputs = val_data[i]
        labels = val_labels[i]
        outputs = net(inputs)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        # Calculate the hamming distance
        hamming = hamming_distance(predicted, labels)
        avg_hamming_distance += hamming

print('Average hamming distance: %.3f' % (avg_hamming_distance/total))
