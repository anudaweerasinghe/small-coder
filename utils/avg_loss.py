import re
import numpy as np
from matplotlib import pyplot as plt

# Read the contents of the file
with open('output.log', 'r') as file:
    lines = file.readlines()

# Filter lines starting with "{'loss':"
loss_lines = [line.strip() for line in lines if line.startswith("{'loss':")]

# Extract loss and epoch values using regular expressions
loss_values = []
epoch_values = []
for line in loss_lines:
    match = re.search(r"'loss':\s*(\d+\.\d+)", line)
    if match:
        loss_values.append(float(match.group(1)))
    
    match = re.search(r"'epoch':\s*(\d+\.\d+)", line)
    if match:
        epoch_values.append(float(match.group(1)))

# Define the epoch bins
epoch_bins = [0, 1, 2, 3, 4, 5, 6]

# Calculate average loss for each epoch bin
avg_losses = []
for i in range(len(epoch_bins) - 1):
    start_epoch = epoch_bins[i]
    end_epoch = epoch_bins[i + 1]
    bin_losses = [loss for loss, epoch in zip(loss_values, epoch_values) if start_epoch <= epoch < end_epoch]
    avg_loss = np.mean(bin_losses) if bin_losses else 0
    avg_losses.append(avg_loss)

# Plot the average loss per epoch bin
plt.figure(figsize=(8, 6))
plt.plot(epoch_bins[:-1], avg_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Average Loss per Epoch Bin')
plt.xticks(epoch_bins[:-1])
plt.grid(True)
plt.show()