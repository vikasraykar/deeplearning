---
title: Training loop
weight: 6
bookToc: true
---

## Training loop

```python
# Load the dataset.
train_dataset = SampleDataset(X_train, y_train)
test_dataset = SampleDataset(X_test, y_test)

# Preparing your data for training with DataLoaders.
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Define the model class.
model = LogisticRegression(num_features=d)

# Loss fucntion.
loss_fn = nn.BCELoss()

# Optimizer.
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Learning rate scheduler.
scheduler = ExponentialLR(optimizer, gamma=0.9)

# Run for a few epochs.
for epoch in range(n_epochs):
    # Iterate through the DataLoader to access mini-batches.
    for batch, (input, target) in enumerate(train_dataloader):
        # Prediction.
        output = model(input)

        # Compute loss.
        loss = loss_fn(output, target)

        # Compute gradient.
        loss.backward()

        # Gradient descent.
        optimizer.step()

        # Prevent gradient accumulation
        optimizer.zero_grad()

    # Adjust learning rate
    scheduler.step()

```


