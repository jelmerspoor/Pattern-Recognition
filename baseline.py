import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# A very basic convolutional neural network
class CNN2D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Three layers of simple 3x3 filters with a 2x2 pooling layer
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Creating a very simple classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    # How the data is passed through the model
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Variables
num_classes = 41
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# The device, model, criterion and optimizer we are using
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN2D(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# The dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Training and validation functions
for epoch in range(num_epochs):

    # Training phase
    model.train()
    running_loss = 0.0

    for spectrograms, labels in train_loader:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        # clear the previously used gradiant
        optimizer.zero_grad()

        # Forward pass
        logits = model(spectrograms)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # Weights for calculating MAP@3
    ranks = torch.tensor([1.0, 0.5, 1.0 / 3.0], device=device)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    correct = 0
    map3_sum = 0.0


    # Disable gradient calculations during validation
    with torch.no_grad():
        for spectrograms, labels in val_loader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(spectrograms)

            # Calculate validation loss
            loss = criterion(logits, labels)

            running_val_loss += loss.item() * labels.size(0)

            # Calculate accuracy
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()

            # Get the three highest scoring predictions
            top3 = logits.topk(3, dim=1).indices

            # Calculate MAP@3
            matches = top3 == labels.unsqueeze(1)
            map3_sum += (matches * ranks).sum().item()

    # Calculate the average validation metrics
    val_loss = running_val_loss / len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)
    map3 = map3_sum / len(val_loader.dataset)

    # Print the results of the current epoch
    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.4f} | "
        f"MAP@3: {map3:.4f}"
    )

# Save the trained model
torch.save(model.state_dict(), "cnn_baseline.pth")