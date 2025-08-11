#only f/t but x 5

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as transforms
import re
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Custom Dataset for loading PNG spectrograms
class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, augment=True):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []

        # Genre mapping
        self.genre_to_idx = {
            'classical': 0, 'country': 1, 'hiphop': 2, 'jazz': 3,
            'metal': 4, 'pop': 5, 'reggae': 6
        }


        # ----------------- NEW (recursive scan) -----------------
        for root, _, files in os.walk(data_dir):                 # walk sub-folders
            genre = os.path.basename(root).lower()               # folder name = genre
            if genre not in self.genre_to_idx:
                continue                                         # skip misc folders
            for filename in files:
                if filename.lower().endswith('.png'):
                    self.image_paths.append(os.path.join(root, filename))
                    self.labels.append(self.genre_to_idx[genre])
        # --------------------------------------------------------

        print(f"Debug: Found {len(self.image_paths)} valid images")

        # Data augmentation transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((227, 227)),  # AlexNet input size
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')  # Convert to RGB (3 channels)
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

# CNN + ANN Model following exact specifications
class CNNWithANN(nn.Module):
    def __init__(self, num_hidden_layers=2, hidden_layer_sizes=[512, 256],
                 dropout_prob=0.5, activation='relu', use_batch_norm=True):
        # Call __init__ method of nn.Module
        super(CNNWithANN, self).__init__()

        # CNN: AlexNet (pre-trained) - Input: 227x227x3, Output: 256x6x6
        # Extracts only the CNN layers, ignores the ANN ones (Transfer Learning )
        alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.cnn = alexnet.features  # Output: [batch_size, 256, 6, 6]

        # Freeze CNN parameters for transfer learning
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Flatten CNN output for ANN input: 256 * 6 * 6 = 9216
        self.flatten = nn.Flatten()

        # ANN: Build dynamic architecture
        self.ann_layers = nn.ModuleList()
        input_size = 256 * 6 * 6  # 9216

        # Build hidden layers
        for i in range(num_hidden_layers):
            layer_size = hidden_layer_sizes[i] if i < len(hidden_layer_sizes) else hidden_layer_sizes[-1]

            # Linear layer
            self.ann_layers.append(nn.Linear(input_size, layer_size))

            # Batch normalization
            if use_batch_norm:
                self.ann_layers.append(nn.BatchNorm1d(layer_size))

            # Activation function
            if activation == 'relu':
                self.ann_layers.append(nn.ReLU())
            elif activation == 'tanh':
                self.ann_layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                self.ann_layers.append(nn.Sigmoid())

            # Dropout
            self.ann_layers.append(nn.Dropout(dropout_prob))

            input_size = layer_size

        # Output layer (7 genres)
        self.output_layer = nn.Linear(input_size, 7)

        # Store hyperparameters
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        # CNN forward pass
        x = self.cnn(x)  # Output: [batch_size, 256, 6, 6]

        # Flatten for ANN
        x = self.flatten(x)  # Output: [batch_size, 9216]

        # ANN forward pass
        for layer in self.ann_layers:
            x = layer(x)

        # Output layer (no softmax here - will be applied in loss function)
        x = self.output_layer(x)  # Output: [batch_size, 7]

        return x

    def get_model_info(self):
        """Return detailed model information for assignment"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'architecture': 'CNN (AlexNet) + ANN',
            'input_size': '227x227x3',
            'cnn_output_shape': '256x6x6',
            'ann_input_size': 9216,
            'num_hidden_layers': self.num_hidden_layers,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation_function': self.activation,
            'dropout_rate': self.dropout_prob,
            'batch_normalization': self.use_batch_norm,
            'output_classes': 7,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }

def train_with_early_stopping(model, train_data, val_data, batch_size=32,
                             num_epochs=50, lr=1e-3, patience=10):
    """Training with early stopping and comprehensive metrics"""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Multi-class cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Track metrics
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for imgs, labels in train_loader:
            # Zero out gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(imgs)
            # Compute loss and compare to true labels
            loss = criterion(outputs, labels)
            # Compute gradients
            loss.backward()
            # Taking a step
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Plot results
    plot_training_results(train_losses, train_accs, val_losses, val_accs)

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }

def plot_training_results(train_losses, train_accs, val_losses, val_accs):
    """Plot comprehensive training results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', marker='o', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', marker='s', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', marker='o', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', marker='s', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # Model architecture diagram
    ax3.text(0.1, 0.9, 'CNN + ANN Architecture:', fontsize=14, fontweight='bold')
    ax3.text(0.1, 0.8, '1. Input: 227×227×3 Spectrogram', fontsize=10)
    ax3.text(0.1, 0.7, '2. CNN: AlexNet Features (Frozen)', fontsize=10)
    ax3.text(0.1, 0.6, '3. Output: 256×6×6 Feature Maps', fontsize=10)
    ax3.text(0.1, 0.5, '4. Flatten: 9216 Features', fontsize=10)
    ax3.text(0.1, 0.4, '5. ANN: Hidden Layers + Dropout', fontsize=10)
    ax3.text(0.1, 0.3, '6. Output: 7 Genre Classes', fontsize=10)
    ax3.text(0.1, 0.2, '7. Loss: Cross-Entropy', fontsize=10)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    # Performance summary
    final_train_acc = train_accs[-1]
    final_val_acc = val_accs[-1]
    ax4.text(0.1, 0.8, 'Performance Summary:', fontsize=14, fontweight='bold')
    ax4.text(0.1, 0.7, f'Final Training Accuracy: {final_train_acc:.2f}%', fontsize=12)
    ax4.text(0.1, 0.6, f'Final Validation Accuracy: {final_val_acc:.2f}%', fontsize=12)
    ax4.text(0.1, 0.5, f'Total Epochs: {len(epochs)}', fontsize=12)
    ax4.text(0.1, 0.4, 'Genres: Classical, Country, Hip-hop,', fontsize=10)
    ax4.text(0.1, 0.3, '        Jazz, Metal, Pop, Reggae', fontsize=10)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.tight_layout()
    plt.show()

def print_model_summary(model):
    """Print detailed model architecture summary"""
    info = model.get_model_info()
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    print(f"Architecture: {info['architecture']}")
    print(f"Input Size: {info['input_size']}")
    print(f"CNN Output Shape: {info['cnn_output_shape']}")
    print(f"ANN Input Size: {info['ann_input_size']:,}")
    print(f"Hidden Layers: {info['num_hidden_layers']}")
    print(f"Hidden Layer Sizes: {info['hidden_layer_sizes']}")
    print(f"Activation Function: {info['activation_function']}")
    print(f"Dropout Rate: {info['dropout_rate']}")
    print(f"Batch Normalization: {info['batch_normalization']}")
    print(f"Output Classes: {info['output_classes']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"Frozen Parameters: {info['frozen_parameters']:,}")
    print("="*60)

def test_single_image(model, image_path, genre_names):
    """Test model on a single spectrogram image"""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, 1)[0]

    predicted_genre = genre_names[predicted.item()]
    confidence_score = confidence[predicted.item()].item()

    print(f"Predicted Genre: {predicted_genre}")
    print(f"Confidence: {confidence_score:.2%}")

    return predicted_genre, confidence_score


# Main execution
if __name__ == "__main__":
    # Load datasets
    train_data = SpectrogramDataset("/content/drive/My Drive/aps360/f_t_data/training", augment=True)
    val_data = SpectrogramDataset("/content/drive/My Drive/aps360/f_t_data/validation", augment=False)

    print(f"Loaded {len(train_data)} training images")
    print(f"Loaded {len(val_data)} validation images")
    if len(train_data) == 0:
        print("ERROR: No images loaded! Check the data directory path.")
        exit()

    genre_names = list(train_data.genre_to_idx.keys())
    print(f"Genres: {genre_names}")

    model = CNNWithANN(
            num_hidden_layers=2,
            hidden_layer_sizes=[512, 256],
            dropout_prob=0.8,
            activation='relu',
            use_batch_norm=True
        )

    # Print model summary
    print_model_summary(model)

    results = train_with_early_stopping(model, train_data, val_data,
                                          batch_size=64, num_epochs=50, lr=5e-4)

    # Test on testing data
    test_data = SpectrogramDataset("/content/drive/My Drive/aps360/f_t_data/testing", augment=False)
    if len(test_data) > 0:
        # Test first image in testing folder
        first_test_img_path = test_data.image_paths[20]
        print(f"\nTesting on: {first_test_img_path}")
        test_single_image(model, first_test_img_path, genre_names)

