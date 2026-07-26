import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import matplotlib.pyplot as plt

class CNNLSTM(nn.Module):
    def __init__(self, input_channels=64, hidden_dim=128, num_classes=5, seq_length=1, 
                 dropout=0.5, bidirectional=True):
        """
        CNN+LSTM model for sleep stage classification
        
        Args:
            input_channels: Number of input channels (latent dimensions from autoencoder)
            hidden_dim: LSTM hidden dimension
            num_classes: Number of output classes
            seq_length: Sequence length for LSTM
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(CNNLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Convolutional layers for spatial feature extraction
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # LSTM layer for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0 if bidirectional else dropout
        )
        
        # Fully connected layers for classification
        lstm_output_dim = hidden_dim * self.num_directions
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: [batch_size, input_channels, seq_length]
        
        # Apply CNN feature extraction
        x = self.conv_block1(x)  # [batch_size, 64, seq_length//2]
        x = self.conv_block2(x)  # [batch_size, 128, seq_length//8]
        
        # Reshape for LSTM
        batch_size = x.size(0)
        seq_len = x.size(2)
        x = x.permute(0, 2, 1)  # [batch_size, seq_length//8, 128]
        
        # Pass through LSTM
        x, _ = self.lstm(x)  # [batch_size, seq_length//8, hidden_dim*num_directions]
        
        # Get the last time step output
        if self.bidirectional:
            # For bidirectional, concatenate the last outputs from both directions
            x = x[:, -1, :]  # [batch_size, hidden_dim*2]
        else:
            x = x[:, -1, :]  # [batch_size, hidden_dim]
            
        # Apply fully connected layers
        x = self.fc(x)  # [batch_size, num_classes]
        
        return x

class UNetLSTM(nn.Module):
    """
    U-Net + LSTM model for sleep stage classification using latent space features
    """
    def __init__(self, input_channels=64, hidden_dim=128, num_classes=5, dropout=0.5, bidirectional=True):
        super(UNetLSTM, self).__init__()
        # Contracting path
        self.encoder1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Expanding path
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        # LSTM for temporal dependencies
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * self.num_directions, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, input_channels, seq_length]
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        up = self.up1(e2)
        # Skip connection
        concat = torch.cat([up, e1], dim=1)
        d1 = self.decoder1(concat)
        # Prepare for LSTM: [batch, channels, seq] -> [batch, seq, channels]
        x_lstm = d1.permute(0, 2, 1)
        out_lstm, _ = self.lstm(x_lstm)
        # Classification on last time step
        out = out_lstm[:, -1, :]
        return self.fc(out)

def train_classifier(model, train_loader, val_loader, device, epochs=50, 
                    lr=0.001, save_path="classifier.pt", class_weights=None, patience=10):
    """
    Train the CNN+LSTM classifier
    
    Args:
        model: The classifier model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to use for training
        epochs: Number of epochs to train
        lr: Learning rate
        save_path: Path to save the model
        class_weights: Class weights for imbalanced data
        patience: Early stopping patience
    """
    model = model.to(device)
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Use weighted loss if class weights are provided
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Use mixed precision training to save memory
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    best_val_acc = 0
    epochs_without_improvement = 0
    # Track metrics for plotting
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(output, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({'loss': train_loss / (batch_idx + 1)})
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(output, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        # Record metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Model saved to {save_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Print confusion matrix for best model
    model.load_state_dict(torch.load(save_path))
    model.eval()
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(target.cpu().numpy())
    
    # Display confusion matrix
    cm = confusion_matrix(val_targets, val_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot and save classifier loss and accuracy curves
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    # Loss curve
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='train_loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Classifier Training Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'classifier_loss.png'))
    plt.close()
    # Accuracy curve
    plt.figure()
    plt.plot(range(1, len(train_accs)+1), train_accs, label='train_acc')
    plt.plot(range(1, len(val_accs)+1), val_accs, label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'classifier_accuracy.png'))
    plt.close()
    
    return model

# Example usage
if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    
    # Generate dummy latent data and labels for testing
    latent_dim = 64
    num_samples = 1000
    num_classes = 5
    
    # Create random latent data and labels
    latent_data = np.random.randn(num_samples, latent_dim, 1)
    labels = np.random.randint(0, num_classes, num_samples)
    
    # Create PyTorch dataset and dataloader
    tensor_x = torch.Tensor(latent_data)
    tensor_y = torch.LongTensor(labels)
    dataset = TensorDataset(tensor_x, tensor_y)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize and train the classifier
    model = CNNLSTM(
        input_channels=latent_dim,
        hidden_dim=128,
        num_classes=num_classes,
        seq_length=1
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Calculate class weights (optional)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / np.sum(class_weights) * len(class_counts)
    
    # Train the classifier
    model = train_classifier(
        model, 
        train_loader, 
        val_loader, 
        device,
        epochs=2,  # Just for testing
        class_weights=class_weights,
        save_path="classifier.pt"
    )