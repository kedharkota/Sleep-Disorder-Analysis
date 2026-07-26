import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler  # Updated import path
import numpy as np
from tqdm import tqdm

class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=4, sequence_length=3000, latent_dim=64):
        """
        Convolutional Autoencoder for EEG signal processing
        
        Args:
            input_channels: Number of input channels (EEG, EOG, EMG)
            sequence_length: Length of signal sequence (30s at 100Hz = 3000)
            latent_dim: Dimension of latent space
        """
        super(ConvAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # First define encoder and decoder, then calculate encoded_size
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=50, stride=6, padding=25),  # -> [16, 500]
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=8, stride=4, padding=4),  # -> [32, 125]
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=4),  # -> [64, 32]
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=4),  # -> [128, 8]
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, latent_dim, kernel_size=8, stride=1, padding=0)  # -> [latent_dim, 1]
        )
        
        # Decoder - carefully adjusted to match input dimensions
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 128, kernel_size=8, stride=1, padding=0),  # -> [128, 8]
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=4, output_padding=3),  # -> [64, 32]
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=4, output_padding=3),  # -> [32, 125]
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=4, padding=4, output_padding=3),  # -> [16, 500]
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, input_channels, kernel_size=50, stride=6, padding=25, output_padding=0),  # -> [input_channels, sequence_length]
            nn.Tanh()  # Normalized data is in range [-1, 1]
        )
        
        # Calculate encoder output size after encoder is defined (optional)
        try:
            self.encoded_size = self._calculate_encoder_output_size()
        except Exception as e:
            print(f"Warning: could not calculate encoded_size: {e}")
            self.encoded_size = None
        
    def _calculate_encoder_output_size(self):
        """Calculate the output size of the encoder to ensure correct dimensions"""
        # Create a dummy input tensor
        x = torch.randn(1, self.input_channels, self.sequence_length)
        # Temporarily disable BatchNorm training statistics
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            x = self.encoder(x)
        # Restore original training mode
        if was_training:
            self.encoder.train()
        return x.shape[-1]  # Return the sequence length after encoding
        
    def forward(self, x):
        # Get original shape for later dimension check
        original_shape = x.shape
        
        # Encode
        z = self.encoder(x)
        
        # Decode
        x_recon = self.decoder(z)
        
        # Ensure the output has the same dimensions as the input
        if x_recon.shape[2] != original_shape[2]:
            # If mismatch, resize using interpolation
            x_recon = F.interpolate(x_recon, size=original_shape[2], mode='linear', align_corners=False)
        
        return x_recon, z
    
    def encode(self, x):
        """Encode input to latent space representation"""
        with torch.no_grad():
            z = self.encoder(x)
        return z

def train_autoencoder(model, train_loader, val_loader, device, epochs=50, 
                      lr=0.001, save_path="autoencoder.pt", patience=10):
    """
    Train the autoencoder model
    
    Args:
        model: The autoencoder model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to use for training
        epochs: Number of epochs to train
        lr: Learning rate
        save_path: Path to save the model
        patience: Early stopping patience
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Use mixed precision training to save memory
    scaler = GradScaler('cuda')  # Updated GradScaler initialization
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    # Lists to store losses for plotting
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (data, _) in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast('cuda'):  # Updated autocast usage
                reconstructed, _ = model(data)
                loss = criterion(reconstructed, data)
            
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'train_loss': train_loss / (batch_idx + 1)})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                
                with autocast('cuda'):  # Updated autocast usage
                    reconstructed, _ = model(data)
                    loss = criterion(reconstructed, data)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save best model and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            epochs_without_improvement = 0
            print(f"Model saved to {save_path}")
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Plot and save loss curves
    import os
    import matplotlib.pyplot as plt
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='train_loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'autoencoder_loss.png'))
    plt.close()
    # Load best model
    model.load_state_dict(torch.load(save_path))
    return model

def encode_dataset(autoencoder, data_loader, device):
    """
    Encode a dataset using the trained autoencoder
    
    Args:
        autoencoder: Trained autoencoder model
        data_loader: DataLoader for the dataset
        device: Device to use for encoding
    
    Returns:
        latent_data: Encoded data in latent space
        labels: Corresponding labels
    """
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    
    latent_data = []
    labels = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Encoding dataset"):
            data = data.to(device)
            
            # Get latent representation
            latent = autoencoder.encode(data)
            
            # Move to CPU and convert to numpy
            latent_data.append(latent.cpu().numpy())
            labels.append(target.numpy())
    
    # Concatenate batches
    latent_data = np.concatenate(latent_data, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return latent_data, labels

# Example usage
if __name__ == "__main__":
    import sys
    from data_processing import prepare_dataloaders
    
    data_dir = "dataset"
    record_file = "dataset/RECORDS"
    
    # Use small batch size and limit for testing
    train_loader, val_loader, dataset = prepare_dataloaders(
        data_dir, 
        record_file, 
        batch_size=8, 
        limit=2
    )
    
    # Get shapes from first batch
    for data, _ in train_loader:
        input_channels = data.shape[1]
        sequence_length = data.shape[2]
        break
    
    # Initialize autoencoder
    autoencoder = ConvAutoencoder(
        input_channels=input_channels,
        sequence_length=sequence_length,
        latent_dim=64
    )
    
    # Train autoencoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    autoencoder = train_autoencoder(
        autoencoder, 
        train_loader, 
        val_loader, 
        device,
        epochs=2,  # Just for testing
        save_path="autoencoder.pt"
    )
    
    # Encode dataset
    latent_data, labels = encode_dataset(autoencoder, train_loader, device)
    print(f"Encoded data shape: {latent_data.shape}")
    print(f"Labels shape: {labels.shape}")