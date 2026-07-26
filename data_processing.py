import os
import numpy as np
import pandas as pd
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Sleep stage mapping
STAGE_MAP = {
    'W': 0,    # Wake
    'R': 1,    # REM
    '1': 2,    # N1
    '2': 3,    # N2
    '3': 4,    # N3
    '4': 4,    # N4 (combining with N3 as SWS)
    'M': -1,   # Movement, will be excluded
    '?': -1    # Unknown/missing, will be excluded
}

class SleepDataset(Dataset):
    def __init__(self, data_dir, record_file=None, transform=None, latent=False):
        """
        Dataset for sleep stage classification
        
        Args:
            data_dir: Directory containing the dataset
            record_file: File containing list of EDF files to load (optional)
            transform: Transform to apply to the data
            latent: If True, assumes data is already encoded in latent space
        """
        self.data_dir = data_dir
        self.transform = transform
        self.latent = latent
        self.data = []
        self.labels = []
        
        # Load record file if provided
        self.edf_files = []
        if record_file and os.path.exists(record_file):
            with open(record_file, 'r') as f:
                self.edf_files = [line.strip() for line in f if line.strip().endswith('-PSG.edf')]
        
    def load_data(self, limit=None, channels=['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']):
        """
        Load data from EDF files
        
        Args:
            limit: Maximum number of files to load (for testing)
            channels: List of channels to extract
        """
        print(f"Loading data from {len(self.edf_files) if self.edf_files else 'all'} EDF files...")
        
        # Get all PSG files if no record file was provided
        if not self.edf_files:
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith('-PSG.edf'):
                        self.edf_files.append(os.path.join(os.path.relpath(root, self.data_dir), file))
        
        # Limit the number of files for testing
        if limit:
            self.edf_files = self.edf_files[:limit]
        
        for i, psg_file in enumerate(tqdm(self.edf_files)):
            try:
                # Get corresponding hypnogram file
                hypno_file = None
                psg_path = os.path.join(self.data_dir, psg_file)
                psg_dir = os.path.dirname(psg_path)
                psg_basename = os.path.basename(psg_file).replace('-PSG.edf', '')
                
                # Find matching hypnogram file (they have different suffixes in their names)
                for f in os.listdir(psg_dir):
                    if f.startswith(psg_basename[:-1]) and 'Hypnogram' in f:
                        hypno_file = os.path.join(psg_dir, f)
                        break
                
                if not hypno_file:
                    print(f"Skipping {psg_file}: No matching hypnogram file found")
                    continue
                
                # Load PSG data using MNE
                raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
                
                # Check if required channels exist
                available_channels = raw.ch_names
                use_channels = [ch for ch in channels if ch in available_channels]
                
                if len(use_channels) != len(channels):
                    missing = set(channels) - set(use_channels)
                    print(f"Warning: File {psg_file} missing channels: {missing}")
                    # If critical channels are missing, skip this file
                    if not any(ch in available_channels for ch in ['EEG Fpz-Cz', 'EEG Pz-Oz']):
                        print(f"Skipping {psg_file}: Missing essential EEG channels")
                        continue
                
                # Pick channels and resample to 100 Hz
                if raw.info['sfreq'] != 100:
                    raw = raw.pick_channels(use_channels).resample(100)
                else:
                    raw = raw.pick_channels(use_channels)
                
                # Load hypnogram data
                hypno = mne.read_annotations(hypno_file)
                
                # Match hypnogram to the data
                raw.set_annotations(hypno)
                events, _ = mne.events_from_annotations(raw, event_id=lambda x: 1)
                
                # Extract 30-second epochs
                epoch_duration = 30  # seconds
                data_array = raw.get_data()
                n_channels, n_samples = data_array.shape
                samples_per_epoch = epoch_duration * int(raw.info['sfreq'])
                n_epochs = n_samples // samples_per_epoch
                
                # Reshape data into epochs
                epochs_data = np.zeros((n_epochs, n_channels, samples_per_epoch))
                for e in range(n_epochs):
                    start = e * samples_per_epoch
                    end = start + samples_per_epoch
                    epochs_data[e] = data_array[:, start:end]
                
                # Extract labels from hypnogram
                epoch_labels = np.full(n_epochs, -1)  # Default to -1 (exclude)
                
                for annotation in hypno:
                    onset = annotation['onset']
                    duration = annotation['duration']
                    description = annotation['description']
                    
                    if description in STAGE_MAP:
                        start_epoch = int(onset / epoch_duration)
                        end_epoch = int((onset + duration) / epoch_duration)
                        
                        # Ensure within array bounds
                        if 0 <= start_epoch < n_epochs:
                            end_epoch = min(end_epoch, n_epochs)
                            epoch_labels[start_epoch:end_epoch] = STAGE_MAP[description]
                
                # Filter out epochs with excluded labels
                valid_mask = epoch_labels >= 0
                if np.sum(valid_mask) > 0:
                    self.data.append(epochs_data[valid_mask])
                    self.labels.append(epoch_labels[valid_mask])
                    
            except Exception as e:
                print(f"Error processing {psg_file}: {str(e)}")
        
        # Concatenate data and labels
        if len(self.data) > 0:
            self.data = np.vstack([d for d in self.data if len(d) > 0])
            self.labels = np.concatenate([l for l in self.labels if len(l) > 0])
            
            # Normalize each channel
            for i in range(self.data.shape[1]):
                scaler = StandardScaler()
                self.data[:, i, :] = scaler.fit_transform(self.data[:, i, :])
            
            print(f"Loaded {len(self.data)} epochs with shape {self.data.shape}")
            print(f"Class distribution: {np.bincount(self.labels.astype(int))}")
        else:
            print("No valid data loaded!")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        # Convert to torch tensors
        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()
        
        return x, y

def prepare_dataloaders(data_dir, record_file=None, batch_size=32, num_workers=4, limit=None):
    """
    Prepare data loaders for training and validation
    
    Args:
        data_dir: Directory containing the dataset
        record_file: File containing list of EDF files to load
        batch_size: Batch size
        num_workers: Number of workers for data loading
        limit: Maximum number of files to load (for testing)
    """
    # Create and load the dataset
    dataset = SleepDataset(data_dir, record_file)
    dataset.load_data(limit=limit)
    
    # Split dataset into train and validation sets
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)), 
        test_size=0.2, 
        random_state=42,
        stratify=dataset.labels
    )
    
    # Create samplers for training and validation
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset

# Example usage
if __name__ == "__main__":
    # Test data loading
    data_dir = "dataset"
    record_file = "dataset/RECORDS"
    
    train_loader, val_loader, dataset = prepare_dataloaders(
        data_dir, 
        record_file,
        batch_size=32, 
        limit=2  # Only load 2 files for testing
    )
    
    # Print sample batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
        break