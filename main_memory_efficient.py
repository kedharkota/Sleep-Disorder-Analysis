import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import gc  # For garbage collection
import scipy.signal as signal

# Import our modules
from autoencoder import ConvAutoencoder, train_autoencoder, encode_dataset
from cnn_lstm import UNetLSTM, train_classifier

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

# Mapping from annotation descriptions to sleep stages
ANNOTATION_TO_STAGE = {
    'Sleep stage W': 'W',
    'Sleep stage R': 'R',
    'Sleep stage 1': '1',
    'Sleep stage 2': '2',
    'Sleep stage 3': '3',
    'Sleep stage 4': '4',
    'Movement time': 'M',
}

class SleepDataset(Dataset):
    """
    Memory-efficient dataset for sleep stage classification.
    Instead of loading all data at once, this loads data on-demand.
    """
    def __init__(self, processed_files, data_dir, transform=None):
        """
        Initialize the dataset with preprocessed file information
        
        Args:
            processed_files: List of tuples (file_path, epochs_data, epochs_labels)
            data_dir: Root directory for data
            transform: Optional transform to apply to data
        """
        self.processed_files = processed_files
        self.data_dir = data_dir
        self.transform = transform
        
        # Create index mapping for quick access
        self.index_map = []
        self.total_epochs = 0
        
        for file_idx, file_info in enumerate(processed_files):
            file_path, _, epochs_labels = file_info
            n_epochs = len(epochs_labels)
            
            # Create mapping from global index to (file_idx, local_epoch_idx)
            for local_epoch_idx in range(n_epochs):
                self.index_map.append((file_idx, local_epoch_idx))
            
            self.total_epochs += n_epochs
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        # Map global index to file and local epoch index
        file_idx, local_epoch_idx = self.index_map[idx]
        file_path, file_data_path, epochs_labels = self.processed_files[file_idx]
        
        # Load data from disk (this is the memory-efficient part)
        # We only load the specific epoch we need, not the entire file
        if os.path.exists(file_data_path):
            # Load the specific epoch from the saved file
            all_epochs = np.load(file_data_path, mmap_mode='r')  # Memory-mapped mode
            x = all_epochs[local_epoch_idx].copy()  # Make a copy of just this epoch
        else:
            raise FileNotFoundError(f"Preprocessed data file not found: {file_data_path}")
        
        # Get label
        y = epochs_labels[local_epoch_idx]
        
        # Apply transforms if any
        if self.transform:
            x = self.transform(x)
        
        # Convert to PyTorch tensors
        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()
        
        return x, y

def preprocess_sleep_data(data_dir, record_file=None, limit=None, 
                   channels=['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental'],
                   output_dir="preprocessed_data"):
    """
    Preprocess sleep data from EDF files and save to disk
    
    Args:
        data_dir: Directory containing the dataset
        record_file: File containing list of EDF files to load
        limit: Maximum number of files to load (for testing)
        channels: List of channels to extract
        output_dir: Directory to save processed data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Suppress specific MNE warnings that are expected and not problematic
    warnings.filterwarnings('ignore', category=RuntimeWarning, 
                          message='Channels contain different .* filters')
    warnings.filterwarnings('ignore', category=RuntimeWarning, 
                          message='Limited .* annotation')
    
    # Load record file if provided
    edf_files = []
    if record_file and os.path.exists(record_file):
        with open(record_file, 'r') as f:
            edf_files = [line.strip() for line in f if line.strip().endswith('-PSG.edf')]
    
    # Get all PSG files if no record file was provided
    if not edf_files:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('-PSG.edf'):
                    edf_files.append(os.path.join(os.path.relpath(root, data_dir), file))
    
    # Limit the number of files for testing
    if limit:
        edf_files = edf_files[:limit]
    
    print(f"Preprocessing {len(edf_files)} EDF files...")
    
    processed_files = []
    processed_file_count = 0
    
    for psg_file in tqdm(edf_files):
        try:
            # Generate output filename
            output_file = os.path.join(output_dir, f"{os.path.basename(psg_file).replace('.edf', '_epochs.npy')}")
            
            # Skip if already processed
            if os.path.exists(output_file):
                # Just load the labels to add to the processed_files list
                labels_file = output_file.replace('_epochs.npy', '_labels.npy')
                if os.path.exists(labels_file):
                    labels = np.load(labels_file)
                    processed_files.append((psg_file, output_file, labels))
                    processed_file_count += 1
                    print(f"Skipping already processed file: {psg_file}")
                    continue
            
            # Get corresponding hypnogram file
            hypno_file = None
            psg_path = os.path.join(data_dir, psg_file)
            psg_dir = os.path.dirname(psg_path)
            psg_basename = os.path.basename(psg_file).replace('-PSG.edf', '')
            
            # Try different locations for hypnogram files based on dataset type
            if 'sleep-cassette' in psg_file:
                # For sleep-cassette, first try in the same directory
                prefix = psg_basename[:-1]  # Remove the last character (usually '0')
                # Look in the same directory first
                for f in os.listdir(psg_dir):
                    if f.startswith(prefix) and 'Hypnogram' in f:
                        hypno_file = os.path.join(psg_dir, f)
                        break
                
                # If not found, check if there's an annotations subdirectory
                if not hypno_file:
                    annotations_dir = os.path.join(data_dir, "sleep-cassette", "annotations")
                    if os.path.exists(annotations_dir):
                        for f in os.listdir(annotations_dir):
                            if f.startswith(prefix) and 'Hypnogram' in f:
                                hypno_file = os.path.join(annotations_dir, f)
                                break
            elif 'sleep-telemetry' in psg_file:
                # For sleep-telemetry, try hypnogram files in the same directory
                # Example: ST7011J0-PSG.edf → ST7011JP-Hypnogram.edf or similar
                prefix = psg_basename[:-1]  # Remove the last character (usually '0')
                for f in os.listdir(psg_dir):
                    if f.startswith(prefix) and 'Hypnogram' in f:
                        hypno_file = os.path.join(psg_dir, f)
                        break
            
            if not hypno_file:
                print(f"Skipping {psg_file}: No matching hypnogram file found")
                continue
            
            # Load PSG data using MNE with filter warnings suppressed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if raw.info['sfreq'] != 100:
                    raw = raw.pick(use_channels).resample(100)
                else:
                    raw = raw.pick(use_channels)
            
            # Get data duration before loading annotations
            data_duration = raw.times[-1]
            
            # Load hypnogram data
            hypno = mne.read_annotations(hypno_file)
            
            # Trim annotations that exceed data duration
            valid_annotations = []
            for annot in hypno:
                onset = annot['onset']
                duration = annot['duration']
                description = annot['description']
                
                # Skip annotations that start after data ends
                if onset >= data_duration:
                    continue
                
                # Trim annotation duration if it extends beyond data
                if onset + duration > data_duration:
                    duration = data_duration - onset
                
                valid_annotations.append({
                    'onset': onset,
                    'duration': duration,
                    'description': description
                })
            
            # Create new annotations object from valid annotations
            if valid_annotations:
                onsets = [a['onset'] for a in valid_annotations]
                durations = [a['duration'] for a in valid_annotations]
                descriptions = [a['description'] for a in valid_annotations]
                trimmed_hypno = mne.Annotations(onsets, durations, descriptions)
                
                # Match hypnogram to the data
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    raw.set_annotations(trimmed_hypno)
                
                # Print unique annotation descriptions for debugging
                unique_descriptions = list(set([a['description'] for a in valid_annotations]))
                print(f"Found {len(unique_descriptions)} stage types: {unique_descriptions}")
                
                # Extract 30-second epochs
                epoch_duration = 30  # seconds
                data_array = raw.get_data()
                n_channels, n_samples = data_array.shape
                samples_per_epoch = epoch_duration * int(raw.info['sfreq'])
                n_epochs = n_samples // samples_per_epoch
                
                # Reshape data into epochs
                epochs_data = np.zeros((n_epochs, n_channels, samples_per_epoch), dtype=np.float32)  # Use float32 instead of float64
                for e in range(n_epochs):
                    start = e * samples_per_epoch
                    end = start + samples_per_epoch
                    epochs_data[e] = data_array[:, start:end]
                
                # Free up memory
                del data_array
                gc.collect()
                
                # Extract labels from hypnogram
                epoch_labels = np.full(n_epochs, -1)  # Default to -1 (exclude)
                
                for annotation in valid_annotations:
                    onset = annotation['onset']
                    duration = annotation['duration']
                    description = annotation['description']
                    
                    # Map annotation description to sleep stage
                    stage = ANNOTATION_TO_STAGE.get(description)
                    
                    if stage and stage in STAGE_MAP:
                        stage_label = STAGE_MAP[stage]
                        start_epoch = int(onset / epoch_duration)
                        end_epoch = int((onset + duration) / epoch_duration)
                        
                        # Ensure within array bounds
                        if 0 <= start_epoch < n_epochs:
                            end_epoch = min(end_epoch, n_epochs)
                            epoch_labels[start_epoch:end_epoch] = stage_label
                
                # Filter out epochs with excluded labels
                valid_mask = epoch_labels >= 0
                if np.sum(valid_mask) > 0:
                    valid_epochs = epochs_data[valid_mask]
                    valid_labels = epoch_labels[valid_mask]
                    
                    # Normalize each channel within this file
                    for i in range(valid_epochs.shape[1]):
                        scaler = StandardScaler()
                        channel_data = valid_epochs[:, i, :]
                        valid_epochs[:, i, :] = scaler.fit_transform(channel_data)
                    
                    # Save to disk
                    np.save(output_file, valid_epochs)
                    labels_file = output_file.replace('_epochs.npy', '_labels.npy')
                    np.save(labels_file, valid_labels)
                    
                    # Add to processed files list
                    processed_files.append((psg_file, output_file, valid_labels))
                    processed_file_count += 1
                    
                    # Free up memory
                    del valid_epochs, valid_labels
                    gc.collect()
                    
                else:
                    print(f"No valid sleep stages found in {psg_file}")
            else:
                print(f"No valid annotations found in {hypno_file}")
                
            # Free up memory after each file
            del raw
            gc.collect()
                
        except Exception as e:
            print(f"Error processing {psg_file}: {str(e)}")
    
    print(f"Successfully preprocessed {processed_file_count} files")
    return processed_files

# Perform STFT on each channel and flatten frequency bins into channels
def stft_transform(x):
    # x: numpy array [channels, samples]
    f, t, Zxx = signal.stft(x, fs=100, nperseg=256, noverlap=128, axis=-1, boundary=None)
    mag = np.abs(Zxx)  # [channels, freq_bins, time_bins]
    c, fb, tb = mag.shape
    return mag.reshape(c*fb, tb)

def create_dataloaders(processed_files, data_dir, batch_size=32, num_workers=4, transform=None):
    """
    Create data loaders from preprocessed files
    
    Args:
        processed_files: List of tuples (file_path, data_path, labels)
        data_dir: Root directory for data
        batch_size: Batch size
        num_workers: Number of workers for data loading
    """
    # Create datasets
    dataset = SleepDataset(processed_files, data_dir, transform=transform)
    
    # Get all labels
    all_labels = np.concatenate([labels for _, _, labels in processed_files])
    
    # Split dataset into train and validation sets
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)), 
        test_size=0.2, 
        random_state=42,
        stratify=all_labels
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
    
    return train_loader, val_loader, all_labels

def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Configuration
    data_dir = "dataset"
    record_file = "dataset/RECORDS"
    batch_size = 64  # Larger batch size with memory-mapped data
    latent_dim = 64
    hidden_dim = 128
    num_classes = 5
    ae_epochs = 30
    cls_epochs = 30
    limit_files = None  # Set to a number to limit files for testing, None to use all
    output_dir = "preprocessed_data"  # Directory for preprocessed data
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Step 1: Preprocess the data and save to disk
    print("Preprocessing data...")
    processed_files = preprocess_sleep_data(
        data_dir, 
        record_file, 
        limit=limit_files,
        output_dir=output_dir
    )
    
    if not processed_files:
        print("No files were successfully processed. Exiting.")
        return
    
    # Step 2: Create data loaders from preprocessed files
    print("Creating data loaders...")
    # Use raw signals without STFT transform
    train_loader, val_loader, all_labels = create_dataloaders(
        processed_files, 
        data_dir,
        batch_size=batch_size,
        transform=None  # No transform - use raw signals
    )
    
    # Print dataset statistics
    label_counts = np.bincount(all_labels.astype(int))
    print(f"Dataset contains {len(all_labels)} epochs with class distribution: {label_counts}")
    
    # Get data shape from first batch
    sample_batch = next(iter(train_loader))
    data_shape = sample_batch[0].shape
    input_channels = data_shape[1]
    sequence_length = data_shape[2]
    print(f"Data shape: [batch_size={data_shape[0]}, channels={input_channels}, sequence_length={sequence_length}]")
    
    # Step 3: Train or load ConvAutoencoder
    autoencoder_path = "autoencoder.pt"
    autoencoder = ConvAutoencoder(
        input_channels=input_channels,
        sequence_length=sequence_length,
        latent_dim=latent_dim
    ).to(device)
    if os.path.exists(autoencoder_path):
        print(f"Loading pretrained autoencoder from {autoencoder_path}")
        try:
            autoencoder.load_state_dict(torch.load(autoencoder_path))
        except RuntimeError as e:
            print("Warning: pretrained autoencoder shape mismatch, retraining:", e)
            autoencoder = train_autoencoder(
                autoencoder,
                train_loader,
                val_loader,
                device,
                epochs=ae_epochs,
                save_path=autoencoder_path
            )
    else:
        print("Training autoencoder...")
        autoencoder = train_autoencoder(
            autoencoder,
            train_loader,
            val_loader,
            device,
            epochs=ae_epochs,
            save_path=autoencoder_path
        )
    
    # Step 4: Encode dataset to latent space
    print("Encoding dataset to latent space...")
    latent_train_data, train_labels = encode_dataset(autoencoder, train_loader, device)
    latent_val_data, val_labels = encode_dataset(autoencoder, val_loader, device)
    
    print(f"Latent train data shape: {latent_train_data.shape}")
    print(f"Latent validation data shape: {latent_val_data.shape}")
    
    # Convert to PyTorch TensorDataset
    tensor_train_x = torch.Tensor(latent_train_data)
    tensor_train_y = torch.LongTensor(train_labels)
    tensor_val_x = torch.Tensor(latent_val_data)
    tensor_val_y = torch.LongTensor(val_labels)
    
    train_dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)
    val_dataset = torch.utils.data.TensorDataset(tensor_val_x, tensor_val_y)
    
    # Create data loaders for encoded data
    latent_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    latent_val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Step 5: Train U-Net+LSTM classifier
    classifier_path = "classifier.pt"
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(train_labels.astype(int))
    class_weights = 1.0 / class_counts
    class_weights = class_weights / np.sum(class_weights) * len(class_counts)
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    # Initialize U-Net+LSTM classifier
    classifier = UNetLSTM(
        input_channels=latent_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=0.5,
        bidirectional=True
    )
    
    if os.path.exists(classifier_path):
        print(f"Loading pretrained classifier from {classifier_path}")
        classifier.load_state_dict(torch.load(classifier_path))
    else:
        print("Training U-Net+LSTM classifier...")
        classifier = train_classifier(
            classifier,
            latent_train_loader,
            latent_val_loader,
            device,
            epochs=cls_epochs,
            class_weights=class_weights,
            save_path=classifier_path
        )
    
    # Step 6: Evaluate model
    print("Evaluating model...")
    classifier.eval()
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(latent_val_loader, desc="Evaluating"):
            data = data.to(device)
            output = classifier(data)
            _, preds = torch.max(output, 1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    
    accuracy = accuracy_score(val_targets, val_preds)
    f1 = f1_score(val_targets, val_preds, average='macro')
    cm = confusion_matrix(val_targets, val_preds)
    
    # Reverse STAGE_MAP to get class names
    reverse_stage_map = {v: k for k, v in STAGE_MAP.items() if v >= 0}
    class_names = [reverse_stage_map[i] for i in range(num_classes)]
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(val_targets, val_preds, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    main()