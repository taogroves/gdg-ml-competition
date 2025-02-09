# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader

class MovieDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def create_data_loaders(features, target, batch_size=32, train_split=0.8):
    # Determine split index
    split_idx = int(len(features) * train_split)
    
    # Create random indices
    indices = torch.randperm(len(features))
    
    # Split into train and validation sets
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create datasets
    train_dataset = MovieDataset(
        features[train_indices], 
        target[train_indices]
    )
    val_dataset = MovieDataset(
        features[val_indices], 
        target[val_indices]
    )
    
    # Create dataloaders
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
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Import necessary functions to get data
    from prep_data import get_prepared_data
    
    # Test 1: Basic Dataset Creation and Inspection
    def test_dataset():
        # Get the preprocessed data
        features, target = get_prepared_data()
        
        # Create dataset instance
        dataset = MovieDataset(features, target)
        
        # Test length
        print(f"\nTest 1: Dataset Length")
        print(f"Dataset length: {len(dataset)}")
        
        # Test getting single item
        print(f"\nTest 2: Single Item Access")
        first_feature, first_target = dataset[0]
        print(f"First feature shape: {first_feature.shape}")
        print(f"First target shape: {first_target.shape}")
        print(f"First few feature values: {first_feature[:5]}")
        print(f"First target value: {first_target}")
        
        return dataset

    # Test 2: DataLoader Creation and Inspection
    def test_dataloaders():
        features, target = get_prepared_data()
        
        # Create dataloaders
        train_loader, val_loader = create_data_loaders(
            features, 
            target,
            batch_size=32,
            train_split=0.8
        )
        
        print(f"\nTest 3: DataLoader Inspection")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        
        # Get a batch from train_loader
        print(f"\nTest 4: Batch Inspection")
        first_batch_features, first_batch_targets = next(iter(train_loader))
        print(f"Batch feature shape: {first_batch_features.shape}")
        print(f"Batch target shape: {first_batch_targets.shape}")
        
        return train_loader, val_loader

    # Test 3: Data Distribution
    def test_data_distribution(train_loader, val_loader):
        print(f"\nTest 5: Data Distribution")
        
        # Count total samples in each loader
        train_samples = sum(len(features) for features, _ in train_loader)
        val_samples = sum(len(features) for features, _ in val_loader)
        
        print(f"Training samples: {train_samples}")
        print(f"Validation samples: {val_samples}")
        print(f"Split ratio: {train_samples/(train_samples + val_samples):.2f}")
    # Additional tests you might want to add:

    try:
        # Run all tests
        print("Starting Dataset Tests...")
        dataset = test_dataset()
        train_loader, val_loader = test_dataloaders()
        test_data_distribution(train_loader, val_loader)
        print("\nAll tests completed successfully! ✅")
        
    except Exception as e:
        print(f"Error during testing: {str(e)} ❌")