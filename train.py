import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import time
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler

# ========== Constants ==========
BATCH_SIZE = 32
# Updated for EfficientNet-B2 (260x260 is recommended size)
IMG_HEIGHT = 260
IMG_WIDTH = 260
EPOCHS = 10
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
LR = 1e-4
SAVE_PATH = 'skin_lesion_model_b2.pth'
EARLY_STOPPING_PATIENCE = 3  # Number of epochs to wait for improvement

# ========== Check GPU Availability ==========
#print(f"Using device: {DEVICE}")
#if torch.cuda.is_available():
#    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ========== Load & Prepare Data ==========
try:
    df = pd.read_csv("HAM10000_metadata.csv")
    #print(f"Dataset loaded with {len(df)} samples")
    
    # Create a label encoder for the diagnosis
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['dx'])
    class_names = le.classes_
    num_classes = len(class_names)
    
    # Update image path
    df['image_path'] = df['image_id'].apply(lambda x: f"HAM10_images/{x}.jpg")
    
    # Process metadata - Encode categorical variables
    df['sex'] = df['sex'].map({'male': 1, 'female': 0, 'unknown': 0.5})
    
    # Fill missing values
    df['age'] = df['age'].fillna(df['age'].median())
    
    # Encode localization
    localization_encoder = LabelEncoder()
    df['localization_encoded'] = localization_encoder.fit_transform(df['localization'])
    num_localizations = len(localization_encoder.classes_)
    
    # Split data with stratification
    train_df, testval_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(testval_df, test_size=0.50, stratify=testval_df['label'], random_state=RANDOM_STATE)
    
    #print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# ========== Custom Dataset Class ==========
class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (600, 450), (0, 0, 0))
        
        # Get metadata features
        age = self.dataframe.iloc[idx]['age'] / 100.0  # Simple normalization
        sex = self.dataframe.iloc[idx]['sex']
        loc_encoded = self.dataframe.iloc[idx]['localization_encoded'] / num_localizations  # Simple normalization
        
        # Get label
        label = self.dataframe.iloc[idx]['label']
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
            
        # Create metadata tensor
        metadata = torch.tensor([age, sex, loc_encoded], dtype=torch.float)
            
        return img, metadata, label

# ========== Data Augmentation & Preprocessing ==========
# Enhanced augmentations for better generalization and robustness to different image sizes
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT + 20, IMG_WIDTH + 20)),  # Resize slightly larger than target
    transforms.RandomCrop((IMG_HEIGHT, IMG_WIDTH)),  # Random crop to target size
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),  # Add rotation for more robustness
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# For validation, use a center crop to handle different image sizes consistently
valid_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT + 20, IMG_WIDTH + 20)),  # Resize slightly larger than target
    transforms.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),  # Center crop to target size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== Create DataLoaders ==========
train_dataset = SkinLesionDataset(train_df, transform=train_transform)
val_dataset = SkinLesionDataset(val_df, transform=valid_transform)
test_dataset = SkinLesionDataset(test_df, transform=valid_transform)

# Calculate sample weights (higher weight for minority classes)
class_sample_counts = train_df['label'].value_counts().sort_index().values
weights = 1. / class_sample_counts
samples_weights = weights[train_df['label'].values]

# Create sampler
sampler = WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(samples_weights),
    replacement=True  # Allows oversampling
)

# Modify train_loader to use sampler
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,  # Instead of shuffle=True
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False
)

# ========== Custom Model ==========
class EfficientNetWithMeta(nn.Module):
    def __init__(self, model_name='efficientnet-b2', num_classes=7, num_metadata_features=3, dropout_rate=0.3):
        super(EfficientNetWithMeta, self).__init__()
        # Load pretrained EfficientNet
        self.efficientnet = EfficientNet.from_pretrained(model_name)
        
        # Get the number of features from the last layer
        eff_last_dim = self.efficientnet._fc.in_features
        
        # Replace the final fully connected layer with Identity
        self.efficientnet._fc = nn.Identity()
        
        # Create new classifier that includes metadata with improved dropout
        self.classifier = nn.Sequential(
            nn.Linear(eff_last_dim + num_metadata_features, 512),
            nn.BatchNorm1d(512),  # Add batch normalization to help with overfitting
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, image, metadata):
        # Extract features from the image
        features = self.efficientnet(image)
        
        # Concatenate image features with metadata
        combined = torch.cat((features, metadata), dim=1)
        
        # Pass through classifier
        output = self.classifier(combined)
        
        return output

# ========== Training Function ==========
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Training")
    for i, (images, metadata, labels) in enumerate(progress_bar):
        images = images.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        batch_loss = loss.item()
        batch_acc = (predicted == labels).sum().item() / labels.size(0)
        progress_bar.set_postfix({
            'loss': f"{batch_loss:.4f}",
            'acc': f"{batch_acc:.4f}"
        })
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# ========== Validation Function ==========
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for images, metadata, labels in progress_bar:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            
            # Track statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            batch_loss = loss.item()
            batch_acc = (predicted == labels).sum().item() / labels.size(0)
            progress_bar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'acc': f"{batch_acc:.4f}"
            })
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# ========== Test Function ==========
def test_model(model, test_loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(len(class_names), len(class_names))
    
    with torch.no_grad():
        for images, metadata, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            
            # Track statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct / total
    
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_correct = confusion_matrix[i, i]
        class_total = confusion_matrix[i, :].sum()
        print(f"{class_name}: {class_correct/class_total:.4f}")
    
    return test_loss, test_acc, confusion_matrix

# ========== Training Loop with Early Stopping ==========
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs, patience=3):
    best_val_acc = 0.0
    best_epoch = -1
    
    start_time = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            print(f"Saving best model with accuracy: {best_val_acc:.4f}")
            torch.save(model.state_dict(), SAVE_PATH)
        
        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break
    
    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    
    return history

# ========== Function to load best model and evaluate on test set ==========
def evaluate_best_model(model, test_loader, criterion, device, class_names):
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    return test_model(model, test_loader, criterion, device, class_names)

# ========== Main Execution ==========
if __name__ == "__main__":
    print("Starting training process...")

    # Initialize model with B2 instead of B0
    model = EfficientNetWithMeta(model_name='efficientnet-b2', num_classes=num_classes)
    model = model.to(DEVICE)

    # ========== Loss & Optimizer ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)  # Added weight decay to combat overfitting
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)  # Learning rate scheduler
    
    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))

    # Train the model with early stopping
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=DEVICE,
        num_epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE
    )
    
    # Evaluate best model on test set
    test_loss, test_acc, conf_matrix = evaluate_best_model(model, test_loader, criterion, DEVICE, class_names)
    
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Model saved to {SAVE_PATH}")
    print("Training and evaluation complete!")