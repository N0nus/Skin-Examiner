import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random

# ========== Constants ==========
IMG_HEIGHT = 260
IMG_WIDTH = 260
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'skin_lesion_model_b2.pth'
CSV_PATH = "HAM10000_metadata.csv"  # Path to the metadata CSV file

# ========== Model Definition ==========
class EfficientNetWithMeta(nn.Module):
    def __init__(self, model_name='efficientnet-b2', num_classes=7, num_metadata_features=3, dropout_rate=0.3):
        super(EfficientNetWithMeta, self).__init__()
        # Load pretrained EfficientNet-B2
        self.efficientnet = EfficientNet.from_pretrained(model_name)
        
        # Get the number of features from the last layer
        eff_last_dim = self.efficientnet._fc.in_features
        
        # Replace the final fully connected layer with Identity
        self.efficientnet._fc = nn.Identity()
        
        # Create new classifier that includes metadata with improved architecture
        self.classifier = nn.Sequential(
            nn.Linear(eff_last_dim + num_metadata_features, 512),
            nn.BatchNorm1d(512),
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

# ========== Load Classes and Metadata Info ==========
def load_metadata_info(metadata_csv_path=CSV_PATH):
    print("Loading class and metadata information...")
    df = pd.read_csv(metadata_csv_path)
    
    # Load class names
    le = LabelEncoder()
    le.fit(df['dx'])
    class_names = le.classes_
    
    # Get localization info
    localization_encoder = LabelEncoder()
    localization_encoder.fit(df['localization'])
    num_localizations = len(localization_encoder.classes_)
    
    # Get median age for missing values
    median_age = df['age'].median()
    
    return class_names, localization_encoder, median_age, num_localizations, df

# ========== Prediction Function ==========
def predict_lesion(image_path, age, sex, localization, model, class_names, localization_encoder, num_localizations):
    # Updated transform to match validation transform from training
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT + 20, IMG_WIDTH + 20)),  # Resize slightly larger
        transforms.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),  # Center crop to target size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Process metadata
    if sex.lower() == 'male':
        sex_value = 1.0
    elif sex.lower() == 'female':
        sex_value = 0.0
    else:
        sex_value = 0.5
    
    # Normalize age
    age_value = float(age) / 100.0
    
    # Encode localization
    try:
        loc_encoded = localization_encoder.transform([localization])[0]
    except ValueError:
        print(f"Unknown localization: {localization}. Using default.")
        loc_encoded = 0
    
    loc_value = loc_encoded / num_localizations
    
    # Create metadata tensor
    metadata = torch.tensor([[age_value, sex_value, loc_value]], dtype=torch.float).to(DEVICE)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor, metadata)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get predicted class and probabilities
    pred_class_idx = torch.argmax(probabilities).item()
    pred_class = class_names[pred_class_idx]
    
    return pred_class, probabilities.cpu().numpy()

# ========== Main Function ==========

def main():
    print(f"Using device: {DEVICE}")
    
    # Get class names and metadata info
    class_names, localization_encoder, median_age, num_localizations, df = load_metadata_info()
    
    # Load model with correct architecture
    print("Loading EfficientNet-B2 model...")
    model = EfficientNetWithMeta(
        model_name='efficientnet-b2',
        num_classes=len(class_names)
    )
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)
        print(f"Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"Detected classes: {', '.join(class_names)}")
    
    # Get 10 random image paths from the dataset
    random_images = random.sample(list(df['image_id']), 10)  # Get 10 random image IDs
    
    for img_id in random_images:
        # Get image path and metadata for each random image
        image_path = f"HAM10_images/{img_id}.jpg"
        
        # Get metadata for the current image
        metadata = df[df['image_id'] == img_id].iloc[0]
        actual_class = metadata['dx']  # The actual class from the dataset
        age = metadata['age'] if not pd.isna(metadata['age']) else median_age
        sex = metadata['sex']
        localization = metadata['localization']
        
        print(f"\nTesting image: {img_id}")
        
        # Make prediction
        pred_class, probabilities = predict_lesion(
            image_path=image_path,
            age=age,
            sex=sex,
            localization=localization,
            model=model,
            class_names=class_names,
            localization_encoder=localization_encoder,
            num_localizations=num_localizations
        )
        
        # Print prediction results
        print(f"Actual class: {actual_class}")
        print(f"Predicted class: {pred_class}")
        print("Class probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"{class_names[i]}: {prob*100:.2f}%")
        
        # If predicted class is melanoma, give a warning
        if pred_class == 'mel':
            print("\n⚠️ WARNING: This lesion is predicted as MELANOMA.")
        
if __name__ == "__main__":
    main()
