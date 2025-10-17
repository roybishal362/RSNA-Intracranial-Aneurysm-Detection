import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import cv2
import pydicom
import nibabel as nib
from scipy import ndimage
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import shutil

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    DATA_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection"
    SERIES_PATH = f"{DATA_PATH}/series"
    SEGMENTATION_PATH = f"{DATA_PATH}/segmentations"
    LOCALIZER_PATH = f"{DATA_PATH}/train_localizers.csv"
    
    # Pretrained model path (from your uploaded dataset)
    PRETRAINED_MODEL_PATH = "/kaggle/input/efficientnet-b3-rwightman-b3899882-pth/efficientnet_b3_rwightman-b3899882.pth"
    
    # Stratified subset for full coverage
    USE_SUBSET = True
    SUBSET_SIZE = 800  # Larger subset for better coverage
    
    # Model settings
    IMAGE_SIZE = 256  # Larger for better detail
    BATCH_SIZE = 16
    NUM_EPOCHS = 15
    LR = 1e-4
    NUM_FOLDS = 5
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Labels
    LABEL_COLS = [
        'Left Infraclinoid Internal Carotid Artery',
        'Right Infraclinoid Internal Carotid Artery',
        'Left Supraclinoid Internal Carotid Artery',
        'Right Supraclinoid Internal Carotid Artery',
        'Left Middle Cerebral Artery',
        'Right Middle Cerebral Artery',
        'Anterior Communicating Artery',
        'Left Anterior Cerebral Artery',
        'Right Anterior Cerebral Artery',
        'Left Posterior Communicating Artery',
        'Right Posterior Communicating Artery',
        'Basilar Tip',
        'Other Posterior Circulation',
        'Aneurysm Present'
    ]
    
    LOCATION_COLS = LABEL_COLS[:-1]  # All except "Aneurysm Present"

config = Config()
print(f"Device: {config.DEVICE}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Check if pretrained model exists
if os.path.exists(config.PRETRAINED_MODEL_PATH):
    print(f"✓ Pretrained model found: {config.PRETRAINED_MODEL_PATH}")
else:
    print(f"⚠ Pretrained model not found at: {config.PRETRAINED_MODEL_PATH}")
    print("  Model will download from PyTorch (requires internet)")

# ============================================================================
# STRATIFIED DATA SAMPLING FOR FULL COVERAGE
# ============================================================================

print("\n" + "="*70)
print("PHASE 1: INTELLIGENT DATA SAMPLING")
print("="*70)

train_df = pd.read_csv(f"{config.DATA_PATH}/train.csv")
localizer_df = pd.read_csv(config.LOCALIZER_PATH)

print(f"Total samples: {len(train_df)}")
print(f"Samples with localizations: {len(localizer_df['SeriesInstanceUID'].unique())}")

def stratified_subset_selection(df, subset_size=800):
    """
    Create a BALANCED stratified subset with improved class distribution:
    - Take ALL aneurysm cases (100% positive samples)
    - Take 30-40% MORE negative cases than positives
    - This gives ~40-45% positive rate instead of 22%
    - Stratify negatives by modality for diversity
    """
    
    print("\nCreating BALANCED stratified subset...")
    
    # 1. Take ALL positive cases (aneurysm present)
    positive_cases = df[df['Aneurysm Present'] == 1].copy()
    n_positive = len(positive_cases)
    
    print(f"✓ Taking ALL {n_positive} aneurysm cases (100% of positives)")
    
    # Show location diversity in positive cases
    location_counts = positive_cases[config.LOCATION_COLS].sum()
    print(f"  Location distribution in aneurysm cases:")
    for loc, count in location_counts.items():
        if count > 0:
            print(f"    - {loc[:40]}: {count}")
    
    selected_samples = [positive_cases]
    
    # 2. Calculate negative samples (30-40% more than positives)
    # Using 35% more for good balance
    n_negative = int(n_positive * 1.35)  # 35% more negatives
    
    print(f"\n✓ Selecting {n_negative} negative cases (35% more than positives)")
    print(f"  This gives {n_positive/(n_positive+n_negative)*100:.1f}% positive rate")
    
    # 3. Stratify negative cases by modality for diversity
    negative_cases = df[df['Aneurysm Present'] == 0].copy()
    
    # Calculate samples per modality proportionally
    modality_distribution = negative_cases['Modality'].value_counts()
    negative_by_modality = []
    
    print(f"\n  Negative samples by modality:")
    for modality, count in modality_distribution.items():
        # Proportional sampling from each modality
        n_samples = int(n_negative * (count / len(negative_cases)))
        n_samples = min(n_samples, count)  # Don't exceed available
        
        if n_samples > 0:
            sampled = negative_cases[negative_cases['Modality'] == modality].sample(
                n=n_samples, random_state=42
            )
            negative_by_modality.append(sampled)
            print(f"    - {modality}: {n_samples} samples")
    
    negative_subset = pd.concat(negative_by_modality)
    selected_samples.append(negative_subset)
    
    # 4. Combine
    final_subset = pd.concat(selected_samples).drop_duplicates(subset='SeriesInstanceUID')
    
    # Calculate final statistics
    final_positive_rate = final_subset['Aneurysm Present'].mean()
    
    print(f"\n{'='*60}")
    print(f"FINAL BALANCED SUBSET STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(final_subset)}")
    print(f"Aneurysm cases: {final_subset['Aneurysm Present'].sum()} ({final_positive_rate*100:.1f}%)")
    print(f"No aneurysm: {(final_subset['Aneurysm Present']==0).sum()} ({(1-final_positive_rate)*100:.1f}%)")
    print(f"\nModality distribution:")
    print(final_subset['Modality'].value_counts())
    print(f"\nAge range: {final_subset['PatientAge'].min():.0f}-{final_subset['PatientAge'].max():.0f}")
    print(f"Gender: Female={((final_subset['PatientSex']=='Female').sum())}, Male={((final_subset['PatientSex']=='Male').sum())}")
    
    return final_subset

if config.USE_SUBSET:
    train_df = stratified_subset_selection(train_df, config.SUBSET_SIZE)

# ============================================================================
# COMPREHENSIVE EDA WITH MEDICAL VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: EXPLORATORY DATA ANALYSIS")
print("="*70)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# 1. Aneurysm prevalence
ax1 = fig.add_subplot(gs[0, 0])
prevalence = train_df['Aneurysm Present'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax1.pie(prevalence, labels=['No Aneurysm', 'Aneurysm'], autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax1.set_title('Aneurysm Prevalence', fontsize=12, fontweight='bold')

# 2. Age distribution by aneurysm status
ax2 = fig.add_subplot(gs[0, 1])
no_aneurysm = train_df[train_df['Aneurysm Present'] == 0]['PatientAge']
with_aneurysm = train_df[train_df['Aneurysm Present'] == 1]['PatientAge']
ax2.hist([no_aneurysm, with_aneurysm], bins=20, label=['No Aneurysm', 'Aneurysm'], 
         alpha=0.7, color=['#3498db', '#e74c3c'])
ax2.set_xlabel('Age')
ax2.set_ylabel('Count')
ax2.set_title('Age Distribution by Status', fontsize=12, fontweight='bold')
ax2.legend()

# 3. Modality distribution
ax3 = fig.add_subplot(gs[0, 2])
modality_counts = train_df['Modality'].value_counts()
ax3.barh(modality_counts.index, modality_counts.values, color='#9b59b6')
ax3.set_xlabel('Count')
ax3.set_title('Imaging Modality Distribution', fontsize=12, fontweight='bold')

# 4. Gender vs Aneurysm
ax4 = fig.add_subplot(gs[0, 3])
gender_aneurysm = pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'])
gender_aneurysm.plot(kind='bar', ax=ax4, color=['#3498db', '#e74c3c'])
ax4.set_xlabel('Gender')
ax4.set_ylabel('Count')
ax4.set_title('Aneurysm by Gender', fontsize=12, fontweight='bold')
ax4.legend(['No Aneurysm', 'Aneurysm'])
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)

# 5. Location-wise prevalence
ax5 = fig.add_subplot(gs[1, :2])
location_counts = train_df[config.LOCATION_COLS].sum().sort_values(ascending=True)
ax5.barh(range(len(location_counts)), location_counts.values, color='#e67e22')
ax5.set_yticks(range(len(location_counts)))
ax5.set_yticklabels([col.replace('Internal Carotid Artery', 'ICA').replace(' Artery', '')[:30] 
                     for col in location_counts.index], fontsize=9)
ax5.set_xlabel('Number of Cases')
ax5.set_title('Aneurysm Location Distribution', fontsize=12, fontweight='bold')

# 6. Co-occurrence heatmap
ax6 = fig.add_subplot(gs[1, 2:])
location_corr = train_df[config.LOCATION_COLS].corr()
sns.heatmap(location_corr, annot=False, cmap='coolwarm', center=0, ax=ax6,
            xticklabels=[col[:10] for col in config.LOCATION_COLS],
            yticklabels=[col[:10] for col in config.LOCATION_COLS])
ax6.set_title('Location Co-occurrence Correlation', fontsize=12, fontweight='bold')

# 7. Age box plot
ax7 = fig.add_subplot(gs[2, 0])
age_data = [train_df[train_df['Aneurysm Present']==0]['PatientAge'],
            train_df[train_df['Aneurysm Present']==1]['PatientAge']]
bp = ax7.boxplot(age_data, labels=['No Aneurysm', 'Aneurysm'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
    patch.set_facecolor(color)
ax7.set_ylabel('Age')
ax7.set_title('Age Distribution by Status', fontsize=12, fontweight='bold')

# 8. Modality vs Aneurysm
ax8 = fig.add_subplot(gs[2, 1])
modality_aneurysm = pd.crosstab(train_df['Modality'], train_df['Aneurysm Present'], normalize='index')
modality_aneurysm.plot(kind='bar', stacked=True, ax=ax8, color=['#3498db', '#e74c3c'])
ax8.set_xlabel('Modality')
ax8.set_ylabel('Proportion')
ax8.set_title('Aneurysm Rate by Modality', fontsize=12, fontweight='bold')
ax8.legend(['No Aneurysm', 'Aneurysm'])
ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45, ha='right')

# 9. Summary statistics
ax9 = fig.add_subplot(gs[2, 2:])
ax9.axis('off')
summary_text = f"""
DATASET SUMMARY
{'='*40}
Total Samples: {len(train_df)}
Aneurysm Cases: {train_df['Aneurysm Present'].sum()} ({train_df['Aneurysm Present'].mean()*100:.1f}%)
No Aneurysm: {(train_df['Aneurysm Present']==0).sum()} ({(train_df['Aneurysm Present']==0).mean()*100:.1f}%)

Age Range: {train_df['PatientAge'].min():.0f} - {train_df['PatientAge'].max():.0f}
Mean Age: {train_df['PatientAge'].mean():.1f} ± {train_df['PatientAge'].std():.1f}

Gender Distribution:
  Female: {(train_df['PatientSex']=='Female').sum()} ({(train_df['PatientSex']=='Female').mean()*100:.1f}%)
  Male: {(train_df['PatientSex']=='Male').sum()} ({(train_df['PatientSex']=='Male').mean()*100:.1f}%)

Most Common Locations:
  {location_counts.index[-1][:30]}: {location_counts.values[-1]}
  {location_counts.index[-2][:30]}: {location_counts.values[-2]}
  {location_counts.index[-3][:30]}: {location_counts.values[-3]}
"""
ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Comprehensive Medical Imaging Dataset Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.show()

# ============================================================================
# MEDICAL IMAGE VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("PHASE 3: MEDICAL IMAGE VISUALIZATION")
print("="*70)

def visualize_dicom_series(series_path, title="", num_slices=6):
    """Visualize DICOM series with proper medical imaging display"""
    try:
        dcm_files = sorted([f for f in os.listdir(series_path) if f.endswith('.dcm')])
        if not dcm_files:
            return None
        
        # Select evenly spaced slices
        indices = np.linspace(0, len(dcm_files)-1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes):
            if idx < len(indices):
                dcm_path = os.path.join(series_path, dcm_files[indices[idx]])
                ds = pydicom.dcmread(dcm_path, force=True)
                img = ds.pixel_array.astype(np.float32)
                
                # Handle multi-dimensional
                while len(img.shape) > 2:
                    img = img[0]
                
                # Apply HU scaling
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                
                # Display with medical colormap
                ax.imshow(img, cmap='bone', aspect='auto')
                ax.set_title(f'Slice {indices[idx]+1}/{len(dcm_files)}', fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return True
    except Exception as e:
        print(f"Error visualizing: {e}")
        return None

# Visualize samples with and without aneurysms
print("\n1. Normal Brain Scan (No Aneurysm)")
normal_samples = train_df[train_df['Aneurysm Present'] == 0].sample(1)
for _, row in normal_samples.iterrows():
    series_path = os.path.join(config.SERIES_PATH, row['SeriesInstanceUID'])
    visualize_dicom_series(series_path, 
                          f"Normal Scan | {row['Modality']} | Age: {row['PatientAge']} | Sex: {row['PatientSex']}")

print("\n2. Aneurysm Case")
aneurysm_samples = train_df[train_df['Aneurysm Present'] == 1].sample(1)
for _, row in aneurysm_samples.iterrows():
    series_path = os.path.join(config.SERIES_PATH, row['SeriesInstanceUID'])
    locations = [col for col in config.LOCATION_COLS if row[col] == 1]
    loc_str = ", ".join([loc.replace('Internal Carotid Artery', 'ICA')[:20] for loc in locations])
    visualize_dicom_series(series_path,
                          f"Aneurysm Case | {row['Modality']} | Locations: {loc_str}")

# ============================================================================
# ADVANCED DATA LOADING
# ============================================================================

def load_dicom_enhanced(series_path: str, target_slices=3) -> np.ndarray:
    """Enhanced DICOM loading with medical imaging best practices"""
    try:
        dcm_files = sorted([f for f in os.listdir(series_path) if f.endswith('.dcm')])
        
        if not dcm_files:
            return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.float32)
        
        # Load multiple representative slices
        n_files = len(dcm_files)
        slice_indices = [
            n_files // 4,      # First quarter
            n_files // 2,      # Middle
            3 * n_files // 4   # Third quarter
        ]
        
        slices = []
        for idx in slice_indices:
            idx = min(idx, n_files - 1)
            dcm_path = os.path.join(series_path, dcm_files[idx])
            
            ds = pydicom.dcmread(dcm_path, force=True)
            img = ds.pixel_array.astype(np.float32)
            
            # Handle dimensions
            while len(img.shape) > 2:
                img = img[0]
            
            # Validate shape
            if img.shape[0] == 0 or img.shape[1] == 0:
                img = np.zeros((512, 512), dtype=np.float32)
            
            # Apply HU windowing (brain window: 40 ± 80)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                # Brain window
                img = np.clip(img, 0, 80)
            
            # Normalize
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Resize
            if img.shape[0] > 0 and img.shape[1] > 0:
                img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
            else:
                img = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
            
            slices.append(img)
        
        # Stack as RGB-like
        return np.stack(slices, axis=-1).astype(np.float32)
        
    except Exception as e:
        return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.float32)

# ============================================================================
# DATASET
# ============================================================================

class MedicalDataset(Dataset):
    def __init__(self, dataframe, series_dir, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.series_dir = series_dir
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_id = row['SeriesInstanceUID']
        series_path = os.path.join(self.series_dir, series_id)
        
        # Load image
        img = load_dicom_enhanced(series_path)
        
        # Validate shape
        if len(img.shape) != 3 or img.shape[-1] != 3:
            img = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.float32)
        
        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-10, 10)
                img = ndimage.rotate(img, angle, axes=(0, 1), reshape=False, mode='constant')
        
        # To tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Labels
        labels = row[config.LABEL_COLS].values.astype(np.float32)
        labels = torch.from_numpy(labels)
        
        return img, labels

# ============================================================================
# COMPETITION-READY ARCHITECTURE
# ============================================================================

class AneurysmDetectionModel(nn.Module):
    """Production-grade model with attention and multi-scale features"""
    
    def __init__(self, num_classes=14, pretrained_path=None):
        super().__init__()
        
        # Backbone: EfficientNet-B3 (better than ResNet50 for medical imaging)
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
        
        # Load from local dataset if path provided, otherwise download
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from: {pretrained_path}")
            self.backbone = efficientnet_b3(weights=None)  # Initialize without weights
            
            # Load state dict
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict)
            print("Successfully loaded pretrained weights!")
        else:
            print("Loading pretrained weights from PyTorch (requires internet)")
            self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # Get feature dimension
        backbone_out = 1536  # EfficientNet-B3 output
        
        # Remove classifier
        self.backbone.classifier = nn.Identity()
        
        # Attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_out, backbone_out // 16),
            nn.ReLU(),
            nn.Linear(backbone_out // 16, backbone_out),
            nn.Sigmoid()
        )
        
        # Multi-task head
        self.location_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_out, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 13),  # 13 locations
            nn.Sigmoid()
        )
        
        self.main_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_out, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1),  # Main target
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        if len(features.shape) == 4:
            b, c, h, w = features.shape
            attn = self.channel_attention(features).view(b, c, 1, 1)
            features = features * attn
            features = F.adaptive_avg_pool2d(features, 1).view(b, -1)
        
        # Predictions
        location_preds = self.location_classifier(features)
        main_pred = self.main_classifier(features)
        
        # Concatenate
        return torch.cat([location_preds, main_pred], dim=1)

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

print("\n" + "="*70)
print("PHASE 4: MODEL TRAINING")
print("="*70)

# Create fold splits
skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)
train_df['fold'] = -1

for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df['Aneurysm Present'])):
    train_df.loc[train_df.index[val_idx], 'fold'] = fold

# Train first fold for demonstration
TRAIN_FOLD = 0
print(f"\nTraining Fold {TRAIN_FOLD + 1}/{config.NUM_FOLDS}")

train_data = train_df[train_df['fold'] != TRAIN_FOLD].reset_index(drop=True)
val_data = train_df[train_df['fold'] == TRAIN_FOLD].reset_index(drop=True)

print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Datasets
train_dataset = MedicalDataset(train_data, config.SERIES_PATH, augment=True)
val_dataset = MedicalDataset(val_data, config.SERIES_PATH, augment=False)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

# Model
model = AneurysmDetectionModel(
    num_classes=len(config.LABEL_COLS),
    pretrained_path=config.PRETRAINED_MODEL_PATH
)
model = model.to(config.DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Loss with BALANCED class weights for improved handling
pos_weights = []
for col in config.LABEL_COLS:
    pos = (train_data[col] == 1).sum()
    neg = (train_data[col] == 0).sum()
    
    if pos > 0:
        # Calculate balanced weight
        weight = neg / pos
        # Cap weight to prevent extreme values
        weight = min(weight, 10.0)
    else:
        weight = 1.0
    
    pos_weights.append(weight)

pos_weights = torch.FloatTensor(pos_weights).to(config.DEVICE)

print(f"\nClass weights (for handling remaining imbalance):")
for col, weight in zip(config.LABEL_COLS, pos_weights):
    if weight > 2.0:  # Only show significant weights
        print(f"  {col[:40]}: {weight:.2f}x")

criterion = nn.BCELoss()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

# Training
train_losses = []
val_losses = []
val_aucs = []
best_auc = 0

for epoch in range(config.NUM_EPOCHS):
    # Train
    model.train()
    train_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        
        # Weighted loss
        loss = F.binary_cross_entropy(outputs, labels, weight=pos_weights.unsqueeze(0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # Calculate AUC
    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    
    try:
        auc_main = roc_auc_score(labels[:, -1], preds[:, -1])
        val_aucs.append(auc_main)
        
        if auc_main > best_auc:
            best_auc = auc_main
            torch.save(model.state_dict(), 'best_model.pth')
    except:
        auc_main = 0.5
        val_aucs.append(auc_main)
    
    scheduler.step()
    
    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {auc_main:.4f}")

print(f"\nBest Validation AUC: {best_auc:.4f}")

# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

print("\n" + "="*70)
print("PHASE 5: COMPREHENSIVE EVALUATION")
print("="*70)

# Calculate per-label AUCs
label_aucs = []
for i, col in enumerate(config.LABEL_COLS):
    try:
        auc = roc_auc_score(labels[:, i], preds[:, i])
        label_aucs.append(auc)
    except:
        label_aucs.append(0.5)

# Competition metric
comp_metric = (label_aucs[-1] * 13 + sum(label_aucs[:-1])) / (13 + len(label_aucs) - 1)

print(f"\nPer-Label AUC Scores:")
print("="*50)
for col, auc in zip(config.LABEL_COLS, label_aucs):
    print(f"{col[:45]:45s} | AUC: {auc:.4f}")

print(f"\n{'='*50}")
print(f"Competition Metric (Weighted): {comp_metric:.4f}")
print(f"Mean Location AUC: {np.mean(label_aucs[:-1]):.4f}")
print(f"Main Target AUC: {label_aucs[-1]:.4f}")

# Visualization
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Training curves
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(train_losses, label='Train Loss', linewidth=2)
ax1.plot(val_losses, label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Curves', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. AUC progression
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(val_aucs, marker='o', linewidth=2, markersize=8, color='#e74c3c')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('AUC')
ax2.set_title('Validation AUC Progression', fontweight='bold')
ax2.axhline(y=best_auc, color='g', linestyle='--', label=f'Best: {best_auc:.4f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Per-label AUC
ax3 = fig.add_subplot(gs[0, 2])
colors = plt.cm.viridis(np.linspace(0, 1, len(label_aucs)))
bars = ax3.barh(range(len(label_aucs)), label_aucs, color=colors)
ax3.set_yticks(range(len(label_aucs)))
ax3.set_yticklabels([col[:25] for col in config.LABEL_COLS], fontsize=8)
ax3.set_xlabel('AUC Score')
ax3.set_title('Per-Label AUC Scores', fontweight='bold')
ax3.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
ax3.grid(True, alpha=0.3, axis='x')

# 4. ROC Curve
ax4 = fig.add_subplot(gs[1, 0])
fpr, tpr, _ = roc_curve(labels[:, -1], preds[:, -1])
ax4.plot(fpr, tpr, linewidth=3, label=f'AUC = {label_aucs[-1]:.3f}', color='#e74c3c')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curve - Aneurysm Present', fontweight='bold')
ax4.legend(fontsize=12)
ax4.grid(True, alpha=0.3)

# 5. Precision-Recall Curve
ax5 = fig.add_subplot(gs[1, 1])
from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, _ = precision_recall_curve(labels[:, -1], preds[:, -1])
avg_precision = average_precision_score(labels[:, -1], preds[:, -1])
ax5.plot(recall, precision, linewidth=3, label=f'AP = {avg_precision:.3f}', color='#3498db')
ax5.set_xlabel('Recall')
ax5.set_ylabel('Precision')
ax5.set_title('Precision-Recall Curve', fontweight='bold')
ax5.legend(fontsize=12)
ax5.grid(True, alpha=0.3)

# 6. Prediction Distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(preds[:, -1][labels[:, -1] == 0], bins=30, alpha=0.6, 
         label='No Aneurysm', color='#2ecc71', edgecolor='black')
ax6.hist(preds[:, -1][labels[:, -1] == 1], bins=30, alpha=0.6, 
         label='Aneurysm', color='#e74c3c', edgecolor='black')
ax6.set_xlabel('Prediction Score')
ax6.set_ylabel('Count')
ax6.set_title('Prediction Distribution', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Confusion Matrix
ax7 = fig.add_subplot(gs[2, 0])
y_pred_binary = (preds[:, -1] > 0.5).astype(int)
cm = confusion_matrix(labels[:, -1], y_pred_binary)
im = ax7.imshow(cm, cmap='Blues', aspect='auto')
ax7.set_xticks([0, 1])
ax7.set_yticks([0, 1])
ax7.set_xticklabels(['No Aneurysm', 'Aneurysm'])
ax7.set_yticklabels(['No Aneurysm', 'Aneurysm'])
ax7.set_xlabel('Predicted')
ax7.set_ylabel('True')
ax7.set_title('Confusion Matrix', fontweight='bold')

# Add counts
for i in range(2):
    for j in range(2):
        text = ax7.text(j, i, f'{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)',
                       ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                       fontsize=12, fontweight='bold')

# 8. Calibration curve
ax8 = fig.add_subplot(gs[2, 1])
from sklearn.calibration import calibration_curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    labels[:, -1], preds[:, -1], n_bins=10)
ax8.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=2, markersize=8, label='Model')
ax8.plot([0, 1], [0, 1], "k--", linewidth=2, label='Perfect Calibration')
ax8.set_xlabel('Mean Predicted Probability')
ax8.set_ylabel('Fraction of Positives')
ax8.set_title('Calibration Curve', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Location correlation heatmap
ax9 = fig.add_subplot(gs[2, 2])
location_corr = np.corrcoef(preds[:, :-1].T)
im = ax9.imshow(location_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax9.set_xticks(range(len(config.LOCATION_COLS)))
ax9.set_yticks(range(len(config.LOCATION_COLS)))
ax9.set_xticklabels([col[:10] for col in config.LOCATION_COLS], rotation=45, ha='right', fontsize=7)
ax9.set_yticklabels([col[:10] for col in config.LOCATION_COLS], fontsize=7)
ax9.set_title('Prediction Correlation Matrix', fontweight='bold')
plt.colorbar(im, ax=ax9)

plt.suptitle('Comprehensive Model Evaluation - Aneurysm Detection', 
             fontsize=16, fontweight='bold', y=0.995)
plt.show()

# ============================================================================
# PREDICTION VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("PHASE 6: PREDICTION VISUALIZATION")
print("="*70)

def visualize_predictions(model, dataset, num_samples=4):
    """Visualize model predictions on actual cases"""
    
    model.eval()
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 10))
    
    # Get samples (2 correct, 2 incorrect if possible)
    with torch.no_grad():
        for idx in range(num_samples):
            img, label = dataset[idx]
            img_tensor = img.unsqueeze(0).to(config.DEVICE)
            pred = model(img_tensor).cpu().numpy()[0]
            
            # Get true and predicted main label
            true_label = label[-1].item()
            pred_label = pred[-1]
            
            # Display image
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # Top row: images
            axes[0, idx].imshow(img_np)
            axes[0, idx].axis('off')
            
            color = 'green' if abs(true_label - pred_label) < 0.5 else 'red'
            axes[0, idx].set_title(f'True: {int(true_label)}, Pred: {pred_label:.2f}', 
                                  color=color, fontweight='bold', fontsize=12)
            
            # Bottom row: prediction bars
            pred_locations = pred[:-1]
            true_locations = label[:-1].numpy()
            
            x = np.arange(len(config.LOCATION_COLS))
            width = 0.35
            
            axes[1, idx].barh(x - width/2, true_locations, width, label='True', alpha=0.7)
            axes[1, idx].barh(x + width/2, pred_locations, width, label='Pred', alpha=0.7)
            
            axes[1, idx].set_yticks(x)
            axes[1, idx].set_yticklabels([col[:15] for col in config.LOCATION_COLS], fontsize=7)
            axes[1, idx].set_xlabel('Probability')
            axes[1, idx].set_xlim([0, 1])
            axes[1, idx].grid(True, alpha=0.3, axis='x')
            
            if idx == 0:
                axes[1, idx].legend(loc='lower right')
    
    plt.suptitle('Model Predictions on Validation Cases', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

visualize_predictions(model, val_dataset, num_samples=4)

# ============================================================================
# KAGGLE SUBMISSION SETUP
# ============================================================================

print("\n" + "="*70)
print("PHASE 7: KAGGLE SUBMISSION PREPARATION")
print("="*70)

# Save model
torch.save(model.state_dict(), 'aneurysm_detection_model.pth')
print("Model saved: aneurysm_detection_model.pth")

# Create inference model
inference_model = model
inference_model.eval()

def predict(series_path: str):
    """Production inference function for Kaggle evaluation"""
    import polars as pl
    
    series_id = os.path.basename(series_path)
    
    try:
        # Load image
        img = load_dicom_enhanced(series_path)
        
        # Validate shape
        if len(img.shape) != 3 or img.shape[-1] != 3:
            img = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.float32)
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        img = img.to(config.DEVICE)
        
        # Predict
        with torch.no_grad():
            pred = inference_model(img).cpu().numpy()[0]
        
        # Clip to valid range
        pred = np.clip(pred, 0.001, 0.999)
        
        # Create result
        result_df = pl.DataFrame(
            data=[pred.tolist()],
            schema=config.LABEL_COLS,
            orient='row'
        )
        
    except Exception as e:
        print(f"Error predicting {series_id}: {e}")
        # Smart fallback based on prevalence
        fallback = [0.1] * 13 + [0.4]  # Low prob for locations, moderate for main
        result_df = pl.DataFrame(
            data=[fallback],
            schema=config.LABEL_COLS,
            orient='row'
        )
    
    # Cleanup
    if os.path.exists('/kaggle/shared'):
        shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    return result_df

# Setup inference server
try:
    import kaggle_evaluation.rsna_inference_server
    
    inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
    
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("Running in competition mode...")
        inference_server.serve()
    else:
        print("Running local gateway for testing...")
        inference_server.run_local_gateway()
        
        try:
            import polars as pl
            submission = pl.read_parquet('/kaggle/working/submission.parquet')
            print("\n✓ Submission created successfully!")
            print(f"Shape: {submission.shape}")
            print("\nSample predictions:")
            print(submission.head())
            
            # Validate submission
            print("\nSubmission Validation:")
            print(f"  - All values in [0, 1]: {submission.select(pl.all().is_between(0, 1)).to_numpy().all()}")
            print(f"  - No NaN values: {not submission.null_count().sum(axis=1)[0] > 0}")
            
        except Exception as e:
            print(f"Submission file not available: {e}")
            
except ImportError as e:
    print(f"Kaggle evaluation module not available: {e}")
    print("Running in development mode")
except Exception as e:
    print(f"Error during inference setup: {e}")
    print("This is expected without test data")

