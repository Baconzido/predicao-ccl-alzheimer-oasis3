#!/usr/bin/env python3
"""
CNN 3D (5 blocos) com dataset cacheado em RAM
TCC - Samuel Augusto Souza Alves Santana

Variante que pré-carrega todas as imagens na RAM antes do treino
(mais rápida quando a máquina tem RAM suficiente para o dataset
inteiro) e usa saída em logits + BCEWithLogitsLoss para maior
estabilidade numérica. NUM_THREADS e NUM_WORKERS abaixo são
parâmetros de hardware/SO: ajuste para os núcleos/threads da sua
máquina. NUM_WORKERS=0 é o valor seguro no Windows (workers>0 trava
por causa do multiprocessing); em Linux/macOS costuma ser seguro
usar 2-4.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÕES DE HARDWARE/SO (ajuste para sua máquina)
# ============================================

# Threads para CPU - ajuste para o número de núcleos disponíveis
torch.set_num_threads(12)

# Configurações
BATCH_SIZE = 4
NUM_WORKERS = 0  # 0 = seguro no Windows; em Linux/macOS pode subir para 2-4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 5e-4

PREPROCESSED_DIR = "./oasis3_preprocessed"
LABELS_FILE = "mci_subjects_for_download.csv"
DEVICE = torch.device('cpu')


# ============================================
# DATASET COM CACHE
# ============================================

class MRIDataset(Dataset):
    def __init__(self, subject_ids, labels, data_dir, augment=False):
        self.subject_ids = subject_ids
        self.labels = labels
        self.data_dir = data_dir
        self.augment = augment
        
        # Cache em RAM
        self.cache = {}
        print("  Carregando imagens na RAM...", end=" ", flush=True)
        for sid in subject_ids:
            filepath = os.path.join(data_dir, f"{sid}.npy")
            self.cache[sid] = np.load(filepath).astype(np.float32)
        print(f"✓ ({len(self.cache)} imagens)")
    
    def __len__(self):
        return len(self.subject_ids)
    
    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        label = self.labels[idx]
        
        image = self.cache[subject_id].copy()
        image = np.expand_dims(image, axis=0)
        
        if self.augment:
            image = self._augment(image)
        
        # Garantir range [0, 1]
        image = np.clip(image, 0, 1)
        
        return torch.FloatTensor(image), torch.FloatTensor([label])
    
    def _augment(self, image):
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=3).copy()
        
        if np.random.rand() > 0.7:
            image = np.flip(image, axis=1).copy()
        
        if np.random.rand() > 0.6:
            noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
            image = image + noise
        
        if np.random.rand() > 0.6:
            image = image * np.random.uniform(0.9, 1.1)
        
        image = np.clip(image, 0, 1)
        return image.astype(np.float32)


# ============================================
# CNN 3D
# ============================================

class CNN3D(nn.Module):
    def __init__(self, dropout=0.5):
        super(CNN3D, self).__init__()
        
        self.features = nn.Sequential(
            # Bloco 1: 128 -> 64
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Dropout3d(0.1),
            
            # Bloco 2: 64 -> 32
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Dropout3d(0.1),
            
            # Bloco 3: 32 -> 16
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Dropout3d(0.15),
            
            # Bloco 4: 16 -> 8
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Dropout3d(0.2),
            
            # Bloco 5: 8 -> 4
            nn.Conv3d(256, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),
            nn.Linear(128, 1)
            # SEM Sigmoid aqui - usar BCEWithLogitsLoss
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================
# TREINAMENTO
# ============================================

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(outputs).detach().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.numpy())
    
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_labels)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.sigmoid(outputs).numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
    
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_labels)


def train_fold(model, train_loader, val_loader, fold_num):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    
    best_auc = 0
    patience_counter = 0
    best_state = None
    
    pbar = tqdm(range(NUM_EPOCHS), desc=f"Fold {fold_num}", ncols=80)
    
    for epoch in pbar:
        train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion)
        
        try:
            val_auc = roc_auc_score(val_labels.flatten(), val_preds.flatten())
        except:
            val_auc = 0.5
        
        scheduler.step()
        
        pbar.set_postfix({'loss': f'{train_loss:.3f}', 'auc': f'{val_auc:.3f}'})
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_auc


# ============================================
# VALIDAÇÃO CRUZADA
# ============================================

def run_cv(subject_ids, labels, data_dir):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    all_preds = []
    all_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(subject_ids, labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/5")
        print(f"{'='*60}")
        
        train_subj = [subject_ids[i] for i in train_idx]
        train_lab = [labels[i] for i in train_idx]
        val_subj = [subject_ids[i] for i in val_idx]
        val_lab = [labels[i] for i in val_idx]
        
        print(f"  Train: {len(train_subj)} (pMCI: {sum(train_lab)})")
        print(f"  Val:   {len(val_subj)} (pMCI: {sum(val_lab)})")
        
        train_ds = MRIDataset(train_subj, train_lab, data_dir, augment=True)
        val_ds = MRIDataset(val_subj, val_lab, data_dir, augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = CNN3D(dropout=0.5)
        
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parâmetros: {n_params:,}")
        
        model, _ = train_fold(model, train_loader, val_loader, fold+1)
        
        _, preds, labs = evaluate(model, val_loader, nn.BCEWithLogitsLoss())
        
        all_preds.extend(preds.flatten())
        all_labels.extend(labs.flatten())
        
        preds_bin = (preds.flatten() > 0.5).astype(int)
        labs_flat = labs.flatten().astype(int)
        
        auc = roc_auc_score(labs_flat, preds.flatten())
        acc = accuracy_score(labs_flat, preds_bin)
        
        cm = confusion_matrix(labs_flat, preds_bin)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
        sens = tp/(tp+fn) if (tp+fn) > 0 else 0
        spec = tn/(tn+fp) if (tn+fp) > 0 else 0
        
        results.append({'fold': fold+1, 'auc': auc, 'accuracy': acc, 
                       'sensitivity': sens, 'specificity': spec})
        
        print(f"\n  ✓ Fold {fold+1}: AUC={auc:.4f}, Acc={acc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")
        
        del model, train_ds, val_ds
    
    return results, np.array(all_preds), np.array(all_labels)


# ============================================
# MAIN
# ============================================

def main():
    print("="*60)
    print("CNN 3D - i5-10400 + 16GB RAM")
    print("="*60)
    print(f"Threads: {torch.get_num_threads()}")
    print(f"Batch: {BATCH_SIZE}, Épocas: {NUM_EPOCHS}")
    
    if not os.path.exists(PREPROCESSED_DIR):
        print(f"\nERRO: Execute primeiro: python preprocessar_mri.py")
        return
    
    import csv
    subject_ids, labels = [], []
    
    with open(LABELS_FILE, 'r') as f:
        for row in csv.DictReader(f):
            sid = row['OASISID']
            if os.path.exists(os.path.join(PREPROCESSED_DIR, f"{sid}.npy")):
                subject_ids.append(sid)
                labels.append(int(row['label']))
    
    print(f"\nSujeitos: {len(subject_ids)} (pMCI: {sum(labels)}, sMCI: {len(labels)-sum(labels)})")
    
    results, all_preds, all_labels = run_cv(subject_ids, labels, PREPROCESSED_DIR)
    
    # Resultados
    print("\n" + "="*60)
    print("RESULTADOS FINAIS")
    print("="*60)
    
    aucs = [r['auc'] for r in results]
    accs = [r['accuracy'] for r in results]
    sens = [r['sensitivity'] for r in results]
    specs = [r['specificity'] for r in results]
    global_auc = roc_auc_score(all_labels, all_preds)
    
    print(f"\nAUC:         {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Accuracy:    {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Sensitivity: {np.mean(sens):.4f} ± {np.std(sens):.4f}")
    print(f"Specificity: {np.mean(specs):.4f} ± {np.std(specs):.4f}")
    print(f"\nAUC Global:  {global_auc:.4f}")
    
    # Salvar
    with open('cnn3d_final_results.json', 'w') as f:
        json.dump({
            'n_subjects': len(subject_ids),
            'results': results,
            'summary': {
                'auc_mean': float(np.mean(aucs)),
                'auc_std': float(np.std(aucs)),
                'auc_global': float(global_auc),
                'accuracy_mean': float(np.mean(accs)),
                'sensitivity_mean': float(np.mean(sens)),
                'specificity_mean': float(np.mean(specs))
            }
        }, f, indent=2)
    
    # Gráfico ROC
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'CNN 3D (AUC = {global_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Curva ROC - CNN 3D')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('cnn3d_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Salvos: cnn3d_final_results.json, cnn3d_roc_curve.png")


if __name__ == "__main__":
    main()