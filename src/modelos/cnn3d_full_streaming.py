#!/usr/bin/env python3
"""
CNN 3D (5 blocos) com leitura sob demanda do disco
TCC - Samuel Augusto Souza Alves Santana

Variante que carrega cada imagem do disco a cada acesso (não cacheia
em RAM) e salva um checkpoint por fold durante a validação cruzada.
Usa batch menor e learning rate mais conservador que a variante
cacheada (src/modelos/cnn3d_full_cached.py) - útil quando o dataset
não cabe inteiro na RAM disponível.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÕES OTIMIZADAS PARA CPU
# ============================================

BATCH_SIZE = 2          # Menor para não estourar memória
LEARNING_RATE = 5e-5    # Menor para convergência mais estável
NUM_EPOCHS = 100        # Mais épocas para melhor aprendizado
EARLY_STOPPING_PATIENCE = 20  # Mais paciência
WEIGHT_DECAY = 1e-3     # Mais regularização

# Diretórios
PREPROCESSED_DIR = "./oasis3_preprocessed"
LABELS_FILE = "mci_subjects_for_download.csv"

# Device
DEVICE = torch.device('cpu')
print(f"Usando: {DEVICE} (otimizado para CPU)")

# ============================================
# DATASET COM AUGMENTATION MELHORADO
# ============================================

class MRIDataset(Dataset):
    """Dataset para imagens MRI 3D com augmentation robusto"""
    
    def __init__(self, subject_ids, labels, data_dir, augment=False):
        self.subject_ids = subject_ids
        self.labels = labels
        self.data_dir = data_dir
        self.augment = augment
    
    def __len__(self):
        return len(self.subject_ids)
    
    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        label = self.labels[idx]
        
        filepath = os.path.join(self.data_dir, f"{subject_id}.npy")
        image = np.load(filepath)
        image = np.expand_dims(image, axis=0)
        
        if self.augment:
            image = self._augment(image)
        
        image = torch.FloatTensor(image)
        label = torch.FloatTensor([label])
        
        return image, label
    
    def _augment(self, image):
        """Augmentation mais agressivo para aumentar diversidade"""
        
        # Flip horizontal (50% chance)
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=3).copy()
        
        # Flip sagital (30% chance)
        if np.random.rand() > 0.7:
            image = np.flip(image, axis=1).copy()
        
        # Ruído gaussiano (40% chance)
        if np.random.rand() > 0.6:
            noise_level = np.random.uniform(0.005, 0.02)
            noise = np.random.normal(0, noise_level, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        # Variação de intensidade (40% chance)
        if np.random.rand() > 0.6:
            factor = np.random.uniform(0.9, 1.1)
            image = image * factor
            image = np.clip(image, 0, 1)
        
        # Pequena rotação via shift (30% chance)
        if np.random.rand() > 0.7:
            shift = np.random.randint(-3, 4)
            image = np.roll(image, shift, axis=2)
        
        return image.astype(np.float32)


# ============================================
# ARQUITETURA CNN 3D COMPLETA
# ============================================

class CNN3D_Full(nn.Module):
    """
    CNN 3D Completa - mais camadas e capacidade
    Input: (batch, 1, 128, 128, 128)
    """
    
    def __init__(self, dropout=0.5):
        super(CNN3D_Full, self).__init__()
        
        # Bloco 1: 128 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(0.1)
        )
        
        # Bloco 2: 64 -> 32
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(0.1)
        )
        
        # Bloco 3: 32 -> 16
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(0.2)
        )
        
        # Bloco 4: 16 -> 8
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(0.2)
        )
        
        # Bloco 5: 8 -> 4
        self.conv5 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(0.3)
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ============================================
# TREINAMENTO
# ============================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping para estabilidade
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, num_epochs, device, fold_num):
    """Pipeline de treinamento com class weights"""
    
    # Calcular class weights
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy().flatten())
    
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Usar BCELoss normal já que temos Sigmoid no modelo
    criterion = nn.BCELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_auc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    pbar = tqdm(range(num_epochs), desc=f"Fold {fold_num}", leave=False)
    
    for epoch in pbar:
        train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        # Calcular AUC
        try:
            val_auc = roc_auc_score(val_labels.flatten(), val_preds.flatten())
        except:
            val_auc = 0.5
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        pbar.set_postfix({'loss': f'{train_loss:.3f}', 'val_auc': f'{val_auc:.3f}'})
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_fold{fold_num}.pt')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping na época {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"\n  Época {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val AUC={val_auc:.4f}")
    
    return history, best_auc


# ============================================
# VALIDAÇÃO CRUZADA
# ============================================

def run_cross_validation(subject_ids, labels, data_dir, n_folds=5):
    """Validação cruzada completa"""
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []
    all_val_preds = []
    all_val_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(subject_ids, labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")
        
        train_subjects = [subject_ids[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_subjects = [subject_ids[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        print(f"  Train: {len(train_subjects)} (pMCI: {sum(train_labels)}, sMCI: {len(train_labels)-sum(train_labels)})")
        print(f"  Val:   {len(val_subjects)} (pMCI: {sum(val_labels)}, sMCI: {len(val_labels)-sum(val_labels)})")
        
        # Datasets
        train_dataset = MRIDataset(train_subjects, train_labels, data_dir, augment=True)
        val_dataset = MRIDataset(val_subjects, val_labels, data_dir, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Modelo completo
        model = CNN3D_Full(dropout=0.5).to(DEVICE)
        
        # Contar parâmetros
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parâmetros do modelo: {n_params:,}")
        
        # Treinar
        history, best_auc = train_model(
            model, train_loader, val_loader,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            fold_num=fold+1
        )
        
        # Carregar melhor modelo
        model.load_state_dict(torch.load(f'best_model_fold{fold+1}.pt'))
        _, val_preds, val_labels_arr = evaluate(model, val_loader, nn.BCELoss(), DEVICE)
        
        # Guardar predições
        all_val_preds.extend(val_preds.flatten())
        all_val_labels.extend(val_labels_arr.flatten())
        
        # Métricas
        val_preds_binary = (val_preds.flatten() > 0.5).astype(int)
        val_labels_flat = val_labels_arr.flatten().astype(int)
        
        accuracy = accuracy_score(val_labels_flat, val_preds_binary)
        auc = roc_auc_score(val_labels_flat, val_preds.flatten())
        
        tn, fp, fn, tp = confusion_matrix(val_labels_flat, val_preds_binary).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append({
            'fold': fold + 1,
            'auc': auc,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity
        })
        
        print(f"\n  ✓ Resultados Fold {fold+1}:")
        print(f"    AUC:         {auc:.4f}")
        print(f"    Accuracy:    {accuracy:.4f}")
        print(f"    Sensitivity: {sensitivity:.4f}")
        print(f"    Specificity: {specificity:.4f}")
    
    return results, all_val_preds, all_val_labels


# ============================================
# GERAR VISUALIZAÇÕES
# ============================================

def plot_results(results, all_preds, all_labels):
    """Gera gráficos dos resultados"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Curva ROC
    ax1 = axes[0]
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    
    ax1.plot(fpr, tpr, 'b-', lw=2, label=f'CNN 3D (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Aleatório')
    ax1.set_xlabel('Taxa de Falso Positivo')
    ax1.set_ylabel('Taxa de Verdadeiro Positivo')
    ax1.set_title('Curva ROC - CNN 3D')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Métricas por fold
    ax2 = axes[1]
    folds = [r['fold'] for r in results]
    aucs = [r['auc'] for r in results]
    accs = [r['accuracy'] for r in results]
    
    x = np.arange(len(folds))
    width = 0.35
    
    ax2.bar(x - width/2, aucs, width, label='AUC', color='#3498db')
    ax2.bar(x + width/2, accs, width, label='Accuracy', color='#e74c3c')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Valor')
    ax2.set_title('Métricas por Fold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Fold {f}' for f in folds])
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('cnn3d_results_optimized.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Gráfico salvo: cnn3d_results_optimized.png")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("CNN 3D OTIMIZADA - PREDIÇÃO MCI → AD")
    print("=" * 60)
    print(f"\nConfigurações:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Épocas: {NUM_EPOCHS}")
    print(f"  Early Stopping: {EARLY_STOPPING_PATIENCE} épocas")
    print(f"  Modelo: CNN3D_Full (completo)")
    
    # Verificar dados
    if not os.path.exists(PREPROCESSED_DIR):
        print(f"\nERRO: Diretório {PREPROCESSED_DIR} não encontrado!")
        print("Execute: python preprocessar_mri.py")
        return
    
    # Carregar labels
    import csv
    subject_ids = []
    labels = []
    
    with open(LABELS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_id = row['OASISID']
            if os.path.exists(os.path.join(PREPROCESSED_DIR, f"{subject_id}.npy")):
                subject_ids.append(subject_id)
                labels.append(int(row['label']))
    
    print(f"\n[1] Sujeitos com imagens: {len(subject_ids)}")
    print(f"    pMCI (conversores): {sum(labels)}")
    print(f"    sMCI (estáveis): {len(labels) - sum(labels)}")
    
    if len(subject_ids) < 20:
        print("\nAVISO: Poucos sujeitos. Resultados podem ser instáveis.")
    
    # Validação cruzada
    print("\n[2] Iniciando treinamento...")
    print("    (Isso pode demorar algumas horas em CPU)")
    
    results, all_preds, all_labels = run_cross_validation(
        subject_ids, labels, PREPROCESSED_DIR, n_folds=5
    )
    
    # Resultados finais
    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS - CNN 3D OTIMIZADA")
    print("=" * 60)
    
    aucs = [r['auc'] for r in results]
    accs = [r['accuracy'] for r in results]
    sens = [r['sensitivity'] for r in results]
    specs = [r['specificity'] for r in results]
    
    # AUC global (todas as predições)
    global_auc = roc_auc_score(all_labels, all_preds)
    
    print(f"\nMétricas por Fold (média ± std):")
    print(f"  AUC:         {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Accuracy:    {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Sensitivity: {np.mean(sens):.4f} ± {np.std(sens):.4f}")
    print(f"  Specificity: {np.mean(specs):.4f} ± {np.std(specs):.4f}")
    print(f"\nAUC Global (pooled): {global_auc:.4f}")
    
    # Salvar resultados
    results_dict = {
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'model': 'CNN3D_Full',
            'n_subjects': len(subject_ids),
            'n_pmci': sum(labels),
            'n_smci': len(labels) - sum(labels)
        },
        'results_per_fold': results,
        'summary': {
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs)),
            'auc_global': float(global_auc),
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'sensitivity_mean': float(np.mean(sens)),
            'specificity_mean': float(np.mean(specs))
        }
    }
    
    with open('cnn3d_results_optimized.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\n✓ Resultados salvos em cnn3d_results_optimized.json")
    
    # Gerar gráficos
    plot_results(results, all_preds, all_labels)
    
    # Limpar arquivos temporários
    for fold in range(1, 6):
        try:
            os.remove(f'best_model_fold{fold}.pt')
        except:
            pass


if __name__ == "__main__":
    main()