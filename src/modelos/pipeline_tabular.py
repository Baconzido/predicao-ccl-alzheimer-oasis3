"""
Pipeline tabular para Predição MCI -> AD (dados clínicos, cognitivos e de neuroimagem)
TCC - Samuel Augusto Souza Alves Santana

Monta o dataset (349 participantes) a partir dos CSVs derivados do OASIS-3
(CDR, demografia, FreeSurfer, avaliações cognitivas, FAQ), seleciona as 20
melhores features por ANOVA F-score, e treina/compara 5 classificadores
(Regressão Logística, SVM, Random Forest, Gradient Boosting, MLP) mais 2
ensembles (Voting, Stacking) com validação cruzada 5-fold. A Regressão
Logística foi o melhor modelo (AUC 0,837 ± 0,041 - resultado oficial do TCC).

Ao final, roda também uma análise de ablação por tipo de variável (Random
Forest fixo, variando o conjunto de features: clínicas / neuroimagem /
cognitivas / combinações) - ver seção "Contribuição de cada grupo de
variáveis" no README principal.

Requer os CSVs derivados do OASIS-3 (CDR, demographics, FreeSurfer,
avaliações cognitivas, FAQ) no diretório de execução - não incluídos
neste repositório (ver data/README.md).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    classification_report, f1_score, balanced_accuracy_score
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Modelos
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression

# SMOTE para balanceamento
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("imblearn não disponível - usando class_weight em vez de SMOTE")

print("=" * 70)
print("MODELO MELHORADO PARA PREDIÇÃO MCI → AD")
print("=" * 70)

# ============================================
# 1. CARREGAR E PREPARAR NOVOS DADOS
# ============================================

print("\n[1] Carregando dados...")

# Carregar datasets
cdr_df = pd.read_csv('OASIS3_UDSb4_cdr.csv')
demo_df = pd.read_csv('OASIS3_demographics.csv')
freesurfer_df = pd.read_csv('OASIS3_Freesurfer_output.csv')
cognitive_df = pd.read_csv('OASIS3_UDSc1_cognitive_assessments.csv')
faq_df = pd.read_csv('OASIS3_UDSb7_faq_fas.csv')

print(f"    CDR: {len(cdr_df)} registros")
print(f"    Cognitive assessments: {len(cognitive_df)} registros")
print(f"    FAQ: {len(faq_df)} registros")

# ============================================
# 2. EXTRAIR FEATURES COGNITIVAS
# ============================================

print("\n[2] Extraindo features cognitivas...")

# Features cognitivas importantes (baseado na literatura)
cog_features = [
    'OASISID', 'days_to_visit',
    'LOGIMEM',      # Logical Memory (memória)
    'MEMUNITS',     # Memory Units
    'ANIMALS',      # Category Fluency - Animals
    'VEG',          # Category Fluency - Vegetables
    'digfor',       # Digit Span Forward
    'digback',      # Digit Span Backward
    'tma',          # Trail Making A (atenção)
    'tmb',          # Trail Making B (função executiva)
    'bnt',          # Boston Naming Test
    'craftvrs',     # Craft Story Recall - Verbatim
    'craftdvr',     # Craft Story Recall - Delayed
    'mocatots',     # MoCA Total Score
    'lmdelay',      # Logical Memory Delayed
]

# Filtrar colunas existentes
cog_features_available = [f for f in cog_features if f in cognitive_df.columns]
cog_df = cognitive_df[cog_features_available].copy()

# Para cada sujeito, pegar a primeira avaliação (baseline)
cog_df = cog_df.sort_values(['OASISID', 'days_to_visit'])
cog_baseline = cog_df.groupby('OASISID').first().reset_index()
cog_baseline = cog_baseline.drop(columns=['days_to_visit'], errors='ignore')

print(f"    Features cognitivas disponíveis: {len(cog_features_available) - 2}")

# ============================================
# 3. EXTRAIR FAQ SCORE
# ============================================

print("\n[3] Extraindo FAQ score...")

# FAQ items (0-3 cada, maior = pior)
faq_items = ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 
             'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']

# Calcular FAQ total (soma dos itens, excluindo valores 8 que significam N/A)
faq_df_clean = faq_df.copy()
for col in faq_items:
    if col in faq_df_clean.columns:
        faq_df_clean[col] = faq_df_clean[col].replace(8, np.nan)

faq_df_clean['FAQ_TOTAL'] = faq_df_clean[faq_items].sum(axis=1, skipna=True)

# Baseline FAQ
faq_df_clean = faq_df_clean.sort_values(['OASISID', 'days_to_visit'])
faq_baseline = faq_df_clean.groupby('OASISID').first().reset_index()
faq_baseline = faq_baseline[['OASISID', 'FAQ_TOTAL']]

print(f"    FAQ baseline calculado para {len(faq_baseline)} sujeitos")

# ============================================
# 4. IDENTIFICAR SUJEITOS MCI
# ============================================

print("\n[4] Identificando sujeitos MCI...")

def identify_mci_subjects(cdr_df, min_followup_days=1095):
    cdr_df = cdr_df.sort_values(['OASISID', 'days_to_visit'])
    mci_subjects = []
    
    for subject_id, group in cdr_df.groupby('OASISID'):
        mci_visits = group[group['CDRTOT'] == 0.5]
        if len(mci_visits) == 0:
            continue
        
        baseline = mci_visits.iloc[0]
        baseline_day = baseline['days_to_visit']
        follow_up = group[group['days_to_visit'] > baseline_day]
        dementia_visits = follow_up[follow_up['CDRTOT'] >= 1]
        
        if len(dementia_visits) > 0:
            conversion = dementia_visits.iloc[0]
            mci_subjects.append({
                'OASISID': subject_id,
                'baseline_day': baseline_day,
                'baseline_age': baseline['age at visit'],
                'baseline_mmse': baseline['MMSE'],
                'baseline_cdrsum': baseline['CDRSUM'],
                'conversion_status': 'pMCI',
                'time_to_conversion': conversion['days_to_visit'] - baseline_day
            })
        else:
            if len(follow_up) > 0:
                followup_duration = follow_up['days_to_visit'].max() - baseline_day
            else:
                followup_duration = 0
            
            if followup_duration >= min_followup_days:
                stable = follow_up[follow_up['CDRTOT'] <= 0.5]
                if len(stable) == len(follow_up):
                    mci_subjects.append({
                        'OASISID': subject_id,
                        'baseline_day': baseline_day,
                        'baseline_age': baseline['age at visit'],
                        'baseline_mmse': baseline['MMSE'],
                        'baseline_cdrsum': baseline['CDRSUM'],
                        'conversion_status': 'sMCI',
                        'time_to_conversion': None
                    })
    
    return pd.DataFrame(mci_subjects)

mci_df = identify_mci_subjects(cdr_df)
print(f"    pMCI: {(mci_df['conversion_status'] == 'pMCI').sum()}")
print(f"    sMCI: {(mci_df['conversion_status'] == 'sMCI').sum()}")

# ============================================
# 5. COMBINAR TODAS AS FEATURES
# ============================================

print("\n[5] Combinando todas as features...")

# Demographics
mci_df = mci_df.merge(
    demo_df[['OASISID', 'GENDER', 'EDUC', 'APOE']],
    on='OASISID', how='left'
)
mci_df['sex_numeric'] = (mci_df['GENDER'] == 1).astype(int)
mci_df['has_apoe4'] = mci_df['APOE'].apply(
    lambda x: 1 if pd.notna(x) and '4' in str(int(x)) else 0
)

# FreeSurfer
fs_features = [
    'Subject', 'MR_session',
    'IntraCranialVol', 'TotalGrayVol', 'CortexVol',
    'Left-Hippocampus_volume', 'Right-Hippocampus_volume',
    'Left-Amygdala_volume', 'Right-Amygdala_volume',
    'lh_entorhinal_volume', 'rh_entorhinal_volume',
    'Left-Lateral-Ventricle_volume', 'Right-Lateral-Ventricle_volume',
    'Left-Thalamus-Proper_volume', 'Right-Thalamus-Proper_volume',
    'lh_parahippocampal_volume', 'rh_parahippocampal_volume',
    'TOTAL_HIPPOCAMPUS_VOLUME'
]
fs_features = [f for f in fs_features if f in freesurfer_df.columns]
fs_df = freesurfer_df[fs_features].copy()
fs_df = fs_df.rename(columns={'Subject': 'OASISID'})
fs_df['mr_day'] = fs_df['MR_session'].str.extract(r'd(\d+)').astype(float)
fs_df = fs_df.sort_values(['OASISID', 'mr_day'])
fs_baseline = fs_df.groupby('OASISID').first().reset_index()

# Merge tudo
dataset = mci_df.merge(fs_baseline, on='OASISID', how='inner')
dataset = dataset.merge(cog_baseline, on='OASISID', how='left')
dataset = dataset.merge(faq_baseline, on='OASISID', how='left')

# Features derivadas
dataset['total_hippocampus'] = (
    dataset['Left-Hippocampus_volume'] + dataset['Right-Hippocampus_volume']
)
dataset['total_entorhinal'] = (
    dataset['lh_entorhinal_volume'] + dataset['rh_entorhinal_volume']
)
dataset['total_ventricles'] = (
    dataset['Left-Lateral-Ventricle_volume'] + dataset['Right-Lateral-Ventricle_volume']
)

# Normalizar por ICV
icv = dataset['IntraCranialVol']
dataset['hippocampus_norm'] = dataset['total_hippocampus'] / icv * 100
dataset['entorhinal_norm'] = dataset['total_entorhinal'] / icv * 100
dataset['ventricles_norm'] = dataset['total_ventricles'] / icv * 100

# Label
dataset['label'] = (dataset['conversion_status'] == 'pMCI').astype(int)

print(f"    Dataset final: {len(dataset)} sujeitos")
print(f"    pMCI: {dataset['label'].sum()} ({dataset['label'].mean()*100:.1f}%)")

# ============================================
# 6. PREPARAR FEATURES PARA MODELO
# ============================================

print("\n[6] Preparando features...")

# Lista de todas as features
feature_cols = [
    # Demográficas/Clínicas
    'baseline_age', 'baseline_mmse', 'baseline_cdrsum', 'EDUC', 
    'sex_numeric', 'has_apoe4',
    # MRI - Volumes normalizados
    'hippocampus_norm', 'entorhinal_norm', 'ventricles_norm',
    # MRI - Volumes absolutos
    'Left-Hippocampus_volume', 'Right-Hippocampus_volume',
    'Left-Amygdala_volume', 'Right-Amygdala_volume',
    'TOTAL_HIPPOCAMPUS_VOLUME',
    # Cognitivas
    'LOGIMEM', 'MEMUNITS', 'ANIMALS', 'VEG', 
    'digfor', 'digback', 'tma', 'tmb', 'bnt',
    'craftvrs', 'craftdvr', 'mocatots', 'lmdelay',
    # Funcional
    'FAQ_TOTAL'
]

# Filtrar features disponíveis
feature_cols = [c for c in feature_cols if c in dataset.columns]
print(f"    Features disponíveis: {len(feature_cols)}")

X = dataset[feature_cols].copy()
y = dataset['label'].values

# Imputar valores faltantes
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

print(f"    Shape: X={X_imputed.shape}")

# ============================================
# 7. FEATURE SELECTION
# ============================================

print("\n[7] Seleção de features...")

# SelectKBest
selector = SelectKBest(f_classif, k=min(20, len(feature_cols)))
selector.fit(X_imputed, y)

# Ranking de features
feature_scores = pd.DataFrame({
    'Feature': feature_cols,
    'Score': selector.scores_,
    'P-value': selector.pvalues_
}).sort_values('Score', ascending=False)

print("\nTop 15 features (ANOVA F-score):")
print(feature_scores.head(15).to_string(index=False))

# Usar top features
top_k = 20
top_features = feature_scores.head(top_k)['Feature'].tolist()
X_selected = X_imputed[top_features]

print(f"\n    Usando top {top_k} features para o modelo")

# ============================================
# 8. VALIDAÇÃO CRUZADA COM MODELOS OTIMIZADOS
# ============================================

print("\n" + "=" * 70)
print("TREINAMENTO COM OTIMIZAÇÃO")
print("=" * 70)

def evaluate_model(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'AUC': roc_auc_score(y_true, y_prob),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Acc': balanced_accuracy_score(y_true, y_pred),
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'F1': f1_score(y_true, y_pred)
    }

def run_cv_with_balancing(X, y, model, n_folds=5, use_smote=False):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Aplicar SMOTE se disponível
        if use_smote and HAS_IMBLEARN:
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        
        # Treinar
        model.fit(X_train_scaled, y_train)
        
        # Predizer
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]
        
        metrics = evaluate_model(y_val, y_pred, y_prob)
        results.append(metrics)
    
    return pd.DataFrame(results)

# Modelos com class_weight para lidar com desbalanceamento
models = {
    'MLP (otimizado)': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.01,  # Regularização L2
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42
    ),
    'SVM (RBF)': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
}

all_results = {}

for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Modelo: {model_name}")
    print(f"{'='*50}")
    
    # Testar com e sem SMOTE
    results = run_cv_with_balancing(X_selected, y, model, use_smote=HAS_IMBLEARN)
    all_results[model_name] = results
    
    print(f"\nMétricas (5-fold CV):")
    print(f"  AUC:          {results['AUC'].mean():.4f} ± {results['AUC'].std():.4f}")
    print(f"  Accuracy:     {results['Accuracy'].mean():.4f} ± {results['Accuracy'].std():.4f}")
    print(f"  Balanced Acc: {results['Balanced_Acc'].mean():.4f} ± {results['Balanced_Acc'].std():.4f}")
    print(f"  Sensitivity:  {results['Sensitivity'].mean():.4f} ± {results['Sensitivity'].std():.4f}")
    print(f"  Specificity:  {results['Specificity'].mean():.4f} ± {results['Specificity'].std():.4f}")
    print(f"  F1-Score:     {results['F1'].mean():.4f} ± {results['F1'].std():.4f}")

# ============================================
# 9. ENSEMBLE (VOTING + STACKING)
# ============================================

print("\n" + "=" * 70)
print("ENSEMBLE MODELS")
print("=" * 70)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, 
                                      class_weight='balanced', random_state=42)),
        ('svm', SVC(kernel='rbf', C=1.0, class_weight='balanced', 
                    probability=True, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=150, max_depth=5, 
                                          random_state=42))
    ],
    voting='soft'
)

print("\nVoting Classifier (RF + SVM + GB):")
results_voting = run_cv_with_balancing(X_selected, y, voting_clf, use_smote=HAS_IMBLEARN)
all_results['Voting Ensemble'] = results_voting
print(f"  AUC:          {results_voting['AUC'].mean():.4f} ± {results_voting['AUC'].std():.4f}")
print(f"  Balanced Acc: {results_voting['Balanced_Acc'].mean():.4f} ± {results_voting['Balanced_Acc'].std():.4f}")
print(f"  Sensitivity:  {results_voting['Sensitivity'].mean():.4f}")
print(f"  Specificity:  {results_voting['Specificity'].mean():.4f}")

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, 
                                      class_weight='balanced', random_state=42)),
        ('svm', SVC(kernel='rbf', C=1.0, class_weight='balanced', 
                    probability=True, random_state=42)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, 
                              random_state=42))
    ],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=3
)

print("\nStacking Classifier (RF + SVM + MLP → LR):")
results_stacking = run_cv_with_balancing(X_selected, y, stacking_clf, use_smote=HAS_IMBLEARN)
all_results['Stacking Ensemble'] = results_stacking
print(f"  AUC:          {results_stacking['AUC'].mean():.4f} ± {results_stacking['AUC'].std():.4f}")
print(f"  Balanced Acc: {results_stacking['Balanced_Acc'].mean():.4f} ± {results_stacking['Balanced_Acc'].std():.4f}")
print(f"  Sensitivity:  {results_stacking['Sensitivity'].mean():.4f}")
print(f"  Specificity:  {results_stacking['Specificity'].mean():.4f}")

# ============================================
# 10. COMPARAÇÃO FINAL
# ============================================

print("\n" + "=" * 70)
print("COMPARAÇÃO FINAL DE TODOS OS MODELOS")
print("=" * 70)

comparison = []
for model_name, results in all_results.items():
    comparison.append({
        'Model': model_name,
        'AUC': f"{results['AUC'].mean():.3f}±{results['AUC'].std():.3f}",
        'Bal_Acc': f"{results['Balanced_Acc'].mean():.3f}±{results['Balanced_Acc'].std():.3f}",
        'Sens': f"{results['Sensitivity'].mean():.3f}",
        'Spec': f"{results['Specificity'].mean():.3f}",
        'F1': f"{results['F1'].mean():.3f}",
        'AUC_mean': results['AUC'].mean()
    })

comparison_df = pd.DataFrame(comparison).sort_values('AUC_mean', ascending=False)
print("\n" + comparison_df[['Model', 'AUC', 'Bal_Acc', 'Sens', 'Spec', 'F1']].to_string(index=False))

# ============================================
# 11. ANÁLISE: IMPACTO DAS NOVAS FEATURES
# ============================================

print("\n" + "=" * 70)
print("ANÁLISE: IMPACTO DOS DIFERENTES TIPOS DE FEATURES")
print("=" * 70)

# Separar features por tipo
clinical_features = ['baseline_age', 'baseline_mmse', 'baseline_cdrsum', 'EDUC', 
                     'sex_numeric', 'has_apoe4']
mri_features = ['hippocampus_norm', 'entorhinal_norm', 'ventricles_norm',
                'Left-Hippocampus_volume', 'Right-Hippocampus_volume',
                'Left-Amygdala_volume', 'Right-Amygdala_volume', 'TOTAL_HIPPOCAMPUS_VOLUME']
cognitive_features = ['LOGIMEM', 'MEMUNITS', 'ANIMALS', 'VEG', 'digfor', 'digback',
                      'tma', 'tmb', 'bnt', 'craftvrs', 'craftdvr', 'mocatots', 'lmdelay']
functional_features = ['FAQ_TOTAL']

feature_sets = {
    'Só Clínicas': [f for f in clinical_features if f in X_imputed.columns],
    'Só MRI': [f for f in mri_features if f in X_imputed.columns],
    'Só Cognitivas': [f for f in cognitive_features if f in X_imputed.columns],
    'Clínicas + MRI': [f for f in clinical_features + mri_features if f in X_imputed.columns],
    'Clínicas + Cognitivas': [f for f in clinical_features + cognitive_features if f in X_imputed.columns],
    'TODAS': top_features
}

best_model = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                    class_weight='balanced', random_state=42)

print("\nComparação por tipo de feature (Random Forest):\n")
for set_name, features in feature_sets.items():
    if len(features) < 2:
        continue
    X_subset = X_imputed[features]
    results = run_cv_with_balancing(X_subset, y, best_model, use_smote=HAS_IMBLEARN)
    print(f"{set_name} ({len(features)} features):")
    print(f"  AUC: {results['AUC'].mean():.4f} | Bal_Acc: {results['Balanced_Acc'].mean():.4f} | "
          f"Sens: {results['Sensitivity'].mean():.3f} | Spec: {results['Specificity'].mean():.3f}")

# ============================================
# 12. SALVAR RESULTADOS
# ============================================

print("\n" + "=" * 70)
print("SALVANDO RESULTADOS")
print("=" * 70)

# Salvar dataset melhorado
dataset.to_csv('mci_dataset_improved.csv', index=False)
print("    Dataset melhorado: mci_dataset_improved.csv")

# Salvar comparação de modelos
comparison_df.to_csv('model_comparison_improved.csv', index=False)
print("    Comparação: model_comparison_improved.csv")

# Salvar ranking de features
feature_scores.to_csv('feature_ranking.csv', index=False)
print("    Ranking de features: feature_ranking.csv")

print("\n" + "=" * 70)
print("TREINAMENTO CONCLUÍDO!")
print("=" * 70)

# Melhor resultado
best_model_name = comparison_df.iloc[0]['Model']
best_auc = comparison_df.iloc[0]['AUC']
print(f"\n🏆 MELHOR MODELO: {best_model_name}")
print(f"   AUC: {best_auc}")
