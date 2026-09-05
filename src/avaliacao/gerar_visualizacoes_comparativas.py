"""
Visualizações Comparativas - TCC
Inclui resultados da CNN 3D vs Modelos Tabulares

Samuel Augusto Souza Alves Santana
Universidade Federal de Sergipe
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Configurar estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150

print("=" * 60)
print("GERANDO VISUALIZAÇÕES COMPARATIVAS")
print("=" * 60)

# ============================================
# DADOS DOS RESULTADOS
# ============================================

# Resultados dos modelos tabulares (features clínicas + cognitivas + MRI)
tabular_results = {
    'Logistic Regression': {'auc': 0.837, 'auc_std': 0.041, 'sens': 0.743, 'spec': 0.768, 'acc': 0.753},
    'Voting Ensemble': {'auc': 0.822, 'auc_std': 0.087, 'sens': 0.838, 'spec': 0.632, 'acc': 0.735},
    'Stacking Ensemble': {'auc': 0.818, 'auc_std': 0.077, 'sens': 0.757, 'spec': 0.711, 'acc': 0.734},
    'SVM (RBF)': {'auc': 0.813, 'auc_std': 0.079, 'sens': 0.748, 'spec': 0.733, 'acc': 0.741},
    'Random Forest': {'auc': 0.813, 'auc_std': 0.088, 'sens': 0.819, 'spec': 0.617, 'acc': 0.739},
    'Gradient Boosting': {'auc': 0.801, 'auc_std': 0.087, 'sens': 0.810, 'spec': 0.625, 'acc': 0.736},
    'MLP': {'auc': 0.800, 'auc_std': 0.065, 'sens': 0.790, 'spec': 0.647, 'acc': 0.733},
}

# Resultados CNN 3D
cnn_results = {
    'CNN 3D (21 subj)': {'auc': 0.783, 'auc_std': 0.194, 'sens': 0.600, 'spec': 0.400, 'acc': 0.480},
    'CNN 3D (61 subj)': {'auc': 0.828, 'auc_std': 0.054, 'sens': 0.733, 'spec': 0.400, 'acc': 0.574},
}

# ============================================
# FIGURA 1: COMPARAÇÃO GERAL DE AUC
# ============================================

print("\n[1] Gerando comparação geral de AUC...")

fig, ax = plt.subplots(figsize=(12, 7))

# Combinar todos os modelos
all_models = {**tabular_results, **cnn_results}
models = list(all_models.keys())
aucs = [all_models[m]['auc'] for m in models]
stds = [all_models[m]['auc_std'] for m in models]

# Cores diferentes para cada tipo
colors = []
for m in models:
    if 'CNN' in m:
        colors.append('#e74c3c')  # Vermelho para CNN
    elif m == 'Logistic Regression':
        colors.append('#27ae60')  # Verde para o melhor
    else:
        colors.append('#3498db')  # Azul para outros tabulares

# Ordenar por AUC
sorted_idx = np.argsort(aucs)[::-1]
models = [models[i] for i in sorted_idx]
aucs = [aucs[i] for i in sorted_idx]
stds = [stds[i] for i in sorted_idx]
colors = [colors[i] for i in sorted_idx]

x = np.arange(len(models))
bars = ax.bar(x, aucs, yerr=stds, capsize=5, color=colors, edgecolor='white', linewidth=1.5)

# Adicionar valores
for i, (bar, auc, std) in enumerate(zip(bars, aucs, stds)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
            f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('AUC (Área sob a Curva ROC)', fontsize=12)
ax.set_ylim([0.65, 0.95])
ax.set_title('Comparação de Performance: Modelos Tabulares vs CNN 3D\nPredição de Conversão MCI → Alzheimer', 
             fontsize=14, fontweight='bold')

# Legenda
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#27ae60', label='Melhor Modelo Tabular'),
    Patch(facecolor='#3498db', label='Outros Modelos Tabulares'),
    Patch(facecolor='#e74c3c', label='CNN 3D (Deep Learning)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_comparacao_geral_auc.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ fig_comparacao_geral_auc.png")

# ============================================
# FIGURA 2: CNN 3D - EVOLUÇÃO COM MAIS DADOS
# ============================================

print("\n[2] Gerando evolução da CNN 3D...")

fig, ax = plt.subplots(figsize=(8, 6))

n_subjects = [21, 61]
auc_values = [0.783, 0.828]
std_values = [0.194, 0.054]

colors = ['#e74c3c', '#27ae60']
bars = ax.bar(['21 sujeitos', '61 sujeitos'], auc_values, yerr=std_values, 
              capsize=8, color=colors, edgecolor='white', linewidth=2, width=0.5)

for bar, auc, std in zip(bars, auc_values, std_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
            f'AUC: {auc:.3f}\n(±{std:.3f})', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylim([0.5, 1.0])
ax.set_ylabel('AUC', fontsize=12)
ax.set_xlabel('Tamanho do Dataset', fontsize=12)
ax.set_title('CNN 3D: Impacto do Tamanho do Dataset\nna Performance do Modelo', fontsize=14, fontweight='bold')

# Anotação
ax.annotate('', xy=(1, 0.828), xytext=(0, 0.783),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(0.5, 0.75, '+5.7% AUC\n-72% variância', ha='center', fontsize=11, 
        color='green', fontweight='bold')

ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_cnn3d_evolucao.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ fig_cnn3d_evolucao.png")

# ============================================
# FIGURA 3: TABULAR VS CNN 3D (LADO A LADO)
# ============================================

print("\n[3] Gerando comparação Tabular vs CNN 3D...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: AUC
ax1 = axes[0]
models_compare = ['Logistic Regression\n(Tabular)', 'CNN 3D\n(Deep Learning)']
aucs_compare = [0.837, 0.828]
stds_compare = [0.041, 0.054]
colors_compare = ['#3498db', '#e74c3c']

bars1 = ax1.bar(models_compare, aucs_compare, yerr=stds_compare, capsize=8,
                color=colors_compare, edgecolor='white', linewidth=2, width=0.5)

for bar, auc, std in zip(bars1, aucs_compare, stds_compare):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
            f'{auc:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylim([0.7, 0.95])
ax1.set_ylabel('AUC', fontsize=12)
ax1.set_title('AUC - Área sob a Curva ROC', fontsize=13, fontweight='bold')
ax1.grid(True, axis='y', alpha=0.3)

# Subplot 2: Métricas detalhadas
ax2 = axes[1]
metrics = ['Sensibilidade', 'Especificidade', 'Acurácia']
tabular_vals = [0.743, 0.768, 0.753]
cnn_vals = [0.733, 0.400, 0.574]

x = np.arange(len(metrics))
width = 0.35

bars_tab = ax2.bar(x - width/2, tabular_vals, width, label='Logistic Regression', color='#3498db')
bars_cnn = ax2.bar(x + width/2, cnn_vals, width, label='CNN 3D', color='#e74c3c')

ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=11)
ax2.set_ylim([0, 1.0])
ax2.set_ylabel('Valor', fontsize=12)
ax2.set_title('Métricas de Classificação', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, axis='y', alpha=0.3)

# Adicionar valores nas barras
for bars in [bars_tab, bars_cnn]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

fig.suptitle('Comparação: Modelo Tabular vs CNN 3D\nPredição de Conversão MCI → Alzheimer', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('fig_tabular_vs_cnn3d.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ fig_tabular_vs_cnn3d.png")

# ============================================
# FIGURA 4: RESUMO EXECUTIVO
# ============================================

print("\n[4] Gerando resumo executivo...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Título
ax.text(0.5, 0.95, 'RESUMO DOS RESULTADOS', fontsize=20, fontweight='bold', 
        ha='center', va='top', transform=ax.transAxes)
ax.text(0.5, 0.89, 'Predição de Conversão MCI → Doença de Alzheimer', fontsize=14, 
        ha='center', va='top', transform=ax.transAxes, style='italic')

# Linha separadora
ax.axhline(y=0.85, xmin=0.1, xmax=0.9, color='gray', linewidth=2)

# Dataset
ax.text(0.5, 0.80, '📊 DATASET', fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
ax.text(0.5, 0.74, 'OASIS-3: 61 sujeitos MCI (31 conversores, 30 estáveis)', fontsize=12, 
        ha='center', transform=ax.transAxes)

# Resultados principais
ax.text(0.25, 0.64, '🧠 MODELO TABULAR', fontsize=13, fontweight='bold', ha='center', transform=ax.transAxes)
ax.text(0.25, 0.58, 'Logistic Regression', fontsize=11, ha='center', transform=ax.transAxes)
ax.text(0.25, 0.52, 'Features: Clínicas + Cognitivas + MRI', fontsize=10, ha='center', transform=ax.transAxes)
ax.text(0.25, 0.44, 'AUC = 0.837', fontsize=18, fontweight='bold', ha='center', 
        transform=ax.transAxes, color='#3498db')
ax.text(0.25, 0.38, 'Sensibilidade: 74.3%\nEspecificidade: 76.8%', fontsize=11, 
        ha='center', transform=ax.transAxes)

ax.text(0.75, 0.64, '🤖 CNN 3D', fontsize=13, fontweight='bold', ha='center', transform=ax.transAxes)
ax.text(0.75, 0.58, 'Deep Learning', fontsize=11, ha='center', transform=ax.transAxes)
ax.text(0.75, 0.52, 'Input: Imagens MRI T1w (128³)', fontsize=10, ha='center', transform=ax.transAxes)
ax.text(0.75, 0.44, 'AUC = 0.828', fontsize=18, fontweight='bold', ha='center', 
        transform=ax.transAxes, color='#e74c3c')
ax.text(0.75, 0.38, 'Sensibilidade: 73.3%\nEspecificidade: 40.0%', fontsize=11, 
        ha='center', transform=ax.transAxes)

# Linha vertical separadora
ax.axvline(x=0.5, ymin=0.35, ymax=0.65, color='gray', linewidth=1, linestyle='--')

# Conclusões
ax.axhline(y=0.28, xmin=0.1, xmax=0.9, color='gray', linewidth=1)
ax.text(0.5, 0.22, '📌 CONCLUSÕES', fontsize=13, fontweight='bold', ha='center', transform=ax.transAxes)

conclusions = [
    '• Ambas abordagens alcançaram AUC > 0.82 (competitivo com literatura)',
    '• Modelo tabular teve melhor especificidade (76.8% vs 40.0%)',
    '• CNN 3D mostrou potencial similar com apenas imagens',
    '• Combinação multimodal pode melhorar resultados'
]
for i, conc in enumerate(conclusions):
    ax.text(0.5, 0.16 - i*0.045, conc, fontsize=11, ha='center', transform=ax.transAxes)

plt.savefig('fig_resumo_executivo.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ fig_resumo_executivo.png")

# ============================================
# FIGURA 5: ARQUITETURA CNN 3D
# ============================================

print("\n[5] Gerando diagrama da arquitetura CNN 3D...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)

# Título
ax.text(7, 5.7, 'Arquitetura CNN 3D para Classificação MCI', fontsize=14, fontweight='bold', ha='center')

# Blocos da arquitetura
blocks = [
    {'x': 0.5, 'w': 1.5, 'h': 3, 'color': '#ecf0f1', 'label': 'Input\n128×128×128\n(1 canal)', 'y': 1.5},
    {'x': 2.5, 'w': 1.5, 'h': 2.5, 'color': '#3498db', 'label': 'Conv3D\n32 filtros\n+Pool', 'y': 1.75},
    {'x': 4.5, 'w': 1.5, 'h': 2.0, 'color': '#3498db', 'label': 'Conv3D\n64 filtros\n+Pool', 'y': 2.0},
    {'x': 6.5, 'w': 1.5, 'h': 1.5, 'color': '#3498db', 'label': 'Conv3D\n128 filtros\n+Pool', 'y': 2.25},
    {'x': 8.5, 'w': 1.5, 'h': 1.0, 'color': '#9b59b6', 'label': 'Global\nAvgPool', 'y': 2.5},
    {'x': 10.5, 'w': 1.5, 'h': 1.5, 'color': '#e74c3c', 'label': 'FC 128\n+Dropout', 'y': 2.25},
    {'x': 12.5, 'w': 1.0, 'h': 1.0, 'color': '#27ae60', 'label': 'Output\npMCI/sMCI', 'y': 2.5},
]

for block in blocks:
    rect = plt.Rectangle((block['x'], block['y']), block['w'], block['h'], 
                         facecolor=block['color'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(block['x'] + block['w']/2, block['y'] + block['h']/2, block['label'],
            ha='center', va='center', fontsize=9, fontweight='bold')

# Setas
arrow_style = dict(arrowstyle='->', color='black', lw=1.5)
for i in range(len(blocks)-1):
    ax.annotate('', xy=(blocks[i+1]['x'], 3), xytext=(blocks[i]['x']+blocks[i]['w'], 3),
                arrowprops=arrow_style)

# Dimensões
dims = ['128³', '64³', '32³', '16³', '128', '128', '1']
for i, (block, dim) in enumerate(zip(blocks, dims)):
    ax.text(block['x'] + block['w']/2, block['y'] - 0.3, dim, ha='center', fontsize=8, color='gray')

# Legenda
ax.text(1, 0.5, 'Conv3D = Convolução 3D + BatchNorm + ReLU', fontsize=9)
ax.text(7, 0.5, 'Pool = MaxPooling 3D (2×2×2)', fontsize=9)

plt.tight_layout()
plt.savefig('fig_arquitetura_cnn3d.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ fig_arquitetura_cnn3d.png")

# ============================================
# FIGURA 6: COMPARAÇÃO POR ABORDAGEM
# ============================================

print("\n[6] Gerando comparação por abordagem...")

fig, ax = plt.subplots(figsize=(10, 7))

# Dados
approaches = ['Só Clínicas\n(6 features)', 'Só MRI\nVolumes\n(8 features)', 
              'Só Cognitivas\n(13 features)', 'Clínicas +\nCognitivas\n(19 features)',
              'Todas\nTabulares\n(20 features)', 'CNN 3D\n(imagens)']
auc_values = [0.787, 0.720, 0.743, 0.823, 0.837, 0.828]
colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6', '#e74c3c']

bars = ax.bar(approaches, auc_values, color=colors, edgecolor='white', linewidth=2)

for bar, auc in zip(bars, auc_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{auc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylim([0.65, 0.90])
ax.set_ylabel('AUC', fontsize=12)
ax.set_title('Comparação de Diferentes Abordagens\npara Predição MCI → AD', fontsize=14, fontweight='bold')
ax.axhline(y=0.837, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='Melhor resultado')
ax.legend(loc='lower right')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('fig_comparacao_abordagens.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ fig_comparacao_abordagens.png")

# ============================================
# FINALIZAÇÃO
# ============================================

print("\n" + "=" * 60)
print("VISUALIZAÇÕES GERADAS COM SUCESSO!")
print("=" * 60)

print("""
Arquivos criados:
  1. fig_comparacao_geral_auc.png    - Todos os modelos
  2. fig_cnn3d_evolucao.png          - Impacto do tamanho do dataset
  3. fig_tabular_vs_cnn3d.png        - Comparação direta
  4. fig_resumo_executivo.png        - Resumo para apresentação
  5. fig_arquitetura_cnn3d.png       - Diagrama da CNN
  6. fig_comparacao_abordagens.png   - Diferentes abordagens

💡 Use estas figuras nas seções de Resultados e Discussão do TCC!
""")
