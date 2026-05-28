# ============================================================
# Task 3: Exploratory Data Analysis (EDA)
# Edutech Solution – AI & ML Internship
# Dataset: Iris (150 rows × 5 columns)
# Tools: Python · Pandas · Matplotlib · Seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="Set2")
COLORS = ["#2ecc71", "#3498db", "#e74c3c"]

print("=" * 65)
print("   TASK 3: EXPLORATORY DATA ANALYSIS (EDA) — IRIS DATASET")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD & OVERVIEW
# ─────────────────────────────────────────────────────────────
print("\n📂 STEP 1: Load Dataset & Overview")
print("─" * 65)

df = pd.read_csv("iris.csv")
print(f"   Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns : {df.columns.tolist()}")
print(f"\n   First 5 rows:")
print(df.head().to_string())
print(f"\n   Data Types:\n{df.dtypes.to_string()}")
print(f"\n   Missing Values: {df.isnull().sum().sum()}")
print(f"   Species Classes: {df['species'].unique().tolist()}")
print(f"   Class Distribution:\n{df['species'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────
print("\n\n📊 STEP 2: Summary Statistics")
print("─" * 65)
print(df.describe().round(3).to_string())

print("\n   Per-species statistics:")
num_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
print(df.groupby("species")[num_cols].mean().round(2).to_string())

# ─────────────────────────────────────────────────────────────
# STEP 3 — UNIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n\n📈 STEP 3: Univariate Analysis")
print("─" * 65)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Univariate Analysis — Iris Dataset", fontsize=18, fontweight='bold', y=1.01)

for i, col in enumerate(num_cols):
    # Histogram
    ax_h = axes[0][i]
    for j, (sp, grp) in enumerate(df.groupby("species")):
        ax_h.hist(grp[col], bins=15, alpha=0.65, color=COLORS[j],
                  label=sp, edgecolor='white')
    ax_h.set_title(f"{col.replace('_', ' ').title()} — Histogram", fontsize=11, fontweight='bold')
    ax_h.set_xlabel(col.replace('_', ' '))
    ax_h.set_ylabel("Count")
    ax_h.legend(fontsize=8)
    mean_val = df[col].mean()
    ax_h.axvline(mean_val, color='black', linestyle='--', linewidth=1.2,
                 label=f"Mean={mean_val:.2f}")

    # Boxplot
    ax_b = axes[1][i]
    bp = ax_b.boxplot(
        [df[df["species"] == sp][col].values for sp in df["species"].unique()],
        labels=df["species"].unique(), patch_artist=True,
        medianprops=dict(color="black", linewidth=2)
    )
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_b.set_title(f"{col.replace('_', ' ').title()} — Boxplot", fontsize=11, fontweight='bold')
    ax_b.set_ylabel(col.replace('_', ' '))
    ax_b.set_xticklabels(df["species"].unique(), rotation=10)

plt.tight_layout()
plt.savefig("01_univariate_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 01_univariate_analysis.png")

# ─────────────────────────────────────────────────────────────
# STEP 4 — BIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n\n🔗 STEP 4: Bivariate Analysis")
print("─" * 65)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Bivariate Analysis — Scatter Plots", fontsize=16, fontweight='bold')

pairs = [
    ("sepal_length", "sepal_width"),
    ("petal_length", "petal_width"),
    ("sepal_length", "petal_length"),
]

for ax, (x, y) in zip(axes, pairs):
    for j, (sp, grp) in enumerate(df.groupby("species")):
        ax.scatter(grp[x], grp[y], label=sp, color=COLORS[j],
                   alpha=0.75, s=60, edgecolors='white', linewidth=0.5)
    ax.set_xlabel(x.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel(y.replace('_', ' ').title(), fontsize=11)
    ax.set_title(f"{x.replace('_',' ').title()} vs {y.replace('_',' ').title()}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("02_bivariate_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 02_bivariate_scatter.png")

# Bar chart – mean values per species
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Mean Feature Values per Species — Bar Charts", fontsize=14, fontweight='bold')
species_means = df.groupby("species")[num_cols].mean()

for ax, col in zip(axes, num_cols):
    bars = ax.bar(species_means.index, species_means[col],
                  color=COLORS, edgecolor='white', width=0.5)
    ax.set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_ylabel("Mean Value (cm)")
    ax.set_xticklabels(species_means.index, rotation=10)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("03_bivariate_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 03_bivariate_bar.png")

# ─────────────────────────────────────────────────────────────
# STEP 5 — CORRELATION MATRIX & HEATMAP
# ─────────────────────────────────────────────────────────────
print("\n\n🌡️  STEP 5: Correlation Matrix & Heatmap")
print("─" * 65)

corr = df[num_cols].corr()
print("   Correlation Matrix:")
print(corr.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Correlation Analysis", fontsize=16, fontweight='bold')

# Full heatmap
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=axes[0],
            cbar_kws={"shrink": 0.8}, square=True)
axes[0].set_title("Correlation Heatmap (All Features)", fontsize=13, fontweight='bold')

# Triangle heatmap
mask2 = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask2, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=axes[1],
            cbar_kws={"shrink": 0.8}, square=True)
axes[1].set_title("Lower Triangle Heatmap", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("04_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 04_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────
# STEP 6 — PAIRPLOT
# ─────────────────────────────────────────────────────────────
print("\n\n🔷 STEP 6: Pairplot — All Numerical Variables")
print("─" * 65)

fig = sns.pairplot(df, hue="species", palette={"setosa": COLORS[0],
                    "versicolor": COLORS[1], "virginica": COLORS[2]},
                   diag_kind="kde", plot_kws={"alpha": 0.65, "s": 40,
                   "edgecolor": "white", "linewidth": 0.3},
                   height=2.5)
fig.fig.suptitle("Pairplot — Iris Dataset", y=1.01, fontsize=16, fontweight='bold')
fig.savefig("05_pairplot.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 05_pairplot.png")

# ─────────────────────────────────────────────────────────────
# STEP 7 — PATTERNS, TRENDS & ANOMALIES
# ─────────────────────────────────────────────────────────────
print("\n\n🔍 STEP 7: Patterns, Trends & Anomalies")
print("─" * 65)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Patterns, Trends & Anomalies", fontsize=16, fontweight='bold')

# KDE plots
for j, (sp, grp) in enumerate(df.groupby("species")):
    axes[0][0].plot(sorted(grp["petal_length"]),
                    np.linspace(0, 1, len(grp)), color=COLORS[j], label=sp, linewidth=2)
axes[0][0].set_title("Petal Length — CDF per Species", fontsize=12, fontweight='bold')
axes[0][0].set_xlabel("Petal Length (cm)")
axes[0][0].set_ylabel("Cumulative Probability")
axes[0][0].legend()

# Violin plot
species_list = df["species"].unique()
data_vio = [df[df["species"] == sp]["petal_length"].values for sp in species_list]
vp = axes[0][1].violinplot(data_vio, positions=[1, 2, 3], showmedians=True)
for i, pc in enumerate(vp['bodies']):
    pc.set_facecolor(COLORS[i])
    pc.set_alpha(0.7)
axes[0][1].set_xticks([1, 2, 3])
axes[0][1].set_xticklabels(species_list)
axes[0][1].set_title("Petal Length — Violin Plot", fontsize=12, fontweight='bold')
axes[0][1].set_ylabel("Petal Length (cm)")

# Species distribution pie
counts = df["species"].value_counts()
axes[1][0].pie(counts, labels=counts.index, colors=COLORS,
               autopct='%1.0f%%', startangle=90,
               wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1][0].set_title("Species Distribution", fontsize=12, fontweight='bold')

# Sepal area analysis
df["sepal_area"] = df["sepal_length"] * df["sepal_width"]
df["petal_area"] = df["petal_length"] * df["petal_width"]
for j, (sp, grp) in enumerate(df.groupby("species")):
    axes[1][1].scatter(grp["sepal_area"], grp["petal_area"],
                       label=sp, color=COLORS[j], alpha=0.7, s=60, edgecolors='white')
axes[1][1].set_xlabel("Sepal Area (cm²)")
axes[1][1].set_ylabel("Petal Area (cm²)")
axes[1][1].set_title("Sepal Area vs Petal Area", fontsize=12, fontweight='bold')
axes[1][1].legend()

plt.tight_layout()
plt.savefig("06_patterns_trends.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 06_patterns_trends.png")

# ─────────────────────────────────────────────────────────────
# STEP 8 — OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────
print("\n\n📦 STEP 8: Outlier Detection")
print("─" * 65)

fig, axes = plt.subplots(1, 4, figsize=(18, 6))
fig.suptitle("Outlier Detection — Boxplots per Feature", fontsize=14, fontweight='bold')

for ax, col in zip(axes, num_cols):
    data = [df[df["species"] == sp][col].values for sp in species_list]
    bp = ax.boxplot(data, patch_artist=True, labels=species_list,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker='o', markersize=6,
                                    markerfacecolor='red', markeredgecolor='black'))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_ylabel("Value (cm)")
    ax.set_xticklabels(species_list, rotation=12)

    # Count outliers
    for i, sp in enumerate(species_list):
        vals = df[df["species"] == sp][col]
        Q1, Q3 = vals.quantile(0.25), vals.quantile(0.75)
        IQR = Q3 - Q1
        n = ((vals < Q1 - 1.5*IQR) | (vals > Q3 + 1.5*IQR)).sum()
        if n > 0:
            print(f"   '{col}' — {sp}: {n} outlier(s)")

plt.tight_layout()
plt.savefig("07_outlier_detection.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 07_outlier_detection.png")

# ─────────────────────────────────────────────────────────────
# KEY INSIGHTS REPORT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("   KEY INSIGHTS REPORT")
print("=" * 65)

corr_vals = corr.unstack().sort_values(ascending=False)
print(f"""
  📌 DATASET OVERVIEW
     • 150 samples | 3 species (50 each) | 4 features | 0 missing

  📌 UNIVARIATE INSIGHTS
     • Petal Length has highest variance (std={df['petal_length'].std():.2f}cm)
     • Sepal Width has smallest range ({df['sepal_width'].min()}–{df['sepal_width'].max()}cm)
     • Setosa is clearly separated from others in petal dimensions

  📌 BIVARIATE INSIGHTS
     • Petal Length vs Petal Width: strongest positive correlation ({corr['petal_length']['petal_width']:.2f})
     • Sepal Length vs Sepal Width: slight negative correlation ({corr['sepal_length']['sepal_width']:.2f})
     • Setosa clusters distinctly; Versicolor & Virginica overlap slightly

  📌 CORRELATION INSIGHTS
     • petal_length ↔ petal_width : {corr['petal_length']['petal_width']:.3f} (very strong +ve)
     • sepal_length ↔ petal_length: {corr['sepal_length']['petal_length']:.3f} (strong +ve)
     • sepal_length ↔ petal_width : {corr['sepal_length']['petal_width']:.3f} (strong +ve)
     • sepal_width  ↔ petal_length: {corr['sepal_width']['petal_length']:.3f} (moderate -ve)

  📌 SPECIES-LEVEL INSIGHTS
     • Setosa    : smallest petals (mean petal length {df[df.species=='setosa']['petal_length'].mean():.2f}cm)
     • Versicolor: medium petals (mean petal length {df[df.species=='versicolor']['petal_length'].mean():.2f}cm)
     • Virginica : largest petals (mean petal length {df[df.species=='virginica']['petal_length'].mean():.2f}cm)

  📌 ANOMALIES
     • A few outliers in Setosa sepal_width (unusually wide)
     • Versicolor & Virginica overlap in sepal dimensions

  📌 ML READINESS
     • Petal features are most discriminative for classification
     • No missing values — dataset is ML-ready
""")
print("=" * 65)
print("   ✅ Task 3 EDA Complete! All charts saved.")
print("=" * 65)
