# 📊 Task 3: Exploratory Data Analysis (EDA)

### Edutech Solution – AI & ML Internship

---

## 🎯 Objective

Develop the ability to **extract meaningful stories from raw data** through comprehensive Exploratory Data Analysis — visualizing distributions, relationships, correlations, and anomalies in the Iris dataset.

---

## 📁 Repository Structure

```
├── iris.csv                        # Dataset (150 rows × 5 columns)
├── eda_analysis.py                 # Full Python EDA script
├── Task3_EDA_Iris.ipynb            # Jupyter Notebook (all 8 steps)
├── 01_univariate_analysis.png      # Histograms + Boxplots
├── 02_bivariate_scatter.png        # Scatter plots
├── 03_bivariate_bar.png            # Bar charts (mean per species)
├── 04_correlation_heatmap.png      # Correlation heatmaps
├── 05_pairplot.png                 # Pairplot (all features)
├── 06_patterns_trends.png          # Violin, CDF, Area scatter, Pie
├── 07_outlier_detection.png        # Outlier boxplots
└── README.md                       # This file
```

---

## 🗂️ Dataset: Iris

| Feature | Type | Description |
|---------|------|-------------|
| sepal_length | float64 | Sepal length in cm |
| sepal_width | float64 | Sepal width in cm |
| petal_length | float64 | Petal length in cm |
| petal_width | float64 | Petal width in cm |
| species | object | setosa / versicolor / virginica |

- **150 rows** — 50 samples each for 3 species
- **0 missing values**

---

## ✅ Tasks Completed

| # | Analysis Type | Method | Status |
|---|--------------|--------|--------|
| 1 | Univariate Analysis | Histograms + Boxplots | ✅ |
| 2 | Bivariate Analysis | Scatter plots + Bar charts | ✅ |
| 3 | Correlation Matrix | Seaborn Heatmap | ✅ |
| 4 | Patterns & Trends | CDF, Violin, Area Scatter | ✅ |
| 5 | Pairplot | All numerical variables | ✅ |
| 6 | Outlier Detection | IQR Boxplots | ✅ |
| 7 | Key Insights | Summary report | ✅ |

---

## 📊 Key Insights

| Finding | Detail |
|---------|--------|
| Best separating features | Petal Length & Petal Width |
| Strongest correlation | petal_length ↔ petal_width = **0.963** |
| Most distinct species | Setosa — clearly linearly separable |
| Overlap region | Versicolor & Virginica in sepal dimensions |
| Negative correlation | sepal_width ↔ petal_length = **-0.428** |
| Outliers | Setosa has a few sepal_width outliers |
| ML readiness | No missing values — ready for modeling |

### Correlation Summary
- **petal_length ↔ petal_width: 0.963** (very strong positive)
- **sepal_length ↔ petal_length: 0.872** (strong positive)
- **sepal_length ↔ petal_width: 0.818** (strong positive)
- **sepal_width ↔ petal_length: -0.428** (moderate negative)

---

## 🛠️ Tools & Technologies

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **IDE:** Jupyter Notebook / VS Code

---

## ▶️ How to Run

```bash
git clone https://github.com/vishnubabalsure/Task3-EDA-Iris.git
cd Task3-EDA-Iris
pip install pandas numpy matplotlib seaborn jupyter
python eda_analysis.py
# OR
jupyter notebook Task3_EDA_Iris.ipynb
```

---

## 💡 Interview Q&A

**Q1. Univariate vs Bivariate vs Multivariate analysis?**  
Univariate = 1 variable (histogram, boxplot). Bivariate = 2 variables (scatter, bar). Multivariate = 3+ variables (pairplot, heatmap, 3D scatter).

**Q2. How to interpret a Correlation Heatmap?**  
Values near +1 = strong positive, near -1 = strong negative, near 0 = no relationship. Green/red shading shows the strength and direction.

**Q3. What is the purpose of a Boxplot?**  
Shows the median (line), IQR (box), range (whiskers), and outliers (dots) — compact summary of a distribution.

**Q4. How to identify outliers visually?**  
Boxplot: dots beyond whiskers. Scatter plot: isolated points far from the cluster. Histogram: extreme bars at tails.

**Q5. Why is EDA the most important step before modeling?**  
EDA reveals missing values, skewed distributions, multicollinearity, and class imbalance — guiding feature engineering, model choice, and preprocessing. Skipping EDA leads to poor and unreliable models.

---

*Submitted as part of Edutech Solution AI & ML Internship – Task 3*
