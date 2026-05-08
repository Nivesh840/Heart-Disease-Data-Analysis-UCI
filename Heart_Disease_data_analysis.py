import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Real Heart Disease Dataset loading from UCI (via URL)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df_real = pd.read_csv(url, names=names, na_values='?')

# 2. Cleaning: Real data mein hamesha 'NaN' (khali jagah) hoti hai
df_real = df_real.dropna()

# 3. Analysis: Cholesterol vs Heart Disease Risk
# Target 0 = No Disease, 1-4 = Risk present
df_real['Risk'] = df_real['target'].apply(lambda x: 'High Risk' if x > 0 else 'Healthy')

plt.figure(figsize=(10,6))
sns.violinplot(x='Risk', y='chol', data=df_real, palette='coolwarm')
plt.title('Real Data Analysis: Cholesterol Levels vs Heart Disease Risk')
plt.ylabel('Cholesterol (mg/dl)')
plt.show()

print(f"Bhai, humne {len(df_real)} real patients ka data analyze kiya hai!")