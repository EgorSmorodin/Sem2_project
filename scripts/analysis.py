import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

file_path = '../data/data_clean.csv'
data = pd.read_csv(file_path, encoding='utf-8')

data['Цена(в $)'] = data['Цена(в $)'].str.replace(' ', '').astype(int)
data['Пробег'] = data['Пробег'].str.replace(' тыс.км', '').str.replace(' ', '').astype(float)

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Распределение цен
plt.figure()
sns.histplot(data['Цена(в $)'], bins=30, kde=True, color='skyblue')
plt.title('Распределение цен на автомобили', fontsize=16)
plt.xlabel('Цена ($)', fontsize=14)
plt.ylabel('Количество автомобилей', fontsize=14)
plt.axvline(data['Цена(в $)'].mean(), color='red', linestyle='--', label=f'Среднее: ${data["Цена(в $)"].mean():,.0f}')
plt.axvline(data['Цена(в $)'].median(), color='green', linestyle='--', label=f'Медиана: ${data["Цена(в $)"].median():,.0f}')
plt.legend()
plt.show()

# 2. Распределение пробега
plt.figure()
sns.histplot(data['Пробег'], bins=30, kde=True, color='salmon')
plt.title('Распределение пробега автомобилей', fontsize=16)
plt.xlabel('Пробег (тыс. км)', fontsize=14)
plt.ylabel('Количество автомобилей', fontsize=14)
plt.axvline(data['Пробег'].mean(), color='red', linestyle='--', label=f'Среднее: {data["Пробег"].mean():.1f} тыс. км')
plt.axvline(data['Пробег'].median(), color='green', linestyle='--', label=f'Медиана: {data["Пробег"].median():.1f} тыс. км')
plt.legend()
plt.show()

# 3. Распределение года выпуска
plt.figure()
sns.histplot(data['Год'], bins=30, kde=True, color='lightgreen')
plt.title('Распределение года выпуска автомобилей', fontsize=16)
plt.xlabel('Год выпуска', fontsize=14)
plt.ylabel('Количество автомобилей', fontsize=14)
plt.axvline(data['Год'].mean(), color='red', linestyle='--', label=f'Среднее: {data["Год"].mean():.0f}')
plt.axvline(data['Год'].median(), color='green', linestyle='--', label=f'Медиана: {data["Год"].median():.0f}')
plt.legend()
plt.show()

# 4. Boxplot цен по маркам (топ-10)
plt.figure(figsize=(14, 8))
top_10_brands = data['Марка'].value_counts().head(10).index
sns.boxplot(data=data[data['Марка'].isin(top_10_brands)], x='Марка', y='Цена(в $)', palette='Set3')
plt.title('Распределение цен по топ-10 маркам', fontsize=16)
plt.xlabel('Марка автомобиля', fontsize=14)
plt.ylabel('Цена ($)', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# 5. Диаграмма рассеивания: цена vs пробег с группировкой по году
plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(data=data, x='Пробег', y='Цена(в $)', hue='Год', palette='viridis', size='Год', sizes=(20, 200))
plt.title('Зависимость цены от пробега и года выпуска', fontsize=16)
plt.xlabel('Пробег (тыс. км)', fontsize=14)
plt.ylabel('Цена ($)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# 6. Корреляционный анализ
correlation_matrix = data[['Цена(в $)', 'Пробег', 'Год']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=.5)
plt.title('Матрица корреляции числовых признаков', fontsize=16)
plt.show()

# 7. Топ-5 марок по средней цене
top_5_expensive = data.groupby('Марка')['Цена(в $)'].mean().sort_values(ascending=False).head(6)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_5_expensive.index, y=top_5_expensive.values, palette='rocket')
plt.title('Топ-5 самых дорогих марок (средняя цена)', fontsize=16)
plt.xlabel('Марка', fontsize=14)
plt.ylabel('Средняя цена ($)', fontsize=14)
plt.xticks(rotation=45)
plt.show()
