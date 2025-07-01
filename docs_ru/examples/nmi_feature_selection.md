## Отбор признаков с помощью NMIFeatureSelector на наборе данных о качестве вина

Этот пример оценивает способность `NMIFeatureSelector` сохранять информативные признаки и отбрасывать шум, используя набор данных о качестве вина из UCI. Мы:
- Загрузим набор данных и предварительно обработаем его.
- Добавим синтетические шумовые признаки для имитации неинформативных атрибутов.
- Сравним производительность модели с отбором признаков и без него по мере увеличения шума.
- Визуализируем результаты.
## Шаг 1: Загрузка и предварительная обработка данных
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from bn_nmi_feature_selector import NMIFeatureSelector

# Загрузка набора данных о качестве вина (красное вино)
data = pd.read_csv('data/feature_selection/WineQT.csv')
data = data.drop(columns=['Id'])
X = data.drop("quality", axis=1)
y = data["quality"].values
```
## Шаг 2: Определение функции для добавления шума и оценки
```python
def evaluate_performance(X, y, noise_features=0, threshold=0.05):
    # Добавление случайных шумовых признаков
    np.random.seed(42)
    noise = np.random.randn(X.shape[0], noise_features)
    X_combined = np.hstack([X.values, noise])
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.3, random_state=42
    )
    
    # С отбором признаков
    selector = NMIFeatureSelector(threshold=threshold, n_bins=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Обучение модели на отобранных признаках
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    acc_selected = accuracy_score(y_test, y_pred)
    
    # Без отбора признаков
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred)
    
    return acc_full, acc_selected, selector.selected_features_.shape[0]
```

## Шаг 3: Запуск эксперимента с увеличением шума
```python
noise_levels = np.arange(0, 50, 10)  # Тестируем от 0 до 40 шумовых признаков
results = []

for noise in noise_levels:
    acc_full, acc_selected, n_selected = evaluate_performance(X, y, noise_features=noise)
    results.append({
        "Noise Features": noise,
        "Accuracy (Full)": acc_full,
        "Accuracy (Selected)": acc_selected,
        "Selected Features": n_selected
    })
    print(f"Noise: {noise:2d} | Full Acc: {acc_full:.3f} | Selected Acc: {acc_selected:.3f} | Features Kept: {n_selected}")

results_df = pd.DataFrame(results)
```
## Шаг 4: Визуализация результатов
```python
plt.figure(figsize=(10, 6))
plt.plot(results_df["Noise Features"], results_df["Accuracy (Full)"], 
         marker="o", label="Without Selection", color="red")
plt.plot(results_df["Noise Features"], results_df["Accuracy (Selected)"], 
         marker="o", label="With Selection", color="green")
plt.xlabel("Number of Noise Features Added")
plt.ylabel("Model Accuracy")
plt.title("Impact of Noise Features on Model Performance")
plt.legend()
plt.grid(True)
plt.show()
```
![output](https://github.com/user-attachments/assets/1023a03f-dcc0-4adf-804f-c0ae2e651268)

## Заключение
`NMIFeatureSelector` эффективно:
- Идентифицирует информативные признаки в реальном наборе данных.
- Фильтрует шум, предотвращая снижение производительности модели.
- Работает надежно, даже когда 80% признаков являются неинформативными (40 шумовых + 11 исходных).

Рекомендация:
Настройте параметр `threshold` для баланса между сохранением признаков и удалением шума. Например, более высокий порог (например, 0.1) может дополнительно уменьшить шум, но рискует исключить слабо информативные признаки.
