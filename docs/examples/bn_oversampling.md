# Пример: работа с сильным дисбалансом классов с помощью BNOverSampler и SMOTE

В этом примере показано, как работать с сильным дисбалансом классов с использованием двух методов ресемплинга:
**BNOverSampler** (метод на основе байесовских сетей) и **SMOTE**. Мы сравниваем их производительность на маркетинговом наборе данных с использованием метрик F1-score и AUC-ROC.

---

## Зависимости
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from applybn.imbalanced.over_sampling import BNOverSampler  # Пользовательский пакет
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
```

## Шаг 1: Загрузка данных и создание дисбаланса
Мы используем набор данных о банковском маркетинге и искусственно создаем соотношение классов 30:1 для имитации сильного дисбаланса:
```python
data = pd.read_csv('data/bn_oversampling/bank-marketing.csv', index_col=[0])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Создаем дисбаланс 5:1
_, X_min, _, y_min = train_test_split(
    X[y == 1], y[y == 1], 
    test_size=0.2, 
    random_state=42
)
X_imb = pd.concat([X[y == 0], X_min])
y_imb = pd.concat([y[y == 0], y_min])
```
## Шаг 2: Разделение на обучающую и тестовую выборки
Разделяем данные на 70% для обучения и 30% для тестирования, сохраняя стратификацию:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_imb, y_imb, 
    test_size=0.3, 
    stratify=y_imb,
    random_state=42
)
```
## Шаг 3: Инициализация ресемплеров
Настраиваем методы ресемплинга:
```python
bn_sampler = BNOverSampler(class_column='class', strategy='max_class')  # Балансирует до размера мажоритарного класса
smote = SMOTE(k_neighbors=5, random_state=42)
```
## Шаг 4: Ресемплинг обучающих данных
Применяем оба метода и измеряем время выполнения:
```python
# BNOverSampler
start_bn = time.time()
X_bn, y_bn = bn_sampler.fit_resample(X_train, y_train)
time_bn = time.time() - start_bn

# SMOTE
start_smote = time.time()
X_smote, y_smote = smote.fit_resample(X_train, y_train)
time_smote = time.time() - start_smote
```
## Шаг 5: Обучение и оценка моделей
Используем **RandomForestClassifier** для обоих наборов данных после ресемплинга:
```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_bn, y_bn)
bn_pred = clf.predict(X_test)
bn_proba = clf.predict_proba(X_test)[:, 1]

clf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
clf_smote.fit(X_smote, y_smote)
smote_pred = clf_smote.predict(X_test)
smote_proba = clf_smote.predict_proba(X_test)[:, 1]
```
## Шаг 6: Результаты
| Метод           | F1-Score | AUC-ROC | Время (с) |
|-----------------|----------|---------|-----------|
| BNOverSampler   | 0.575    | 0.858   | 35.1      |
| SMOTE           | 0.556    | 0.844   | 0.0       |

![pairplot_comparison](https://github.com/user-attachments/assets/fd08ef62-ee89-4f5f-a84a-f7174d350a6e)


Используйте BNOverSampler, когда критически важны интерпретируемость и сохранение распределения, а вычислительные ресурсы достаточны.
