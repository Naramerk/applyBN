## Анализ робастности: каузальный отбор признаков с добавлением неинформативных признаков
Этот пример демонстрирует способность `CausalFeatureSelector` поддерживать производительность модели даже при добавлении нерелевантных признаков второго порядка, используя набор данных о вине.
## Шаг 1: Загрузка и предварительная обработка данных
```python
data = pd.read_csv('data/feature_selection/WineQT.csv')
data = data.drop(columns=['Id'])
X = data.drop("quality", axis=1)
y = data["quality"].values
feature_names = list(data.columns)[:-1]
```
## Шаг 2: Добавление полиномиальных признаков (второго порядка) и перемешивание для создания неинформативных признаков
```python
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
np.random.seed(42)
X_poly = np.random.permutation(X_poly)  # Уничтожение связи с целевой переменной

# Объединение исходных и синтетических признаков
X_combined = np.hstack([X, X_poly])
print(f"Всего признаков: {X_combined.shape[1]} (30 исходных + {X_poly.shape[1]} синтетических)")
```
## Шаг 3: Инициализация контейнеров для результатов
```python
n_uninformative_steps = np.arange(0, 200, 20)
accuracies_full = []
accuracies_selected = []

# Инициализация моделей
selector = CausalFeatureSelector(n_bins=5)
model = LogisticRegression(max_iter=1000, random_state=42)
```
## Шаг 4: Постепенное добавление признаков
```python
for n in n_uninformative_steps:
    # Выбор подмножества неинформативных признаков
    X_current = np.hstack([X, X_poly[:, :n]])
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_current, y, test_size=0.3, random_state=42
    )
    
    # С отбором признаков
    selector.fit(X_train, y_train)
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)
    
    model.fit(X_train_sel, y_train)
    acc_selected = accuracy_score(y_test, model.predict(X_test_sel))
    
    # Без отбора признаков
    model.fit(X_train, y_train)
    acc_full = accuracy_score(y_test, model.predict(X_test))
    
    # Сохранение результатов
    accuracies_full.append(acc_full)
    accuracies_selected.append(acc_selected)
    
    print(f"Добавлено {n} шумовых признаков | "
          f"Отобрано: {X_train_sel.shape[1]} признаков | "
          f"Точность: {acc_selected:.3f} (отобранные) vs {acc_full:.3f} (полные)")
```
## Шаг 5: Визуализация и анализ последней итерации
```python
plt.figure(figsize=(10, 6))
plt.plot(n_uninformative_steps, accuracies_full, 'r--', label='Полный набор признаков')
plt.plot(n_uninformative_steps, accuracies_selected, 'g-', label='Отобранные признаки')
plt.xlabel('Количество добавленных неинформативных признаков')
plt.ylabel('Точность на тесте')
plt.title('Робастность модели к нерелевантным признакам')
plt.legend()
plt.grid(True)
plt.show()

true_features = set(range(30))
selected_features = set(np.where(selector.support_)[0])

print(f"\nАнализ окончательного отбора:")
print(f"- Всего отобрано: {len(selected_features)} признаков")
print(f"- Сохранено истинных признаков: {len(true_features & selected_features)}/{30}")
print(f"- Отобрано шумовых признаков: {len(selected_features - true_features)}")
```
![ce_fs](https://github.com/user-attachments/assets/55f85792-306d-4beb-8d5c-395b841a0f1c)

## Заключение
Этот пример демонстрирует робастность метода в реальных сценариях, где наборы данных часто содержат много нерелевантных или сконструированных признаков. Механизм каузального отбора эффективно выявляет стабильные предиктивные взаимосвязи, несмотря на возрастающее шумовое загрязнение.
