## Comparative Analysis: Bayesian Network Feature Generation vs Traditional Methods

This example demonstrates the `BNFeatureGenerator`'s ability to improve model performance compared to traditional feature engineering methods like polynomial features and feature interactions.

## Step 1: Load and Preprocess Data

```python
# Load the Wilt dataset from local files
training_data = pd.read_csv('data/feature_extraction/wilt/training.csv')
testing_data = pd.read_csv('data/feature_extraction/wilt/testing.csv')

# Combine training and testing data for cross-validation
combined_data = pd.concat([training_data, testing_data], ignore_index=True)

# Separate features and target
X = combined_data.drop("class", axis=1) 
y = combined_data["class"]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[('scale', StandardScaler(), list(X.columns))],
    remainder='passthrough'
)
preprocessor.fit(X)
```

## Step 2: Define Feature Generation Methods
### 2.1: Original Features
```python
def generate_original_features(X_train, X_test, preprocessor):
    """Generate original features with scaling."""
    X_train_scaled = pd.DataFrame(
        preprocessor.transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=X_test.columns
    )
    return X_train_scaled, X_test_scaled
```
### 2.2: Polynomial Features
```python
def generate_polynomial_features(X_train, X_test, preprocessor):
    """Generate polynomial features (degree 2)."""
    # Preprocess data
    X_train_scaled, X_test_scaled = generate_original_features(X_train, X_test, preprocessor)
    
    # Generate polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(X_train_scaled)
    
    X_train_poly = poly.transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    feature_names = poly.get_feature_names_out(X_train_scaled.columns)
    X_train_features = pd.DataFrame(X_train_poly, columns=feature_names)
    X_test_features = pd.DataFrame(X_test_poly, columns=feature_names)
    
    return X_train_features, X_test_features
```
### 2.3: Interaction Features
```python
def generate_interaction_features(X_train, X_test, preprocessor):
    """Generate interaction features (pairwise multiplication)."""
    # Preprocess data
    X_train_scaled, X_test_scaled = generate_original_features(X_train, X_test, preprocessor)
    
    # Create copies to add interaction features
    X_train_interact = X_train_scaled.copy()
    X_test_interact = X_test_scaled.copy()
    
    # Generate pairwise interactions
    features = X_train_scaled.columns
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i < j:  # Avoid duplicates and self-interactions
                X_train_interact[f'interaction_{feat1}_{feat2}'] = X_train_scaled[feat1] * X_train_scaled[feat2]
                X_test_interact[f'interaction_{feat1}_{feat2}'] = X_test_scaled[feat1] * X_test_scaled[feat2]
    
    return X_train_interact, X_test_interact
```
### 2.4: Bayesian Network Features
```python
def generate_bayesian_network_features(X_train, X_test, y_train):
    """Generate features using Bayesian Network."""
    # Initialize BN generator
    bn_generator = BNFeatureGenerator()
    bn_generator.fit(X=X_train, y=y_train)
    
    # Transform data
    X_train_bn = bn_generator.transform(X_train).reset_index(drop=True)
    X_test_bn = bn_generator.transform(X_test).reset_index(drop=True)
    
    # Handle categorical features
    encoders = {}
    for col in X_train_bn.columns:
        if X_train_bn[col].dtype == 'object':
            encoders[col] = LabelEncoder()
            encoders[col].fit(X_train_bn[col].unique()) 
            X_train_bn[col] = encoders[col].transform(X_train_bn[col])
    
    for col in X_test_bn.columns:
        if col in encoders and X_test_bn[col].dtype == 'object':
            X_test_bn[col] = encoders[col].transform(X_test_bn[col])
    
    return X_train_bn, X_test_bn
```

## Step 3: Initialize Models and Evaluation Parameters

```python
# Define models to compare
models = {
    'Decision Tree': (DecisionTreeClassifier, {'random_state': 42}),
    'Logistic Regression': (LogisticRegression, {'random_state': 42, 'max_iter': 1000}),
    'SVC': (SVC, {'random_state': 42})
}

# Feature generation methods
feature_generators = {
    'Original': generate_original_features,
    'Polynomial': generate_polynomial_features,
    'Interaction': generate_interaction_features,
    'Bayesian Network': generate_bayesian_network_features
}

# Evaluation parameters
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Container for results
all_results = {}
```

## Step 4: Cross-Validation and Model Evaluation

```python
# Evaluate all combinations of models and feature types
for model_name, (model_class, model_params) in models.items():
    all_results[model_name] = {}
    
    for feature_type, feature_generator in feature_generators.items():
        accuracy_scores = []
        f1_scores = []
        feature_importance = None
        
        # Cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Generate features
            if feature_type == 'Bayesian Network':
                X_train_features, X_test_features = feature_generator(X_train, X_test, y_train)
            else:
                X_train_features, X_test_features = feature_generator(X_train, X_test, preprocessor)
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train_features, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_features)
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='macro'))
            
            # Store feature importance for first fold
            if fold_idx == 0 and hasattr(model, 'feature_importances_'):
                feature_importance = pd.Series(
                    model.feature_importances_,
                    index=X_train_features.columns
                )
        
        # Calculate average metrics
        all_results[model_name][feature_type] = {
            'accuracy': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'f1': np.mean(f1_scores),
            'f1_std': np.std(f1_scores)
        }
        
        if feature_importance is not None:
            all_results[model_name][feature_type]['importance'] = feature_importance

# Print detailed performance comparison
print("\n" + "="*80)
print("DETAILED PERFORMANCE COMPARISON")
print("="*80)

for feature_name in feature_generators.keys():
    print(f"\n{feature_name} Features:")
    print("-" * 60)
    
    for model_name in models:
        result = all_results[model_name][feature_name]
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {result['accuracy']:.4f} (±{result['accuracy_std']:.4f})")
        print(f"  F1 Score:  {result['f1']:.4f} (±{result['f1_std']:.4f})")
```

## Step 5: Visualization and Analysis

```python
# Find best model for each feature type
print("\n\n" + "="*80)
print("BEST MODELS SUMMARY")
print("="*80)

for feature_name in feature_generators.keys():
    best_model = max(models.keys(), 
                    key=lambda model: all_results[model][feature_name]['accuracy'])
    
    print(f"\nBest model with {feature_name} features: {best_model}")
    print(f"  Accuracy: {all_results[best_model][feature_name]['accuracy']:.4f} (±{all_results[best_model][feature_name]['accuracy_std']:.4f})")
    print(f"  F1 Score: {all_results[best_model][feature_name]['f1']:.4f} (±{all_results[best_model][feature_name]['f1_std']:.4f})")

# Find the overall best model and feature combination
best_feature = None
best_model = None
best_accuracy = 0

for model_name in models:
    for feature_name in feature_generators.keys():
        acc = all_results[model_name][feature_name]['accuracy']
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model_name
            best_feature = feature_name

# Create performance comparison chart
accuracies = []
f1_scores = []
model_names = []
feature_types = []

for model_name in models:
    for feature_type in feature_generators.keys():
        result = all_results[model_name][feature_type]
        accuracies.append(result['accuracy'])
        f1_scores.append(result['f1'])
        model_names.append(model_name)
        feature_types.append(feature_type)

results_df = pd.DataFrame({
    'Model': model_names,
    'Feature Type': feature_types,
    'Accuracy': accuracies,
    'F1 Score': f1_scores
})

plt.figure(figsize=(12, 8))
g = sns.barplot(x='Feature Type', y='Accuracy', hue='Model', data=results_df)
plt.title('Model Accuracy by Feature Generation Method')
plt.xlabel('Feature Generation Method')
plt.ylabel('Accuracy (5-fold CV)')
plt.ylim(0.9, 1.0)  
plt.xticks(rotation=45)
plt.legend(title='Model')
plt.tight_layout()
plt.show()
```

## Example Results

When running this code on the Wilt dataset, the Bayesian Network features consistently outperform traditional feature engineering methods, particularly when combined with SVC:

```
================================================================================
BEST MODELS SUMMARY
================================================================================

Best model with Original features: SVC
  Accuracy: 0.9806 (±0.0047)
  F1 Score: 0.8834 (±0.0390)

Best model with Polynomial features: Logistic Regression
  Accuracy: 0.9795 (±0.0052)
  F1 Score: 0.8798 (±0.0349)

Best model with Interaction features: Decision Tree
  Accuracy: 0.9760 (±0.0024)
  F1 Score: 0.8822 (±0.0170)

Best model with Bayesian Network features: SVC
  Accuracy: 0.9833 (±0.0016)
  F1 Score: 0.9168 (±0.0124)
```
![Figure_111](https://github.com/user-attachments/assets/d3689591-0e05-432a-b00d-1ac256b4e75f)

```python
# Visualize feature importance for the best model
print("\n\n" + "="*80)
print("FEATURE IMPORTANCE VISUALIZATION")
print("="*80)

print(f"\nBest overall model: {best_model} with {best_feature} features")
print(f"Accuracy: {all_results[best_model][best_feature]['accuracy']:.4f} (±{all_results[best_model][best_feature]['accuracy_std']:.4f})")

if 'importance' in all_results[best_model][best_feature]:
    importance = all_results[best_model][best_feature]['importance']
    top_n = min(20, len(importance))
    
    plt.figure(figsize=(12, 8))
    top_importance = importance.sort_values(ascending=False).head(top_n)
    sns.barplot(x=top_importance.values, y=top_importance.index)
    plt.title(f'Top {top_n} Feature Importance - {best_model} with {best_feature} features')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
```
![image](https://github.com/user-attachments/assets/8a9da7ab-9753-4e4b-b38e-1ceb3f1530e6)

## Conclusion

This example demonstrates the effectiveness of Bayesian Network feature generation compared to traditional feature engineering methods. The BNFeatureGenerator captures complex dependencies in the data that are not easily represented by polynomial or interaction features. 
