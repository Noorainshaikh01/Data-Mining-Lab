from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import io
import base64
import time
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Globals to hold dataset, models, and metadata
uploaded_df = None
trained_models = {}
training_data = {}


def preprocess_dataframe(df, target_col):
    df = df.copy()
    # Encode categorical columns dynamically
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)  # exclude target if categorical
    le_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_encoders[col] = le

    # Encode target variable if categorical
    target_le = None
    y = df[target_col]
    if y.dtype == object or y.nunique() < 20:
        target_le = LabelEncoder()
        y = target_le.fit_transform(y.astype(str))
    
    X = df.drop(columns=[target_col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, y, X_scaled, le_encoders, target_le, scaler


def train_models_with_data(X, y, X_scaled):
    models_config = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [8, 10, None], 'min_samples_split': [2, 5]}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'max_depth': [5, 8, 10, 15], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy']}
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {'C': [1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'poly']}
        }
    }

    global trained_models
    trained_models.clear()

    # Stratified split if classification, else simple split
    stratify_param = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    for name, config in models_config.items():
        start_time = time.time()

        grid = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        training_time = time.time() - start_time

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            y_test, test_pred, zero_division=0
        )

        cm = confusion_matrix(y_test, test_pred)

        # Feature importance (only if available)
        if hasattr(model, 'feature_importances_'):
            feature_imp_df = pd.DataFrame({
                'feature': [f'Feature_{i}' for i in range(X.shape[1])],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
        else:
            feature_imp_df = None

        trained_models[name] = {
            'model': model,
            'best_params': grid.best_params_,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'training_time': training_time,
            'predictions': test_pred.tolist(),
            'probabilities': test_proba.tolist() if test_proba is not None else None,
            'confusion_matrix': cm.tolist(),
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'per_class_f1': per_class_f1.tolist(),
            'support': support.tolist(),
            'feature_importance': feature_imp_df.to_dict('records') if feature_imp_df is not None else None,
            'overfitting_gap': train_acc - test_acc
        }
    
    return trained_models, X_train, X_test, y_train, y_test


def generate_charts_for_model(model_name):
    model_data = trained_models.get(model_name)
    if not model_data:
        return None
    
    # For flexible class names, generate a range of class labels
    num_classes = len(model_data['per_class_f1'])
    classes_labels = [f'Class {i}' for i in range(num_classes)]

    charts = {}

    # Confusion Matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(np.array(model_data['confusion_matrix']), annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=classes_labels,
                yticklabels=classes_labels,
                cbar_kws={'label': 'Count'})
    ax.set_title(f'{model_name} - Confusion Matrix\nAccuracy: {model_data["test_acc"]*100:.2f}%', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    charts['confusion_matrix'] = fig_to_base64(fig)
    plt.close()

    # Metrics bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = {
        'Accuracy': model_data['test_acc'],
        'Precision': model_data['precision'],
        'Recall': model_data['recall'],
        'F1-Score': model_data['f1'],
        'CV Score': model_data['cv_mean']
    }
    bars = ax.bar(metrics.keys(), [v*100 for v in metrics.values()],
                  color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
                  edgecolor='black')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Score (%)')
    ax.set_title(f'{model_name} - Performance Metrics', fontweight='bold')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}%', ha='center', va='bottom')
    plt.tight_layout()
    charts['metrics'] = fig_to_base64(fig)
    plt.close()

    # Per-Class Performance chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(num_classes)
    width = 0.25
    ax.bar(x - width, model_data['per_class_precision'], width, label='Precision', color='skyblue', edgecolor='black')
    ax.bar(x, model_data['per_class_recall'], width, label='Recall', color='lightgreen', edgecolor='black')
    ax.bar(x + width, model_data['per_class_f1'], width, label='F1-Score', color='salmon', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(classes_labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_xlabel('Class')
    ax.set_title(f'{model_name} - Per-Class Performance', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    charts['per_class_performance'] = fig_to_base64(fig)
    plt.close()

    # Feature Importance if available
    if model_data['feature_importance']:
        fig, ax = plt.subplots(figsize=(10, 8))
        features = pd.DataFrame(model_data['feature_importance'])
        ax.barh(range(len(features)), features['importance'], color='skyblue', edgecolor='black')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'{model_name} - Top Feature Importances', fontweight='bold')
        for i, row in features.iterrows():
            ax.text(row['importance'], i, f"{row['importance']:.4f}", va='center')
        plt.tight_layout()
        charts['feature_importance'] = fig_to_base64(fig)
        plt.close()
    else:
        charts['feature_importance'] = None

    return charts


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@app.route('/')
def home():
    return render_template('advanced_interface.html')


@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    global uploaded_df, trained_models, training_data
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        uploaded_df = pd.read_csv(file)
        trained_models.clear()
        training_data.clear()
        return jsonify({'message': 'Dataset uploaded successfully', 'columns': uploaded_df.columns.tolist()})
    except Exception as e:
        return jsonify({'error': f'Failed to parse CSV: {str(e)}'}), 400


@app.route('/api/train', methods=['POST'])
def train():
    global uploaded_df, trained_models, training_data
    if uploaded_df is None:
        return jsonify({'error': 'No dataset uploaded'}), 400
    req = request.json
    target_col = req.get('target_column')
    if not target_col or target_col not in uploaded_df.columns:
        return jsonify({'error': 'Invalid or missing target column'}), 400

    X, y, X_scaled, le_encoders, target_le, scaler = preprocess_dataframe(uploaded_df, target_col)
    trained_models, X_train, X_test, y_train, y_test = train_models_with_data(X, y, X_scaled)

    # Save training parts to global training_data for chart functions
    training_data['X_train'] = X_train
    training_data['X_test'] = X_test
    training_data['y_train'] = y_train
    training_data['y_test'] = y_test
    training_data['le_target'] = target_le if target_le else None
    training_data['target_classes'] = target_le.classes_ if target_le else sorted(np.unique(y))

    return jsonify({'message': 'Models trained', 'models': list(trained_models.keys())})


@app.route('/api/models')
def list_models():
    if not trained_models:
        return jsonify({'error': 'No models trained yet'}), 400
    summary = {}
    for name, model in trained_models.items():
        summary[name] = {
            'test_acc': model['test_acc'],
            'precision': model['precision'],
            'recall': model['recall'],
            'f1': model['f1'],
            'cv_mean': model['cv_mean'],
            'training_time': model['training_time']
        }
    return jsonify(summary)


@app.route('/api/model/<model_name>')
def model_details(model_name):
    if model_name not in trained_models:
        return jsonify({'error': 'Model not found'}), 404
    charts = generate_charts_for_model(model_name)
    model_data = trained_models[model_name]
    return jsonify({
        'model': model_name,
        'metrics': {
            'train_acc': model_data['train_acc'],
            'test_acc': model_data['test_acc'],
            'precision': model_data['precision'],
            'recall': model_data['recall'],
            'f1': model_data['f1'],
            'cv_mean': model_data['cv_mean'],
            'cv_std': model_data['cv_std'],
            'training_time': model_data['training_time'],
            'overfitting_gap': model_data['overfitting_gap']
        },
        'best_params': model_data['best_params'],
        'confusion_matrix': model_data['confusion_matrix'],
        'per_class_metrics': {
            'precision': model_data['per_class_precision'],
            'recall': model_data['per_class_recall'],
            'f1': model_data['per_class_f1'],
            'support': model_data['support']
        },
        'feature_importance': model_data['feature_importance'],
        'charts': charts
    })


@app.route('/api/compare/<model1>/<model2>')
def compare_model_metrics(model1, model2):
    if model1 not in trained_models or model2 not in trained_models:
        return jsonify({'error': 'One or both models not found'}), 404

    # Build comparison charts similarly to above, omitted for brevity
    # You can adapt your existing generate_comparison_charts function here

    # For now, provide metrics comparison JSON
    m1 = trained_models[model1]
    m2 = trained_models[model2]

    comp_metrics = ['test_acc', 'precision', 'recall', 'f1', 'cv_mean', 'training_time']

    comparison = {
        'model1': model1,
        'model2': model2,
        'model1_metrics': {metric: m1[metric] for metric in comp_metrics},
        'model2_metrics': {metric: m2[metric] for metric in comp_metrics}
    }

    return jsonify(comparison)


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Advanced ML Model Comparison - Web Application (Dynamic Dataset)")
    print("=" * 80)
    print("Start the Flask server and upload a CSV dataset via the UI or /api/upload_dataset endpoint.")
    print("Select the target column and train models dynamically.\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
