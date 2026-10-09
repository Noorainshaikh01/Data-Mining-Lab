"""
Advanced ML Model Comparison - Interactive Web Application
View individual models or compare any 2 models side-by-side
Next-level features with detailed analysis
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import io
import base64
import time
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Global storage
trained_models = {}
training_data = {}

def train_all_models():
    """Train 3 models with comprehensive metrics"""
    global trained_models, training_data
    
    print("Loading dataset...")
    df = pd.read_csv('student-mat.csv', delimiter=';')
    
    # Preprocess
    df['Performance'] = df['G3'].apply(lambda g: 'Low' if g < 10 else ('Medium' if g < 14 else 'High'))
    
    df_encoded = df.copy()
    categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 
                        'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 
                        'activities', 'nursery', 'higher', 'internet', 'romantic']
    
    for col in categorical_cols:
        df_encoded[col] = LabelEncoder().fit_transform(df[col])
    
    le_target = LabelEncoder()
    df_encoded['Performance_encoded'] = le_target.fit_transform(df_encoded['Performance'])
    
    X = df_encoded.drop(['G3', 'Performance', 'Performance_encoded'], axis=1)
    y = df_encoded['Performance_encoded']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    training_data = {
        'X': X,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_scaled': X_scaled,
        'y': y,
        'le_target': le_target
    }
    
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
    
    for name, config in models_config.items():
        print(f"Training {name}...")
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
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred, average='weighted')
        recall = recall_score(y_test, test_pred, average='weighted')
        f1 = f1_score(y_test, test_pred, average='weighted')
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            y_test, test_pred
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
        else:
            feature_imp = None
        
        # Store everything
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
            'feature_importance': feature_imp.to_dict('records') if feature_imp is not None else None,
            'overfitting_gap': train_acc - test_acc
        }
        
        print(f"  ✓ {name}: {test_acc*100:.2f}% accuracy in {training_time:.2f}s")
    
    print("Training complete!")
    return trained_models

def generate_individual_charts(model_name):
    """Generate comprehensive charts for a single model"""
    model_data = trained_models[model_name]
    y_test = training_data['y_test']
    le_target = training_data['le_target']
    
    charts = {}
    
    # Chart 1: Confusion Matrix
    fig1, ax = plt.subplots(figsize=(8, 6))
    cm = np.array(model_data['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_,
                cbar_kws={'label': 'Count'})
    ax.set_title(f'{model_name} - Confusion Matrix\nAccuracy: {model_data["test_acc"]*100:.2f}%', 
                fontweight='bold', fontsize=14)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    plt.tight_layout()
    charts['confusion'] = fig_to_base64(fig1)
    plt.close()
    
    # Chart 2: Metrics Breakdown
    fig2, ax = plt.subplots(figsize=(10, 6))
    metrics = {
        'Accuracy': model_data['test_acc'],
        'Precision': model_data['precision'],
        'Recall': model_data['recall'],
        'F1-Score': model_data['f1'],
        'CV Score': model_data['cv_mean']
    }
    bars = ax.bar(metrics.keys(), [v*100 for v in metrics.values()], 
                  color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
                  edgecolor='black', linewidth=2)
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    ax.set_title(f'{model_name} - Performance Metrics', fontweight='bold', fontsize=14)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    charts['metrics'] = fig_to_base64(fig2)
    plt.close()
    
    # Chart 3: Per-Class Performance
    fig3, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    width = 0.25
    
    ax.bar(x - width, model_data['per_class_precision'], width, label='Precision', color='skyblue', edgecolor='black')
    ax.bar(x, model_data['per_class_recall'], width, label='Recall', color='lightgreen', edgecolor='black')
    ax.bar(x + width, model_data['per_class_f1'], width, label='F1-Score', color='salmon', edgecolor='black')
    
    ax.set_xlabel('Class', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title(f'{model_name} - Per-Class Performance', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(le_target.classes_)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    charts['perclass'] = fig_to_base64(fig3)
    plt.close()
    
    # Chart 4: Feature Importance (if available)
    if model_data['feature_importance']:
        fig4, ax = plt.subplots(figsize=(10, 8))
        features = pd.DataFrame(model_data['feature_importance'])
        
        ax.barh(range(len(features)), features['importance'], color='skyblue', edgecolor='black')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontweight='bold', fontsize=12)
        ax.set_title(f'{model_name} - Top 15 Features', fontweight='bold', fontsize=14)
        
        for i, row in features.iterrows():
            ax.text(row['importance'], i, f" {row['importance']:.4f}", 
                   va='center', fontweight='bold')
        
        plt.tight_layout()
        charts['features'] = fig_to_base64(fig4)
        plt.close()
    else:
        charts['features'] = None
    
    return charts

def generate_comparison_charts(model1_name, model2_name):
    """Generate side-by-side comparison charts"""
    model1 = trained_models[model1_name]
    model2 = trained_models[model2_name]
    le_target = training_data['le_target']
    
    charts = {}
    
    # Chart 1: Confusion Matrices Comparison
    fig1, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (name, model) in enumerate([(model1_name, model1), (model2_name, model2)]):
        cm = np.array(model['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=le_target.classes_,
                    yticklabels=le_target.classes_)
        axes[idx].set_title(f'{name}\nAccuracy: {model["test_acc"]*100:.2f}%', 
                          fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('Predicted', fontweight='bold')
        axes[idx].set_ylabel('Actual', fontweight='bold')
    
    plt.tight_layout()
    charts['confusion_compare'] = fig_to_base64(fig1)
    plt.close()
    
    # Chart 2: Metrics Comparison
    fig2, ax = plt.subplots(figsize=(12, 6))
    metrics = ['test_acc', 'precision', 'recall', 'f1', 'cv_mean']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    model1_values = [model1[m]*100 for m in metrics]
    model2_values = [model2[m]*100 for m in metrics]
    
    bars1 = ax.bar(x - width/2, model1_values, width, label=model1_name, 
                   color='skyblue', edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, model2_values, width, label=model2_name, 
                   color='lightcoral', edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('Metrics Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15)
    ax.legend()
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    charts['metrics_compare'] = fig_to_base64(fig2)
    plt.close()
    
    # Chart 3: Per-Class Comparison
    fig3, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (name, model) in enumerate([(model1_name, model1), (model2_name, model2)]):
        x = np.arange(3)
        width = 0.25
        
        axes[idx].bar(x - width, model['per_class_precision'], width, 
                     label='Precision', color='skyblue', edgecolor='black')
        axes[idx].bar(x, model['per_class_recall'], width, 
                     label='Recall', color='lightgreen', edgecolor='black')
        axes[idx].bar(x + width, model['per_class_f1'], width, 
                     label='F1-Score', color='salmon', edgecolor='black')
        
        axes[idx].set_xlabel('Class', fontweight='bold')
        axes[idx].set_title(f'{name} - Per-Class', fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(le_target.classes_)
        axes[idx].legend()
        axes[idx].set_ylim([0, 1.1])
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    charts['perclass_compare'] = fig_to_base64(fig3)
    plt.close()
    
    # Chart 4: Feature Importance Comparison
    if model1['feature_importance'] and model2['feature_importance']:
        fig4, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        for idx, (name, model) in enumerate([(model1_name, model1), (model2_name, model2)]):
            features = pd.DataFrame(model['feature_importance']).head(10)
            
            axes[idx].barh(range(len(features)), features['importance'], 
                          color='skyblue', edgecolor='black')
            axes[idx].set_yticks(range(len(features)))
            axes[idx].set_yticklabels(features['feature'])
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Importance', fontweight='bold')
            axes[idx].set_title(f'{name} - Top 10 Features', fontweight='bold')
            
            for i, row in features.iterrows():
                axes[idx].text(row['importance'], i, f" {row['importance']:.3f}", 
                             va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        charts['features_compare'] = fig_to_base64(fig4)
        plt.close()
    else:
        charts['features_compare'] = None
    
    # Chart 5: Training Time and Overfitting
    fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training time
    times = [model1['training_time'], model2['training_time']]
    bars = ax1.bar([model1_name, model2_name], times, 
                   color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
    ax1.set_ylabel('Time (seconds)', fontweight='bold')
    ax1.set_title('Training Time Comparison', fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Overfitting gap
    gaps = [model1['overfitting_gap']*100, model2['overfitting_gap']*100]
    bars2 = ax2.bar([model1_name, model2_name], gaps, 
                    color=['#2ecc71', '#f39c12'], edgecolor='black', linewidth=2)
    ax2.set_ylabel('Gap (%)', fontweight='bold')
    ax2.set_title('Overfitting Analysis (Train-Test Gap)', fontweight='bold')
    ax2.axhline(y=10, color='red', linestyle='--', label='10% threshold')
    ax2.legend()
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    charts['time_overfit'] = fig_to_base64(fig5)
    plt.close()
    
    return charts

def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

@app.route('/')
def index():
    return render_template('advanced_interface.html')

@app.route('/api/models')
def get_models():
    """Get list of trained models with summary"""
    if not trained_models:
        train_all_models()
    
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
def get_model_details(model_name):
    """Get detailed info and charts for one model"""
    if not trained_models:
        train_all_models()
    
    if model_name not in trained_models:
        return jsonify({'error': 'Model not found'}), 404
    
    model_data = trained_models[model_name]
    charts = generate_individual_charts(model_name)
    
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
def compare_models(model1, model2):
    """Compare two models"""
    if not trained_models:
        train_all_models()
    
    if model1 not in trained_models or model2 not in trained_models:
        return jsonify({'error': 'One or both models not found'}), 404
    
    charts = generate_comparison_charts(model1, model2)
    
    return jsonify({
        'model1': model1,
        'model2': model2,
        'model1_metrics': {
            'test_acc': trained_models[model1]['test_acc'],
            'precision': trained_models[model1]['precision'],
            'recall': trained_models[model1]['recall'],
            'f1': trained_models[model1]['f1'],
            'cv_mean': trained_models[model1]['cv_mean'],
            'training_time': trained_models[model1]['training_time']
        },
        'model2_metrics': {
            'test_acc': trained_models[model2]['test_acc'],
            'precision': trained_models[model2]['precision'],
            'recall': trained_models[model2]['recall'],
            'f1': trained_models[model2]['f1'],
            'cv_mean': trained_models[model2]['cv_mean'],
            'training_time': trained_models[model2]['training_time']
        },
        'charts': charts
    })

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Advanced ML Model Comparison - Web Application")
    print("=" * 80)
    print("\nTraining models...")
    
    train_all_models()
    
    print("\n✓ Training complete!")
    print("\nStarting web server...")
    print("\n" + "─" * 80)
    print("Open browser: http://127.0.0.1:5000")
    print("─" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
