"""
═══════════════════════════════════════════════════════════════════════════════
                    ENTERPRISE ML MODEL ANALYTICS PLATFORM
                        Professional Production System
═══════════════════════════════════════════════════════════════════════════════
Features:
- Advanced model training with hyperparameter optimization
- Real-time performance tracking
- Interactive data visualization
- Model persistence and caching
- Comprehensive error handling
- Professional logging system
- API documentation
- Security features
═══════════════════════════════════════════════════════════════════════════════
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_fscore_support, matthews_corrcoef)
import io
import base64
import time
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ml-analytics-enterprise-2025'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_analytics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Matplotlib styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL DATA STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

trained_models = {}
training_data = {}
dataset_stats = {}
training_history = []

# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED MODEL TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class MLPipeline:
    """Advanced ML Pipeline for model training and evaluation"""
    
    def __init__(self, dataset_path='student-mat.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.le_target = None
        self.scaler = None
        
    def load_and_preprocess(self):
        """Load and preprocess the dataset"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        try:
            self.df = pd.read_csv(self.dataset_path, delimiter=';')
            logger.info(f"Dataset loaded: {len(self.df)} records, {self.df.shape[1]} features")
            
            # Create performance categories
            self.df['Performance'] = self.df['G3'].apply(
                lambda g: 'Low' if g < 10 else ('Medium' if g < 14 else 'High')
            )
            
            # Store statistics
            global dataset_stats
            dataset_stats = {
                'total_students': len(self.df),
                'total_features': self.df.shape[1],
                'grade_stats': {
                    'min': float(self.df['G3'].min()),
                    'max': float(self.df['G3'].max()),
                    'mean': float(self.df['G3'].mean()),
                    'median': float(self.df['G3'].median()),
                    'std': float(self.df['G3'].std())
                },
                'performance_distribution': self.df['Performance'].value_counts().to_dict(),
                'features': self.df.columns.tolist()
            }
            
            # Encode categorical variables
            df_encoded = self.df.copy()
            categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 
                                'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 
                                'activities', 'nursery', 'higher', 'internet', 'romantic']
            
            for col in categorical_cols:
                if col in df_encoded.columns:
                    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
            
            self.le_target = LabelEncoder()
            df_encoded['Performance_encoded'] = self.le_target.fit_transform(df_encoded['Performance'])
            
            # Prepare features
            X = df_encoded.drop(['G3', 'Performance', 'Performance_encoded'], axis=1)
            y = df_encoded['Performance_encoded']
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            global training_data
            training_data = {
                'X': X,
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'X_scaled': X_scaled,
                'y': y,
                'le_target': self.le_target,
                'scaler': self.scaler,
                'train_size': len(self.y_train),
                'test_size': len(self.y_test)
            }
            
            logger.info(f"Preprocessing complete: Train={len(self.y_train)}, Test={len(self.y_test)}")
            return True
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return False
    
    def train_model(self, model_name, model, param_grid):
        """Train a single model with GridSearchCV"""
        logger.info(f"Training {model_name}...")
        start_time = time.time()
        
        try:
            # GridSearchCV with extensive cross-validation
            grid = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0,
                return_train_score=True
            )
            
            grid.fit(self.X_train, self.y_train)
            best_model = grid.best_estimator_
            training_time = time.time() - start_time
            
            # Predictions
            train_pred = best_model.predict(self.X_train)
            test_pred = best_model.predict(self.X_test)
            test_proba = best_model.predict_proba(self.X_test) if hasattr(best_model, 'predict_proba') else None
            
            # Calculate comprehensive metrics
            train_acc = accuracy_score(self.y_train, train_pred)
            test_acc = accuracy_score(self.y_test, test_pred)
            precision = precision_score(self.y_test, test_pred, average='weighted')
            recall = recall_score(self.y_test, test_pred, average='weighted')
            f1 = f1_score(self.y_test, test_pred, average='weighted')
            mcc = matthews_corrcoef(self.y_test, test_pred)
            
            # Cross-validation scores
            cv_scores = cross_val_score(best_model, training_data['X_scaled'], training_data['y'], cv=5, scoring='accuracy')
            
            # Per-class metrics
            per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
                self.y_test, test_pred
            )
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, test_pred)
            
            # Feature importance (if available)
            if hasattr(best_model, 'feature_importances_'):
                feature_imp = pd.DataFrame({
                    'feature': training_data['X'].columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False).head(15)
            else:
                feature_imp = None
            
            # Learning curve data
            train_sizes, train_scores, val_scores = learning_curve(
                best_model, self.X_train, self.y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
            )
            
            # Store model data
            model_data = {
                'model': best_model,
                'model_name': model_name,
                'best_params': grid.best_params_,
                'train_acc': float(train_acc),
                'test_acc': float(test_acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'mcc': float(mcc),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'cv_scores': cv_scores.tolist(),
                'training_time': float(training_time),
                'predictions': test_pred.tolist(),
                'probabilities': test_proba.tolist() if test_proba is not None else None,
                'confusion_matrix': cm.tolist(),
                'per_class_precision': per_class_precision.tolist(),
                'per_class_recall': per_class_recall.tolist(),
                'per_class_f1': per_class_f1.tolist(),
                'support': support.tolist(),
                'feature_importance': feature_imp.to_dict('records') if feature_imp is not None else None,
                'overfitting_gap': float(train_acc - test_acc),
                'total_correct': int(np.trace(cm)),
                'total_test': len(self.y_test),
                'learning_curve': {
                    'train_sizes': train_sizes.tolist(),
                    'train_scores_mean': train_scores.mean(axis=1).tolist(),
                    'val_scores_mean': val_scores.mean(axis=1).tolist()
                },
                'trained_at': datetime.now().isoformat()
            }
            
            # Log training history
            training_history.append({
                'model': model_name,
                'accuracy': test_acc,
                'training_time': training_time,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"✓ {model_name}: {test_acc*100:.2f}% accuracy in {training_time:.2f}s")
            
            return model_data
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            return None

def train_all_models():
    """Train all ML models"""
    global trained_models
    
    logger.info("="*80)
    logger.info("STARTING COMPREHENSIVE MODEL TRAINING PIPELINE")
    logger.info("="*80)
    
    # Initialize pipeline
    pipeline = MLPipeline()
    
    if not pipeline.load_and_preprocess():
        logger.error("Failed to load and preprocess data")
        return False
    
    # Model configurations
    models_config = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [8, 10, 12, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [5, 8, 10, 12, 15, 20],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random']
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        }
    }
    
    # Train each model
    trained_models = {}
    for name, config in models_config.items():
        model_data = pipeline.train_model(name, config['model'], config['params'])
        if model_data:
            trained_models[name] = model_data
    
    logger.info("="*80)
    logger.info(f"TRAINING COMPLETE: {len(trained_models)} models trained successfully")
    logger.info("="*80)
    
    return True

# ═══════════════════════════════════════════════════════════════════════════════
# CHART GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str

def generate_individual_charts(model_name):
    """Generate comprehensive charts for individual model"""
    model_data = trained_models[model_name]
    y_test = training_data['y_test']
    le_target = training_data['le_target']
    X = training_data['X']
    
    charts = {}
    
    # Chart 1: Enhanced Confusion Matrix
    fig1, ax = plt.subplots(figsize=(12, 10))
    cm = np.array(model_data['confusion_matrix'])
    
    # Create percentage annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.array([[f'{count}\n({pct:.1f}%)' for count, pct in zip(row_count, row_pct)] 
                      for row_count, row_pct in zip(cm, cm_percent)])
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='RdYlGn', ax=ax,
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_,
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 14, 'weight': 'bold'},
                linewidths=2, linecolor='white')
    
    ax.set_title(f'{model_name} - Confusion Matrix\nAccuracy: {model_data["test_acc"]*100:.2f}% | Correct: {model_data["total_correct"]}/{model_data["total_test"]}', 
                fontweight='bold', fontsize=18, pad=20)
    ax.set_xlabel('Predicted Class', fontweight='bold', fontsize=14)
    ax.set_ylabel('True Class', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    charts['confusion'] = fig_to_base64(fig1)
    
    # Chart 2: Comprehensive Metrics Dashboard
    fig2 = plt.figure(figsize=(16, 10))
    gs = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Metrics bar chart
    ax1 = fig2.add_subplot(gs[0, :])
    metrics = {
        'Accuracy': model_data['test_acc'],
        'Precision': model_data['precision'],
        'Recall': model_data['recall'],
        'F1-Score': model_data['f1'],
        'MCC': (model_data['mcc'] + 1) / 2,  # Normalize to 0-1
        'CV Score': model_data['cv_mean']
    }
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    bars = ax1.bar(metrics.keys(), [v*100 for v in metrics.values()], 
                  color=colors, edgecolor='black', linewidth=2.5, alpha=0.8)
    
    ax1.set_ylabel('Score (%)', fontweight='bold', fontsize=14)
    ax1.set_title(f'{model_name} - Performance Metrics Overview', fontweight='bold', fontsize=16)
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% Threshold')
    ax1.legend()
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Per-class performance
    ax2 = fig2.add_subplot(gs[1, 0])
    x = np.arange(3)
    width = 0.25
    
    ax2.bar(x - width, model_data['per_class_precision'], width, 
           label='Precision', color='skyblue', edgecolor='black', linewidth=1.5)
    ax2.bar(x, model_data['per_class_recall'], width, 
           label='Recall', color='lightgreen', edgecolor='black', linewidth=1.5)
    ax2.bar(x + width, model_data['per_class_f1'], width, 
           label='F1-Score', color='salmon', edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Class', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax2.set_title('Per-Class Performance', fontweight='bold', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(le_target.classes_)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1.1])
    ax2.grid(axis='y', alpha=0.3)
    
    # Training vs Test Accuracy
    ax3 = fig2.add_subplot(gs[1, 1])
    comparison = ['Training', 'Testing']
    accuracies = [model_data['train_acc']*100, model_data['test_acc']*100]
    colors_comp = ['#3498db', '#e74c3c']
    
    bars_comp = ax3.bar(comparison, accuracies, color=colors_comp, 
                       edgecolor='black', linewidth=2, alpha=0.8)
    ax3.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax3.set_title('Training vs Testing Performance', fontweight='bold', fontsize=14)
    ax3.set_ylim([0, 100])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars_comp:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add overfitting gap annotation
    gap = model_data['overfitting_gap'] * 100
    ax3.text(0.5, 50, f'Gap: {gap:.2f}%', transform=ax3.transAxes,
            ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    charts['metrics'] = fig_to_base64(fig2)
    
    # Chart 3: Feature Importance
    if model_data['feature_importance']:
        fig3, ax = plt.subplots(figsize=(14, 10))
        features = pd.DataFrame(model_data['feature_importance'])
        
        colors_feat = plt.cm.viridis(np.linspace(0, 1, len(features)))
        bars_feat = ax.barh(range(len(features)), features['importance'], 
                           color=colors_feat, edgecolor='black', linewidth=1.5)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features['feature'], fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontweight='bold', fontsize=14)
        ax.set_title(f'{model_name} - Top {len(features)} Feature Importance', 
                    fontweight='bold', fontsize=16)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, row in features.iterrows():
            ax.text(row['importance'] + 0.001, i, f" {row['importance']:.4f} ({row['importance']*100:.2f}%)", 
                   va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        charts['features'] = fig_to_base64(fig3)
    else:
        charts['features'] = None
    
    # Chart 4: Learning Curve
    if 'learning_curve' in model_data:
        fig4, ax = plt.subplots(figsize=(12, 8))
        lc = model_data['learning_curve']
        
        ax.plot(lc['train_sizes'], [s*100 for s in lc['train_scores_mean']], 
               'o-', color='blue', label='Training Score', linewidth=2, markersize=8)
        ax.plot(lc['train_sizes'], [s*100 for s in lc['val_scores_mean']], 
               'o-', color='red', label='Validation Score', linewidth=2, markersize=8)
        
        ax.fill_between(lc['train_sizes'], 
                       [(s-0.05)*100 for s in lc['train_scores_mean']],
                       [(s+0.05)*100 for s in lc['train_scores_mean']],
                       alpha=0.2, color='blue')
        
        ax.set_xlabel('Training Set Size', fontweight='bold', fontsize=14)
        ax.set_ylabel('Score (%)', fontweight='bold', fontsize=14)
        ax.set_title(f'{model_name} - Learning Curve', fontweight='bold', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        charts['learning_curve'] = fig_to_base64(fig4)
    else:
        charts['learning_curve'] = None
    
    return charts

def generate_comparison_charts(model1_name, model2_name):
    """Generate comprehensive comparison charts"""
    model1 = trained_models[model1_name]
    model2 = trained_models[model2_name]
    le_target = training_data['le_target']
    
    charts = {}
    
    # Chart 1: Side-by-side Confusion Matrices
    fig1, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    for idx, (name, model) in enumerate([(model1_name, model1), (model2_name, model2)]):
        cm = np.array(model['confusion_matrix'])
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annot = np.array([[f'{count}\n({pct:.1f}%)' for count, pct in zip(row_count, row_pct)] 
                          for row_count, row_pct in zip(cm, cm_percent)])
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='RdYlGn', ax=axes[idx],
                    xticklabels=le_target.classes_,
                    yticklabels=le_target.classes_,
                    annot_kws={'size': 12, 'weight': 'bold'},
                    linewidths=2, linecolor='white')
        
        axes[idx].set_title(f'{name}\nAcc: {model["test_acc"]*100:.2f}% | {model["total_correct"]}/{model["total_test"]} correct', 
                          fontweight='bold', fontsize=14)
        axes[idx].set_xlabel('Predicted', fontweight='bold', fontsize=12)
        axes[idx].set_ylabel('Actual', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    charts['confusion_compare'] = fig_to_base64(fig1)
    
    # Chart 2: Comprehensive Metrics Comparison
    fig2, ax = plt.subplots(figsize=(16, 8))
    
    metrics = ['test_acc', 'precision', 'recall', 'f1', 'cv_mean']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    model1_values = [model1[m]*100 for m in metrics]
    model2_values = [model2[m]*100 for m in metrics]
    
    bars1 = ax.bar(x - width/2, model1_values, width, label=model1_name, 
                   color='#3498db', edgecolor='black', linewidth=2.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, model2_values, width, label=model2_name, 
                   color='#e74c3c', edgecolor='black', linewidth=2.5, alpha=0.8)
    
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=14)
    ax.set_title(f'Head-to-Head Metrics Comparison: {model1_name} vs {model2_name}', 
                fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15, ha='right')
    ax.legend(fontsize=13, loc='upper left')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    charts['metrics_compare'] = fig_to_base64(fig2)
    
    # Chart 3: Per-Class Comparison
    fig3, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for idx, (name, model) in enumerate([(model1_name, model1), (model2_name, model2)]):
        x = np.arange(3)
        width = 0.25
        
        axes[idx].bar(x - width, model['per_class_precision'], width, 
                     label='Precision', color='skyblue', edgecolor='black', linewidth=2)
        axes[idx].bar(x, model['per_class_recall'], width, 
                     label='Recall', color='lightgreen', edgecolor='black', linewidth=2)
        axes[idx].bar(x + width, model['per_class_f1'], width, 
                     label='F1-Score', color='salmon', edgecolor='black', linewidth=2)
        
        axes[idx].set_xlabel('Class', fontweight='bold', fontsize=12)
        axes[idx].set_title(f'{name} - Per-Class Performance', fontweight='bold', fontsize=14)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(le_target.classes_)
        axes[idx].legend(fontsize=11)
        axes[idx].set_ylim([0, 1.1])
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    charts['perclass_compare'] = fig_to_base64(fig3)
    
    # Chart 4: Feature Importance Comparison
    if model1['feature_importance'] and model2['feature_importance']:
        fig4, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        for idx, (name, model) in enumerate([(model1_name, model1), (model2_name, model2)]):
            features = pd.DataFrame(model['feature_importance']).head(10)
            colors_feat = plt.cm.viridis(np.linspace(0, 1, len(features)))
            
            axes[idx].barh(range(len(features)), features['importance'], 
                          color=colors_feat, edgecolor='black', linewidth=1.5)
            axes[idx].set_yticks(range(len(features)))
            axes[idx].set_yticklabels(features['feature'], fontsize=11)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Importance', fontweight='bold', fontsize=12)
            axes[idx].set_title(f'{name} - Top 10 Features', fontweight='bold', fontsize=14)
            axes[idx].grid(axis='x', alpha=0.3)
            
            for i, row in features.iterrows():
                axes[idx].text(row['importance'], i, f" {row['importance']:.3f}", 
                             va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        charts['features_compare'] = fig_to_base64(fig4)
    else:
        charts['features_compare'] = None
    
    # Chart 5: Training Time and Overfitting Analysis
    fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Training time
    times = [model1['training_time'], model2['training_time']]
    bars_time = ax1.bar([model1_name, model2_name], times, 
                       color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=2.5, width=0.5)
    ax1.set_ylabel('Time (seconds)', fontweight='bold', fontsize=13)
    ax1.set_title('Training Time Comparison', fontweight='bold', fontsize=15)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars_time:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Overfitting gap
    gaps = [model1['overfitting_gap']*100, model2['overfitting_gap']*100]
    colors_gap = ['#2ecc71' if g < 10 else '#f39c12' if g < 20 else '#e74c3c' for g in gaps]
    bars_gap = ax2.bar([model1_name, model2_name], gaps, 
                      color=colors_gap, edgecolor='black', linewidth=2.5, width=0.5)
    ax2.set_ylabel('Gap (%)', fontweight='bold', fontsize=13)
    ax2.set_title('Overfitting Analysis (Train-Test Gap)', fontweight='bold', fontsize=15)
    ax2.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Good (<10%)', alpha=0.7)
    ax2.axhline(y=20, color='orange', linestyle='--', linewidth=2, label='Warning (>20%)', alpha=0.7)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars_gap:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    charts['time_overfit'] = fig_to_base64(fig5)
    
    return charts

# ═══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES - API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        with open('templates/professional_interface.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Return embedded HTML if file not found
        return render_template_string(EMBEDDED_HTML)

@app.route('/api/models')
def get_models():
    """Get all trained models summary"""
    try:
        if not trained_models:
            logger.info("No models found, initiating training...")
            train_all_models()
        
        summary = {}
        for name, model in trained_models.items():
            summary[name] = {
                'test_acc': model['test_acc'],
                'precision': model['precision'],
                'recall': model['recall'],
                'f1': model['f1'],
                'cv_mean': model['cv_mean'],
                'training_time': model['training_time'],
                'total_correct': model['total_correct'],
                'total_test': model['total_test']
            }
        
        return jsonify(summary)
    
    except Exception as e:
        logger.error(f"Error in get_models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset_stats')
def get_dataset_stats():
    """Get dataset statistics"""
    try:
        if not dataset_stats:
            train_all_models()
        return jsonify(dataset_stats)
    except Exception as e:
        logger.error(f"Error in get_dataset_stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/<model_name>')
def get_model_details(model_name):
    """Get detailed model information with charts"""
    try:
        if not trained_models:
            train_all_models()
        
        if model_name not in trained_models:
            return jsonify({'error': 'Model not found'}), 404
        
        model_data = trained_models[model_name]
        charts = generate_individual_charts(model_name)
        
        response_data = {
            'model': model_name,
            'metrics': {
                'train_acc': model_data['train_acc'],
                'test_acc': model_data['test_acc'],
                'precision': model_data['precision'],
                'recall': model_data['recall'],
                'f1': model_data['f1'],
                'mcc': model_data['mcc'],
                'cv_mean': model_data['cv_mean'],
                'cv_std': model_data['cv_std'],
                'training_time': model_data['training_time'],
                'overfitting_gap': model_data['overfitting_gap'],
                'total_correct': model_data['total_correct'],
                'total_test': model_data['total_test']
            },
            'best_params': model_data['best_params'],
            'confusion_matrix': model_data['confusion_matrix'],
            'correct_predictions': {
                'High': int(model_data['confusion_matrix'][0][0]),
                'Low': int(model_data['confusion_matrix'][1][1]),
                'Medium': int(model_data['confusion_matrix'][2][2])
            },
            'per_class_metrics': {
                'precision': model_data['per_class_precision'],
                'recall': model_data['per_class_recall'],
                'f1': model_data['per_class_f1'],
                'support': model_data['support']
            },
            'feature_importance': model_data['feature_importance'],
            'charts': charts,
            'trained_at': model_data['trained_at']
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in get_model_details: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare/<model1>/<model2>')
def compare_models_api(model1, model2):
    """Compare two models"""
    try:
        if not trained_models:
            train_all_models()
        
        if model1 not in trained_models or model2 not in trained_models:
            return jsonify({'error': 'One or both models not found'}), 404
        
        charts = generate_comparison_charts(model1, model2)
        
        response_data = {
            'model1': model1,
            'model2': model2,
            'model1_metrics': {
                'test_acc': trained_models[model1]['test_acc'],
                'precision': trained_models[model1]['precision'],
                'recall': trained_models[model1]['recall'],
                'f1': trained_models[model1]['f1'],
                'cv_mean': trained_models[model1]['cv_mean'],
                'training_time': trained_models[model1]['training_time'],
                'total_correct': trained_models[model1]['total_correct']
            },
            'model2_metrics': {
                'test_acc': trained_models[model2]['test_acc'],
                'precision': trained_models[model2]['precision'],
                'recall': trained_models[model2]['recall'],
                'f1': trained_models[model2]['f1'],
                'cv_mean': trained_models[model2]['cv_mean'],
                'training_time': trained_models[model2]['training_time'],
                'total_correct': trained_models[model2]['total_correct']
            },
            'charts': charts
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in compare_models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training_history')
def get_training_history():
    """Get training history"""
    try:
        return jsonify(training_history)
    except Exception as e:
        logger.error(f"Error in get_training_history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_trained': len(trained_models),
        'timestamp': datetime.now().isoformat()
    })

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═"*80)
    print(" " * 20 + "ENTERPRISE ML MODEL ANALYTICS PLATFORM")
    print(" " * 30 + "Professional Edition")
    print("═"*80)
    print("\n📊 Features:")
    print("  ✓ Advanced hyperparameter optimization")
    print("  ✓ Comprehensive model evaluation")
    print("  ✓ Interactive visualizations")
    print("  ✓ Real-time performance tracking")
    print("  ✓ Professional logging system")
    print("\n🚀 Initializing system...")
    
    # Create required directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Train models on startup
    logger.info("Starting initial model training...")
    train_all_models()
    
    print("\n✓ System ready!")
    print("\n" + "─"*80)
    print("🌐 Web Server: http://127.0.0.1:5000")
    print("📖 API Docs: http://127.0.0.1:5000/api/health")
    print("─"*80 + "\n")
    
    # Run Flask app
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
