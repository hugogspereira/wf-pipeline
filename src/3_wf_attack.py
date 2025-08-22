import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

################################################################################
# Constants
MODELS_FOLDER = "./../data/models/"
RESULTS_FOLDER = "./../data/results/"
DATA_PATH = "./../data/features_dataset.csv"

# Create necessary directories
for folder in [MODELS_FOLDER, RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")


################################################################################
def select_model(model_name, n_components_for_pca):
    """Select model with optimized hyperparameter grid, scaling only non-tree models"""
    if model_name == 'RandomForest':
        param_grid = {
            'n_estimators': [100, 250, 500],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced'],
            'random_state': [42]
        }
        model = RandomForestClassifier()
    elif model_name == 'GradientBoosting':
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', GradientBoostingClassifier())
        ])
        param_grid = {
            'model__n_estimators': [100, 250],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [3, 5],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2],
            'model__subsample': [0.8, 1.0],
            'model__random_state': [42]
        }
    elif model_name == 'DecisionTree':
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': [None, 'sqrt'],
            'random_state': [42]
        }
        model = DecisionTreeClassifier()
    elif model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [200, 500],
            'learning_rate': [0.1, 0.2],
            'max_depth': [6, 10],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0.1, 0.5],
            'random_state': [42]
        }
        model = xgb.XGBClassifier()
    elif model_name == 'ExtraTrees':
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', ExtraTreesClassifier())
        ])
        param_grid = {
            'model__n_estimators': [100, 500],
            'model__max_depth': [None, 10],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2],
            'model__max_features': ['sqrt'],
            'model__bootstrap': [True, False],
            'model__class_weight': [None, 'balanced'],
            'model__random_state': [42]
        }
        
    # Non-tree models: wrap in Pipeline with StandardScaler
    elif model_name == 'SVM':
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components_for_pca)),
            ('model', SVC(probability=True))
        ])
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__gamma': ['scale', 'auto'],
            'model__kernel': ['linear', 'rbf'],
            'model__class_weight': [None, 'balanced']
        }
    elif model_name == 'KNN':
        param_grid = {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['minkowski', 'euclidean'],
            'model__algorithm': ['auto', 'ball_tree']
        }
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components_for_pca)),
            ('model', KNeighborsClassifier())
        ])
    elif model_name == 'LogisticRegression':
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__penalty': ['l2'],
            'model__solver': ['lbfgs', 'liblinear'],
            'model__class_weight': [None, 'balanced'],
            'model__max_iter': [500, 1000]
        }
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', SimpleImputer(strategy='median')),
            ('model', LogisticRegression())
        ])
    elif model_name == 'NaiveBayes':
        param_grid = {
            'model__var_smoothing': [1e-09, 1e-08, 1e-07]
        }
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', SimpleImputer(strategy='median')),
            ('model', GaussianNB())
        ])
    else:
        raise ValueError("Unsupported model type")

    return GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)


################################################################################
def load_model(model_name, models_directory):
    """Load a saved model if it exists"""
    model_filename = os.path.join(MODELS_FOLDER, f"{models_directory}_{model_name}.pkl")
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
        return model
    return None

################################################################################
def save_model(model, model_name, models_directory):
    """Save a trained model"""
    model_filename = os.path.join(MODELS_FOLDER, f"{models_directory}_{model_name}.pkl")
    joblib.dump(model, model_filename)

################################################################################
def load_and_prepare_data(file_path, test_size=0.2, random_state=42):
    """Load and prepare the training data with proper per-website train/test split"""
    df = pd.read_csv(file_path)
    
    # Separate features and labels
    X = df.drop(columns=['website'])
    y = df['website']
    
    print(f"Total data shape: {df.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Unique websites: {len(y.unique())}")
    print(f"Samples per website:\n{Counter(y)}")
    
    # CRUCIAL: Split per website to ensure each website has samples in both train and test
    train_indices = []
    test_indices = []
    
    for website in y.unique():
        # Get all indices for this website
        website_indices = df[df['website'] == website].index.tolist()
        
        # Split this website's samples
        if len(website_indices) < 2:
            print(f"Warning: Website '{website}' has only {len(website_indices)} sample(s). Adding to training set.")
            train_indices.extend(website_indices)
        else:
            # Calculate split sizes
            n_test = max(1, int(len(website_indices) * test_size))  # At least 1 sample for test
            n_train = len(website_indices) - n_test
            
            # Randomly split indices for this website
            np.random.seed(random_state)
            shuffled_indices = np.random.permutation(website_indices)
            
            train_indices.extend(shuffled_indices[:n_train])
            test_indices.extend(shuffled_indices[n_train:])
            
            print(f"  {website}: {n_train} train, {n_test} test samples")
    
    # Create train and test sets using the indices
    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)
    
    print(f"\nFinal split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training websites: {len(y_train.unique())}")
    print(f"Test websites: {len(y_test.unique())}")
    
    # Verify all websites are in both sets
    train_websites = set(y_train.unique())
    test_websites = set(y_test.unique())
    
    if train_websites != test_websites:
        missing_in_test = train_websites - test_websites
        missing_in_train = test_websites - train_websites
        if missing_in_test:
            print(f"WARNING: Websites missing in test set: {missing_in_test}")
        if missing_in_train:
            print(f"WARNING: Websites missing in train set: {missing_in_train}")
    else:
        print("All websites present in both training and test sets")
    
    return X_train, X_test, y_train, y_test

################################################################################
def encode_labels(y_train, y_test):
    """Encode string labels to integers"""
    label_encoder = LabelEncoder()
    
    # Fit on training data only
    y_train_encoded = label_encoder.fit_transform(y_train)
    # Transform test data using the same encoder
    y_test_encoded = label_encoder.transform(y_test)
    
    return y_train_encoded, y_test_encoded, label_encoder

################################################################################
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, models_directory):
    """Train and evaluate a single model with proper train/test methodology"""
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL: {model_name}")
    print(f"{'='*60}")
    
    # Check if model already exists
    model = load_model(model_name, models_directory)
    
    if model is None:
        print(f"Training new {model_name} model...")
        
        model = select_model(model_name, min(50, X_train.shape[1]))
        model.fit(X_train, y_train)
        save_model(model, model_name, models_directory)
        
        # Print best parameters if it's a GridSearchCV
        if hasattr(model, 'best_params_'):
            print(f"Best parameters: {model.best_params_}")
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'best_params_'):
            print(f"Best parameters: {model.named_steps['model'].best_params_}")
    else:
        print(f"Using existing {model_name} model...")
    
    # CROSS-VALIDATION ON TRAINING DATA ONLY
    print("Performing cross-validation on training data...")
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Get the actual model (handle GridSearchCV and Pipeline)
    eval_model = model
    if hasattr(model, 'best_estimator_'):
        eval_model = model.best_estimator_
    
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_results = cross_validate(
        eval_model, X_train, y_train,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calculate and display CV results
    cv_metrics = {}
    print("\nCROSS-VALIDATION RESULTS (Training Data):")
    for metric in scoring:
        test_mean = cv_results[f'test_{metric}'].mean()
        test_std = cv_results[f'test_{metric}'].std()
        train_mean = cv_results[f'train_{metric}'].mean()
        train_std = cv_results[f'train_{metric}'].std()
        
        cv_metrics[metric] = {
            'test_mean': test_mean,
            'test_std': test_std,
            'train_mean': train_mean,
            'train_std': train_std
        }
        
        print(f"{metric.upper()}:")
        print(f"  CV Test:  {test_mean:.4f} (+/- {test_std * 2:.4f})")
        print(f"  CV Train: {train_mean:.4f} (+/- {train_std * 2:.4f})")
    
    # FINAL EVALUATION ON HELD-OUT TEST SET
    print("\nFINAL EVALUATION ON TEST SET:")
    y_test_pred = model.predict(X_test)
    
    # Handle multi-dimensional predictions
    if len(y_test_pred.shape) > 1 and y_test_pred.shape[1] > 1:
        y_test_pred = np.argmax(y_test_pred, axis=1)
    
    # Calculate test set metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)
    test_kappa = cohen_kappa_score(y_test, y_test_pred)
    
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test MCC: {test_mcc:.4f}")
    print(f"  Test Kappa: {test_kappa:.4f}")
    
    # Detailed classification report
    print(f"\nDETAILED CLASSIFICATION REPORT (Test Set):")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    print(classification_report(y_test, y_test_pred))
    
    # Store all results
    results = {
        'model_name': model_name,
        'cv_metrics': cv_metrics,
        'test_accuracy': test_accuracy,
        'test_mcc': test_mcc,
        'test_kappa': test_kappa,
        'test_precision_macro': report['macro avg']['precision'],
        'test_recall_macro': report['macro avg']['recall'],
        'test_f1_macro': report['macro avg']['f1-score'],
        'classification_report': report
    }
    
    return model, results

################################################################################
def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Plot feature importance for tree-based models"""
    # Get the actual model (handle GridSearchCV and Pipeline)
    actual_model = model
    if hasattr(model, 'best_estimator_'):
        actual_model = model.best_estimator_
    if hasattr(actual_model, 'named_steps') and 'model' in actual_model.named_steps:
        actual_model = actual_model.named_steps['model']
        if hasattr(actual_model, 'best_estimator_'):
            actual_model = actual_model.best_estimator_
    
    if hasattr(actual_model, 'feature_importances_'):
        print(f"\nGenerating feature importance plot for {model_name}...")
        
        importance = actual_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(top_n)
        
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features - {model_name}')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        plt.savefig(os.path.join(RESULTS_FOLDER, f'feature_importance_{model_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        feature_importance.to_csv(os.path.join(RESULTS_FOLDER, f'feature_importance_{model_name}.csv'), 
                                 index=False)
        print(f"Feature importance saved for {model_name}")
        
        return feature_importance
    else:
        print(f"Model {model_name} does not have feature importances.")
        return None

################################################################################
def compare_models(all_results):
    """Compare all models and identify the best performer"""
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    comparison_df = []
    
    for result in all_results:
        comparison_df.append({
            'Model': result['model_name'],
            'CV_Accuracy': f"{result['cv_metrics']['accuracy']['test_mean']:.4f} ± {result['cv_metrics']['accuracy']['test_std']:.4f}",
            'Test_Accuracy': f"{result['test_accuracy']:.4f}",
            'Test_Precision': f"{result['test_precision_macro']:.4f}",
            'Test_Recall': f"{result['test_recall_macro']:.4f}",
            'Test_F1': f"{result['test_f1_macro']:.4f}",
            'Test_MCC': f"{result['test_mcc']:.4f}",
            'Test_Kappa': f"{result['test_kappa']:.4f}",
            'Test_Accuracy_Value': result['test_accuracy']  # For sorting
        })
    
    comparison_df = pd.DataFrame(comparison_df)
    comparison_df = comparison_df.sort_values('Test_Accuracy_Value', ascending=False)
    
    print(comparison_df.drop('Test_Accuracy_Value', axis=1).to_string(index=False))
    
    # Save comparison
    comparison_df.drop('Test_Accuracy_Value', axis=1).to_csv(
        os.path.join(RESULTS_FOLDER, 'model_comparison.csv'), index=False
    )
    
    # Identify best model
    best_model = comparison_df.iloc[0]['Model']
    best_accuracy = comparison_df.iloc[0]['Test_Accuracy_Value']
    
    print(f"\nBEST MODEL: {best_model}")
    print(f"BEST TEST ACCURACY: {best_accuracy:.4f}")
    
    return best_model, comparison_df

################################################################################
def plot_model_comparison(comparison_df):
    """Plot model comparison"""
    print("\nGenerating model comparison plot...")
    
    plt.figure(figsize=(14, 8))
    
    models = comparison_df['Model']
    accuracies = comparison_df['Test_Accuracy_Value']
    
    bars = plt.bar(models, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Model Performance Comparison - Test Set Accuracy', fontsize=16)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Test Set Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULTS_FOLDER, 'model_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()

################################################################################
def main():
    """Main execution function"""
    print("="*80)
    print("COMPREHENSIVE WEBSITE FINGERPRINTING ATTACK")
    print("="*80)
    
    # Load and prepare data with proper train/test split
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        DATA_PATH, test_size=0.2, random_state=42
    )
    
    # Encode labels
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)
    
    # Define models to test
    models_directory = "wf_models"
    model_names = [
        'GradientBoosting', 'DecisionTree', 'RandomForest', 'XGBoost', 'ExtraTrees', 'LogisticRegression', 'NaiveBayes', 'KNN', 'SVM'
    ]
    
    print(f"\nTesting {len(model_names)} different models...")
    
    # Train and evaluate all models
    all_results = []
    trained_models = {}
    
    for model_name in model_names:
        model, results = train_and_evaluate_model(
            X_train, X_test, y_train_encoded, y_test_encoded, 
            model_name, models_directory
        )
        all_results.append(results)
        trained_models[model_name] = model
        
        # Plot feature importance for tree-based models
        if model_name in ['RandomForest', 'XGBoost', 'GradientBoosting', 'ExtraTrees', 'DecisionTree']:
            plot_feature_importance(model, X_train.columns, model_name)
    
    # Compare all models
    best_model_name, comparison_df = compare_models(all_results)
    
    # Plot comparison
    plot_model_comparison(comparison_df)
    
    # Save comprehensive results
    with open(os.path.join(RESULTS_FOLDER, 'comprehensive_results.txt'), 'w') as f:
        f.write("Website Fingerprinting - Comprehensive Model Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features\n")
        f.write(f"Test set: {X_test.shape[0]} samples\n")
        f.write(f"Websites: {len(label_encoder.classes_)}\n")
        f.write(f"Best Model: {best_model_name}\n\n")
        
        f.write("Test Set Results:\n")
        f.write("-" * 40 + "\n")
        for result in all_results:
            f.write(f"{result['model_name']}:\n")
            f.write(f"  Test Accuracy: {result['test_accuracy']:.4f}\n")
            f.write(f"  Test F1-Score: {result['test_f1_macro']:.4f}\n")
            f.write(f"  Test MCC: {result['test_mcc']:.4f}\n\n")
    
    print(f"\nCOMPREHENSIVE EVALUATION COMPLETED!")
    print(f"All results saved to: {RESULTS_FOLDER}")
    print(f"Best performing model: {best_model_name}")

################################################################################
if __name__ == "__main__":
    main()