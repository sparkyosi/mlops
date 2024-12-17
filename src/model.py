import os
import mlflow
import dagshub
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from mlflow.models.signature import infer_signature
from itertools import cycle
from imblearn.over_sampling import SMOTE
import warnings

# Initialize DagsHub and MLflow integration
dagshub.init(repo_owner='asmaa.hnaien', repo_name='mlops', mlflow=True)

mlflow.set_experiment("MLOps")

# Set the tracking URI for MLflow to log the experiment in DagsHub
mlflow.set_tracking_uri("https://dagshub.com/asmaa.hnaien/mlops.mlflow") 

def load_data(train_path: str, test_path: str):
    """Load train and test datasets"""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Add debug prints
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Train columns: {train_data.columns.tolist()}")
        
        # Verify 'score' column exists
        if 'score' not in train_data.columns or 'score' not in test_data.columns:
            raise ValueError("'score' column not found in the dataset")
            
        X_train = train_data.drop(columns=['score'])  # Removed axis parameter
        y_train = train_data['score']
        X_test = test_data.drop(columns=['score'])    # Removed axis parameter
        y_test = test_data['score']
        
        # Add more debug prints
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise Exception(f"Error in data loading: {str(e)}")

def scale_data(train_data: pd.DataFrame, test_data: pd.DataFrame,models_dir) -> tuple:
    """Scale training and test data using StandardScaler"""
    try:
        print(f"Input train_data shape: {train_data.shape}")
        print(f"Input test_data shape: {test_data.shape}")
        
        if not isinstance(train_data, pd.DataFrame) or not isinstance(test_data, pd.DataFrame):
            raise ValueError("Input data must be pandas DataFrames")
            
        scaler = StandardScaler()
        
        # Convert to numpy array, scale, and convert back to DataFrame
        train_scaled = pd.DataFrame(
            scaler.fit_transform(train_data),
            columns=train_data.columns,
            index=train_data.index
        )
        test_scaled = pd.DataFrame(
            scaler.transform(test_data),
            columns=test_data.columns,
            index=test_data.index
        )
        
        print(f"Output train_scaled shape: {train_scaled.shape}")
        print(f"Output test_scaled shape: {test_scaled.shape}")

         # Save scaler to models directory
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        with open(scaler_path, "wb") as file:
            pickle.dump(scaler, file)
        print(f"Scaler saved to {scaler_path}")
        
        return train_scaled, test_scaled
    except Exception as e:
        raise Exception(f"Error during scaling: {str(e)}")

def create_models():
    """Create dictionary of models with their parameters"""
    try:
        models = {
            'DT':{
                'model': DecisionTreeClassifier(),
                'params': {
                    'criterion':'gini'
                }
            }
            ,
            'XGBoost': {
                'model': XGBClassifier(),
                'params': {
                    'n_estimators': 10,
                    'max_depth': 50,
                    'learning_rate': 0.01,
                    'random_state': 42
               }
            },
            'RandomForest': {
                'model': RandomForestClassifier(),
                'params': {
                    'n_estimators': 10,
                    'max_depth': 50,
                    "bootstrap":False, 
                    "max_features":'sqrt',
                    'min_samples_split': 2,
                    'random_state': 42
                }
            }
        }
        return models
    except Exception as e:
        raise Exception(f"Error creating models: {str(e)}")


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curves(y_test, y_proba, class_names, output_path):
    """Plot ROC curves for all classes."""
    plt.figure(figsize=(10, 8))
    n_classes = len(class_names)
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_model(model, X_test, y_test, class_names, output_dir):
    """Evaluate model and log confusion matrix, ROC curves."""
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr')
        }

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(cm, class_names, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

        # ROC Curves
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plot_roc_curves(y_test, y_proba, class_names, roc_path)
        mlflow.log_artifact(roc_path, artifact_path="roc_curves")

        return metrics
    except Exception as e:
        raise Exception(f"Error during model evaluation: {str(e)}")

def train_model(X_train, y_train, model, params, sampling='none'):
    """Train a model with optional SMOTE sampling"""
    try:
        print(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
        print(f"Sampling method: {sampling}")
        
        if sampling == 'smote':
            smote = SMOTE(random_state=42)
            X_train_sample, y_train_sample = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - X: {X_train_sample.shape}, y: {y_train_sample.shape}")
        else:
            X_train_sample, y_train_sample = X_train, y_train
        
        clf = model.__class__(**params)
        clf.fit(X_train_sample, y_train_sample)
        return clf
    except Exception as e:
        raise Exception(f"Error during model training: {str(e)}")

def main():
    try:
        # Load and prepare data
        train_path = "../data/processed/train_processed.csv"
        test_path = "../data/processed/test_processed.csv"

        # Create models directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        models_dir = os.path.join(project_root, "backend/models")
        output_dir = os.path.join(project_root, "outputs")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nStep 1: Loading data...")
        X_train, X_test, y_train, y_test = load_data(train_path, test_path)
        
        print("\nStep 2: Scaling data...")
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test,models_dir)
        

        class_names = np.unique(y_train)  # Get class names

        print("\nStep 3: Creating models...")
        models = create_models()

        best_model = None
        best_score = -1
        best_config = None

        print("\nStep 4: Training and evaluating models...")
        for model_name, model_info in models.items():
            for sampling in ['none' ,'smote']:
                print(f"\nTraining {model_name} with {sampling} sampling...")
                
                with mlflow.start_run(run_name=f"{model_name}_{sampling}") as run:
                    # Récupération du run_id
                    run_id = run.info.run_id

                    model = train_model(
                        X_train_scaled, y_train,
                        model_info['model'],
                        model_info['params'],
                        sampling
                    )

                    metrics = evaluate_model(model, X_test_scaled, y_test, class_names, output_dir)

                    mlflow.log_params(model_info['params'])
                    mlflow.log_param("input_rows", X_train.shape[0])
                    mlflow.log_param("input_cols", X_train.shape[1])
                    mlflow.log_param('sampling_method', sampling)
                    mlflow.log_metrics(metrics)

                    # Create signature for the best model
                    signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))
                    if model_name == 'XGBoost':
                        mlflow.xgboost.log_model(
                            model, 
                            artifact_path=model_name,
                            signature=signature
                            )
                    else:
                        # Log the best model
                        mlflow.sklearn.log_model(
                            model,
                            artifact_path=model_name,
                            signature=signature
                        )


                    if metrics['f1'] > best_score:
                        best_score = metrics['f1']
                        best_model = model
                        best_config = {
                            'model_name': model_name,
                            'sampling': sampling,
                            'metrics': metrics,
                            'run_id': run_id  # Sauvegarde du run_id
                        }

        print("\nStep 5: Saving best model and metrics...")
        if best_model is not None:
            
            metrics_path = os.path.join(models_dir, "best_model_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(best_config['metrics'], f, indent=4)
            print(f"Metrics saved to {metrics_path}")

            metrics_path = os.path.join(models_dir, "best_model_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(best_config['metrics'], f, indent=4)
            print(f"Metrics saved to {metrics_path}")
            
            # Enregistrer le run_id dans run_info.json
            run_info_path = os.path.join(models_dir, "run_info.json")
            with open(run_info_path, "w") as f:
                json.dump({'run_id': best_config['run_id'], 'model_name': model_name}, f, indent=4)
            print(f"Run info saved to {run_info_path}")

            print("\nBest Model Configuration:")
            print(f"Model: {best_config['model_name']}")
            print(f"Sampling: {best_config['sampling']}")
            print("Metrics:", best_config['metrics'])
        else:
            raise ValueError("No valid model found to save.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
    