"""
Model training module for optimized MLP Classifier with MLflow tracking
Uses pre-tuned hyperparameters from Optuna optimization
"""
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn


class ModelTrainer:
    """Train MLP model with optimized hyperparameters for cart abandonment prediction"""
    
    # Best hyperparameters from Optuna optimization
    BEST_PARAMS = {
        'hidden_layer_sizes': (128, 64),
        'activation': 'relu',
        'solver': 'adam',
        'learning_rate_init': 0.000793,
        'alpha': 1.06e-05,
        'batch_size': 64,
        'max_iter': 401,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 20,
    }
    
    def __init__(self, random_state=42, use_mlflow=True):
        self.random_state = random_state
        self.use_mlflow = use_mlflow
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def split_data(self, df, target_col='cart_abandonment_flag', test_size=0.2):
        """
        Split data into train and test sets
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with features and target
        target_col : str
            Name of target column
        test_size : float
            Proportion of test data
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*70)
        print("TRAIN-TEST SPLIT")
        print("="*70)
        
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        print(f"\nFeatures (X): {X.shape}")
        print(f"Target (y): {y.shape}")
        print(f"\nTarget distribution:")
        print(y.value_counts())
        print(f"\nClass balance:")
        print(y.value_counts(normalize=True))
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"\n✅ Data split complete!")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, experiment_name="cart_abandonment_mlp"):
        """
        Train MLP model with optimized hyperparameters and log to MLflow
        
        Parameters:
        -----------
        experiment_name : str
            MLflow experiment name
            
        Returns:
        --------
        tuple
            (trained_model, f1_score)
        """
        if self.X_train is None:
            raise ValueError("Run split_data() first!")
        
        print("\n" + "="*70)
        print("TRAINING OPTIMIZED MLP MODEL")
        print("="*70)
        
        # Display hyperparameters
        print("\nUsing optimized hyperparameters:")
        print(f"  hidden_layer_sizes: {self.BEST_PARAMS['hidden_layer_sizes']}")
        print(f"  activation: {self.BEST_PARAMS['activation']}")
        print(f"  solver: {self.BEST_PARAMS['solver']}")
        print(f"  learning_rate_init: {self.BEST_PARAMS['learning_rate_init']:.6f}")
        print(f"  alpha: {self.BEST_PARAMS['alpha']:.2e}")
        print(f"  batch_size: {self.BEST_PARAMS['batch_size']}")
        print(f"  max_iter: {self.BEST_PARAMS['max_iter']}")
        
        # Add random state to params
        params = self.BEST_PARAMS.copy()
        params['random_state'] = self.random_state
        
        # Setup MLflow
        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)
            print(f"\n📊 MLflow experiment: {experiment_name}")
        
        # Start MLflow run
        with mlflow.start_run() if self.use_mlflow else self._dummy_context():
            
            # Train model
            print("\nTraining model...")
            self.model = MLPClassifier(**params)
            self.model.fit(self.X_train, self.y_train)
            
            # Evaluate
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            print(f"\n✅ Training complete!")
            print(f"\nModel Performance:")
            print(f"  F1-Score:  {f1:.6f}")
            print(f"  Precision: {precision:.6f}")
            print(f"  Recall:    {recall:.6f}")
            print(f"  Accuracy:  {accuracy:.6f}")
            
            # Log to MLflow
            if self.use_mlflow:
                # Log parameters
                mlflow.log_params({
                    'hidden_layer_sizes': str(self.BEST_PARAMS['hidden_layer_sizes']),
                    'activation': self.BEST_PARAMS['activation'],
                    'solver': self.BEST_PARAMS['solver'],
                    'learning_rate_init': self.BEST_PARAMS['learning_rate_init'],
                    'alpha': self.BEST_PARAMS['alpha'],
                    'batch_size': self.BEST_PARAMS['batch_size'],
                    'max_iter': self.BEST_PARAMS['max_iter'],
                    'random_state': self.random_state,
                })
                
                # Log metrics
                mlflow.log_metrics({
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                })
                
                # Log dataset info
                mlflow.log_params({
                    'train_samples': self.X_train.shape[0],
                    'test_samples': self.X_test.shape[0],
                    'n_features': self.X_train.shape[1],
                })
                
                # Log model
                mlflow.sklearn.log_model(
                    self.model, 
                    "model",
                    registered_model_name="cart_abandonment_mlp"
                )
                
                print(f"\n✅ Model and metrics logged to MLflow")
            
            # Detailed metrics
            print(f"\nConfusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))
            
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
        
        return self.model, f1
    
    def _dummy_context(self):
        """Dummy context manager when MLflow is disabled"""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("Train model first!")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\n✅ Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded from {filepath}")
        return model
    
    @staticmethod
    def load_model_from_mlflow(model_name="cart_abandonment_mlp", stage="Production"):
        """
        Load model from MLflow Model Registry
        
        Parameters:
        -----------
        model_name : str
            Registered model name in MLflow
        stage : str
            Model stage (None, Staging, Production, Archived)
            
        Returns:
        --------
        model
            Loaded sklearn model
        """
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Model loaded from MLflow: {model_uri}")
        return model