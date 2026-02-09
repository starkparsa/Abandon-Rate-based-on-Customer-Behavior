"""
Main pipeline for cart abandonment prediction with MLflow tracking
Run this script to execute the entire ML pipeline
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data.load_data import load_raw_data, filter_usa
from features.feature_engineering import FeatureEngineer
from features.encoding import DataEncoder
from models.train import ModelTrainer
from utils.config import Config


def main():
    """Main pipeline for cart abandonment prediction"""
    
    print("="*70)
    print("CART ABANDONMENT PREDICTION PIPELINE")
    print("="*70)
    
    # Create directories
    Config.create_directories()
    
    # Setup MLflow
    if Config.USE_MLFLOW:
        Config.setup_mlflow()
    
    # 1. Load and filter data
    print("\n" + "="*70)
    print("1. LOADING DATA")
    print("="*70)
    df = load_raw_data(Config.RAW_DATA_FILE)
    df = filter_usa(df)
    
    # 2. Feature engineering
    print("\n" + "="*70)
    print("2. FEATURE ENGINEERING")
    print("="*70)
    engineer = FeatureEngineer()
    df = engineer.fit_transform(df)
    
    # 3. Encoding
    encoder = DataEncoder()
    df = encoder.encode(df)
    
    # 4. Train MLP model
    print("\n" + "="*70)
    print("4. MODEL TRAINING")
    print("="*70)
    
    trainer = ModelTrainer(
        random_state=Config.RANDOM_STATE,
        use_mlflow=Config.USE_MLFLOW
    )
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(
        df, 
        target_col='cart_abandonment_flag',
        test_size=Config.TEST_SIZE
    )
    
    # Train model with optimized hyperparameters
    model, f1_score = trainer.train(experiment_name=Config.MLFLOW_EXPERIMENT_NAME)
    
    # 5. Save model
    print("\n" + "="*70)
    print("5. SAVING MODEL")
    print("="*70)
    trainer.save_model(Config.FINAL_MODEL_FILE)
    
    # Save processed data (optional)
    df.to_csv(Config.PROCESSED_DATA_FILE, index=False)
    print(f"✅ Processed data saved to {Config.PROCESSED_DATA_FILE}")
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  MLP F1-Score: {f1_score:.6f}")
    print(f"\nOutputs:")
    print(f"  Model (pickle): {Config.FINAL_MODEL_FILE}")
    print(f"  Model (MLflow): Registered as 'cart_abandonment_mlp'")
    print(f"  Data: {Config.PROCESSED_DATA_FILE}")
    
    if Config.USE_MLFLOW:
        print(f"\n📊 View MLflow UI:")
        print(f"  Run: mlflow ui")
        print(f"  Open: http://localhost:5000")
    

if __name__ == "__main__":
    main()