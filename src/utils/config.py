"""
Configuration settings for the project
"""
from pathlib import Path


class Config:
    """Project configuration"""
    
    # Paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = ROOT_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = ROOT_DIR / 'models'
    NOTEBOOKS_DIR = ROOT_DIR / 'notebooks'
    MLFLOW_DIR = ROOT_DIR / 'mlruns'
    
    # Data files
    RAW_DATA_FILE = RAW_DATA_DIR / 'e_commerce_shopper_behaviour_and_lifestyle.csv'
    PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / 'processed_data.csv'
    
    # Model files
    FINAL_MODEL_FILE = MODELS_DIR / 'mlp_model.pkl'
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CART_ABANDONMENT_THRESHOLD = 0.3
    
    # MLflow settings
    USE_MLFLOW = True
    MLFLOW_EXPERIMENT_NAME = "cart_abandonment_mlp"
    MLFLOW_TRACKING_URI = None  # None = local mlruns folder, or set to remote URI
    
    # Feature engineering
    ENGAGEMENT_COLS = ['daily_session_time_minutes', 'product_views_per_day', 'app_usage_frequency']
    DISCOUNT_COLS = ['coupon_usage_frequency', 'impulse_purchases_per_month']
    ADVOCACY_COLS = ['brand_loyalty_score', 'review_writing_frequency', 'social_sharing_frequency', 'referral_count']
    STRESS_COLS = ['stress_from_financial_decisions', 'overall_stress_level', 'mental_health_score', 'sleep_quality']
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
        print("✅ Directories created")
    
    @classmethod
    def setup_mlflow(cls):
        """Setup MLflow tracking URI"""
        import mlflow
        if cls.MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(cls.MLFLOW_TRACKING_URI)
            print(f"✅ MLflow tracking URI set to: {cls.MLFLOW_TRACKING_URI}")
        else:
            print(f"✅ MLflow using local tracking (mlruns folder)")