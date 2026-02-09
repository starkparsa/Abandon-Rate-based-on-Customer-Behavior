"""
Data encoding module for converting categorical variables to numeric
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataEncoder:
    """Encode categorical variables for ML models"""
    
    def __init__(self):
        self.label_encoders = {}
    
    def encode(self, df):
        """
        Encode all categorical variables
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with categorical columns
            
        Returns:
        --------
        pd.DataFrame
            Fully encoded numeric dataframe
        """
        df_encoded = df.copy()
        
        print("\n" + "="*70)
        print("SMART ENCODING STRATEGY")
        print("="*70)
        
        # Step 1: Convert booleans to int
        bool_cols = df_encoded.select_dtypes(include='bool').columns.tolist()
        if bool_cols:
            for col in bool_cols:
                df_encoded[col] = df_encoded[col].astype(int)
            print(f"\n1. Converted {len(bool_cols)} boolean columns to 1/0")
        else:
            print(f"\n1. Converting {len(bool_cols)} boolean columns to 1/0")
        
        # Step 2: Ordinal encoding
        print(f"\n2. Ordinal Encoding (preserves order)")
        
        if 'customer_value_tier' in df_encoded.columns:
            value_tier_mapping = {'Low': 0, 'Mid': 1, 'High': 2}
            df_encoded['customer_value_tier'] = df_encoded['customer_value_tier'].map(value_tier_mapping)
            print(f"   ✅ customer_value_tier: {value_tier_mapping}")
        
        if 'recency_bucket' in df_encoded.columns:
            recency_mapping = {'Active': 0, 'Warm': 1, 'Cold': 2, 'Dormant': 3}
            df_encoded['recency_bucket'] = df_encoded['recency_bucket'].map(recency_mapping)
            print(f"   ✅ recency_bucket: {recency_mapping}")
        
        # Step 3: Label encoding for remaining categoricals
        print(f"\n3. Label Encoding for nominal categories")
        
        object_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if 'cart_abandonment_flag' in object_cols:
            object_cols.remove('cart_abandonment_flag')
        
        print(f"   Found {len(object_cols)} categorical columns")
        
        for col in object_cols:
            n_unique = df_encoded[col].nunique()
            
            if n_unique <= 10:
                # Use Label Encoding for low cardinality
                print(f"   {col} ({n_unique} categories) → Label Encoding")
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Use One-Hot for high cardinality
                print(f"   {col} ({n_unique} categories) → One-Hot Encoding")
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
        
        # Step 4: Final numeric conversion
        print(f"\n4. Final conversion - ALL to numeric")
        
        # Convert remaining booleans
        bool_cols_remaining = df_encoded.select_dtypes(include='bool').columns.tolist()
        if bool_cols_remaining:
            print(f"   Converting {len(bool_cols_remaining)} remaining boolean columns to int")
            for col in bool_cols_remaining:
                df_encoded[col] = df_encoded[col].astype(int)
        
        # Convert remaining non-numeric
        for col in df_encoded.columns:
            if df_encoded[col].dtype in ['object', 'category', 'bool']:
                print(f"   Converting {col} from {df_encoded[col].dtype} to numeric")
                try:
                    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                except:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        # Convert uint to int64
        uint_cols = df_encoded.select_dtypes(include=['uint8', 'uint16', 'uint32']).columns.tolist()
        if uint_cols:
            print(f"   Converting {len(uint_cols)} uint columns to int64")
            for col in uint_cols:
                df_encoded[col] = df_encoded[col].astype('int64')
        
        print(f"\n✅ Encoding complete: {df_encoded.shape}")
        print(f"   Total features: {df_encoded.shape[1] - 1} (excluding target)")
        print(f"   Total samples: {df_encoded.shape[0]}")
        
        return df_encoded