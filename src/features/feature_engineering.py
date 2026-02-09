"""
Feature engineering for cart abandonment prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer:
    """Feature engineering class for cart abandonment prediction"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def create_engagement_features(self, df):
        """Create engagement intensity features"""
        df = df.copy()
        
        # Normalize engagement metrics
        engagement_cols = ['daily_session_time_minutes', 'product_views_per_day', 'app_usage_frequency']
        df[engagement_cols] = self.scaler.fit_transform(df[engagement_cols])
        
        # Average daily engagement
        df['avg_daily_engagement_score'] = (
            df['daily_session_time_minutes'] +
            df['product_views_per_day'] +
            df['app_usage_frequency']
        ) / 3
        
        # Normalize purchase conversion
        df[['purchase_conversion_rate_normalized']] = self.scaler.fit_transform(
            df[['purchase_conversion_rate']]
        )
        
        # Engagement to purchase score
        df['engagement_to_purchase_score'] = (
            df['avg_daily_engagement_score'] * df['purchase_conversion_rate_normalized']
        )
        
        # Weekly engagement index
        df['weekly_engagement_index'] = df['engagement_to_purchase_score'] * 7
        
        return df
    
    def create_advertising_features(self, df):
        """Create advertising responsiveness features"""
        df = df.copy()
        
        # Ad response rate
        df['ad_response_rate'] = df['ad_clicks_per_day'] / df['ad_views_per_day'].replace(0, np.nan)
        df['ad_response_rate'].fillna(0, inplace=True)
        
        # Normalize ad views
        df[['ad_views_per_day']] = self.scaler.fit_transform(df[['ad_views_per_day']])
        
        # Ad exposure score
        df['ad_exposure_score'] = df['ad_views_per_day'] * df['ad_response_rate']
        
        return df
    
    def create_purchase_intent_features(self, df):
        """Create purchase intent features (no target leakage)"""
        df = df.copy()
        
        # Browse to buy inverse
        df['browse_to_buy_inverse'] = 1 / df['browse_to_buy_ratio'].replace(0, np.nan)
        df['browse_to_buy_inverse'].fillna(0, inplace=True)
        
        # Normalize and aggregate
        temp_intent = pd.DataFrame({
            'cart': df['cart_items_average'],
            'browse': df['browse_to_buy_inverse'],
            'weekly': df['weekly_purchases']
        })
        temp_intent_scaled = self.scaler.fit_transform(temp_intent)
        
        df['purchase_intent_score'] = (
            temp_intent_scaled[:, 0] +
            temp_intent_scaled[:, 1] +
            temp_intent_scaled[:, 2]
        ) / 3
        
        return df
    
    def create_discount_features(self, df):
        """Create discount sensitivity features"""
        df = df.copy()
        
        discount_cols = ['coupon_usage_frequency', 'impulse_purchases_per_month']
        df[discount_cols] = self.scaler.fit_transform(df[discount_cols])
        
        df['discount_sensitivity_index'] = (
            df['coupon_usage_frequency'] +
            df['impulse_purchases_per_month']
        ) / 2
        
        return df
    
    def create_revenue_features(self, df):
        """Create revenue strength features"""
        df = df.copy()
        
        # Normalize spend
        df[['monthly_spend']] = self.scaler.fit_transform(df[['monthly_spend']])
        df['normalized_spend_score'] = df['monthly_spend']
        
        # Customer value tier
        df['customer_value_tier'] = pd.cut(
            df['normalized_spend_score'],
            bins=[-np.inf, 0.40, 0.75, np.inf],
            labels=['Low', 'Mid', 'High']
        )
        
        return df
    
    def create_recency_features(self, df):
        """Create recency features"""
        df = df.copy()
        
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
        today = pd.Timestamp.now()
        df['days_since_last_purchase'] = (today - df['last_purchase_date']).dt.days
        
        df['recency_bucket'] = pd.cut(
            df['days_since_last_purchase'],
            bins=[-np.inf, 7, 30, 90, np.inf],
            labels=['Active', 'Warm', 'Cold', 'Dormant']
        )
        
        return df
    
    def create_advocacy_features(self, df):
        """Create loyalty and advocacy features"""
        df = df.copy()
        
        advocacy_cols = ['brand_loyalty_score', 'review_writing_frequency', 
                        'social_sharing_frequency', 'referral_count']
        df[advocacy_cols] = self.scaler.fit_transform(df[advocacy_cols])
        
        df['advocacy_score'] = (
            df['brand_loyalty_score'] +
            df['review_writing_frequency'] +
            df['social_sharing_frequency'] +
            df['referral_count']
        ) / 4
        
        return df
    
    def create_stress_features(self, df):
        """Create stress impact features"""
        df = df.copy()
        
        stress_cols = ['stress_from_financial_decisions', 'overall_stress_level', 
                      'mental_health_score', 'sleep_quality']
        df[stress_cols] = self.scaler.fit_transform(df[stress_cols])
        
        df['stress_impact_index'] = (
            df['stress_from_financial_decisions'] +
            df['overall_stress_level'] +
            (1 - df['mental_health_score']) +
            (1 - df['sleep_quality'])
        ) / 4
        
        return df
    
    def create_shopping_regularity_features(self, df):
        """Create shopping consistency features"""
        df = df.copy()
        
        # Map shopping time to numeric
        shopping_time_mapping = {time: idx for idx, time in enumerate(df['shopping_time_of_day'].unique())}
        df['shopping_time_numeric'] = df['shopping_time_of_day'].map(shopping_time_mapping)
        
        df['shopping_consistency_score'] = df['shopping_time_numeric'] * (1 + df['weekend_shopper'])
        
        return df
    
    def create_target(self, df, threshold=0.3):
        """Create binary target variable"""
        df = df.copy()
        df['cart_abandonment_flag'] = (df['cart_abandonment_rate'] >= threshold).astype(int)
        print(f"✅ Target created with threshold {threshold}")
        print(f"   Target distribution: {df['cart_abandonment_flag'].value_counts().to_dict()}")
        return df
    
    def drop_raw_features(self, df):
        """Drop raw columns after aggregation"""
        columns_to_drop = [
            # Identifiers
            'user_id',
            
            # Engagement
            'daily_session_time_minutes', 'product_views_per_day', 'app_usage_frequency',
            
            # Advertising
            'ad_views_per_day', 'ad_clicks_per_day', 'notification_response_rate',
            
            # Purchase Intent
            'cart_items_average', 'browse_to_buy_ratio', 'weekly_purchases', 'browse_to_buy_inverse',
            
            # Discount
            'coupon_usage_frequency', 'impulse_purchases_per_month',
            
            # Revenue
            'monthly_spend', 'average_order_value',
            
            # Recency
            'last_purchase_date', 'days_since_last_purchase', 'account_age_months',
            
            # Advocacy
            'brand_loyalty_score', 'review_writing_frequency', 'social_sharing_frequency', 'referral_count',
            
            # Stress
            'stress_from_financial_decisions', 'overall_stress_level', 'mental_health_score', 'sleep_quality',
            
            # Shopping
            'shopping_time_of_day', 'weekend_shopper', 'shopping_time_numeric',
            
            # Low-value demographics
            'ethnicity', 'language_preference', 'occupation', 'relationship_status', 'urban_rural', 'household_size',
            
            # Lifestyle noise
            'reading_habits', 'hobby_count', 'travel_frequency', 'exercise_frequency', 'physical_activity_level',
            
            # Additional drops
            'purchase_conversion_rate', 'return_rate', 'return_frequency', 'wishlist_items_count',
            
            # TARGET LEAKAGE - CRITICAL
            'cart_abandonment_rate', 'checkout_abandonments_per_month',
        ]
        
        df_final = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"✅ Dropped {len(columns_to_drop)} raw feature columns")
        return df_final
    
    def fit_transform(self, df):
        """Apply all feature engineering steps"""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING")
        print("="*70)
        
        df = self.create_engagement_features(df)
        print("  ✅ Engagement features created")
        
        df = self.create_advertising_features(df)
        print("  ✅ Advertising features created")
        
        df = self.create_purchase_intent_features(df)
        print("  ✅ Purchase intent features created")
        
        df = self.create_discount_features(df)
        print("  ✅ Discount features created")
        
        df = self.create_revenue_features(df)
        print("  ✅ Revenue features created")
        
        df = self.create_recency_features(df)
        print("  ✅ Recency features created")
        
        df = self.create_advocacy_features(df)
        print("  ✅ Advocacy features created")
        
        df = self.create_stress_features(df)
        print("  ✅ Stress features created")
        
        df = self.create_shopping_regularity_features(df)
        print("  ✅ Shopping regularity features created")
        
        df = self.create_target(df)
        
        df = self.drop_raw_features(df)
        
        print(f"\n✅ Feature engineering complete: {df.shape}")
        return df