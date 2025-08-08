import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UnitedAirlinesEnhancedPredictor:
    def __init__(self):
        self.models = {}
        self.df = None
        self.X = None
        self.y = None
        self.preprocessor = None
        self.train_idx = None
        self.test_idx = None
        self.results = {}
        self.feature_importance = {}
        self.best_model = None

    def load_united_airlines_data(self, flight_file="united_airlines_flights.csv", weather_file="merged_flight_weather.csv"):
        """Load United Airlines flight data with weather data"""
        logging.info(f"Loading United Airlines data from {flight_file}...")
        
        # Load United Airlines data
        try:
            df_flights = pd.read_csv(flight_file, low_memory=False)
            # Filter for United Airlines flights only
            df_flights = df_flights[df_flights['Reporting_Airline'] == 'UA'].copy()
            logging.info(f"United Airlines flights found: {len(df_flights):,}")
        except Exception as e:
            logging.error(f"Could not load {flight_file}: {e}")
            logging.error("This model requires United Airlines flight data!")
            return None
        
        # Load weather data
        logging.info(f"Loading weather data from {weather_file}...")
        try:
            df_weather = pd.read_csv(weather_file)
            logging.info(f"Weather data loaded: {len(df_weather):,} records")
        except Exception as e:
            logging.warning(f"Could not load {weather_file}: {e}")
            df_weather = None
        
        # Prepare flight data for merging
        df_flights['FlightDate'] = pd.to_datetime(df_flights['FlightDate'], errors='coerce')
        
        # If we have weather data, merge it
        if df_weather is not None:
            df_weather['FlightDate'] = pd.to_datetime(df_weather['FlightDate'], errors='coerce')
            
            # Merge on common columns
            merge_cols = ['Origin', 'Dest', 'FlightDate']
            df_merged = pd.merge(df_flights, df_weather, on=merge_cols, how='left', suffixes=('', '_weather'))
            logging.info(f"After merging with weather: {len(df_merged):,} records")
        else:
            df_merged = df_flights.copy()
            logging.info("Using flight data only (no weather data available)")
        
        # Enhanced data cleaning
        logging.info("Performing enhanced data cleaning...")
        
        # Remove extreme delays using IQR method
        Q1 = df_merged['DepDelay'].quantile(0.25)
        Q3 = df_merged['DepDelay'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # More lenient than 1.5 * IQR
        upper_bound = Q3 + 3 * IQR
        
        initial_count = len(df_merged)
        df_merged = df_merged[df_merged['DepDelay'].between(lower_bound, upper_bound)]
        removed_count = initial_count - len(df_merged)
        logging.info(f"Removed {removed_count:,} outliers using IQR method")
        
        # Remove flights with missing critical data
        df_merged = df_merged.dropna(subset=['DepDelay', 'Origin', 'Dest'])
        
        # Keep essential columns
        essential_cols = [
            'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime', 'DepDelay', 'ArrDelay', 'FlightDate',
            'Distance', 'LateAircraftDelay', 'WeatherDelay', 'CarrierDelay', 
            'NASDelay', 'SecurityDelay', 'DayOfWeek', 'Month', 'Quarter'
        ]
        
        # Add weather columns if available
        weather_cols = ['temperature_c', 'precip_mm', 'cloud_pct', 'wind_speed_mps', 'OriginCity']
        for col in weather_cols:
            if col in df_merged.columns:
                essential_cols.append(col)
        
        # Filter for columns that exist
        available_cols = [col for col in essential_cols if col in df_merged.columns]
        df_merged = df_merged[available_cols].copy()
        
        self.df = df_merged.reset_index(drop=True)
        logging.info(f"Final United Airlines dataset shape: {self.df.shape}")
        logging.info(f"Columns: {list(self.df.columns)}")
        return self.df

    def engineer_enhanced_features(self):
        """Engineer comprehensive features for United Airlines"""
        logging.info("Engineering enhanced features for United Airlines...")
        df = self.df.copy()
        
        # Time-based features
        df['hour'] = df['CRSDepTime'] // 100
        df['minute'] = df['CRSDepTime'] % 100
        df['departure_time_minutes'] = df['hour'] * 60 + df['minute']
        
        # Arrival time features
        df['arrival_hour'] = df['CRSArrTime'] // 100
        df['arrival_minute'] = df['CRSArrTime'] % 100
        df['arrival_time_minutes'] = df['arrival_hour'] * 60 + df['arrival_minute']
        
        # Flight duration
        df['scheduled_duration'] = df['arrival_time_minutes'] - df['departure_time_minutes']
        df.loc[df['scheduled_duration'] < 0, 'scheduled_duration'] += 1440  # Add 24 hours if negative
        
        # Enhanced time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                  bins=[0, 6, 9, 12, 15, 18, 21, 24], 
                                  labels=['Early_Morning', 'Morning', 'Late_Morning', 'Afternoon', 
                                         'Late_Afternoon', 'Evening', 'Night'], 
                                  right=False, include_lowest=True)
        
        df['arrival_time_of_day'] = pd.cut(df['arrival_hour'], 
                                          bins=[0, 6, 9, 12, 15, 18, 21, 24], 
                                          labels=['Early_Morning', 'Morning', 'Late_Morning', 'Afternoon', 
                                                 'Late_Afternoon', 'Evening', 'Night'], 
                                          right=False, include_lowest=True)
        
        # Date features
        df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')
        df['day_of_week'] = df['FlightDate'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday_season'] = df['FlightDate'].dt.month.isin([11, 12]).astype(int)
        df['is_summer'] = df['FlightDate'].dt.month.isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['FlightDate'].dt.month.isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['FlightDate'].dt.month.isin([3, 4, 5]).astype(int)
        df['is_fall'] = df['FlightDate'].dt.month.isin([9, 10, 11]).astype(int)
        
        # Route complexity based on distance
        df['route_complexity'] = pd.cut(df['Distance'], 
                                       bins=[0, 300, 600, 1200, 2000, 5000], 
                                       labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long'])
        
        # United Airlines major hubs
        major_hubs = ['ORD', 'DEN', 'IAH', 'EWR', 'SFO', 'LAX', 'IAD', 'CLT', 'ATL', 'MIA', 'BOS', 'SEA']
        df['origin_is_hub'] = df['Origin'].isin(major_hubs).astype(int)
        df['dest_is_hub'] = df['Dest'].isin(major_hubs).astype(int)
        df['hub_to_hub'] = (df['origin_is_hub'] & df['dest_is_hub']).astype(int)
        df['hub_connection'] = (df['origin_is_hub'] | df['dest_is_hub']).astype(int)
        
        # Seasonality
        df['season'] = pd.cut(df['Month'], 
                             bins=[0, 3, 6, 9, 12], 
                             labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
        # Weather-based features (if available)
        if 'temperature_c' in df.columns:
            # Temperature categories
            df['temp_category'] = pd.cut(df['temperature_c'], 
                                       bins=[-50, -10, 0, 10, 20, 30, 40, 50], 
                                       labels=['Extreme_Cold', 'Very_Cold', 'Cold', 'Cool', 'Warm', 'Hot', 'Very_Hot'])
            
            # Extreme weather conditions
            df['extreme_cold'] = (df['temperature_c'] < -10).astype(int)
            df['extreme_hot'] = (df['temperature_c'] > 35).astype(int)
            df['freezing_temp'] = (df['temperature_c'] < 0).astype(int)
            df['comfortable_temp'] = ((df['temperature_c'] >= 15) & (df['temperature_c'] <= 25)).astype(int)
            df['temp_deviation'] = abs(df['temperature_c'] - 20)
        
        if 'precip_mm' in df.columns:
            # Precipitation categories
            df['precip_category'] = pd.cut(df['precip_mm'], 
                                         bins=[0, 0.1, 1, 2.5, 7.5, 15, 50], 
                                         labels=['None', 'Trace', 'Light', 'Moderate', 'Heavy', 'Very_Heavy'])
            
            # Rain/snow indicators
            df['has_precipitation'] = (df['precip_mm'] > 0).astype(int)
            df['heavy_precipitation'] = (df['precip_mm'] > 7.5).astype(int)
            df['moderate_precipitation'] = (df['precip_mm'] > 2.5).astype(int)
            df['light_precipitation'] = ((df['precip_mm'] > 0.1) & (df['precip_mm'] <= 2.5)).astype(int)
        
        if 'cloud_pct' in df.columns:
            # Cloud cover categories
            df['cloud_category'] = pd.cut(df['cloud_pct'], 
                                        bins=[0, 10, 25, 50, 75, 90, 100], 
                                        labels=['Clear', 'Mostly_Clear', 'Partly_Cloudy', 'Mostly_Cloudy', 'Cloudy', 'Overcast'])
            
            # Visibility conditions
            df['low_visibility'] = (df['cloud_pct'] > 80).astype(int)
            df['clear_sky'] = (df['cloud_pct'] < 25).astype(int)
            df['overcast'] = (df['cloud_pct'] > 90).astype(int)
        
        if 'wind_speed_mps' in df.columns:
            # Wind speed categories
            df['wind_category'] = pd.cut(df['wind_speed_mps'], 
                                       bins=[0, 3, 7, 12, 18, 25, 30], 
                                       labels=['Calm', 'Light', 'Gentle', 'Moderate', 'Strong', 'Very_Strong'])
            
            # Wind conditions
            df['high_wind'] = (df['wind_speed_mps'] > 15).astype(int)
            df['moderate_wind'] = (df['wind_speed_mps'] > 10).astype(int)
            df['calm_wind'] = (df['wind_speed_mps'] < 5).astype(int)
        
        # Weather severity index
        weather_severity = 0
        if 'temperature_c' in df.columns:
            weather_severity += np.abs(df['temperature_c'] - 20) / 20 * 0.3
        if 'precip_mm' in df.columns:
            weather_severity += df['precip_mm'] / 10 * 0.4
        if 'cloud_pct' in df.columns:
            weather_severity += df['cloud_pct'] / 100 * 0.2
        if 'wind_speed_mps' in df.columns:
            weather_severity += df['wind_speed_mps'] / 20 * 0.1
        
        df['weather_severity_index'] = weather_severity
        
        # Historical arrival delay patterns (if ArrDelay is available)
        if 'ArrDelay' in df.columns:
            # Calculate historical arrival delay statistics by route
            route_arr_delays = df.groupby(['Origin', 'Dest'])['ArrDelay'].agg([
                'mean', 'std', 'median', 'count'
            ]).reset_index()
            
            route_arr_delays.columns = ['Origin', 'Dest', 'route_arr_delay_mean', 
                                      'route_arr_delay_std', 'route_arr_delay_median', 'route_flight_count']
            
            # Merge back to main dataframe
            df = pd.merge(df, route_arr_delays, on=['Origin', 'Dest'], how='left')
            
            # Fill missing values with global statistics
            global_mean = df['ArrDelay'].mean()
            global_std = df['ArrDelay'].std()
            global_median = df['ArrDelay'].median()
            
            df['route_arr_delay_mean'].fillna(global_mean, inplace=True)
            df['route_arr_delay_std'].fillna(global_std, inplace=True)
            df['route_arr_delay_median'].fillna(global_median, inplace=True)
            df['route_flight_count'].fillna(1, inplace=True)
            
            # Historical patterns by time of day
            time_arr_delays = df.groupby('hour')['ArrDelay'].agg(['mean', 'std']).reset_index()
            time_arr_delays.columns = ['hour', 'hour_arr_delay_mean', 'hour_arr_delay_std']
            df = pd.merge(df, time_arr_delays, on='hour', how='left')
            
            # Historical patterns by day of week
            dow_arr_delays = df.groupby('day_of_week')['ArrDelay'].agg(['mean', 'std']).reset_index()
            dow_arr_delays.columns = ['day_of_week', 'dow_arr_delay_mean', 'dow_arr_delay_std']
            df = pd.merge(df, dow_arr_delays, on='day_of_week', how='left')
            
            # Historical patterns by month
            month_arr_delays = df.groupby('Month')['ArrDelay'].agg(['mean', 'std']).reset_index()
            month_arr_delays.columns = ['Month', 'month_arr_delay_mean', 'month_arr_delay_std']
            df = pd.merge(df, month_arr_delays, on='Month', how='left')
            
            # Route reliability score
            df['route_reliability'] = df['route_arr_delay_std'] / (abs(df['route_arr_delay_mean']) + 1)
            df['route_reliability'] = df['route_reliability'].replace([np.inf, -np.inf], np.nan)
        
        # Interaction features
        if 'temperature_c' in df.columns and 'wind_speed_mps' in df.columns:
            df['temp_wind_interaction'] = df['temperature_c'] * df['wind_speed_mps']
            df['temp_wind_ratio'] = df['temperature_c'] / (df['wind_speed_mps'] + 1)
            df['temp_wind_ratio'] = df['temp_wind_ratio'].replace([np.inf, -np.inf], np.nan)
        else:
            df['temp_wind_interaction'] = 0
            df['temp_wind_ratio'] = 0
            
        if 'precip_mm' in df.columns and 'wind_speed_mps' in df.columns:
            df['precip_wind_interaction'] = df['precip_mm'] * df['wind_speed_mps']
        else:
            df['precip_wind_interaction'] = 0
        
        # Flight characteristics
        df['is_red_eye'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 16) & (df['hour'] <= 18))
        df['is_peak_hour'] = df['is_peak_hour'].astype(int)
        df['is_off_peak'] = ((df['hour'] >= 10) & (df['hour'] <= 15)).astype(int)
        
        # Distance-based features
        df['distance_category'] = pd.cut(df['Distance'], 
                                       bins=[0, 500, 1000, 2000, 5000], 
                                       labels=['Short', 'Medium', 'Long', 'Very_Long'])
        
        # Delay type features
        delay_types = ['LateAircraftDelay', 'WeatherDelay', 'CarrierDelay', 'NASDelay', 'SecurityDelay']
        for delay_type in delay_types:
            if delay_type in df.columns:
                df[f'{delay_type}_present'] = (df[delay_type] > 0).astype(int)
                df[f'{delay_type}_ratio'] = df[delay_type] / (df['DepDelay'] + 1)
                df[f'{delay_type}_ratio'] = df[f'{delay_type}_ratio'].replace([np.inf, -np.inf], np.nan)
        
        self.df = df
        logging.info("Enhanced feature engineering completed!")
        return self.df

    def prepare_features(self):
        """Prepare features for the model"""
        logging.info("Preparing features...")
        df = self.df
        
        # Define feature categories
        categorical_features = [
            'Origin', 'Dest', 'time_of_day', 'arrival_time_of_day', 'route_complexity', 'season',
            'origin_is_hub', 'dest_is_hub', 'hub_to_hub', 'hub_connection', 'is_weekend', 'is_holiday_season',
            'temp_category', 'precip_category', 'cloud_category', 'wind_category', 'distance_category'
        ]
        
        numerical_features = [
            'departure_time_minutes', 'arrival_time_minutes', 'scheduled_duration', 'Distance', 
            'DayOfWeek', 'Month', 'Quarter', 'hour', 'arrival_hour'
        ]
        
        # Add weather features
        weather_numerical = [
            'temperature_c', 'precip_mm', 'cloud_pct', 'wind_speed_mps', 'weather_severity_index',
            'extreme_cold', 'extreme_hot', 'freezing_temp', 'comfortable_temp', 'temp_deviation',
            'has_precipitation', 'heavy_precipitation', 'moderate_precipitation', 'light_precipitation',
            'low_visibility', 'clear_sky', 'overcast', 'high_wind', 'moderate_wind', 'calm_wind',
            'temp_wind_interaction', 'temp_wind_ratio', 'precip_wind_interaction', 'is_red_eye', 
            'is_peak_hour', 'is_off_peak', 'is_summer', 'is_winter', 'is_spring', 'is_fall'
        ]
        
        # Add historical arrival delay features
        historical_features = [
            'route_arr_delay_mean', 'route_arr_delay_std', 'route_arr_delay_median', 'route_flight_count',
            'route_reliability', 'hour_arr_delay_mean', 'hour_arr_delay_std',
            'dow_arr_delay_mean', 'dow_arr_delay_std',
            'month_arr_delay_mean', 'month_arr_delay_std'
        ]
        
        # Add delay type features
        delay_type_features = []
        delay_types = ['LateAircraftDelay', 'WeatherDelay', 'CarrierDelay', 'NASDelay', 'SecurityDelay']
        for delay_type in delay_types:
            if delay_type in df.columns:
                delay_type_features.extend([f'{delay_type}_present', f'{delay_type}_ratio'])
        
        # Filter for columns that exist in the dataset
        categorical_features = [col for col in categorical_features if col in df.columns]
        numerical_features = [col for col in numerical_features + weather_numerical + historical_features + delay_type_features if col in df.columns]
        
        logging.info(f"Categorical features: {len(categorical_features)}")
        logging.info(f"Numerical features: {len(numerical_features)}")
        
        X = df[categorical_features + numerical_features].copy()
        y = df['DepDelay'].copy()
        
        # Handle missing values
        for col in categorical_features:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        for col in numerical_features:
            # Handle infinite values first
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(X[col].median())
        
        # Final data validation
        logging.info("Performing final data validation...")
        
        # Check for infinite values
        inf_mask = np.isinf(X.select_dtypes(include=[np.number]))
        if inf_mask.any().any():
            inf_cols = inf_mask.any()[inf_mask.any()].index.tolist()
            logging.warning(f"Found infinite values in columns: {inf_cols}")
            for col in inf_cols:
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                X[col] = X[col].fillna(X[col].median())
        
        # Check for extremely large values
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].abs().max() > 1e10:
                logging.warning(f"Found extremely large values in {col}, capping at 99th percentile")
                q99 = X[col].quantile(0.99)
                X[col] = X[col].clip(upper=q99)
        
        self.X, self.y = X, y
        logging.info(f"Final feature matrix shape: {self.X.shape}")
        return self.X, self.y

    def create_preprocessing_pipeline(self):
        """Create preprocessing pipeline"""
        X = self.X
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Categorical transformer
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
        ])
        
        # Numerical transformer
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        self.preprocessor = ColumnTransformer([
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])
        
        return self.preprocessor

    def train_models(self):
        """Train multiple models and compare performance"""
        logging.info("Training United Airlines flight delay prediction models...")
        
        # Split data
        self.train_idx, self.test_idx = train_test_split(
            self.df.index, test_size=0.2, random_state=42
        )
        
        X_train, X_test = self.X.loc[self.train_idx], self.X.loc[self.test_idx]
        y_train, y_test = self.y.loc[self.train_idx], self.y.loc[self.test_idx]
        
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline()
        
        # Define models
        models = {
            'RandomForest': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1))
            ]),
            'GradientBoosting': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42))
            ])
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            logging.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            logging.info(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]['model']
        
        self.models = results
        self.results = results
        
        logging.info(f"Best model: {best_model_name} with R² = {results[best_model_name]['r2']:.3f}")
        
        return results

    def analyze_feature_importance(self):
        """Analyze feature importance for the best model"""
        logging.info("Analyzing feature importance...")
        
        if self.best_model is None:
            logging.warning("No best model available for feature importance analysis")
            return
        
        # Get feature names after preprocessing
        preprocessor = self.create_preprocessing_pipeline()
        X_train = self.X.loc[self.train_idx]
        preprocessor.fit(X_train)
        
        # Get feature names
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        num_features = self.X.select_dtypes(include=[np.number]).columns.tolist()
        feature_names = list(cat_features) + num_features
        
        # Get feature importance from the best model
        if hasattr(self.best_model.named_steps['regressor'], 'feature_importances_'):
            importance = self.best_model.named_steps['regressor'].feature_importances_
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.feature_importance['Best_Model'] = importance_df
            
            logging.info(f"\nTop 15 features for Best Model:")
            for i, row in importance_df.head(15).iterrows():
                logging.info(f"  {row['feature']}: {row['importance']:.4f}")

    def plot_results(self):
        """Create visualizations"""
        logging.info("Creating visualizations...")
        plt.style.use('seaborn-v0_8')
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        best_result = self.results[best_model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['r2'] for name in model_names]
        
        axes[0, 0].bar(model_names, r2_scores, color=['skyblue', 'lightgreen'])
        axes[0, 0].set_title('Model Comparison - R² Scores')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Actual vs Predicted (best model)
        y_test = self.y.loc[self.test_idx]
        y_pred = best_result['predictions']
        
        axes[0, 1].scatter(y_test, y_pred, alpha=0.5, color='blue')
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Delay (minutes)')
        axes[0, 1].set_ylabel('Predicted Delay (minutes)')
        axes[0, 1].set_title(f'{best_model_name}: Actual vs Predicted')
        
        # 3. Feature importance (if available)
        if 'Best_Model' in self.feature_importance:
            importance_df = self.feature_importance['Best_Model'].head(10)
            axes[1, 0].barh(importance_df['feature'], importance_df['importance'], color='lightcoral')
            axes[1, 0].set_title(f'{best_model_name}: Top 10 Feature Importance')
            axes[1, 0].set_xlabel('Importance')
        
        # 4. Delay distribution
        axes[1, 1].hist(self.y, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Departure Delay (minutes)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of United Airlines Delays')
        axes[1, 1].axvline(self.y.mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.y.mean():.1f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

    def generate_report(self):
        """Generate comprehensive report"""
        logging.info("\n" + "="*80)
        logging.info("UNITED AIRLINES FLIGHT DELAY PREDICTION MODEL REPORT")
        logging.info("="*80)
        
        logging.info(f"\nDataset Information:")
        logging.info(f"- Total United Airlines flights: {len(self.df):,}")
        logging.info(f"- Features used: {len(self.X.columns)}")
        logging.info(f"- Average delay: {self.y.mean():.2f} minutes")
        logging.info(f"- Delay standard deviation: {self.y.std():.2f} minutes")
        
        # Weather statistics
        weather_features = ['temperature_c', 'precip_mm', 'cloud_pct', 'wind_speed_mps']
        for feature in weather_features:
            if feature in self.X.columns:
                data = self.X[feature]
                logging.info(f"- {feature}: mean={data.mean():.2f}, std={data.std():.2f}")
        
        # Historical arrival delay statistics
        if 'route_arr_delay_mean' in self.X.columns:
            data = self.X['route_arr_delay_mean']
            logging.info(f"- Historical route arrival delays: mean={data.mean():.2f}, std={data.std():.2f}")
        
        logging.info(f"\nModel Performance:")
        for name, result in self.results.items():
            logging.info(f"- {name}:")
            logging.info(f"  R² = {result['r2']:.3f}")
            logging.info(f"  RMSE = {result['rmse']:.2f} minutes")
            logging.info(f"  MAE = {result['mae']:.2f} minutes")
            logging.info(f"  CV R² = {result['cv_r2_mean']:.3f} (+/- {result['cv_r2_std'] * 2:.3f})")
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        best_result = self.results[best_model_name]
        
        logging.info(f"\nBest Model: {best_model_name}")
        logging.info(f"- R² = {best_result['r2']:.3f}")
        logging.info(f"- RMSE = {best_result['rmse']:.2f} minutes")
        logging.info(f"- MAE = {best_result['mae']:.2f} minutes")
        
        # Feature importance summary
        if 'Best_Model' in self.feature_importance:
            logging.info(f"\nTop 10 Most Important Features ({best_model_name}):")
            importance_df = self.feature_importance['Best_Model'].head(10)
            for i, row in importance_df.iterrows():
                logging.info(f"- {row['feature']}: {row['importance']:.4f}")
        
        logging.info("\n" + "="*80)

def main():
    """Main function to run United Airlines model"""
    predictor = UnitedAirlinesEnhancedPredictor()
    
    # Load and process data
    df = predictor.load_united_airlines_data()
    if df is None:
        logging.error("Failed to load United Airlines data. Exiting.")
        return None
    
    df = predictor.engineer_enhanced_features()
    X, y = predictor.prepare_features()
    
    # Train models
    results = predictor.train_models()
    
    # Analyze feature importance
    predictor.analyze_feature_importance()
    
    # Generate visualizations and report
    predictor.plot_results()
    predictor.generate_report()
    
    return predictor

if __name__ == "__main__":
    predictor = main() 