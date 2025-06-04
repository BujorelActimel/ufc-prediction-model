import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImprovedUFCFightPredictor:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.fighter_stats = {}
        self.feature_names = []
        self.feature_selector = None
        self.best_features = None
        
    def load_and_prepare_data(self, csv_file_path):
        """Load UFC data and prepare for modeling with enhanced features"""
        print("Loading UFC dataset...")
        df = pd.read_csv(csv_file_path)
        
        # Convert numeric columns to proper data types
        numeric_columns = [
            'fighter1_height', 'fighter1_curr_weight', 'fighter1_reach',
            'fighter1_sig_strikes_landed_pm', 'fighter1_sig_strikes_accuracy',
            'fighter1_sig_strikes_absorbed_pm', 'fighter1_sig_strikes_defended',
            'fighter1_takedown_avg_per15m', 'fighter1_takedown_accuracy',
            'fighter1_takedown_defence', 'fighter1_submission_avg_attempted_per15m',
            'fighter2_height', 'fighter2_curr_weight', 'fighter2_reach',
            'fighter2_sig_strikes_landed_pm', 'fighter2_sig_strikes_accuracy',
            'fighter2_sig_strikes_absorbed_pm', 'fighter2_sig_strikes_defended',
            'fighter2_takedown_avg_per15m', 'fighter2_takedown_accuracy',
            'fighter2_takedown_defence', 'fighter2_submission_avg_attempted_per15m',
            'favourite_odds', 'underdog_odds'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter for fights with clear outcomes
        df = df[df['outcome'] == 'fighter1'].copy()
        print(f"Loaded {len(df)} fights with clear winners")
        
        # Store fighter statistics for lookup
        self._build_enhanced_fighter_database(df)
        
        # Create training data with enhanced features
        training_data = self._create_enhanced_training_data(df)
        return training_data
    
    def _build_enhanced_fighter_database(self, df):
        """Build an enhanced database of fighter statistics including win rates and streaks"""
        print("Building enhanced fighter statistics database...")
        
        # Helper function to safely convert to numeric
        def safe_numeric(value, default):
            if pd.isna(value):
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # First, calculate win/loss records for each fighter
        fighter_records = {}
        
        # Process all fights to build records
        for _, row in df.iterrows():
            fighter1 = row['fighter1']
            fighter2 = row['fighter2']
            winner = row['winner'] if 'winner' in row else fighter1  # Assuming fighter1 won if outcome='fighter1'
            
            # Initialize records if needed
            if fighter1 not in fighter_records:
                fighter_records[fighter1] = {'wins': 0, 'losses': 0, 'fights': []}
            if fighter2 not in fighter_records:
                fighter_records[fighter2] = {'wins': 0, 'losses': 0, 'fights': []}
            
            # Update records
            if winner == fighter1:
                fighter_records[fighter1]['wins'] += 1
                fighter_records[fighter2]['losses'] += 1
                fighter_records[fighter1]['fights'].append('W')
                fighter_records[fighter2]['fights'].append('L')
            else:
                fighter_records[fighter2]['wins'] += 1
                fighter_records[fighter1]['losses'] += 1
                fighter_records[fighter2]['fights'].append('W')
                fighter_records[fighter1]['fights'].append('L')
        
        # Calculate streaks
        for fighter, record in fighter_records.items():
            fights = record['fights']
            if fights:
                # Current streak
                current_result = fights[-1]
                streak = 1
                for i in range(len(fights) - 2, -1, -1):
                    if fights[i] == current_result:
                        streak += 1
                    else:
                        break
                record['current_streak'] = streak if current_result == 'W' else -streak
                
                # Longest win streak
                max_streak = 0
                current_win_streak = 0
                for result in fights:
                    if result == 'W':
                        current_win_streak += 1
                        max_streak = max(max_streak, current_win_streak)
                    else:
                        current_win_streak = 0
                record['longest_win_streak'] = max_streak
            else:
                record['current_streak'] = 0
                record['longest_win_streak'] = 0
        
        # Combine fighter data
        fighter_data = []
        
        # Add fighter1 data
        for _, row in df.iterrows():
            if pd.notna(row['fighter1']):
                fighter_name = row['fighter1']
                record = fighter_records.get(fighter_name, {'wins': 0, 'losses': 0, 'current_streak': 0, 'longest_win_streak': 0})
                
                fighter_data.append({
                    'name': fighter_name,
                    'height': safe_numeric(row['fighter1_height'], None),
                    'weight': safe_numeric(row['fighter1_curr_weight'], None),
                    'reach': safe_numeric(row['fighter1_reach'], None),
                    'dob': row['fighter1_dob'],
                    'stance': row.get('fighter1_stance', 'Orthodox'),
                    'sig_strikes_landed_pm': safe_numeric(row['fighter1_sig_strikes_landed_pm'], None),
                    'sig_strikes_accuracy': safe_numeric(row['fighter1_sig_strikes_accuracy'], None),
                    'sig_strikes_absorbed_pm': safe_numeric(row['fighter1_sig_strikes_absorbed_pm'], None),
                    'sig_strikes_defended': safe_numeric(row['fighter1_sig_strikes_defended'], None),
                    'takedown_avg_per15m': safe_numeric(row['fighter1_takedown_avg_per15m'], None),
                    'takedown_accuracy': safe_numeric(row['fighter1_takedown_accuracy'], None),
                    'takedown_defence': safe_numeric(row['fighter1_takedown_defence'], None),
                    'submission_avg_attempted_per15m': safe_numeric(row['fighter1_submission_avg_attempted_per15m'], None),
                    'wins': record['wins'],
                    'losses': record['losses'],
                    'current_streak': record['current_streak'],
                    'longest_win_streak': record['longest_win_streak']
                })
        
        # Add fighter2 data
        for _, row in df.iterrows():
            if pd.notna(row['fighter2']):
                fighter_name = row['fighter2']
                record = fighter_records.get(fighter_name, {'wins': 0, 'losses': 0, 'current_streak': 0, 'longest_win_streak': 0})
                
                fighter_data.append({
                    'name': fighter_name,
                    'height': safe_numeric(row['fighter2_height'], None),
                    'weight': safe_numeric(row['fighter2_curr_weight'], None),
                    'reach': safe_numeric(row['fighter2_reach'], None),
                    'dob': row['fighter2_dob'],
                    'stance': row.get('fighter2_stance', 'Orthodox'),
                    'sig_strikes_landed_pm': safe_numeric(row['fighter2_sig_strikes_landed_pm'], None),
                    'sig_strikes_accuracy': safe_numeric(row['fighter2_sig_strikes_accuracy'], None),
                    'sig_strikes_absorbed_pm': safe_numeric(row['fighter2_sig_strikes_absorbed_pm'], None),
                    'sig_strikes_defended': safe_numeric(row['fighter2_sig_strikes_defended'], None),
                    'takedown_avg_per15m': safe_numeric(row['fighter2_takedown_avg_per15m'], None),
                    'takedown_accuracy': safe_numeric(row['fighter2_takedown_accuracy'], None),
                    'takedown_defence': safe_numeric(row['fighter2_takedown_defence'], None),
                    'submission_avg_attempted_per15m': safe_numeric(row['fighter2_submission_avg_attempted_per15m'], None),
                    'wins': record['wins'],
                    'losses': record['losses'],
                    'current_streak': record['current_streak'],
                    'longest_win_streak': record['longest_win_streak']
                })
        
        # Create fighter stats DataFrame and aggregate
        fighter_df = pd.DataFrame(fighter_data)
        
        # Aggregate stats by fighter (using median for numeric stats, sum for wins/losses)
        self.fighter_stats = fighter_df.groupby('name').agg({
            'height': 'median',
            'weight': 'median', 
            'reach': 'median',
            'dob': 'first',
            'stance': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Orthodox',
            'sig_strikes_landed_pm': 'median',
            'sig_strikes_accuracy': 'median',
            'sig_strikes_absorbed_pm': 'median',
            'sig_strikes_defended': 'median',
            'takedown_avg_per15m': 'median',
            'takedown_accuracy': 'median',
            'takedown_defence': 'median',
            'submission_avg_attempted_per15m': 'median',
            'wins': 'max',
            'losses': 'max',
            'current_streak': 'last',
            'longest_win_streak': 'max'
        }).to_dict('index')
        
        # Calculate additional derived stats
        for fighter, stats in self.fighter_stats.items():
            total_fights = stats['wins'] + stats['losses']
            stats['win_rate'] = stats['wins'] / total_fights if total_fights > 0 else 0.5
            stats['experience'] = total_fights
            
            # Calculate striking efficiency
            if stats['sig_strikes_landed_pm'] and stats['sig_strikes_absorbed_pm']:
                stats['striking_differential'] = stats['sig_strikes_landed_pm'] - stats['sig_strikes_absorbed_pm']
            else:
                stats['striking_differential'] = 0
        
        print(f"Created enhanced database for {len(self.fighter_stats)} unique fighters")
    
    def _create_enhanced_training_data(self, df):
        """Create enhanced training data with more features"""
        print("Creating enhanced training features...")
        
        training_data = []
        
        for _, row in df.iterrows():
            # Skip if missing critical data
            if pd.isna(row['fighter1']) or pd.isna(row['fighter2']):
                continue
                
            # Create features for fighter1 winning
            features_f1_wins = self._extract_enhanced_fight_features(row, fighter1_wins=True)
            if features_f1_wins is not None:
                training_data.append(features_f1_wins + [1])  # fighter1 wins
            
            # Create features for fighter2 winning (swap fighters)
            features_f2_wins = self._extract_enhanced_fight_features(row, fighter1_wins=False)
            if features_f2_wins is not None:
                training_data.append(features_f2_wins + [0])  # fighter1 loses (fighter2 wins)
        
        # Enhanced feature names
        self.feature_names = [
            # Physical advantages
            'height_diff', 'reach_diff', 'age_diff', 'weight_diff',
            'reach_height_ratio_diff',  # New: reach/height ratio difference
            
            # Striking features
            'striking_accuracy_diff', 'striking_defense_diff', 'striking_output_diff',
            'striking_differential_diff',  # New: strikes landed - absorbed
            'striking_efficiency_diff',  # New: accuracy * output
            
            # Grappling features
            'takedown_accuracy_diff', 'takedown_defense_diff', 'takedown_output_diff',
            'submission_diff', 'grappling_score_diff',  # New: combined grappling metric
            
            # Experience and momentum
            'win_rate_diff',  # New: historical win percentage
            'experience_diff',  # New: total fights difference
            'current_streak_diff',  # New: current win/loss streak
            'longest_streak_diff',  # New: longest win streak
            
            # Stance and betting
            'stance_advantage', 'is_betting_favorite',
            
            # Interaction features
            'size_striking_interaction',  # New: size advantage * striking differential
            'experience_momentum_interaction',  # New: experience * current streak
            
            # Weight class specific (one-hot encoded)
            'is_heavyweight', 'is_lightweight', 'is_welterweight', 'is_middleweight',
            'is_womens_division'
        ]
        
        columns = self.feature_names + ['target']
        training_df = pd.DataFrame(training_data, columns=columns)
        
        # Remove any rows with too many NaN values
        training_df = training_df.dropna(thresh=len(training_df.columns) * 0.7)
        
        print(f"Created {len(training_df)} enhanced training examples with {len(self.feature_names)} features")
        return training_df
    
    def _extract_enhanced_fight_features(self, row, fighter1_wins=True):
        """Extract enhanced features for a fight"""
        try:
            if fighter1_wins:
                f1_name, f2_name = row['fighter1'], row['fighter2']
                is_favorite = 1 if row.get('favourite') == row['fighter1'] else 0 if pd.notna(row.get('favourite')) else 0.5
            else:
                f1_name, f2_name = row['fighter2'], row['fighter1']
                is_favorite = 1 if row.get('favourite') == row['fighter2'] else 0 if pd.notna(row.get('favourite')) else 0.5
            
            # Get fighter stats from database
            if f1_name not in self.fighter_stats or f2_name not in self.fighter_stats:
                return None
            
            f1_stats = self.fighter_stats[f1_name]
            f2_stats = self.fighter_stats[f2_name]
            
            # Helper function
            def safe_get(stats, key, default):
                value = stats.get(key, default)
                return value if pd.notna(value) else default
            
            # Calculate ages
            try:
                event_date = pd.to_datetime(row['event_date'])
                f1_age = (event_date - pd.to_datetime(f1_stats['dob'])).days / 365.25 if pd.notna(f1_stats['dob']) else 30
                f2_age = (event_date - pd.to_datetime(f2_stats['dob'])).days / 365.25 if pd.notna(f2_stats['dob']) else 30
            except:
                f1_age, f2_age = 30, 30
            
            # Extract all stats
            f1_height = safe_get(f1_stats, 'height', 175)
            f2_height = safe_get(f2_stats, 'height', 175)
            f1_reach = safe_get(f1_stats, 'reach', 180)
            f2_reach = safe_get(f2_stats, 'reach', 180)
            f1_weight = safe_get(f1_stats, 'weight', 70)
            f2_weight = safe_get(f2_stats, 'weight', 70)
            
            # Calculate reach/height ratio
            f1_reach_height_ratio = f1_reach / f1_height if f1_height > 0 else 1.03
            f2_reach_height_ratio = f2_reach / f2_height if f2_height > 0 else 1.03
            
            # Striking stats
            f1_striking_acc = safe_get(f1_stats, 'sig_strikes_accuracy', 0.45)
            f2_striking_acc = safe_get(f2_stats, 'sig_strikes_accuracy', 0.45)
            f1_striking_def = safe_get(f1_stats, 'sig_strikes_defended', 0.55)
            f2_striking_def = safe_get(f2_stats, 'sig_strikes_defended', 0.55)
            f1_striking_out = safe_get(f1_stats, 'sig_strikes_landed_pm', 4.0)
            f2_striking_out = safe_get(f2_stats, 'sig_strikes_landed_pm', 4.0)
            f1_striking_diff = safe_get(f1_stats, 'striking_differential', 0)
            f2_striking_diff = safe_get(f2_stats, 'striking_differential', 0)
            
            # Striking efficiency
            f1_striking_eff = f1_striking_acc * f1_striking_out
            f2_striking_eff = f2_striking_acc * f2_striking_out
            
            # Grappling stats
            f1_takedown_acc = safe_get(f1_stats, 'takedown_accuracy', 0.35)
            f2_takedown_acc = safe_get(f2_stats, 'takedown_accuracy', 0.35)
            f1_takedown_def = safe_get(f1_stats, 'takedown_defence', 0.65)
            f2_takedown_def = safe_get(f2_stats, 'takedown_defence', 0.65)
            f1_takedown_out = safe_get(f1_stats, 'takedown_avg_per15m', 1.5)
            f2_takedown_out = safe_get(f2_stats, 'takedown_avg_per15m', 1.5)
            f1_submission = safe_get(f1_stats, 'submission_avg_attempted_per15m', 0.5)
            f2_submission = safe_get(f2_stats, 'submission_avg_attempted_per15m', 0.5)
            
            # Grappling score (combined metric)
            f1_grappling = (f1_takedown_acc * f1_takedown_out + f1_takedown_def + f1_submission) / 3
            f2_grappling = (f2_takedown_acc * f2_takedown_out + f2_takedown_def + f2_submission) / 3
            
            # Experience and record stats
            f1_win_rate = safe_get(f1_stats, 'win_rate', 0.5)
            f2_win_rate = safe_get(f2_stats, 'win_rate', 0.5)
            f1_experience = safe_get(f1_stats, 'experience', 10)
            f2_experience = safe_get(f2_stats, 'experience', 10)
            f1_streak = safe_get(f1_stats, 'current_streak', 0)
            f2_streak = safe_get(f2_stats, 'current_streak', 0)
            f1_longest_streak = safe_get(f1_stats, 'longest_win_streak', 0)
            f2_longest_streak = safe_get(f2_stats, 'longest_win_streak', 0)
            
            # Stance advantage
            stance_adv = 0
            f1_stance = f1_stats.get('stance', 'Orthodox')
            f2_stance = f2_stats.get('stance', 'Orthodox')
            if isinstance(f1_stance, str) and isinstance(f2_stance, str):
                if f1_stance == 'Southpaw' and f2_stance == 'Orthodox':
                    stance_adv = 1
                elif f1_stance == 'Orthodox' and f2_stance == 'Southpaw':
                    stance_adv = -1
            
            # Weight class encoding
            weight_class = row.get('weight_class', 'Unknown').lower()
            is_heavyweight = 1 if 'heavyweight' in weight_class and 'light' not in weight_class else 0
            is_lightweight = 1 if 'lightweight' in weight_class else 0
            is_welterweight = 1 if 'welterweight' in weight_class else 0
            is_middleweight = 1 if 'middleweight' in weight_class else 0
            is_womens = 1 if "women" in weight_class or "female" in weight_class else 0
            
            # Calculate all feature differences
            height_diff = f1_height - f2_height
            reach_diff = f1_reach - f2_reach
            size_advantage = (height_diff + reach_diff) / 2
            
            features = [
                # Physical
                height_diff,
                reach_diff,
                f1_age - f2_age,
                f1_weight - f2_weight,
                f1_reach_height_ratio - f2_reach_height_ratio,
                
                # Striking
                f1_striking_acc - f2_striking_acc,
                f1_striking_def - f2_striking_def,
                f1_striking_out - f2_striking_out,
                f1_striking_diff - f2_striking_diff,
                f1_striking_eff - f2_striking_eff,
                
                # Grappling
                f1_takedown_acc - f2_takedown_acc,
                f1_takedown_def - f2_takedown_def,
                f1_takedown_out - f2_takedown_out,
                f1_submission - f2_submission,
                f1_grappling - f2_grappling,
                
                # Experience/momentum
                f1_win_rate - f2_win_rate,
                f1_experience - f2_experience,
                f1_streak - f2_streak,
                f1_longest_streak - f2_longest_streak,
                
                # Other
                stance_adv,
                is_favorite,
                
                # Interactions
                size_advantage * (f1_striking_diff - f2_striking_diff),
                (f1_experience - f2_experience) * (f1_streak - f2_streak),
                
                # Weight classes
                is_heavyweight,
                is_lightweight,
                is_welterweight,
                is_middleweight,
                is_womens
            ]
            
            return features
            
        except Exception as e:
            print(f"Error processing enhanced fight features: {e}")
            return None
    
    def train_model(self, training_data, model_type='ensemble', optimize_hyperparameters=False):
        """Train the prediction model with enhanced techniques"""
        print(f"Training enhanced {model_type} model...")
        
        # Prepare data
        X = training_data[self.feature_names]
        y = training_data['target']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature selection
        print("Performing feature selection...")
        selector = SelectKBest(f_classif, k=min(20, len(self.feature_names)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"Selected {len(selected_features)} best features")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Train model based on type
        if model_type == 'ensemble':
            # Create ensemble of models
            models = []
            
            # Random Forest
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            models.append(('rf', rf))
            
            # Gradient Boosting
            gb = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            models.append(('gb', gb))
            
            # XGBoost (if available)
            if HAS_XGBOOST:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                models.append(('xgb', xgb_model))
            
            # Logistic Regression
            lr = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
            models.append(('lr', lr))
            
            # Create voting classifier
            self.model = VotingClassifier(estimators=models, voting='soft')
            
        elif model_type == 'xgboost':
            if not HAS_XGBOOST:
                print("XGBoost not available, falling back to Random Forest")
                model_type = 'random_forest'
            else:
                if optimize_hyperparameters:
                    print("Optimizing XGBoost hyperparameters...")
                    param_grid = {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [4, 6],
                    }
                    
                    xgb_model = xgb.XGBClassifier(
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                    
                    grid_search = GridSearchCV(
                        xgb_model, param_grid, cv=3, 
                        scoring='accuracy', n_jobs=-1, verbose=1
                    )
                    
                    grid_search.fit(X_train_scaled, y_train)
                    self.model = grid_search.best_estimator_
                    print(f"Best parameters: {grid_search.best_params_}")
                else:
                    self.model = xgb.XGBClassifier(
                        n_estimators=200,
                        learning_rate=0.1,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
        
        if model_type == 'random_forest':
            if optimize_hyperparameters:
                print("Optimizing Random Forest hyperparameters...")
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                
                rf = RandomForestClassifier(random_state=42)
                grid_search = GridSearchCV(
                    rf, param_grid, cv=3, 
                    scoring='accuracy', n_jobs=-1, verbose=1
                )
                
                grid_search.fit(X_train_scaled, y_train)
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
        
        # Train the model
        print("Training final model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Store feature selector and selected features
        self.feature_selector = selector
        self.best_features = selected_features
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel training complete!")
        print(f"Test Accuracy: {accuracy:.3f}")
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"AUC-ROC Score: {auc:.3f}")
        except:
            pass
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        return accuracy
    
    def get_fighter_names(self):
        """Get list of all available fighters"""
        return list(self.fighter_stats.keys())
    
    def search_fighters(self, name_query):
        """Search for fighters by partial name match"""
        name_query = name_query.lower()
        matches = [name for name in self.fighter_stats.keys() 
                  if name_query in name.lower()]
        return sorted(matches)
    
    def predict_fight(self, fighter1_name, fighter2_name, fighter1_is_favorite=None):
        """Predict outcome of a fight between two fighters with confidence calibration"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Check if fighters exist
        if fighter1_name not in self.fighter_stats:
            raise ValueError(f"Fighter '{fighter1_name}' not found in database")
        if fighter2_name not in self.fighter_stats:
            raise ValueError(f"Fighter '{fighter2_name}' not found in database")
        
        # Create a dummy row to extract features
        dummy_row = {
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'event_date': datetime.now(),
            'favourite': fighter1_name if fighter1_is_favorite else (fighter2_name if fighter1_is_favorite is False else None),
            'weight_class': 'Unknown'  # Could be enhanced by inferring from fighter weights
        }
        
        # Extract features
        features = self._extract_enhanced_fight_features(dummy_row, fighter1_wins=True)
        
        if features is None:
            raise ValueError("Could not extract features for this matchup")
        
        # Create DataFrame with proper feature names
        features_df = pd.DataFrame([features], columns=self.feature_names)
        
        # Fill missing values
        features_df = features_df.fillna(features_df.median())
        
        # Select best features
        if self.feature_selector:
            features_selected = self.feature_selector.transform(features_df)
        else:
            features_selected = features_df
        
        # Scale features
        features_scaled = self.scaler.transform(features_selected)
        
        # Get probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        fighter1_win_prob = probabilities[1]  # Probability of class 1 (fighter1 wins)
        fighter2_win_prob = probabilities[0]  # Probability of class 0 (fighter2 wins)
        
        # Calibrate confidence (ensemble models tend to be overconfident)
        # Apply a slight smoothing towards 0.5
        calibration_factor = 0.8
        fighter1_win_prob = 0.5 + (fighter1_win_prob - 0.5) * calibration_factor
        fighter2_win_prob = 1 - fighter1_win_prob
        
        # Determine prediction
        predicted_winner = fighter1_name if fighter1_win_prob > 0.5 else fighter2_name
        confidence = max(fighter1_win_prob, fighter2_win_prob)
        
        # Get feature values for analysis
        feature_values = dict(zip(self.feature_names, features))
        
        # Identify key advantages
        advantages = self._analyze_advantages(feature_values, fighter1_name, fighter2_name)
        
        return {
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'fighter1_win_probability': fighter1_win_prob,
            'fighter2_win_probability': fighter2_win_prob,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'features': feature_values,
            'key_advantages': advantages
        }
    
    def _analyze_advantages(self, features, fighter1_name, fighter2_name):
        """Analyze key advantages for each fighter"""
        advantages = {fighter1_name: [], fighter2_name: []}
        
        # Physical advantages
        if features['height_diff'] > 5:
            advantages[fighter1_name].append(f"Height advantage ({features['height_diff']:.1f}cm)")
        elif features['height_diff'] < -5:
            advantages[fighter2_name].append(f"Height advantage ({-features['height_diff']:.1f}cm)")
        
        if features['reach_diff'] > 5:
            advantages[fighter1_name].append(f"Reach advantage ({features['reach_diff']:.1f}cm)")
        elif features['reach_diff'] < -5:
            advantages[fighter2_name].append(f"Reach advantage ({-features['reach_diff']:.1f}cm)")
        
        # Experience advantages
        if features['experience_diff'] > 5:
            advantages[fighter1_name].append(f"More experienced ({features['experience_diff']:.0f} more fights)")
        elif features['experience_diff'] < -5:
            advantages[fighter2_name].append(f"More experienced ({-features['experience_diff']:.0f} more fights)")
        
        # Momentum advantages
        if features['current_streak_diff'] > 2:
            advantages[fighter1_name].append(f"Better momentum (streak diff: {features['current_streak_diff']:.0f})")
        elif features['current_streak_diff'] < -2:
            advantages[fighter2_name].append(f"Better momentum (streak diff: {-features['current_streak_diff']:.0f})")
        
        # Striking advantages
        if features['striking_differential_diff'] > 1:
            advantages[fighter1_name].append("Superior striking differential")
        elif features['striking_differential_diff'] < -1:
            advantages[fighter2_name].append("Superior striking differential")
        
        # Win rate advantages
        if features['win_rate_diff'] > 0.1:
            advantages[fighter1_name].append(f"Better win rate ({features['win_rate_diff']*100:.1f}% higher)")
        elif features['win_rate_diff'] < -0.1:
            advantages[fighter2_name].append(f"Better win rate ({-features['win_rate_diff']*100:.1f}% higher)")
        
        return advantages
    
    def analyze_matchup(self, fighter1_name, fighter2_name):
        """Provide detailed analysis of the matchup"""
        if fighter1_name not in self.fighter_stats or fighter2_name not in self.fighter_stats:
            return "One or both fighters not found in database"
        
        f1_stats = self.fighter_stats[fighter1_name]
        f2_stats = self.fighter_stats[fighter2_name]
        
        # Make prediction
        prediction = self.predict_fight(fighter1_name, fighter2_name)
        
        analysis = f"\n=== ENHANCED FIGHT ANALYSIS: {fighter1_name} vs {fighter2_name} ===\n"
        
        # Prediction summary
        analysis += f"\n🎯 PREDICTION:\n"
        analysis += f"Winner: {prediction['predicted_winner']} ({prediction['confidence']:.1%} confidence)\n"
        analysis += f"Probabilities: {fighter1_name} {prediction['fighter1_win_probability']:.1%} vs {fighter2_name} {prediction['fighter2_win_probability']:.1%}\n"
        
        # Key advantages
        analysis += "\n🔑 KEY ADVANTAGES:\n"
        for fighter, advantages in prediction['key_advantages'].items():
            if advantages:
                analysis += f"{fighter}:\n"
                for adv in advantages:
                    analysis += f"  • {adv}\n"
        
        # Physical comparison
        analysis += "\n🥊 PHYSICAL ATTRIBUTES:\n"
        analysis += f"Height: {fighter1_name} {f1_stats.get('height', 'N/A')}cm vs {fighter2_name} {f2_stats.get('height', 'N/A')}cm\n"
        analysis += f"Reach: {fighter1_name} {f1_stats.get('reach', 'N/A')}cm vs {fighter2_name} {f2_stats.get('reach', 'N/A')}cm\n"
        analysis += f"Stance: {fighter1_name} {f1_stats.get('stance', 'N/A')} vs {fighter2_name} {f2_stats.get('stance', 'N/A')}\n"
        
        # Performance comparison
        analysis += "\n📊 PERFORMANCE STATS:\n"
        analysis += f"Record: {fighter1_name} {f1_stats.get('wins', 0)}-{f1_stats.get('losses', 0)} vs {fighter2_name} {f2_stats.get('wins', 0)}-{f2_stats.get('losses', 0)}\n"
        analysis += f"Win Rate: {fighter1_name} {f1_stats.get('win_rate', 0)*100:.1f}% vs {fighter2_name} {f2_stats.get('win_rate', 0)*100:.1f}%\n"
        analysis += f"Current Streak: {fighter1_name} {f1_stats.get('current_streak', 0)} vs {fighter2_name} {f2_stats.get('current_streak', 0)}\n"
        
        # Striking stats
        analysis += "\n🥊 STRIKING:\n"
        analysis += f"Strikes Landed/Min: {fighter1_name} {f1_stats.get('sig_strikes_landed_pm', 'N/A')} vs {fighter2_name} {f2_stats.get('sig_strikes_landed_pm', 'N/A')}\n"
        analysis += f"Striking Accuracy: {fighter1_name} {f1_stats.get('sig_strikes_accuracy', 'N/A')} vs {fighter2_name} {f2_stats.get('sig_strikes_accuracy', 'N/A')}\n"
        analysis += f"Striking Defense: {fighter1_name} {f1_stats.get('sig_strikes_defended', 'N/A')} vs {fighter2_name} {f2_stats.get('sig_strikes_defended', 'N/A')}\n"
        
        # Grappling stats
        analysis += "\n🤼 GRAPPLING:\n"
        analysis += f"Takedown Accuracy: {fighter1_name} {f1_stats.get('takedown_accuracy', 'N/A')} vs {fighter2_name} {f2_stats.get('takedown_accuracy', 'N/A')}\n"
        analysis += f"Takedown Defense: {fighter1_name} {f1_stats.get('takedown_defence', 'N/A')} vs {fighter2_name} {f2_stats.get('takedown_defence', 'N/A')}\n"
        
        return analysis
    
    def predict_event(self, fights_list):
        """Predict outcomes for an entire fight card"""
        predictions = []
        
        for fight in fights_list:
            try:
                if isinstance(fight, tuple) and len(fight) >= 2:
                    fighter1, fighter2 = fight[0], fight[1]
                    is_favorite = fight[2] if len(fight) > 2 else None
                elif isinstance(fight, dict):
                    fighter1 = fight.get('fighter1')
                    fighter2 = fight.get('fighter2')
                    is_favorite = fight.get('is_favorite', None)
                else:
                    continue
                
                # Search for fighters
                f1_matches = self.search_fighters(fighter1)
                f2_matches = self.search_fighters(fighter2)
                
                if f1_matches and f2_matches:
                    result = self.predict_fight(f1_matches[0], f2_matches[0], is_favorite)
                    predictions.append(result)
                else:
                    print(f"Could not find one or both fighters: {fighter1}, {fighter2}")
                    
            except Exception as e:
                print(f"Error predicting {fight}: {e}")
                continue
        
        return predictions


# Make it compatible with the original class name
UFCFightPredictor = ImprovedUFCFightPredictor


# Example usage
if __name__ == "__main__":
    # Initialize the improved predictor
    predictor = ImprovedUFCFightPredictor()
    
    # Load and prepare data
    csv_file_path = "data/complete_ufc_data.csv"
    
    try:
        training_data = predictor.load_and_prepare_data(csv_file_path)
        
        # Train the model with ensemble approach
        accuracy = predictor.train_model(training_data, model_type='ensemble', optimize_hyperparameters=False)
        
        print("\n" + "="*50)
        print("IMPROVED MODEL READY FOR USE!")
        print("="*50)
        
        # Example predictions
        print("\n📋 EXAMPLE PREDICTIONS:")
        
        # Test some recent fights
        test_fights = [
            ("Jon Jones", "Stipe Miocic"),
            ("Alex Pereira", "Khalil Rountree"),
            ("Islam Makhachev", "Arman Tsarukyan"),
            ("Merab Dvalishvili", "Sean O'Malley"),
        ]
        
        for fighter1, fighter2 in test_fights:
            try:
                # Search for exact fighter names
                f1_matches = predictor.search_fighters(fighter1)
                f2_matches = predictor.search_fighters(fighter2)
                
                if f1_matches and f2_matches:
                    print(f"\n🥊 {f1_matches[0]} vs {f2_matches[0]}")
                    result = predictor.predict_fight(f1_matches[0], f2_matches[0])
                    print(f"   Predicted Winner: {result['predicted_winner']} ({result['confidence']:.1%})")
                    print(f"   Probabilities: {result['fighter1']} {result['fighter1_win_probability']:.1%}, {result['fighter2']} {result['fighter2_win_probability']:.1%}")
                    
                    # Show key advantages
                    if result['key_advantages'][result['predicted_winner']]:
                        print(f"   Key Advantages for {result['predicted_winner']}:")
                        for adv in result['key_advantages'][result['predicted_winner']][:3]:
                            print(f"     • {adv}")
                
            except Exception as e:
                print(f"Error predicting {fighter1} vs {fighter2}: {e}")
        
        # Detailed analysis example
        print("\n" + "="*50)
        print("DETAILED MATCHUP ANALYSIS EXAMPLE:")
        print("="*50)
        
        if len(predictor.search_fighters("McGregor")) > 0 and len(predictor.search_fighters("Khabib")) > 0:
            analysis = predictor.analyze_matchup(
                predictor.search_fighters("McGregor")[0],
                predictor.search_fighters("Khabib")[0]
            )
            print(analysis)
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file_path}'")
        print("Please make sure the CSV file is in the correct directory.")
    except Exception as e:
        print(f"Error: {e}")