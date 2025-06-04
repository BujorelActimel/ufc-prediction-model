import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class UFCFightPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.fighter_stats = {}
        self.feature_names = []
        
    def load_and_prepare_data(self, csv_file_path):
        """Load UFC data and prepare for modeling"""
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
        self._build_fighter_database(df)
        
        # Create training data
        training_data = self._create_training_data(df)
        return training_data
    
    def _build_fighter_database(self, df):
        """Build a database of fighter statistics"""
        print("Building fighter statistics database...")
        
        # Helper function to safely convert to numeric
        def safe_numeric(value, default):
            if pd.isna(value):
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Combine fighter1 and fighter2 data
        fighter_data = []
        
        # Add fighter1 data
        for _, row in df.iterrows():
            if pd.notna(row['fighter1']):
                fighter_data.append({
                    'name': row['fighter1'],
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
                    'submission_avg_attempted_per15m': safe_numeric(row['fighter1_submission_avg_attempted_per15m'], None)
                })
        
        # Add fighter2 data
        for _, row in df.iterrows():
            if pd.notna(row['fighter2']):
                fighter_data.append({
                    'name': row['fighter2'],
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
                    'submission_avg_attempted_per15m': safe_numeric(row['fighter2_submission_avg_attempted_per15m'], None)
                })
        
        # Create fighter stats DataFrame and aggregate
        fighter_df = pd.DataFrame(fighter_data)
        
        # Aggregate stats by fighter (using median to handle multiple entries)
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
            'submission_avg_attempted_per15m': 'median'
        }).to_dict('index')
        
        print(f"Created database for {len(self.fighter_stats)} unique fighters")
    
    def _create_training_data(self, df):
        """Create training data from fight records"""
        print("Creating training features...")
        
        training_data = []
        
        for _, row in df.iterrows():
            # Skip if missing critical data
            if pd.isna(row['fighter1']) or pd.isna(row['fighter2']):
                continue
                
            # Create features for fighter1 winning
            features_f1_wins = self._extract_fight_features(row, fighter1_wins=True)
            if features_f1_wins is not None:
                training_data.append(features_f1_wins + [1])  # fighter1 wins
            
            # Create features for fighter2 winning (swap fighters)
            features_f2_wins = self._extract_fight_features(row, fighter1_wins=False)
            if features_f2_wins is not None:
                training_data.append(features_f2_wins + [0])  # fighter1 loses (fighter2 wins)
        
        # Convert to DataFrame
        self.feature_names = [
            'height_diff', 'reach_diff', 'age_diff', 'weight_diff',
            'striking_accuracy_diff', 'striking_defense_diff', 'striking_output_diff',
            'takedown_accuracy_diff', 'takedown_defense_diff', 'submission_diff',
            'stance_advantage', 'is_betting_favorite'
        ]
        
        columns = self.feature_names + ['target']
        training_df = pd.DataFrame(training_data, columns=columns)
        
        print(f"Created {len(training_df)} training examples")
        return training_df
    
    def _extract_fight_features(self, row, fighter1_wins=True):
        """Extract features for a fight"""
        try:
            if fighter1_wins:
                f1_name, f2_name = row['fighter1'], row['fighter2']
                is_favorite = 1 if row.get('favourite') == row['fighter1'] else 0 if pd.notna(row.get('favourite')) else 0.5
            else:
                f1_name, f2_name = row['fighter2'], row['fighter1']
                is_favorite = 1 if row.get('favourite') == row['fighter2'] else 0 if pd.notna(row.get('favourite')) else 0.5
            
            # Helper function to safely get numeric values
            def safe_numeric(value, default):
                if pd.isna(value):
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            # Get fighter data based on who we're treating as fighter1
            if fighter1_wins:
                # Calculate age from DOB
                try:
                    event_date = pd.to_datetime(row['event_date'])
                    f1_age = (event_date - pd.to_datetime(row['fighter1_dob'])).days / 365.25 if pd.notna(row['fighter1_dob']) else 30
                    f2_age = (event_date - pd.to_datetime(row['fighter2_dob'])).days / 365.25 if pd.notna(row['fighter2_dob']) else 30
                except:
                    f1_age, f2_age = 30, 30
                
                # Extract physical stats with safe conversion
                f1_height = safe_numeric(row['fighter1_height'], 175)
                f2_height = safe_numeric(row['fighter2_height'], 175)
                f1_weight = safe_numeric(row['fighter1_curr_weight'], 70)
                f2_weight = safe_numeric(row['fighter2_curr_weight'], 70)
                f1_reach = safe_numeric(row['fighter1_reach'], 180)
                f2_reach = safe_numeric(row['fighter2_reach'], 180)
                
                # Extract performance stats with safe conversion
                f1_striking_acc = safe_numeric(row['fighter1_sig_strikes_accuracy'], 0.45)
                f2_striking_acc = safe_numeric(row['fighter2_sig_strikes_accuracy'], 0.45)
                f1_striking_def = safe_numeric(row['fighter1_sig_strikes_defended'], 0.55)
                f2_striking_def = safe_numeric(row['fighter2_sig_strikes_defended'], 0.55)
                f1_striking_out = safe_numeric(row['fighter1_sig_strikes_landed_pm'], 4.0)
                f2_striking_out = safe_numeric(row['fighter2_sig_strikes_landed_pm'], 4.0)
                f1_takedown_acc = safe_numeric(row['fighter1_takedown_accuracy'], 0.35)
                f2_takedown_acc = safe_numeric(row['fighter2_takedown_accuracy'], 0.35)
                f1_takedown_def = safe_numeric(row['fighter1_takedown_defence'], 0.65)
                f2_takedown_def = safe_numeric(row['fighter2_takedown_defence'], 0.65)
                f1_submission = safe_numeric(row['fighter1_submission_avg_attempted_per15m'], 0.5)
                f2_submission = safe_numeric(row['fighter2_submission_avg_attempted_per15m'], 0.5)
                
                # Stance info
                f1_stance = row.get('fighter1_stance', 'Orthodox')
                f2_stance = row.get('fighter2_stance', 'Orthodox')
            else:
                # Swap fighters for negative examples
                try:
                    event_date = pd.to_datetime(row['event_date'])
                    f1_age = (event_date - pd.to_datetime(row['fighter2_dob'])).days / 365.25 if pd.notna(row['fighter2_dob']) else 30
                    f2_age = (event_date - pd.to_datetime(row['fighter1_dob'])).days / 365.25 if pd.notna(row['fighter1_dob']) else 30
                except:
                    f1_age, f2_age = 30, 30
                
                # Extract physical stats with safe conversion (swapped)
                f1_height = safe_numeric(row['fighter2_height'], 175)
                f2_height = safe_numeric(row['fighter1_height'], 175)
                f1_weight = safe_numeric(row['fighter2_curr_weight'], 70)
                f2_weight = safe_numeric(row['fighter1_curr_weight'], 70)
                f1_reach = safe_numeric(row['fighter2_reach'], 180)
                f2_reach = safe_numeric(row['fighter1_reach'], 180)
                
                # Extract performance stats with safe conversion (swapped)
                f1_striking_acc = safe_numeric(row['fighter2_sig_strikes_accuracy'], 0.45)
                f2_striking_acc = safe_numeric(row['fighter1_sig_strikes_accuracy'], 0.45)
                f1_striking_def = safe_numeric(row['fighter2_sig_strikes_defended'], 0.55)
                f2_striking_def = safe_numeric(row['fighter1_sig_strikes_defended'], 0.55)
                f1_striking_out = safe_numeric(row['fighter2_sig_strikes_landed_pm'], 4.0)
                f2_striking_out = safe_numeric(row['fighter1_sig_strikes_landed_pm'], 4.0)
                f1_takedown_acc = safe_numeric(row['fighter2_takedown_accuracy'], 0.35)
                f2_takedown_acc = safe_numeric(row['fighter1_takedown_accuracy'], 0.35)
                f1_takedown_def = safe_numeric(row['fighter2_takedown_defence'], 0.65)
                f2_takedown_def = safe_numeric(row['fighter1_takedown_defence'], 0.65)
                f1_submission = safe_numeric(row['fighter2_submission_avg_attempted_per15m'], 0.5)
                f2_submission = safe_numeric(row['fighter1_submission_avg_attempted_per15m'], 0.5)
                
                # Stance info (swapped)
                f1_stance = row.get('fighter2_stance', 'Orthodox')
                f2_stance = row.get('fighter1_stance', 'Orthodox')
            
            # Stance advantage calculation
            stance_adv = 0
            if isinstance(f1_stance, str) and isinstance(f2_stance, str):
                if f1_stance == 'Southpaw' and f2_stance == 'Orthodox':
                    stance_adv = 1
                elif f1_stance == 'Orthodox' and f2_stance == 'Southpaw':
                    stance_adv = -1
            
            # Calculate feature differences
            features = [
                f1_height - f2_height,
                f1_reach - f2_reach,
                f1_age - f2_age,
                f1_weight - f2_weight,
                f1_striking_acc - f2_striking_acc,
                f1_striking_def - f2_striking_def,
                f1_striking_out - f2_striking_out,
                f1_takedown_acc - f2_takedown_acc,
                f1_takedown_def - f2_takedown_def,
                f1_submission - f2_submission,
                stance_adv,
                is_favorite
            ]
            
            return features
            
        except Exception as e:
            print(f"Error processing fight features: {e}")
            return None
    
    def train_model(self, training_data, model_type='random_forest'):
        """Train the prediction model"""
        print(f"Training {model_type} model...")
        
        X = training_data[self.feature_names]
        y = training_data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42)
        else:
            raise ValueError("Model type must be 'random_forest' or 'logistic_regression'")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model training complete!")
        print(f"Test Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance:")
            print(feature_importance)
        
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
        """Predict outcome of a fight between two fighters"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Check if fighters exist
        if fighter1_name not in self.fighter_stats:
            raise ValueError(f"Fighter '{fighter1_name}' not found in database")
        if fighter2_name not in self.fighter_stats:
            raise ValueError(f"Fighter '{fighter2_name}' not found in database")
        
        # Get fighter stats
        f1_stats = self.fighter_stats[fighter1_name]
        f2_stats = self.fighter_stats[fighter2_name]
        
        # Calculate ages (approximate if DOB available)
        try:
            from datetime import datetime
            current_date = datetime.now()
            f1_age = (current_date - pd.to_datetime(f1_stats['dob'])).days / 365.25 if pd.notna(f1_stats['dob']) else 30
            f2_age = (current_date - pd.to_datetime(f2_stats['dob'])).days / 365.25 if pd.notna(f2_stats['dob']) else 30
        except:
            f1_age, f2_age = 30, 30
        
        # Handle missing values with defaults
        def get_stat(stats, key, default):
            return stats[key] if pd.notna(stats[key]) else default
        
        # Calculate features
        features = [
            get_stat(f1_stats, 'height', 175) - get_stat(f2_stats, 'height', 175),
            get_stat(f1_stats, 'reach', 180) - get_stat(f2_stats, 'reach', 180),
            f1_age - f2_age,
            get_stat(f1_stats, 'weight', 70) - get_stat(f2_stats, 'weight', 70),
            get_stat(f1_stats, 'sig_strikes_accuracy', 0.45) - get_stat(f2_stats, 'sig_strikes_accuracy', 0.45),
            get_stat(f1_stats, 'sig_strikes_defended', 0.55) - get_stat(f2_stats, 'sig_strikes_defended', 0.55),
            get_stat(f1_stats, 'sig_strikes_landed_pm', 4.0) - get_stat(f2_stats, 'sig_strikes_landed_pm', 4.0),
            get_stat(f1_stats, 'takedown_accuracy', 0.35) - get_stat(f2_stats, 'takedown_accuracy', 0.35),
            get_stat(f1_stats, 'takedown_defence', 0.65) - get_stat(f2_stats, 'takedown_defence', 0.65),
            get_stat(f1_stats, 'submission_avg_attempted_per15m', 0.5) - get_stat(f2_stats, 'submission_avg_attempted_per15m', 0.5),
            1 if (f1_stats['stance'] == 'Southpaw' and f2_stats['stance'] == 'Orthodox') else -1 if (f1_stats['stance'] == 'Orthodox' and f2_stats['stance'] == 'Southpaw') else 0,
            1 if fighter1_is_favorite else 0 if fighter1_is_favorite is False else 0.5
        ]
        
        # Scale features and predict
        features_scaled = self.scaler.transform([features])
        
        # Get probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        fighter1_win_prob = probabilities[1]  # Probability of class 1 (fighter1 wins)
        fighter2_win_prob = probabilities[0]  # Probability of class 0 (fighter2 wins)
        
        # Determine prediction
        predicted_winner = fighter1_name if fighter1_win_prob > 0.5 else fighter2_name
        confidence = max(fighter1_win_prob, fighter2_win_prob)
        
        return {
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'fighter1_win_probability': fighter1_win_prob,
            'fighter2_win_probability': fighter2_win_prob,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'features': dict(zip(self.feature_names, features))
        }
    
    def analyze_matchup(self, fighter1_name, fighter2_name):
        """Provide detailed analysis of the matchup"""
        if fighter1_name not in self.fighter_stats or fighter2_name not in self.fighter_stats:
            return "One or both fighters not found in database"
        
        f1_stats = self.fighter_stats[fighter1_name]
        f2_stats = self.fighter_stats[fighter2_name]
        
        analysis = f"\n=== FIGHT ANALYSIS: {fighter1_name} vs {fighter2_name} ===\n"
        
        # Physical comparison
        analysis += "\n🥊 PHYSICAL ATTRIBUTES:\n"
        analysis += f"Height: {fighter1_name} {f1_stats.get('height', 'N/A')}cm vs {fighter2_name} {f2_stats.get('height', 'N/A')}cm\n"
        analysis += f"Reach: {fighter1_name} {f1_stats.get('reach', 'N/A')}cm vs {fighter2_name} {f2_stats.get('reach', 'N/A')}cm\n"
        analysis += f"Weight: {fighter1_name} {f1_stats.get('weight', 'N/A')}kg vs {fighter2_name} {f2_stats.get('weight', 'N/A')}kg\n"
        analysis += f"Stance: {fighter1_name} {f1_stats.get('stance', 'N/A')} vs {fighter2_name} {f2_stats.get('stance', 'N/A')}\n"
        
        # Performance comparison
        analysis += "\n📊 PERFORMANCE STATS:\n"
        analysis += f"Striking Accuracy: {fighter1_name} {f1_stats.get('sig_strikes_accuracy', 'N/A')} vs {fighter2_name} {f2_stats.get('sig_strikes_accuracy', 'N/A')}\n"
        analysis += f"Striking Defense: {fighter1_name} {f1_stats.get('sig_strikes_defended', 'N/A')} vs {fighter2_name} {f2_stats.get('sig_strikes_defended', 'N/A')}\n"
        analysis += f"Takedown Accuracy: {fighter1_name} {f1_stats.get('takedown_accuracy', 'N/A')} vs {fighter2_name} {f2_stats.get('takedown_accuracy', 'N/A')}\n"
        analysis += f"Takedown Defense: {fighter1_name} {f1_stats.get('takedown_defence', 'N/A')} vs {fighter2_name} {f2_stats.get('takedown_defence', 'N/A')}\n"
        
        return analysis


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the predictor
    predictor = UFCFightPredictor()
    
    # Load and prepare data (replace with your CSV file path)
    csv_file_path = "complete_ufc_data.csv"
    
    try:
        training_data = predictor.load_and_prepare_data(csv_file_path)
        
        # Train the model
        accuracy = predictor.train_model(training_data, model_type='random_forest')
        
        # Get list of available fighters
        fighters = predictor.get_fighter_names()
        print(f"\nAvailable fighters: {len(fighters)} total")
        print("Sample fighters:", fighters[:10])
        
        # Example predictions
        print("\n" + "="*50)
        print("EXAMPLE PREDICTIONS")
        print("="*50)
        
        # Search for specific fighters
        print("\nSearching for 'Jones':")
        jones_fighters = predictor.search_fighters("jones")
        print(jones_fighters[:5])
        
        print("\nSearching for 'Silva':")
        silva_fighters = predictor.search_fighters("silva")
        print(silva_fighters[:5])
        
        # Make some example predictions if fighters are found
        if len(jones_fighters) > 0 and len(silva_fighters) > 0:
            fighter1 = jones_fighters[0]
            fighter2 = silva_fighters[0]
            
            print(f"\n🥊 PREDICTION: {fighter1} vs {fighter2}")
            result = predictor.predict_fight(fighter1, fighter2)
            
            print(f"Fighter 1 ({fighter1}): {result['fighter1_win_probability']:.1%}")
            print(f"Fighter 2 ({fighter2}): {result['fighter2_win_probability']:.1%}")
            print(f"Predicted Winner: {result['predicted_winner']}")
            print(f"Confidence: {result['confidence']:.1%}")
            
            # Detailed analysis
            analysis = predictor.analyze_matchup(fighter1, fighter2)
            print(analysis)
        
        print("\n" + "="*50)
        print("MODEL READY FOR USE!")
        print("="*50)
        print("\nHow to use:")
        print("1. predictor.search_fighters('name') - Find fighters")
        print("2. predictor.predict_fight(fighter1, fighter2) - Make prediction")
        print("3. predictor.analyze_matchup(fighter1, fighter2) - Detailed analysis")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file_path}'")
        print("Please make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Error: {e}")