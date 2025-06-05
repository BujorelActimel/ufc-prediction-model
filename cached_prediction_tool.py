"""
Cached UFC Fight Prediction Tool
Uses model caching to avoid retraining every time
"""

import sys
import os
import pickle
import hashlib
from datetime import datetime
import json

try:
    from improved_model import ImprovedUFCFightPredictor
    print("✓ Improved model imported successfully")
except ImportError as e:
    print(f"✗ Error importing improved model: {e}")
    sys.exit(1)


class CachedPredictionTool:
    """Fast prediction tool with model caching"""
    
    def __init__(self, data_file="data/complete_ufc_data.csv", cache_dir="model_cache"):
        self.predictor = None
        self.data_file = data_file
        self.cache_dir = cache_dir
        self.cache_info_file = os.path.join(cache_dir, "cache_info.json")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_data_hash(self):
        """Get hash of the data file to detect changes"""
        if not os.path.exists(self.data_file):
            return None
        
        with open(self.data_file, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    def get_cache_path(self, model_type='ensemble'):
        """Get the cache file path for the model"""
        return os.path.join(self.cache_dir, f"ufc_model_{model_type}.pkl")
    
    def save_model_cache(self, model_type='ensemble'):
        """Save the trained model to cache"""
        if not self.predictor:
            return False
        
        try:
            cache_path = self.get_cache_path(model_type)
            
            # Save the entire predictor object
            with open(cache_path, 'wb') as f:
                pickle.dump(self.predictor, f)
            
            # Save cache info
            cache_info = {
                'model_type': model_type,
                'data_file': self.data_file,
                'data_hash': self.get_data_hash(),
                'cached_at': datetime.now().isoformat(),
                'fighter_count': len(self.predictor.get_fighter_names()),
                'feature_count': len(self.predictor.feature_names)
            }
            
            with open(self.cache_info_file, 'w') as f:
                json.dump(cache_info, f, indent=2)
            
            print(f"✅ Model cached successfully at {cache_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model cache: {e}")
            return False
    
    def load_model_cache(self, model_type='ensemble'):
        """Load the trained model from cache"""
        cache_path = self.get_cache_path(model_type)
        
        if not os.path.exists(cache_path):
            print(f"📝 No cache found for {model_type} model")
            return False
        
        if not os.path.exists(self.cache_info_file):
            print(f"📝 No cache info found")
            return False
        
        try:
            # Load cache info
            with open(self.cache_info_file, 'r') as f:
                cache_info = json.load(f)
            
            # Check if data file has changed
            current_hash = self.get_data_hash()
            if current_hash != cache_info.get('data_hash'):
                print(f"📝 Data file has changed since cache was created")
                return False
            
            # Load the model
            with open(cache_path, 'rb') as f:
                self.predictor = pickle.load(f)
            
            print(f"🚀 Model loaded from cache!")
            print(f"   📊 Cached at: {cache_info.get('cached_at', 'Unknown')}")
            print(f"   👥 Fighters: {cache_info.get('fighter_count', 'Unknown')}")
            print(f"   🎯 Features: {cache_info.get('feature_count', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model cache: {e}")
            print(f"📝 Will retrain model...")
            return False
    
    def load_model(self, model_type='ensemble', force_retrain=False):
        """Load model from cache or train new one"""
        print("🤖 Loading UFC Prediction Model...")
        print("-" * 40)
        
        if not os.path.exists(self.data_file):
            print(f"✗ Data file not found: {self.data_file}")
            return False
        
        # Try to load from cache first (unless force retrain)
        if not force_retrain and self.load_model_cache(model_type):
            return True
        
        # Train new model
        print(f"🏋️ Training new {model_type} model...")
        print("⏳ This may take a few minutes...")
        
        try:
            self.predictor = ImprovedUFCFightPredictor()
            training_data = self.predictor.load_and_prepare_data(self.data_file)
            accuracy = self.predictor.train_model(training_data, model_type=model_type)
            
            print(f"✅ Model trained successfully!")
            print(f"📊 Training accuracy: {accuracy:.1%}")
            print(f"👥 Database contains {len(self.predictor.get_fighter_names())} fighters")
            
            # Save to cache
            self.save_model_cache(model_type)
            
            return True
            
        except Exception as e:
            print(f"✗ Error training model: {e}")
            return False
    
    def clear_cache(self):
        """Clear all cached models"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            
            for cache_file in cache_files:
                os.remove(os.path.join(self.cache_dir, cache_file))
            
            if os.path.exists(self.cache_info_file):
                os.remove(self.cache_info_file)
            
            print(f"🗑️ Cleared {len(cache_files)} cached models")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            return False
    
    def show_cache_info(self):
        """Show information about cached models"""
        print(f"\n💾 CACHE INFORMATION")
        print(f"{'='*40}")
        print(f"Cache Directory: {self.cache_dir}")
        
        if not os.path.exists(self.cache_dir):
            print("❌ Cache directory doesn't exist")
            return
        
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        
        if not cache_files:
            print("📝 No cached models found")
            return
        
        print(f"📁 Found {len(cache_files)} cached models:")
        
        for cache_file in cache_files:
            cache_path = os.path.join(self.cache_dir, cache_file)
            file_size = os.path.getsize(cache_path) / 1024 / 1024  # MB
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            
            print(f"   • {cache_file} ({file_size:.1f} MB, {file_time.strftime('%Y-%m-%d %H:%M')})")
        
        # Show cache info if available
        if os.path.exists(self.cache_info_file):
            try:
                with open(self.cache_info_file, 'r') as f:
                    cache_info = json.load(f)
                
                print(f"\n📋 Last Cache Info:")
                print(f"   Model Type: {cache_info.get('model_type', 'Unknown')}")
                print(f"   Data File: {cache_info.get('data_file', 'Unknown')}")
                print(f"   Cached At: {cache_info.get('cached_at', 'Unknown')}")
                print(f"   Fighters: {cache_info.get('fighter_count', 'Unknown')}")
                print(f"   Features: {cache_info.get('feature_count', 'Unknown')}")
                
            except Exception as e:
                print(f"❌ Error reading cache info: {e}")
    
    def search_fighter(self, name_query, show_details=True):
        """Search for a fighter and return best match"""
        matches = self.predictor.search_fighters(name_query)
        
        if not matches:
            if show_details:
                print(f"❌ No fighters found matching '{name_query}'")
            return None
        elif len(matches) == 1:
            if show_details:
                print(f"✅ Found: {matches[0]}")
            return matches[0]
        else:
            if show_details:
                print(f"📋 Found {len(matches)} matches:")
                for i, fighter in enumerate(matches[:10], 1):
                    print(f"  {i}. {fighter}")
                
                if len(matches) > 10:
                    print(f"  ... and {len(matches) - 10} more")
            
            return matches[0]  # Return best match for quick predictions
    
    def predict_fight(self, fighter1_query, fighter2_query, show_details=True, show_metrics=True):
        """Make a prediction for a fight with detailed metrics"""
        if not self.predictor:
            print("❌ Model not loaded!")
            return None
        
        # Find fighters
        fighter1 = self.search_fighter(fighter1_query, show_details=False)
        fighter2 = self.search_fighter(fighter2_query, show_details=False)
        
        if not fighter1:
            print(f"❌ Could not find fighter: {fighter1_query}")
            # Show suggestions
            suggestions = self.predictor.search_fighters(fighter1_query.split()[0])[:5]
            if suggestions:
                print(f"   💡 Did you mean: {', '.join(suggestions)}")
            return None
        
        if not fighter2:
            print(f"❌ Could not find fighter: {fighter2_query}")
            # Show suggestions
            suggestions = self.predictor.search_fighters(fighter2_query.split()[0])[:5]
            if suggestions:
                print(f"   💡 Did you mean: {', '.join(suggestions)}")
            return None
        
        if fighter1 == fighter2:
            print("❌ Cannot predict a fight between the same fighter!")
            return None
        
        try:
            # Make prediction (this is now fast!)
            result = self.predictor.predict_fight(fighter1, fighter2)
            
            if show_details:
                print(f"\n🥊 FIGHT PREDICTION")
                print(f"{'='*60}")
                print(f"🔴 {fighter1}")
                print(f"🔵 {fighter2}")
                print(f"{'='*60}")
                print(f"🏆 PREDICTED WINNER: {result['predicted_winner']}")
                print(f"📊 CONFIDENCE: {result['confidence']:.1%}")
                print(f"\n📈 WIN PROBABILITIES:")
                print(f"   {fighter1}: {result['fighter1_win_probability']:.1%}")
                print(f"   {fighter2}: {result['fighter2_win_probability']:.1%}")
                
                # Show detailed metrics that influenced the decision
                if show_metrics:
                    self._show_decision_metrics(result, fighter1, fighter2)
                
                # Show key advantages
                if 'key_advantages' in result:
                    winner_advantages = result['key_advantages'][result['predicted_winner']]
                    if winner_advantages:
                        print(f"\n🔑 KEY ADVANTAGES for {result['predicted_winner']}:")
                        for adv in winner_advantages[:5]:
                            print(f"   • {adv}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def _show_decision_metrics(self, result, fighter1, fighter2):
        """Show the key metrics that influenced the prediction decision"""
        print(f"\n🎯 DECISION ANALYSIS - Why {result['predicted_winner']} is favored:")
        print(f"{'='*60}")
        
        # Get feature values from the prediction
        features = result.get('features', {})
        if not features:
            print("   ⚠️ Feature details not available")
            return
        
        # Get fighter stats for comparison
        f1_stats = self.predictor.fighter_stats.get(fighter1, {})
        f2_stats = self.predictor.fighter_stats.get(fighter2, {})
        
        # Analyze key feature differences and their impact
        significant_features = []
        
        # Physical advantages
        if abs(features.get('height_diff', 0)) > 3:
            advantage = "Height" 
            diff = features['height_diff']
            beneficiary = fighter1 if diff > 0 else fighter2
            significant_features.append({
                'category': '📏 Physical',
                'metric': f"{advantage} advantage",
                'value': f"{abs(diff):.1f}cm",
                'beneficiary': beneficiary,
                'impact': 'High' if abs(diff) > 8 else 'Medium'
            })
        
        if abs(features.get('reach_diff', 0)) > 3:
            diff = features['reach_diff']
            beneficiary = fighter1 if diff > 0 else fighter2
            significant_features.append({
                'category': '📏 Physical',
                'metric': "Reach advantage",
                'value': f"{abs(diff):.1f}cm",
                'beneficiary': beneficiary,
                'impact': 'High' if abs(diff) > 8 else 'Medium'
            })
        
        # Experience and momentum
        if abs(features.get('experience_diff', 0)) > 3:
            diff = features['experience_diff']
            beneficiary = fighter1 if diff > 0 else fighter2
            significant_features.append({
                'category': '🎖️ Experience',
                'metric': "Fight experience",
                'value': f"{abs(diff):.0f} more fights",
                'beneficiary': beneficiary,
                'impact': 'Medium'
            })
        
        if abs(features.get('win_rate_diff', 0)) > 0.1:
            diff = features['win_rate_diff']
            beneficiary = fighter1 if diff > 0 else fighter2
            significant_features.append({
                'category': '📊 Record',
                'metric': "Win rate advantage",
                'value': f"{abs(diff)*100:.1f}% higher",
                'beneficiary': beneficiary,
                'impact': 'High'
            })
        
        if abs(features.get('current_streak_diff', 0)) > 1:
            diff = features['current_streak_diff']
            beneficiary = fighter1 if diff > 0 else fighter2
            streak_type = "win streak" if diff > 0 else "better momentum"
            significant_features.append({
                'category': '🔥 Momentum',
                'metric': "Current streak",
                'value': f"{abs(diff):.0f} fight {streak_type}",
                'beneficiary': beneficiary,
                'impact': 'Medium'
            })
        
        # Striking advantages
        if abs(features.get('striking_accuracy_diff', 0)) > 0.05:
            diff = features['striking_accuracy_diff']
            beneficiary = fighter1 if diff > 0 else fighter2
            significant_features.append({
                'category': '👊 Striking',
                'metric': "Striking accuracy",
                'value': f"{abs(diff)*100:.1f}% better",
                'beneficiary': beneficiary,
                'impact': 'Medium'
            })
        
        if abs(features.get('striking_output_diff', 0)) > 1:
            diff = features['striking_output_diff']
            beneficiary = fighter1 if diff > 0 else fighter2
            significant_features.append({
                'category': '👊 Striking',
                'metric': "Strike output",
                'value': f"{abs(diff):.1f} more/min",
                'beneficiary': beneficiary,
                'impact': 'Medium'
            })
        
        if abs(features.get('striking_defense_diff', 0)) > 0.05:
            diff = features['striking_defense_diff']
            beneficiary = fighter1 if diff > 0 else fighter2
            significant_features.append({
                'category': '🛡️ Defense',
                'metric': "Striking defense",
                'value': f"{abs(diff)*100:.1f}% better",
                'beneficiary': beneficiary,
                'impact': 'Medium'
            })
        
        # Grappling advantages
        if abs(features.get('takedown_accuracy_diff', 0)) > 0.1:
            diff = features['takedown_accuracy_diff']
            beneficiary = fighter1 if diff > 0 else fighter2
            significant_features.append({
                'category': '🤼 Grappling',
                'metric': "Takedown accuracy",
                'value': f"{abs(diff)*100:.1f}% better",
                'beneficiary': beneficiary,
                'impact': 'Medium'
            })
        
        if abs(features.get('takedown_defense_diff', 0)) > 0.1:
            diff = features['takedown_defense_diff']
            beneficiary = fighter1 if diff > 0 else fighter2
            significant_features.append({
                'category': '🤼 Grappling',
                'metric': "Takedown defense",
                'value': f"{abs(diff)*100:.1f}% better",
                'beneficiary': beneficiary,
                'impact': 'Medium'
            })
        
        # Stance advantage
        if abs(features.get('stance_advantage', 0)) > 0:
            stance_val = features['stance_advantage']
            if stance_val > 0:
                significant_features.append({
                    'category': '⚔️ Style',
                    'metric': "Southpaw vs Orthodox",
                    'value': "Stance advantage",
                    'beneficiary': fighter1,
                    'impact': 'Low'
                })
            elif stance_val < 0:
                significant_features.append({
                    'category': '⚔️ Style',
                    'metric': "Orthodox vs Southpaw",
                    'value': "Stance advantage",
                    'beneficiary': fighter2,
                    'impact': 'Low'
                })
        
        # Show the most significant factors
        if significant_features:
            # Sort by impact and beneficiary
            winner_factors = [f for f in significant_features if f['beneficiary'] == result['predicted_winner']]
            loser_factors = [f for f in significant_features if f['beneficiary'] != result['predicted_winner']]
            
            if winner_factors:
                print(f"\n✅ FACTORS FAVORING {result['predicted_winner']}:")
                for factor in winner_factors:
                    impact_emoji = "🔥" if factor['impact'] == 'High' else "⚡" if factor['impact'] == 'Medium' else "💡"
                    print(f"   {impact_emoji} {factor['category']} {factor['metric']}: {factor['value']}")
            
            if loser_factors:
                other_fighter = fighter2 if result['predicted_winner'] == fighter1 else fighter1
                print(f"\n⚠️ FACTORS FAVORING {other_fighter}:")
                for factor in loser_factors:
                    impact_emoji = "🔥" if factor['impact'] == 'High' else "⚡" if factor['impact'] == 'Medium' else "💡"
                    print(f"   {impact_emoji} {factor['category']} {factor['metric']}: {factor['value']}")
        
        # Show raw stats comparison for context
        print(f"\n📊 FIGHTER COMPARISON:")
        print(f"{'='*60}")
        
        stats_to_show = [
            ('height', 'Height', 'cm'),
            ('reach', 'Reach', 'cm'), 
            ('win_rate', 'Win Rate', '%'),
            ('experience', 'Total Fights', ''),
            ('current_streak', 'Current Streak', ''),
            ('sig_strikes_accuracy', 'Strike Accuracy', '%'),
            ('sig_strikes_landed_pm', 'Strikes/Min', ''),
            ('takedown_accuracy', 'TD Accuracy', '%'),
            ('takedown_defence', 'TD Defense', '%')
        ]
        
        print(f"{'Metric':<20} {'':>15} {'':>15} {'Advantage':<15}")
        print(f"{'-'*65}")
        
        for stat_key, stat_name, unit in stats_to_show:
            f1_val = f1_stats.get(stat_key)
            f2_val = f2_stats.get(stat_key)
            
            if f1_val is not None and f2_val is not None:
                # Format values
                if unit == '%':
                    f1_display = f"{f1_val*100:.1f}%" if f1_val <= 1 else f"{f1_val:.1f}%"
                    f2_display = f"{f2_val*100:.1f}%" if f2_val <= 1 else f"{f2_val:.1f}%"
                elif unit == '':
                    f1_display = f"{f1_val:.0f}"
                    f2_display = f"{f2_val:.0f}"
                else:
                    f1_display = f"{f1_val:.1f}{unit}"
                    f2_display = f"{f2_val:.1f}{unit}"
                
                # Determine advantage
                if f1_val > f2_val:
                    advantage = f"👆 {fighter1}"
                elif f2_val > f1_val:
                    advantage = f"👆 {fighter2}"
                else:
                    advantage = "🤝 Even"
                
                print(f"{stat_name:<20} {f1_display:>15} {f2_display:>15} {advantage:<15}")
        
        # Show model confidence factors
        confidence = result['confidence']
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"{'='*40}")
        
        if confidence > 0.75:
            print(f"   🔥 Very High Confidence ({confidence:.1%})")
            print(f"      Model sees clear advantages for {result['predicted_winner']}")
        elif confidence > 0.65:
            print(f"   ⚡ High Confidence ({confidence:.1%})")
            print(f"      Strong indicators favor {result['predicted_winner']}")
        elif confidence > 0.55:
            print(f"   ⚖️ Moderate Confidence ({confidence:.1%})")
            print(f"      Close fight with slight edge to {result['predicted_winner']}")
        else:
            print(f"   🤔 Low Confidence ({confidence:.1%})")
            print(f"      Very close fight, could go either way")
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print(f"\n🎯 INTERACTIVE PREDICTION MODE")
        print(f"{'='*50}")
        print(f"Enter fighter names to predict their matchup")
        print(f"Type 'quit' or 'exit' to stop")
        print(f"{'='*50}")
        
        while True:
            print(f"\n🥊 New Fight Prediction:")
            
            # Get Fighter 1
            fighter1_query = input("Enter Fighter 1 name: ").strip()
            if fighter1_query.lower() in ['quit', 'exit', '']:
                break
            
            # Get Fighter 2  
            fighter2_query = input("Enter Fighter 2 name: ").strip()
            if fighter2_query.lower() in ['quit', 'exit', '']:
                break
            
            # Make prediction
            result = self.predict_fight(fighter1_query, fighter2_query, show_metrics=False)
            
            if result:
                # Ask if user wants detailed analysis
                detail_choice = input(f"\n❓ Show detailed analysis with decision metrics? (y/n): ").strip().lower()
                if detail_choice in ['y', 'yes']:
                    try:
                        # Show the detailed metrics
                        self._show_decision_metrics(result, result['fighter1'], result['fighter2'])
                        
                        # Also show the full analysis
                        analysis = self.predictor.analyze_matchup(
                            result['fighter1'], 
                            result['fighter2']
                        )
                        print(analysis)
                    except Exception as e:
                        print(f"❌ Error getting detailed analysis: {e}")
            
            # Ask to continue
            continue_choice = input(f"\n❓ Make another prediction? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                break
        
        print(f"\n👋 Thanks for using the UFC Prediction Tool!")
    
    def quick_predict(self, fighter1, fighter2):
        """Quick prediction without detailed output"""
        result = self.predict_fight(fighter1, fighter2, show_details=False)
        if result:
            winner = result['predicted_winner']
            confidence = result['confidence']
            print(f"   🏆 {winner} ({confidence:.1%})")
        return result
    
    def batch_predict(self, fight_list):
        """Predict multiple fights at once"""
        print(f"\n📊 BATCH PREDICTIONS")
        print(f"{'='*50}")
        
        results = []
        for i, (fighter1, fighter2) in enumerate(fight_list, 1):
            print(f"\n{i}. {fighter1} vs {fighter2}")
            print(f"{'-'*30}")
            
            result = self.quick_predict(fighter1, fighter2)
            if result:
                results.append(result)
        
        return results
    
    def quick_demo(self):
        """Quick demonstration with some known fighters"""
        demo_fights = [
            ("Jon Jones", "Stipe Miocic"),
            ("Alex Pereira", "Khalil Rountree"),
            ("Islam Makhachev", "Arman Tsarukyan"),
            ("Merab Dvalishvili", "Sean O'Malley"),
            ("Tom Aspinall", "Curtis Blaydes"),
        ]
        
        print(f"\n🎬 QUICK DEMO - Predicting Recent UFC Fights")
        print(f"{'='*50}")
        
        self.batch_predict(demo_fights)


def main():
    """Main function"""
    print("🚀 CACHED UFC FIGHT PREDICTION TOOL")
    print("="*50)
    
    # Initialize tool
    tool = CachedPredictionTool()
    
    # Menu system
    while True:
        print(f"\n{'='*40}")
        print("MAIN MENU")
        print("="*40)
        print("1. Load Model (auto-detect cache)")
        print("2. Single Fight Prediction")
        print("3. Interactive Mode")
        print("4. Quick Demo")
        print("5. Batch Predictions")
        print("6. Search Fighter")
        print("7. Cache Management")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            model_loaded = tool.load_model()
            if model_loaded:
                print("🎉 Model ready! You can now make predictions.")
            else:
                print("❌ Failed to load model.")
        
        elif choice == '2':
            if not tool.predictor:
                print("❌ Please load the model first (option 1)")
                continue
                
            print(f"\n🎯 SINGLE FIGHT PREDICTION")
            fighter1 = input("Fighter 1: ").strip()
            fighter2 = input("Fighter 2: ").strip()
            
            if fighter1 and fighter2:
                # Ask if user wants detailed metrics
                show_metrics = input("Show detailed decision metrics? (y/n, default=y): ").strip().lower()
                show_metrics = show_metrics != 'n'  # Default to yes unless explicitly no
                
                tool.predict_fight(fighter1, fighter2, show_metrics=show_metrics)
        
        elif choice == '3':
            if not tool.predictor:
                print("❌ Please load the model first (option 1)")
                continue
            tool.interactive_mode()
        
        elif choice == '4':
            if not tool.predictor:
                print("❌ Please load the model first (option 1)")
                continue
            tool.quick_demo()
        
        elif choice == '5':
            if not tool.predictor:
                print("❌ Please load the model first (option 1)")
                continue
                
            print(f"\n📊 BATCH PREDICTIONS")
            print("Enter fights one by one (format: Fighter1 vs Fighter2)")
            print("Type 'done' when finished")
            
            fights = []
            while True:
                fight_input = input("Fight: ").strip()
                if fight_input.lower() == 'done':
                    break
                
                if ' vs ' in fight_input:
                    parts = fight_input.split(' vs ')
                    if len(parts) == 2:
                        fights.append((parts[0].strip(), parts[1].strip()))
                    else:
                        print("Invalid format. Use: Fighter1 vs Fighter2")
                else:
                    print("Invalid format. Use: Fighter1 vs Fighter2")
            
            if fights:
                tool.batch_predict(fights)
        
        elif choice == '6':
            if not tool.predictor:
                print("❌ Please load the model first (option 1)")
                continue
                
            query = input("Search for fighter: ").strip()
            if query:
                matches = tool.predictor.search_fighters(query)
                if matches:
                    print(f"\n📋 Found {len(matches)} matches:")
                    for i, fighter in enumerate(matches[:20], 1):
                        print(f"  {i:2d}. {fighter}")
                    if len(matches) > 20:
                        print(f"  ... and {len(matches) - 20} more")
                else:
                    print(f"❌ No fighters found matching '{query}'")
        
        elif choice == '7':
            # Cache management submenu
            while True:
                print(f"\n💾 CACHE MANAGEMENT")
                print("="*30)
                print("1. Show Cache Info")
                print("2. Force Retrain Model")
                print("3. Clear Cache")
                print("4. Back to Main Menu")
                
                cache_choice = input("\nSelect option (1-4): ").strip()
                
                if cache_choice == '1':
                    tool.show_cache_info()
                
                elif cache_choice == '2':
                    print("🏋️ Force retraining model...")
                    model_loaded = tool.load_model(force_retrain=True)
                    if model_loaded:
                        print("✅ Model retrained and cached!")
                
                elif cache_choice == '3':
                    confirm = input("❓ Are you sure you want to clear cache? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        tool.clear_cache()
                
                elif cache_choice == '4':
                    break
                
                else:
                    print("❌ Invalid choice. Please select 1-4.")
        
        elif choice == '8':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-8.")


if __name__ == "__main__":
    main()