#!/usr/bin/env python3
"""
Test UFC Model on 2024-2025 Fights
Uses data/fights_2024_2025.csv with structure:
event_name,event_date,weight_class,fighter1,fighter2,method,round,outcome,timestamp
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

try:
    from cached_prediction_tool import CachedPredictionTool
    print("✓ Cached prediction tool imported")
except ImportError as e:
    print(f"✗ Error importing cached tool: {e}")
    sys.exit(1)


def test_model_on_2024_2025_fights():
    """Test the cached model on 2024-2025 fights"""
    
    print("🥊 UFC MODEL TEST - 2024-2025 FIGHTS")
    print("="*50)
    
    # Load the cached model
    print("1. Loading cached model...")
    tool = CachedPredictionTool()
    if not tool.load_model():
        print("❌ Failed to load cached model")
        print("💡 Try running: python cached_prediction_tool.py first")
        return
    
    print("✅ Model loaded successfully!")
    
    # Load the 2024-2025 fights dataset
    data_file = "data/fights_2024_2025.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("💡 Make sure you have the 2024-2025 fights file in the data/ directory")
        return
    
    print(f"\n2. Loading 2024-2025 fight data...")
    
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Loaded {len(df):,} total fights from 2024-2025")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Show data structure
    print(f"\n📊 Dataset structure:")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    
    # Show sample events
    top_events = df['event_name'].value_counts().head(5)
    print(f"\n📅 Sample 2024-2025 events:")
    for event, count in top_events.items():
        print(f"   • {event}: {count} fights")
    
    # Filter out draws and no contests
    print(f"\n3. Filtering for clear outcomes...")
    
    # Show outcome distribution
    outcome_counts = df['outcome'].value_counts()
    print(f"📊 Outcome distribution:")
    for outcome, count in outcome_counts.items():
        print(f"   {outcome}: {count} fights")
    
    # Keep only clear wins (fighter1 or fighter2 wins)
    clear_outcomes = df[df['outcome'].isin(['fighter1', 'fighter2'])].copy()
    
    print(f"✅ Using {len(clear_outcomes)} fights with clear outcomes")
    print(f"❌ Skipping {len(df) - len(clear_outcomes)} draws/no contests")
    
    if clear_outcomes.empty:
        print("❌ No fights with clear outcomes found")
        return
    
    # Make predictions on all clear fights
    print(f"\n4. Making predictions on 2024-2025 fights...")
    
    predictions = []
    successful = 0
    fighter_not_found = 0
    prediction_errors = 0
    
    for i, (_, fight) in enumerate(clear_outcomes.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(clear_outcomes)} fights...")
        
        try:
            fighter1 = fight['fighter1']
            fighter2 = fight['fighter2']
            outcome = fight['outcome']  # 'fighter1' or 'fighter2'
            
            # Determine actual winner based on outcome
            if outcome == 'fighter1':
                actual_winner = fighter1
            elif outcome == 'fighter2':
                actual_winner = fighter2
            else:
                continue  # Skip if unclear
            
            # Skip if missing fighter names
            if pd.isna(fighter1) or pd.isna(fighter2) or not fighter1 or not fighter2:
                continue
            
            # Find fighters in model database
            f1_matches = tool.predictor.search_fighters(str(fighter1))
            f2_matches = tool.predictor.search_fighters(str(fighter2))
            
            if not f1_matches:
                fighter_not_found += 1
                continue
            
            if not f2_matches:
                fighter_not_found += 1
                continue
            
            # Use best matches
            f1_name = f1_matches[0]
            f2_name = f2_matches[0]
            
            # Make prediction
            result = tool.predictor.predict_fight(f1_name, f2_name)
            
            # Map actual winner to model fighter names
            actual_mapped = None
            if actual_winner == fighter1:
                actual_mapped = f1_name
            elif actual_winner == fighter2:
                actual_mapped = f2_name
            
            # Check if prediction is correct
            correct = (result['predicted_winner'] == actual_mapped) if actual_mapped else None
            
            prediction_data = {
                'event_name': fight['event_name'],
                'event_date': fight['event_date'],
                'weight_class': fight.get('weight_class', 'Unknown'),
                'original_fighter1': fighter1,
                'original_fighter2': fighter2,
                'mapped_fighter1': f1_name,
                'mapped_fighter2': f2_name,
                'outcome': outcome,
                'actual_winner_original': actual_winner,
                'actual_winner_mapped': actual_mapped,
                'predicted_winner': result['predicted_winner'],
                'confidence': result['confidence'],
                'fighter1_probability': result['fighter1_win_probability'],
                'fighter2_probability': result['fighter2_win_probability'],
                'correct_prediction': correct,
                'method': fight.get('method', 'Unknown'),
                'round': fight.get('round', 'Unknown')
            }
            
            predictions.append(prediction_data)
            successful += 1
            
        except Exception as e:
            prediction_errors += 1
            continue
    
    # Analyze results
    print(f"\n5. Analysis Results:")
    print("="*40)
    
    print(f"📊 Processing Summary:")
    print(f"   Total 2024-2025 fights: {len(clear_outcomes)}")
    print(f"   Successful predictions: {successful}")
    print(f"   Fighter not found: {fighter_not_found}")
    print(f"   Prediction errors: {prediction_errors}")
    
    if not predictions:
        print("❌ No predictions made")
        return
    
    # Convert to DataFrame for analysis
    df_predictions = pd.DataFrame(predictions)
    
    # Filter for valid predictions (where we could map the winner)
    valid_predictions = df_predictions[df_predictions['correct_prediction'].notna()]
    
    if valid_predictions.empty:
        print("❌ No valid predictions with clear outcomes")
        return
    
    # Calculate accuracy
    total_valid = len(valid_predictions)
    correct_count = valid_predictions['correct_prediction'].sum()
    accuracy = correct_count / total_valid
    
    print(f"\n🎯 ACCURACY RESULTS:")
    print(f"   Valid predictions: {total_valid}")
    print(f"   Correct predictions: {correct_count}")
    print(f"   Overall accuracy: {accuracy:.1%}")
    
    # Confidence-based analysis
    print(f"\n📈 CONFIDENCE BREAKDOWN:")
    
    confidence_ranges = [
        (0.8, 1.0, "Very High (80-100%)"),
        (0.7, 0.8, "High (70-80%)"),
        (0.6, 0.7, "Medium (60-70%)"),
        (0.5, 0.6, "Low (50-60%)")
    ]
    
    for min_conf, max_conf, label in confidence_ranges:
        subset = valid_predictions[
            (valid_predictions['confidence'] >= min_conf) & 
            (valid_predictions['confidence'] < max_conf)
        ]
        
        if not subset.empty:
            subset_correct = subset['correct_prediction'].sum()
            subset_total = len(subset)
            subset_accuracy = subset_correct / subset_total
            
            print(f"   {label}: {subset_correct}/{subset_total} ({subset_accuracy:.1%})")
    
    # Weight class analysis
    print(f"\n🏆 WEIGHT CLASS PERFORMANCE:")
    
    weight_performance = valid_predictions.groupby('weight_class').agg({
        'correct_prediction': ['count', 'sum', 'mean']
    }).round(3)
    
    for weight_class in weight_performance.index:
        if weight_class != 'Unknown':
            count = weight_performance.loc[weight_class, ('correct_prediction', 'count')]
            correct_wc = weight_performance.loc[weight_class, ('correct_prediction', 'sum')]
            accuracy_wc = weight_performance.loc[weight_class, ('correct_prediction', 'mean')]
            
            if count >= 3:  # Only show classes with enough data
                status = "🔥" if accuracy_wc > 0.7 else "👍" if accuracy_wc > 0.6 else "⚠️"
                print(f"   {status} {weight_class}: {correct_wc}/{count} ({accuracy_wc:.1%})")
    
    # Monthly performance analysis
    print(f"\n📅 PERFORMANCE BY TIME PERIOD:")
    
    # Convert event_date to datetime for analysis
    valid_predictions['event_datetime'] = pd.to_datetime(valid_predictions['event_date'], errors='coerce')
    valid_with_dates = valid_predictions.dropna(subset=['event_datetime'])
    
    if not valid_with_dates.empty:
        # Group by year-month
        valid_with_dates['year_month'] = valid_with_dates['event_datetime'].dt.to_period('M')
        monthly_performance = valid_with_dates.groupby('year_month').agg({
            'correct_prediction': ['count', 'sum', 'mean']
        }).round(3)
        
        for period in monthly_performance.index:
            count = monthly_performance.loc[period, ('correct_prediction', 'count')]
            correct_period = monthly_performance.loc[period, ('correct_prediction', 'sum')]
            accuracy_period = monthly_performance.loc[period, ('correct_prediction', 'mean')]
            
            if count >= 5:  # Only show months with enough fights
                status = "🔥" if accuracy_period > 0.7 else "👍" if accuracy_period > 0.6 else "⚠️"
                print(f"   {status} {period}: {correct_period}/{count} ({accuracy_period:.1%})")
    
    # Show best predictions
    print(f"\n✅ BEST PREDICTIONS (High Confidence + Correct):")
    best_predictions = valid_predictions[
        (valid_predictions['correct_prediction'] == True) & 
        (valid_predictions['confidence'] > 0.7)
    ].nlargest(5, 'confidence')
    
    for _, pred in best_predictions.iterrows():
        print(f"   🎯 {pred['mapped_fighter1']} vs {pred['mapped_fighter2']}")
        print(f"      ✅ Predicted: {pred['predicted_winner']} ({pred['confidence']:.1%})")
        print(f"      📅 {pred['event_name']} ({pred['event_date']})")
    
    # Show worst predictions
    print(f"\n❌ MISSED HIGH-CONFIDENCE PREDICTIONS:")
    worst_predictions = valid_predictions[
        (valid_predictions['correct_prediction'] == False) & 
        (valid_predictions['confidence'] > 0.65)
    ].nlargest(3, 'confidence')
    
    for _, pred in worst_predictions.iterrows():
        print(f"   💥 {pred['mapped_fighter1']} vs {pred['mapped_fighter2']}")
        print(f"      ❌ Predicted: {pred['predicted_winner']} ({pred['confidence']:.1%})")
        print(f"      ✅ Actual: {pred['actual_winner_mapped']}")
        print(f"      📅 {pred['event_name']} ({pred['event_date']})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ufc_2024_2025_test_results_{timestamp}.csv"
    df_predictions.to_csv(filename, index=False)
    
    print(f"\n💾 Results saved to: {filename}")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"   🥊 Model Performance: {accuracy:.1%}")
    print(f"   🎲 Random chance baseline: 50%")
    print(f"   📈 Improvement over random: +{(accuracy - 0.5)*100:.1f} percentage points")
    
    if accuracy >= 0.75:
        print(f"   🔥 EXCELLENT: Strong predictive performance!")
    elif accuracy >= 0.65:
        print(f"   👍 VERY GOOD: Solid predictive power!")
    elif accuracy >= 0.55:
        print(f"   👌 GOOD: Meaningful improvement over random!")
    else:
        print(f"   📝 FAIR: Some predictive ability, room for improvement")
    
    # Strategic insights
    print(f"\n💡 STRATEGIC INSIGHTS:")
    
    # Find best confidence threshold
    best_threshold = None
    best_threshold_accuracy = 0
    best_threshold_count = 0
    
    for threshold in [0.8, 0.75, 0.7, 0.65, 0.6]:
        high_conf_subset = valid_predictions[valid_predictions['confidence'] >= threshold]
        if len(high_conf_subset) >= 10:  # Need reasonable sample size
            thresh_accuracy = high_conf_subset['correct_prediction'].mean()
            if thresh_accuracy > best_threshold_accuracy:
                best_threshold = threshold
                best_threshold_accuracy = thresh_accuracy
                best_threshold_count = len(high_conf_subset)
    
    if best_threshold:
        print(f"   🎯 Optimal strategy: Only bet on predictions with ≥{best_threshold:.0%} confidence")
        print(f"   📈 This gives you {best_threshold_accuracy:.1%} accuracy on {best_threshold_count} fights")
        print(f"   💰 Skip {total_valid - best_threshold_count} lower-confidence predictions")
    
    return accuracy, total_valid


def main():
    """Main function"""
    print("UFC Model Test - 2024-2025 Fights")
    print("="*40)
    
    response = input("Proceed with testing? (y/n): ").strip().lower()
    if response != 'y':
        print("Test cancelled.")
        return
    
    accuracy, total = test_model_on_2024_2025_fights()
    
    if accuracy is not None:
        print(f"\n🏁 TEST COMPLETE!")
        print(f"🎯 Final Result: {accuracy:.1%} accuracy on {total} fights from 2024-2025")
        
        # Performance rating
        if accuracy >= 0.75:
            rating = "🔥 EXCELLENT"  
        elif accuracy >= 0.65:
            rating = "👍 VERY GOOD"
        elif accuracy >= 0.55:
            rating = "👌 GOOD"
        else:
            rating = "📝 FAIR"
            
        print(f"📊 Performance Rating: {rating}")
    else:
        print(f"\n❌ Test failed")


if __name__ == "__main__":
    main()