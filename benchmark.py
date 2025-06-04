"""
UFC Model Test Script - Only tests on completed fights with known results
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Import your existing model and the working scraper
try:
    from model import UFCFightPredictor
    print("✓ UFC Predictor model imported successfully")
except ImportError as e:
    print(f"✗ Error importing UFC Predictor: {e}")
    print("Make sure model.py is in the same directory")
    sys.exit(1)

try:
    from scraper import UFCStatsScraper
    print("✓ Working UFC Scraper imported successfully")
except ImportError as e:
    print(f"✗ Error importing UFC Scraper: {e}")
    print("Make sure you saved the working scraper as 'scraper.py'")
    sys.exit(1)


def test_model_on_completed_fights():
    """
    Test the UFC model only on completed fights with known results
    """
    
    print("="*60)
    print("UFC FIGHT PREDICTOR - MODEL TESTING (COMPLETED FIGHTS ONLY)")
    print("="*60)
    
    # Step 1: Initialize and train the model
    print("\n1. Loading and training UFC prediction model...")
    
    try:
        predictor = UFCFightPredictor()
        
        # Check if data file exists
        data_file = "data/complete_ufc_data.csv"
        if not os.path.exists(data_file):
            print(f"✗ Data file not found: {data_file}")
            print("Please ensure the data file is in the correct location")
            return
        
        # Load and train model
        training_data = predictor.load_and_prepare_data(data_file)
        accuracy = predictor.train_model(training_data, model_type='random_forest')
        
        print(f"✓ Model trained successfully with {accuracy:.1%} training accuracy")
        
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return
    
    # Step 2: Scrape only completed fights
    print("\n2. Scraping completed UFC fights (skipping upcoming events)...")
    
    try:
        scraper = UFCStatsScraper(existing_csv_path=data_file)
        
        # Get recent completed events only
        recent_fights = scraper.scrape_recent_fights(max_events=30)
        
        if recent_fights.empty:
            print("✗ No completed fights found")
            return
        
        print(f"✓ Scraped {len(recent_fights)} completed fights")
        
        # Show sample of what we got
        completed_with_winners = recent_fights[recent_fights['winner'] != 'Unknown']
        print(f"✓ Found {len(completed_with_winners)} fights with known winners")
        
        if completed_with_winners.empty:
            print("✗ No fights with known winners found for testing")
            return
        
        # Find fights missing from training data
        missing_fights = scraper.find_missing_fights(completed_with_winners)
        
        if missing_fights.empty:
            print("✓ No missing completed fights found - using recent completed fights instead")
            test_fights = completed_with_winners.head(10)  # Use recent fights for testing
        else:
            print(f"✓ Found {len(missing_fights)} missing completed fights for testing")
            test_fights = missing_fights
        
    except Exception as e:
        print(f"✗ Error scraping fights: {e}")
        return
    
    # Step 3: Create predictions for test fights
    print(f"\n3. Creating predictions for {len(test_fights)} completed fights...")
    
    try:
        test_predictions = scraper.create_test_predictions(test_fights, predictor)
        
        if test_predictions.empty:
            print("✗ No predictions could be created")
            return
        
        print(f"✓ Created {len(test_predictions)} predictions")
        
        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_file = f"ufc_model_test_completed_{timestamp}.csv"
        test_predictions.to_csv(predictions_file, index=False)
        print(f"✓ Predictions saved to {predictions_file}")
        
    except Exception as e:
        print(f"✗ Error creating predictions: {e}")
        return
    
    # Step 4: Analyze results
    print("\n4. Analyzing prediction results...")
    
    try:
        analyze_prediction_results(test_predictions)
        
    except Exception as e:
        print(f"✗ Error analyzing results: {e}")
        return
    
    print("\n" + "="*60)
    print("MODEL TESTING COMPLETE!")
    print("="*60)


def analyze_prediction_results(predictions_df):
    """
    Analyze the prediction results and show performance metrics
    """
    
    # Filter predictions with known outcomes
    valid_predictions = predictions_df[predictions_df['correct_prediction'].notna()]
    
    if valid_predictions.empty:
        print("⚠️  No predictions with known outcomes to analyze")
        print("\nAll Predictions Made:")
        display_predictions(predictions_df)
        return
    
    total_predictions = len(valid_predictions)
    correct_predictions = valid_predictions['correct_prediction'].sum()
    accuracy = correct_predictions / total_predictions
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   Total Test Predictions: {total_predictions}")
    print(f"   Correct Predictions: {correct_predictions}")
    print(f"   Test Accuracy: {accuracy:.1%}")
    
    # Confidence-based analysis
    high_confidence = valid_predictions[valid_predictions['confidence'] > 0.7]
    if not high_confidence.empty:
        high_conf_accuracy = high_confidence['correct_prediction'].mean()
        print(f"   High Confidence Accuracy (>70%): {high_conf_accuracy:.1%} ({len(high_confidence)} predictions)")
    
    medium_confidence = valid_predictions[
        (valid_predictions['confidence'] > 0.6) & 
        (valid_predictions['confidence'] <= 0.7)
    ]
    if not medium_confidence.empty:
        med_conf_accuracy = medium_confidence['correct_prediction'].mean()
        print(f"   Medium Confidence Accuracy (60-70%): {med_conf_accuracy:.1%} ({len(medium_confidence)} predictions)")
    
    low_confidence = valid_predictions[valid_predictions['confidence'] <= 0.6]
    if not low_confidence.empty:
        low_conf_accuracy = low_confidence['correct_prediction'].mean()
        print(f"   Low Confidence Accuracy (≤60%): {low_conf_accuracy:.1%} ({len(low_confidence)} predictions)")
    
    # Weight class analysis
    print(f"\n📈 WEIGHT CLASS BREAKDOWN:")
    weight_class_performance = valid_predictions.groupby('weight_class').agg({
        'correct_prediction': ['count', 'sum', 'mean']
    }).round(3)
    
    for weight_class in weight_class_performance.index:
        count = weight_class_performance.loc[weight_class, ('correct_prediction', 'count')]
        correct = weight_class_performance.loc[weight_class, ('correct_prediction', 'sum')]
        accuracy = weight_class_performance.loc[weight_class, ('correct_prediction', 'mean')]
        print(f"   {weight_class}: {correct}/{count} ({accuracy:.1%})")
    
    # Show detailed results
    print(f"\n📋 DETAILED PREDICTIONS:")
    display_predictions(valid_predictions)
    
    # Show incorrect predictions for analysis
    incorrect_predictions = valid_predictions[valid_predictions['correct_prediction'] == False]
    if not incorrect_predictions.empty:
        print(f"\n❌ INCORRECT PREDICTIONS ({len(incorrect_predictions)}) - ANALYZE THESE:")
        for _, pred in incorrect_predictions.iterrows():
            print(f"\n   Event: {pred['event_name']}")
            print(f"   Fight: {pred['mapped_fighter1']} vs {pred['mapped_fighter2']}")
            print(f"   Predicted: {pred['predicted_winner']} ({pred['confidence']:.1%} confidence)")
            print(f"   Actual: {pred['actual_winner_mapped']} (Original: {pred['actual_winner_original']})")
            print(f"   Method: {pred['method']}, Round: {pred['round']}")
            print(f"   Weight Class: {pred['weight_class']}")


def display_predictions(predictions_df, max_display=15):
    """
    Display prediction results in a readable format
    """
    
    for i, (_, pred) in enumerate(predictions_df.head(max_display).iterrows()):
        result_emoji = "✅" if pred.get('correct_prediction') == True else "❌" if pred.get('correct_prediction') == False else "❓"
        
        print(f"\n{result_emoji} {pred['event_name']} ({pred['event_date']})")
        print(f"   {pred['mapped_fighter1']} vs {pred['mapped_fighter2']}")
        print(f"   Predicted: {pred['predicted_winner']} ({pred['confidence']:.1%})")
        
        if pd.notna(pred.get('actual_winner_mapped')):
            print(f"   Actual: {pred['actual_winner_mapped']}")
        
        print(f"   Probabilities: {pred['mapped_fighter1']} {pred['fighter1_probability']:.1%}, {pred['mapped_fighter2']} {pred['fighter2_probability']:.1%}")
        
        if pred.get('weight_class') and pred.get('weight_class') != 'Unknown':
            print(f"   Details: {pred['weight_class']}, {pred['method']}, Round {pred['round']}")
    
    if len(predictions_df) > max_display:
        print(f"\n... and {len(predictions_df) - max_display} more predictions")


def quick_scrape_test():
    """
    Quick test to see what fights are available
    """
    print("Quick test of completed fights scraping...")
    print("=" * 50)
    
    try:
        scraper = UFCStatsScraper(existing_csv_path="data/complete_ufc_data.csv")
        
        # Get recent events (completed only)
        events = scraper.get_recent_events(max_pages=1, only_completed=True)
        
        print(f"Found {len(events)} completed events:")
        for i, event in enumerate(events[:5]):
            print(f"  {i+1}. {event['name']} ({event['date']}) - {event['location']}")
        
        if events:
            # Test first event
            print(f"\nTesting fights from: {events[0]['name']}")
            fights = scraper.get_fight_details(events[0]['url'])
            
            completed_fights = [f for f in fights if f['winner'] != 'Unknown']
            upcoming_fights = [f for f in fights if f['winner'] == 'Unknown']
            
            print(f"  Completed fights: {len(completed_fights)}")
            print(f"  Upcoming fights: {len(upcoming_fights)}")
            
            if completed_fights:
                print(f"\n  Sample completed fights:")
                for fight in completed_fights[:3]:
                    print(f"    {fight['fighter1']} vs {fight['fighter2']} -> {fight['winner']}")
                    print(f"    Method: {fight['method']}, Round: {fight['round']}")
            
            if upcoming_fights:
                print(f"\n  Sample upcoming fights (will be skipped):")
                for fight in upcoming_fights[:3]:
                    print(f"    {fight['fighter1']} vs {fight['fighter2']} -> {fight['winner']}")
        
    except Exception as e:
        print(f"Error in quick test: {e}")


def manual_prediction_test():
    """
    Manual test with known recent fights
    """
    print("\nManual prediction test...")
    print("=" * 30)
    
    try:
        # Load model
        predictor = UFCFightPredictor()
        training_data = predictor.load_and_prepare_data("data/complete_ufc_data.csv")
        predictor.train_model(training_data)
        
        # Test with some recent well-known fights (you can adjust these)
        test_fights = [
            ("Jon Jones", "Stipe Miocic", "Jon Jones"),  # UFC 309
            ("Alex Pereira", "Khalil Rountree Jr.", "Alex Pereira"),  # Recent fight
            ("Islam Makhachev", "Arman Tsarukyan", "Islam Makhachev"),  # UFC 311
            ("Merab Dvalishvili", "Sean O'Malley", "Merab Dvalishvili"),  # Recent bantamweight
        ]
        
        print("Testing predictions on known recent fights:")
        correct = 0
        total = 0
        
        for fighter1, fighter2, actual_winner in test_fights:
            try:
                # Search for fighters
                f1_matches = predictor.search_fighters(fighter1)
                f2_matches = predictor.search_fighters(fighter2)
                
                if f1_matches and f2_matches:
                    f1_name = f1_matches[0]
                    f2_name = f2_matches[0]
                    
                    result = predictor.predict_fight(f1_name, f2_name)
                    predicted_winner = result['predicted_winner']
                    
                    # Determine if correct
                    is_correct = False
                    if actual_winner.lower() in f1_name.lower() and predicted_winner == f1_name:
                        is_correct = True
                    elif actual_winner.lower() in f2_name.lower() and predicted_winner == f2_name:
                        is_correct = True
                    
                    emoji = "✅" if is_correct else "❌"
                    total += 1
                    if is_correct:
                        correct += 1
                    
                    print(f"\n{emoji} {f1_name} vs {f2_name}")
                    print(f"   Predicted: {predicted_winner} ({result['confidence']:.1%})")
                    print(f"   Actual: {actual_winner}")
                    print(f"   Probabilities: {f1_name} {result['fighter1_win_probability']:.1%}, {f2_name} {result['fighter2_win_probability']:.1%}")
                
            except Exception as e:
                print(f"❌ Error testing {fighter1} vs {fighter2}: {e}")
        
        if total > 0:
            print(f"\nManual Test Accuracy: {correct}/{total} ({correct/total:.1%})")
        
    except Exception as e:
        print(f"Error in manual test: {e}")


def main():
    """
    Main menu for model testing
    """
    print("UFC MODEL TESTING - COMPLETED FIGHTS ONLY")
    print("=========================================")
    print()
    
    while True:
        print("\nSelect an option:")
        print("1. Full Model Test (scrape completed fights + test)")
        print("2. Quick Scrape Test (see what's available)")
        print("3. Manual Prediction Test (test on known fights)")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            test_model_on_completed_fights()
        
        elif choice == '2':
            quick_scrape_test()
        
        elif choice == '3':
            manual_prediction_test()
        
        elif choice == '4':
            print("\nGoodbye! 🥊")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()