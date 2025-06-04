# test_improved_model.py
from improved_model import ImprovedUFCFightPredictor

# Initialize the improved predictor
predictor = ImprovedUFCFightPredictor()

# Load and prepare data
print("Loading data...")
training_data = predictor.load_and_prepare_data("data/complete_ufc_data.csv")

# Train the model (this will take a few minutes)
print("Training improved model...")
accuracy = predictor.train_model(
    training_data, 
    model_type='ensemble',  # Use ensemble for best performance
    optimize_hyperparameters=False  # Set to True for even better results (but slower)
)

# Test on some recent fights
print("\n🥊 Testing predictions on recent fights:")
test_fights = [
    ("Merab Dvalishvili", "Sean O'Malley"),
    ("Alex Pereira", "Khalil Rountree"),
    ("Jon Jones", "Stipe Miocic"),
]

for fighter1, fighter2 in test_fights:
    # Search for fighters
    f1_matches = predictor.search_fighters(fighter1)
    f2_matches = predictor.search_fighters(fighter2)
    
    if f1_matches and f2_matches:
        result = predictor.predict_fight(f1_matches[0], f2_matches[0])
        print(f"\n{f1_matches[0]} vs {f2_matches[0]}")
        print(f"Predicted Winner: {result['predicted_winner']} ({result['confidence']:.1%})")
        print(f"Key Advantages: {result['key_advantages'][result['predicted_winner']][:2]}")