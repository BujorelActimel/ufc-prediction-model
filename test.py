from model import UFCFightPredictor

predictor = UFCFightPredictor()
training_data = predictor.load_and_prepare_data("data/complete_ufc_data.csv")
predictor.train_model(training_data, model_type='random_forest')

fighter1 = predictor.search_fighters("Dvalishvili")[0]
fighter2 = predictor.search_fighters("O'Malley")[0]

result = predictor.predict_fight(fighter1, fighter2)
print(f"Predicted winner: {result['predicted_winner']}")
print(f"Confidence: {result['confidence']:.1%}")

analysis = predictor.analyze_matchup(fighter1, fighter2)
print(analysis)