#!/usr/bin/env python3
"""
Instant UFC Prediction CLI
Super fast command line predictions using cached model

Usage:
  python instant_predict.py "Jon Jones" "Stipe Miocic"
  python instant_predict.py --interactive
  python instant_predict.py --setup
"""

import sys
import os
import argparse
import pickle
import json
from datetime import datetime

# Import the cached tool
try:
    from cached_prediction_tool import CachedPredictionTool
except ImportError:
    print("❌ Could not import CachedPredictionTool")
    print("Make sure cached_prediction_tool.py is in the same directory")
    sys.exit(1)


class InstantPredictor:
    """Ultra-fast UFC predictions using pre-cached model"""
    
    def __init__(self):
        self.tool = CachedPredictionTool()
        self.model_loaded = False
    
    def ensure_model_loaded(self):
        """Ensure model is loaded, load from cache if needed"""
        if not self.model_loaded:
            print("🚀 Loading model...")
            if self.tool.load_model():
                self.model_loaded = True
                print("✅ Ready for predictions!")
                return True
            else:
                print("❌ Failed to load model")
                return False
        return True
    
    def quick_predict(self, fighter1, fighter2):
        """Super quick prediction with minimal output"""
        if not self.ensure_model_loaded():
            return None
        
        # Find fighters (silent)
        f1 = self.tool.search_fighter(fighter1, show_details=False)
        f2 = self.tool.search_fighter(fighter2, show_details=False)
        
        if not f1:
            print(f"❌ Fighter not found: {fighter1}")
            return None
        
        if not f2:
            print(f"❌ Fighter not found: {fighter2}")
            return None
        
        if f1 == f2:
            print("❌ Same fighter!")
            return None
        
        try:
            result = self.tool.predictor.predict_fight(f1, f2)
            return result
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def format_result(self, result, style='full'):
        """Format prediction result"""
        if not result:
            return ""
        
        if style == 'minimal':
            return f"{result['predicted_winner']} ({result['confidence']:.0%})"
        
        elif style == 'compact':
            f1 = result['fighter1']
            f2 = result['fighter2']
            winner = result['predicted_winner']
            conf = result['confidence']
            f1_prob = result['fighter1_win_probability']
            f2_prob = result['fighter2_win_probability']
            
            return f"""🥊 {f1} vs {f2}
🏆 {winner} ({conf:.1%})
📊 {f1}: {f1_prob:.1%} | {f2}: {f2_prob:.1%}"""
        
        else:  # full
            f1 = result['fighter1']
            f2 = result['fighter2']
            winner = result['predicted_winner']
            conf = result['confidence']
            f1_prob = result['fighter1_win_probability']
            f2_prob = result['fighter2_win_probability']
            
            output = f"""
🥊 FIGHT PREDICTION
{'='*40}
🔴 {f1}
🔵 {f2}
{'='*40}
🏆 WINNER: {winner}
📊 CONFIDENCE: {conf:.1%}

📈 PROBABILITIES:
   {f1}: {f1_prob:.1%}
   {f2}: {f2_prob:.1%}"""
            
            # Add key advantages if available
            if 'key_advantages' in result:
                advantages = result['key_advantages'][winner]
                if advantages:
                    output += f"\n\n🔑 KEY ADVANTAGES:\n"
                    for adv in advantages[:3]:
                        output += f"   • {adv}\n"
            
            return output


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Instant UFC Fight Predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Jon Jones" "Stipe Miocic"
  %(prog)s --interactive
  %(prog)s --setup
  %(prog)s "McGregor" "Khabib" --style minimal
        """
    )
    
    parser.add_argument('fighter1', nargs='?', help='First fighter name')
    parser.add_argument('fighter2', nargs='?', help='Second fighter name')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--setup', '-s', action='store_true', help='Setup/train model')
    parser.add_argument('--style', choices=['minimal', 'compact', 'full'], default='compact', 
                       help='Output style (default: compact)')
    parser.add_argument('--batch', '-b', help='File with list of fights')
    parser.add_argument('--search', help='Search for fighters')
    
    args = parser.parse_args()
    
    predictor = InstantPredictor()
    
    # Setup mode
    if args.setup:
        print("🏋️ SETUP MODE - Training/Caching Model")
        print("="*50)
        tool = CachedPredictionTool()
        if tool.load_model(force_retrain=True):
            print("✅ Model setup complete!")
            print("🚀 You can now use instant predictions!")
        else:
            print("❌ Setup failed")
        return
    
    # Search mode
    if args.search:
        if predictor.ensure_model_loaded():
            matches = predictor.tool.predictor.search_fighters(args.search)
            if matches:
                print(f"📋 Found {len(matches)} fighters matching '{args.search}':")
                for i, fighter in enumerate(matches[:20], 1):
                    print(f"  {i:2d}. {fighter}")
                if len(matches) > 20:
                    print(f"  ... and {len(matches) - 20} more")
            else:
                print(f"❌ No fighters found matching '{args.search}'")
        return
    
    # Interactive mode
    if args.interactive:
        if not predictor.ensure_model_loaded():
            return
        
        print("🎯 INTERACTIVE MODE")
        print("Type 'quit' to exit")
        print("="*30)
        
        while True:
            try:
                fighter1 = input("\nFighter 1: ").strip()
                if fighter1.lower() in ['quit', 'exit', 'q']:
                    break
                
                fighter2 = input("Fighter 2: ").strip()
                if fighter2.lower() in ['quit', 'exit', 'q']:
                    break
                
                if fighter1 and fighter2:
                    result = predictor.quick_predict(fighter1, fighter2)
                    if result:
                        print(predictor.format_result(result, args.style))
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
        return
    
    # Batch mode
    if args.batch:
        if not predictor.ensure_model_loaded():
            return
        
        try:
            with open(args.batch, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            print(f"📊 BATCH PREDICTIONS ({len(lines)} fights)")
            print("="*50)
            
            for i, line in enumerate(lines, 1):
                if ' vs ' in line:
                    fighters = line.split(' vs ')
                    if len(fighters) == 2:
                        f1, f2 = fighters[0].strip(), fighters[1].strip()
                        print(f"\n{i}. {f1} vs {f2}")
                        print("-" * 30)
                        
                        result = predictor.quick_predict(f1, f2)
                        if result:
                            if args.style == 'minimal':
                                print(f"   🏆 {predictor.format_result(result, 'minimal')}")
                            else:
                                print(predictor.format_result(result, args.style))
                        else:
                            print("   ❌ Could not predict")
                    else:
                        print(f"{i}. ❌ Invalid format: {line}")
                else:
                    print(f"{i}. ❌ Invalid format: {line}")
        
        except FileNotFoundError:
            print(f"❌ File not found: {args.batch}")
        except Exception as e:
            print(f"❌ Error reading batch file: {e}")
        
        return
    
    # Single prediction mode
    if args.fighter1 and args.fighter2:
        result = predictor.quick_predict(args.fighter1, args.fighter2)
        if result:
            print(predictor.format_result(result, args.style))
        return
    
    # No arguments - show help
    parser.print_help()


if __name__ == "__main__":
    main()