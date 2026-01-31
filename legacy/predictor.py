import torch
import pandas as pd
import torch
from model_checkpoints.utils import load_dataset, Tokenizer, en_dict
from model_checkpoints.model import Model
from typing import List

class IELTSPredictor:
    def __init__(self, model_path="model_checkpoints/best_model.pt"):
        # Initialize dictionary by loading dataset first
        load_dataset()  # This builds the en_dict
        
        self.device = torch.device("cpu")  # Use CPU for prediction
        self.model = Model(vocab_size=len(en_dict)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.tokenizer = Tokenizer(own_dict=en_dict)

    def predict(self, essay: str) -> float:
        # Tokenize the essay
        tokens = self.tokenizer.tokenize(essay.split())
        # Convert to tensor
        input_ids = torch.tensor([tokens["input_ids"]]).long().to(self.device)
        
        # Make prediction
        with torch.no_grad():
            pred = self.model(input_ids)
            # Calibrate to valid IELTS range
            pred = max(0, min(pred.item(), 9))  # clamp between 0 and 9
            return pred

            pred = max(0, min(pred, 9))

def main():
    try:
        # Initialize predictor
        print("Initializing predictor...")
        print(f"Initial dictionary size: {len(en_dict)}")
        predictor = IELTSPredictor()
        print(f"Final dictionary size: {len(en_dict)}")
        print("Model loaded successfully!")
        
        # Example essays to test
        test_essays = [
            "The government should invest more in renewable energy sources like solar and wind power. "
            "This investment would help reduce carbon emissions and combat climate change effectively. "
            "Moreover, it would create new jobs in the green energy sector.",
            
            "Education is crucial for personal development. Students need to focus on both academic "
            "and practical skills to succeed in today's competitive world. Schools should provide "
            "more hands-on learning opportunities."
        ]
        
        # Process each essay
        for i, essay in enumerate(test_essays, 1):
            score = predictor.predict(essay)
            print(f"\nEssay {i}:")
            print(f"Text: {essay[:100]}...")
            print(f"Predicted Band Score: {score:.2f}")
            
            # Generate feedback
            feedback = generate_feedback(score)
            print("Feedback:")
            for point in feedback:
                print(f" - {point}")
                
    except Exception as e:
        print(f"\nError occurred in main: {str(e)}")
        raise

def generate_feedback(pred_score: float) -> List[str]:
    feedback = []
    
    # Threshold-based adaptive rules
    if pred_score < 5.5:
        feedback.append("Focus on organizing ideas and addressing the task more directly.")
        feedback.append("Revise sentence structure to reduce grammar errors.")
        feedback.append("Work on basic vocabulary and grammar accuracy.")
    elif 5.5 <= pred_score < 6.5:
        feedback.append("Add more linking phrases and clear transitions between ideas.")
        feedback.append("Use a wider range of vocabulary to improve lexical variety.")
        feedback.append("Develop more complex sentence structures.")
    elif 6.5 <= pred_score < 7.5:
        feedback.append("Your essay is strong. Aim for more precise word choice and complex sentences.")
        feedback.append("Consider adding more nuanced arguments and examples.")
    else:
        feedback.append("Excellent coherence and structure. Maintain this standard!")
        feedback.append("Continue using sophisticated vocabulary and complex structures appropriately.")
    
    return feedback

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        if "size mismatch" in str(e):
            print("\nTroubleshooting tips:")
            print("1. Make sure you've run training first: python main.py")
            print("2. Check if best_model.pt exists in model_checkpoints/")
            print("3. Verify that the dataset at data/ielts_clean.csv is accessible")
            print("\nCurrent dictionary size:", len(en_dict))


