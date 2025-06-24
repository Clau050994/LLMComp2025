import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def test_traveler_risk(model_path, text_input):
    """
    Test the traveler risk classification model on custom input.
    
    Args:
        model_path: Path to the saved model
        text_input: Text input to classify
    
    Returns:
        Classification result with risk level
    """
    # Load model + tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Tokenize input
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
    
    # Create human-readable output
    risk_levels = ["Low Risk", "Medium Risk", "High Risk"]
    risk_level = risk_levels[prediction]
    
    # Define risk descriptions
    risk_descriptions = {
        0: "No special precautions needed. Standard entry procedures apply.",
        1: "Some caution advised. Additional screening may be required.",
        2: "High alert. Thorough screening and possible additional verification required."
    }
    
    # Format prediction probabilities as percentages
    prob_percentages = [f"{prob * 100:.1f}%" for prob in probabilities]
    
    result = {
        "input_text": text_input,
        "prediction": prediction,
        "risk_level": risk_level,
        "risk_description": risk_descriptions[prediction],
        "confidence": {
            "Low Risk": prob_percentages[0],
            "Medium Risk": prob_percentages[1],
            "High Risk": prob_percentages[2]
        }
    }
    
    return result

def format_result(result):
    """Format the classification result for display"""
    print("\n" + "=" * 60)
    print("TRAVELER RISK ASSESSMENT")
    print("=" * 60)
    print(f"INPUT: {result['input_text']}")
    print("-" * 60)
    print(f"RISK CLASSIFICATION: {result['risk_level']}")
    print(f"DESCRIPTION: {result['risk_description']}")
    print("-" * 60)
    print("CONFIDENCE SCORES:")
    for level, score in result['confidence'].items():
        marker = "→" if level == result['risk_level'] else " "
        print(f" {marker} {level}: {score}")
    print("=" * 60)

if __name__ == "__main__":
    model_path = "models/traveler_classification/distilbert_unified_20250620_110136"
    
    # Test examples covering different risk levels and scenarios
    test_examples = [
        # Basic examples
        "Traveler from Canada arriving via direct flight.",
        "Passenger with Mexico passport, entering from Brazil.",
        "Individual from Syria with recent travel to Iraq.",
        
        # Confounding context examples
        "Passenger from Japan who enjoys Syrian cuisine.",
        "Traveler from Venezuela who studied literature about Denmark.",
        "Canadian citizen who exports goods to Afghanistan.",
        
        # Indirect descriptors
        "Traveler coming from a region with frequent terrorist attacks.",
        "Person with citizenship in a stable democracy with strong rule of law.",
        
        # Spelling variations
        "Pass3ng3r from M3x1co with T@iw@n visa.",
        
        # Custom test - add your own example here
        "Traveler from Cuba that likes Mexican food."
    ]
    
    # Run classification for each example
    print(f"Testing model: {model_path}")
    for example in test_examples:
        result = test_traveler_risk(model_path, example)
        format_result(result)
        
        # Optional: pause between examples
        if example != test_examples[-1]:
            input("Press Enter to continue to the next example...")