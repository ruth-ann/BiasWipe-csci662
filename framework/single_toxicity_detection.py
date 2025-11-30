#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def main():
    parser = argparse.ArgumentParser(description="Test RoBERTa Toxicity Classifier")
    parser.add_argument(
        "-i", "--input", type=str, default="You are amazing!",
        help="Input sentence to classify"
    )
    args = parser.parse_args()
    sentence = args.input

    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
    model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')

    # Tokenize input
    batch = tokenizer.encode(sentence, return_tensors="pt")

    # Forward pass
    output = model(batch)

    # Convert logits to probabilities
    probs = F.softmax(output.logits, dim=1)
    predicted_idx = torch.argmax(output.logits, dim=1).item()
    classes = ["neutral", "toxic"]
    predicted_class = classes[predicted_idx]

    # Print results
    print(f"Input: {sentence}")
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: neutral={probs[0,0].item():.4f}, toxic={probs[0,1].item():.4f}")

if __name__ == "__main__":
    main()

