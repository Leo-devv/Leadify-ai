from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load the fine-tuned model and tokenizer
model_path = "G:/AI/leadify-ai/results/final_model"
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create the NER pipeline
extractor = pipeline("ner", model=model, tokenizer=tokenizer)

# Test with sample inputs
test_sentences = [
    "Find recruitment agencies in London",
    "i need contractor services in london",
    "who are the best recruitment agencies in katowice",
    "give me contact details for construction companies in birmingham"
]

for sentence in test_sentences:
    result = extractor(sentence)
    print(f"Input: {sentence}")
    print("Extracted Keyphrases:", result)
    print()
