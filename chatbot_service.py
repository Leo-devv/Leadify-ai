# chatbot_service.py
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import sys
import json
import os
import traceback
import re
import logging

# Set up logging to file instead of stdout
logging.basicConfig(filename='chatbot_service.log', level=logging.DEBUG)

def load_model():
    try:
        model_path = os.path.join("G:\\", "AI", "leadify-ai", "results", "final_model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return pipeline("ner", model=model, tokenizer=tokenizer)
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def is_search_query(text):
    search_keywords = ['find', 'search', 'looking for', 'where', 'how to find', 'need']
    return any(keyword in text.lower() for keyword in search_keywords)

def is_email_validation_query(text):
    validation_keywords = ['validate', 'check', 'verify', 'test', 'email']
    return any(keyword in text.lower() for keyword in validation_keywords)

def process_input(text, extractor):
    try:
        if is_email_validation_query(text):
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
            if email_match:
                return {
                    'type': 'email_validation',
                    'email': email_match.group(0)
                }
            return {
                'type': 'email_validation',
                'message': "I couldn't find an email address in your message. Please provide an email to validate."
            }
        elif is_search_query(text):
            result = extractor(text)
            keywords = []
            location = []
            current_entity = []
            current_type = None
            
            # Log the raw results
            logging.debug(f"Raw NER results: {result}")
            
            # Process tokens sequentially to handle multi-word entities
            for item in result:
                if item['entity'].startswith('B-'):
                    if current_entity:
                        # Store previous entity
                        if current_type == 'BUSINESS':
                            keywords.append(' '.join(current_entity))
                        elif current_type == 'CITY':
                            location.append(' '.join(current_entity))
                    # Start new entity
                    current_entity = [item['word']]
                    current_type = item['entity'][2:]
                elif item['entity'].startswith('I-'):
                    if current_entity:
                        current_entity.append(item['word'])
            
            # Don't forget the last entity
            if current_entity:
                if current_type == 'BUSINESS':
                    keywords.append(' '.join(current_entity))
                elif current_type == 'CITY':
                    location.append(' '.join(current_entity))
            
            # If no entities were found, try to extract from the text directly
            if not keywords and not location:
                words = text.split()
                for i, word in enumerate(words):
                    if word.lower() in ['in', 'at', 'near'] and i + 1 < len(words):
                        location = [words[i + 1]]
                        keywords = [' '.join(words[1:i])]
                        break
            
            logging.debug(f"Processed keywords: {keywords}")
            logging.debug(f"Processed location: {location}")
            
            if not keywords and not location:
                return {
                    'type': 'general_question',
                    'message': "I couldn't understand what you're looking for. Please specify what type of business and location. For example: 'find vape shops in London'"
                }
            
            return {
                'type': 'search_query',
                'keywords': ' '.join(keywords) if keywords else '',
                'location': ' '.join(location) if location else ''
            }
        
        return {
            'type': 'general_question',
            'message': "I can help you search for businesses or validate emails. What would you like to do? For example:\n- 'find vape shops in London'\n- 'validate email test@example.com'"
        }
    except Exception as e:
        logging.error(f"Error processing input: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            'type': 'error',
            'message': str(e)
        }

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            sys.stdout.write(json.dumps({"error": "Invalid number of arguments"}))
            sys.exit(1)

        extractor = load_model()
        result = process_input(sys.argv[1], extractor)
        sys.stdout.write(json.dumps(result))
        sys.stdout.flush()
    except Exception as e:
        error_response = {
            "type": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        sys.stdout.write(json.dumps(error_response))
        sys.stdout.flush()
        sys.exit(1)
