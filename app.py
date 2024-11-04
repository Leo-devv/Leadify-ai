# app.py
from chatbot_service import load_model, process_input

def main():
    print("Loading model...")
    extractor = load_model()
    print("Model loaded. Ready to process queries.")

    while True:
        user_input = input("Enter your query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        result = process_input(user_input, extractor)
        if result['type'] == 'search_query':
            print(f"Keywords: {result['keywords']}")
            print(f"Location: {result['location']}")
        else:
            print(result['message'])

        print()

if __name__ == "__main__":
    main()

