import json
from datasets import load_dataset
import random

# Load existing datasets
with open('data/business_keywords.json', 'r') as f:
    business_keywords = json.load(f)

with open('data/expanded_business_keywords.json', 'r') as f:
    expanded_business_keywords = json.load(f)

# Combine datasets
combined_dataset = business_keywords + expanded_business_keywords

# Load world cities dataset
cities_dataset = load_dataset("jamescalam/world-cities-geo")

# Load keyword extraction dataset
keyword_dataset = load_dataset("zino36/keyword-extraction-dataset")

# Load LMSYS-Chat-1M dataset
try:
    lmsys_dataset = load_dataset("lmsys/lmsys-chat-1m")
    print("Successfully loaded LMSYS-Chat-1M dataset")
except Exception as e:
    print(f"Error loading LMSYS-Chat-1M dataset: {e}")
    print("Proceeding without LMSYS-Chat-1M dataset")
    lmsys_dataset = None

if lmsys_dataset:
    print("LMSYS-Chat-1M Dataset Info:")
    print(lmsys_dataset)
    print(lmsys_dataset['train'].features)

    # Print a few examples
    for i, example in enumerate(lmsys_dataset['train']):
        print(f"Example {i}:")
        print(example)
        if i >= 2:  # Print only the first 3 examples
            break

# Function to create a sentence with a city
def create_city_sentence(city, keyword):
    return {
        "sentence": ["Find", keyword, "in", city],
        "labels": ["O", "B-KEY", "O", "B-LOC"]
    }

# Add city examples
for city in random.sample(cities_dataset['train']['city'], 1000):  # Add 1000 random cities
    keyword = random.choice(["businesses", "companies", "services", "firms"])
    combined_dataset.append(create_city_sentence(city, keyword))

# Add business keyword examples
keyword_list = keyword_dataset['train']['Keywords']
for keywords in random.sample(keyword_list, 1000):  # Add 1000 random business keywords
    for keyword in keywords.split(','):  # Split multiple keywords
        keyword = keyword.strip()
        keyword_words = keyword.split()
        sentence = ["Find"] + keyword_words + ["in", "City"]
        labels = ["O"] + ["B-KEY"] + ["I-KEY"] * (len(keyword_words) - 1) + ["O", "B-LOC"]
        combined_dataset.append({"sentence": sentence, "labels": labels})

# Add LMSYS-Chat-1M examples
if lmsys_dataset:
    # Convert the dataset to a list if it's not already
    train_data = list(lmsys_dataset['train'])
    # If there are fewer than 5000 items, use all of them
    sample_size = min(5000, len(train_data))
    for conversation in random.sample(train_data, sample_size):
        for turn in conversation['conversation']:
            if turn['role'] == 'user':
                human_query = turn['content']
                words = human_query.split()
                sentence = words[:20]  # Limit to first 20 words to keep sentences manageable
                labels = ["O"] * len(sentence)  # Initially label all as "O"
                combined_dataset.append({"sentence": sentence, "labels": labels})

# Save the combined dataset
with open('data/combined_dataset.json', 'w') as f:
    json.dump(combined_dataset, f, indent=2)

print("Combined dataset created and saved as 'data/combined_dataset.json'")
print(f"Total number of examples: {len(combined_dataset)}")
