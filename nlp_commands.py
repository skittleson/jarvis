import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")

# Define the common task templates and corresponding actions
task_templates = [
    ("set a timer for {duration} {unit}", "set_timer"),
    ("remind me {when} about {task}", "set_reminder"),
    ("play this song", "play_song"),
    ("turn off {location} light", "turn_off_light"),
    ("turn on {location} light", "turn_on_light"),
    ("dim {location} light by {percentage}%", "dim_light"),
    ("open garage door", "open_garage_door"),
    ("close garage door", "close_garage_door"),
]

# Create a simple function to extract entities
def extract_entities(doc, template):
    entities = {}
    for token in doc:
        if token.dep_ == "nummod":
            entities["duration"] = int(token.text)
        elif token.dep_ == "units":
            entities["unit"] = token.text
        elif token.pos_ == "PROPN" or token.pos_ == "NOUN":
            if "location" not in entities:
                entities["location"] = token.text
            else:
                entities["task"] = token.text
        elif token.dep_ == "advmod":
            entities["when"] = token.text
        elif token.dep_ == "quantmod":
            entities["percentage"] = int(token.text)
    
    # Handle cases where "location" might be a compound noun
    if "location" not in entities:
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ == "pobj" and chunk.root.head.text in ["off", "on", "dim"]:
                entities["location"] = chunk.text
    
    return entities

# Function to classify and extract data from user input
def classify_and_extract(input_text):
    # Process the input text with spaCy
    doc = nlp(input_text.lower())
    
    # Initialize vectors for cosine similarity
    vectorizer = TfidfVectorizer()
    input_vector = vectorizer.fit_transform([input_text.lower()])
    
    # Check against each template
    for template, action in task_templates:
        template_vector = vectorizer.transform([template])
        similarity = cosine_similarity(input_vector, template_vector)[0][0]
        
        if similarity > 0.5:  # Threshold for similarity
            entities = extract_entities(doc, template)
            return action, entities
    
    return None, {}