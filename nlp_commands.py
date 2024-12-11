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
    ("start {location} routine", "start_routine"),
    ("execute {task}", "execute_task"),
]

# Create a simple function to extract entities
def extract_entities(doc, template):
    entities = {}
    if 'execute_task' in template:
        start_token = None
        for token in doc:
            if token.text.lower() == "execute":
                start_token = token
                break
        if start_token:
            entities["task"] = " ".join([t.text for t in doc[start_token.i + 1:]])
        return entities
    
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

def is_command_or_question(sentence):
    # Check if the sentence ends with a question mark
    if sentence.text.endswith('?'):
        return sentence, "question"
    
    # Check if the sentence starts with a common command verb
    common_commands = {'go', 'come', 'stop', 'start', 'open', 'close', 'turn', 'please', 'kindly','get', 'resolve'}
    first_token = sentence[0]
    if first_token.pos_ == 'VERB' and first_token.lemma_.lower() in common_commands:
        return sentence, "command"
    return None, None

def extract_commands_and_questions(text):
    # Process the text with spaCy
    doc = nlp(text)
    # Filter sentences that are commands or questions
    commands_and_questions = []
    for sent in doc.sents:
        sentence, sent_type = is_command_or_question(sent)
        if sentence and sent_type:
            commands_and_questions.append((sentence, sent_type))
    return commands_and_questions

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


# # Example text
# text = "Please close the door. Get that for me. Can you open the window? The meeting is scheduled for 10 AM. Go to the store. What time is it?"

# # Extract commands and questions
# commands_and_questions = extract_commands_and_questions(text)

# # Print the results
# for sent, sent_type in commands_and_questions:
#     print(f"Type: {sent_type}, Sentence: {sent.text}")