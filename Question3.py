import os
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def read_file(filename):
    with open(filename, 'r') as file:
        return file.read()

def write_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

# Text preprocessing functions
def lowercase_text(text):
    return text.lower()

def tokenize_text(text):
    return text.split()  # We'll use simple splitting instead of NLTK's word_tokenize

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def remove_punctuation(tokens):
    return [token.strip(string.punctuation) for token in tokens]

def remove_blank_tokens(tokens):
    return [token for token in tokens if token.strip() != '']

def preprocess_text(text):
    text = lowercase_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = remove_punctuation(tokens)
    tokens = remove_blank_tokens(tokens)
    return tokens  # Returning list of tokens instead of joined string

# Building positional index
def build_positional_index(preprocessed_directory):
    positional_index = {}
    for file_name in os.listdir(preprocessed_directory):
        file_path = os.path.join(preprocessed_directory, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            tokens = file.read().split()
            for position, token in enumerate(tokens):
                if token not in positional_index:
                    positional_index[token] = {}
                if file_name not in positional_index[token]:
                    positional_index[token][file_name] = []
                positional_index[token][file_name].append(position)
    return positional_index

# Save and load index functions
def save_index(index, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(index, file)

def load_index(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)



def find_phrase_in_document(phrase, document_positions):
    if not phrase:
        return False
    if len(phrase) == 1:
        return True if phrase[0] in document_positions else False
    
    if phrase[0] in document_positions and phrase[1] in document_positions:
        positions_first_word = document_positions[phrase[0]]
        for pos in positions_first_word:
            if all((int(pos) + i) in document_positions[phrase[i]] for i in range(1, len(phrase))):
                return True
    return False




# Main function
def main():
    
    dataset_directory = 'E:/IIITD/Sem 2/IR/Assignment1/Data Set/BeforePreprocessingFiles'
    preprocessed_directory = 'E:/IIITD/Sem 2/IR/Assignment1/Data Set/AfterPreprocessingFiles'
    
    # Building indexes
    positional_index = build_positional_index(preprocessed_directory)
    
   

    # Save indexes
    save_index(positional_index, 'positional_index.pkl')

    # Load positional index
    positional_index = load_index('positional_index.pkl')

    #  # Print the positional index
    # print("Positional Index:")
    # print(positional_index)

     # Positional query input
    phrase_queries = [
        "Not rugged enough for",
        "Coffee brewing techniques in cookbook"
    ]
    
    # Process phrase queries with positional index
    for query in phrase_queries:
        preprocessed_query = preprocess_text(query)
        matching_documents = []
        for term in preprocessed_query:
            if term in positional_index:
                for document, positions in positional_index[term].items():
                    if find_phrase_in_document(preprocessed_query, positional_index[term]):
                        if document not in matching_documents:
                            matching_documents.append(document)
        print(f"Phrase Query: {query}")
        print(f"Number of documents retrieved: {len(matching_documents)}")
        print(f"Names of documents retrieved: {', '.join(matching_documents)}")



if __name__ == "__main__":
    main()
