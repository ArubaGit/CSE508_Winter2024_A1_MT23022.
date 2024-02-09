import os
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

all_documents = set()

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
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def remove_blank_tokens(tokens):
    return [token for token in tokens if token.strip() != '']

def remove_punctuation(tokens):
    return [token for token in tokens if token not in string.punctuation]

def preprocess_text(text):
    text = lowercase_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = remove_punctuation(tokens)
    tokens = remove_blank_tokens(tokens)
    return ' '.join(tokens)

# Supporting functions for inverted index
def build_inverted_index(preprocessed_directory):
    inverted_index = {}
    for file_name in os.listdir(preprocessed_directory):
        file_path = os.path.join(preprocessed_directory, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            tokens = file.read().split()
            for token in tokens:
                if token in inverted_index:
                    if file_name not in inverted_index[token]:
                        inverted_index[token].append(file_name)
                else:
                    inverted_index[token] = [file_name]
    return inverted_index

def save_index(index, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(index, file)

def load_index(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

# Boolean query operations
def perform_operation(docs_set1, docs_set2, operation, all_documents):
    if operation == 'AND':
        return docs_set1.intersection(docs_set2)
    elif operation == 'OR':
        return docs_set1.union(docs_set2)
    elif operation == 'AND NOT':
        return docs_set1 - docs_set2
    elif operation == 'OR NOT':
        return docs_set1.union(all_documents - docs_set2)
    else:
        raise ValueError("Unsupported operation")
    
def evaluate_query(inverted_index, preprocessed_query, operations):
    result_docs = set()
    for term in preprocessed_query:
        if term in inverted_index:
            docs_set = set(inverted_index[term])
            if not result_docs:
                result_docs = docs_set
            else:
                result_docs = perform_operation(result_docs, docs_set, operations.pop(0), all_documents)

    return result_docs


# Main function
def main():
    dataset_directory = 'E:/IIITD/Sem 2/IR/Assignment1/Data Set/BeforePreprocessingFiles'
    preprocessed_directory = 'E:/IIITD/Sem 2/IR/Assignment1/Data Set/AfterPreprocessingFiles'

    # Ensure the preprocessed directory exists
    os.makedirs(preprocessed_directory, exist_ok=True)

    # List all text files in the dataset
    file_paths = [os.path.join(dataset_directory, f) for f in os.listdir(dataset_directory) if f.endswith('.txt')]

    # Preprocess text and build inverted index
    for file_path in file_paths:
        text = read_file(file_path)
        preprocessed_text = preprocess_text(text)
        preprocessed_file_path = os.path.join(preprocessed_directory, os.path.basename(file_path))
        write_file(preprocessed_file_path, preprocessed_text)

    # Build inverted index
    inverted_index = build_inverted_index(preprocessed_directory)

    # Save inverted index
    save_index(inverted_index, 'inverted_index.pkl')

    # Load inverted index
    inverted_index = load_index('inverted_index.pkl')

    # Read the number of queries from standard input
    num_queries = int(input("Enter the number of queries: "))
    
    # Process boolean queries with inverted index
    for i in range(num_queries):
        # Read query and operations from standard input
        query_sequence = input("Enter query sequence: ")
        operation_sequence = input("Enter operation sequence: ")
        
        # Preprocess query sequence
        preprocessed_query = preprocess_text(query_sequence).split()
        operations = operation_sequence.split(', ')
        
        # Evaluate query
        result_docs = evaluate_query(inverted_index, preprocessed_query, operations)
        print(f"Query {i+1}: {' '.join(preprocessed_query)}")
        print(f"Number of documents retrieved for query {i+1}: {len(result_docs)}")
        print(f"Names of the documents retrieved for query {i+1}: {', '.join(sorted(result_docs))}")

if __name__ == "__main__":
    main()
