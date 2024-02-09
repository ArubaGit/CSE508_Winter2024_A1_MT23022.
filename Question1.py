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
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def remove_blank_tokens(tokens):
    return [token for token in tokens if token.strip() != '']

def remove_punctuation(tokens):
    return [token for token in tokens if token not in string.punctuation]

def remove_blank_tokens(tokens):
    return [token for token in tokens if token.strip() != '']

def preprocess_text(text):
    text = lowercase_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = remove_punctuation(tokens)
    tokens = remove_blank_tokens(tokens)
    return ' '.join(tokens)


def main():
    dataset_directory = 'E:/IIITD/Sem 2/IR/Assignment1/Data Set/BeforePreprocessingFiles'
    preprocessed_directory = 'E:/IIITD/Sem 2/IR/Assignment1/Data Set/AfterPreprocessingFiles'
    
    # Ensure the preprocessed directory exists
    os.makedirs(preprocessed_directory, exist_ok=True)
    
    # List all text files in the dataset
    file_paths = [os.path.join(dataset_directory, f) for f in os.listdir(dataset_directory) if f.endswith('.txt')]
    
    # Preprocessing steps
    preprocessing_functions = [
        ("Lowercasing", lowercase_text),
        ("Tokenization", tokenize_text),
        ("Stopwords Removal", remove_stopwords),
        ("Punctuation Removal", remove_punctuation),
        ("Blank Tokens Removal", remove_blank_tokens)
    ]
    
    # Process and print the contents of 5 sample files before and after each preprocessing step
    for file_path in file_paths[:5]:  # Limiting to 5 files for demonstration
        print("File:", file_path)
        text = read_file(file_path)
        
        # Print the original content before preprocessing
        #print("\nBefore preprocessing:")
        #print(text[:200])  # Print first 200 characters before preprocessing
        
        processed_text = text
        for preprocessing_step, preprocessing_function in preprocessing_functions:
            print(f"\nBefore {preprocessing_step}:")
            print(processed_text[:200])  # Print first 200 characters before preprocessing step

            # Apply the preprocessing step
            processed_text = preprocessing_function(processed_text)

            print(f"\nAfter {preprocessing_step}:")
            print(processed_text[:200])  # Print first 200 characters after preprocessing step
        
        if isinstance(processed_text, list):
            processed_text = ' '.join(processed_text)
        # Save the preprocessed text
        preprocessed_file_path = os.path.join(preprocessed_directory, f"{os.path.basename(file_path)}_preprocessed.txt")
        write_file(preprocessed_file_path, processed_text)

if __name__ == "__main__":
    main()


