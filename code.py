import re  # For regex-based punctuation removal
import os  # For file path validation
from nltk.sentiment import SentimentIntensityAnalyzer  # VADER for advanced sentiment
from nltk.corpus import stopwords  # For stopword removal
from nltk.stem import WordNetLemmatizer  # For lemmatization
from textblob import TextBlob  # For subjectivity detection (innovative add-on)
import nltk  # Ensure NLTK data is downloaded

# Download required NLTK data (run once if needed)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """
    Advanced preprocessing: lowercase, remove punctuation, remove stopwords, and lemmatize.
    
    Args:
        text (str): The raw input text.
    
    Returns:
        str: The cleaned and processed text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize into words
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Rejoin into a string
    return ' '.join(words)

def analyze_sentiment(text):
    """
    Performs advanced sentiment analysis using NLTK's VADER.
    Also detects subjectivity using TextBlob for innovation.
    
    Args:
        text (str): The preprocessed text.
    
    Returns:
        dict: Contains sentiment label, compound score, detailed scores, and subjectivity.
    """
    # VADER analysis
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    # Classify based on compound score (thresholds: >0.05 positive, <-0.05 negative, else neutral)
    if compound > 0.05:
        label = "Positive"
    elif compound < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    
    # Subjectivity detection (innovative: from TextBlob)
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
    subj_label = "Subjective" if subjectivity > 0.5 else "Objective"
    
    return {
        'label': label,
        'compound': compound,
        'pos': scores['pos'],
        'neg': scores['neg'],
        'neu': scores['neu'],
        'subjectivity': subjectivity,
        'subj_label': subj_label
    }

def process_batch_file(file_path):
    """
    Innovative batch processing: Reads a file (one text per line), analyzes each, and writes results to output.txt.
    
    Args:
        file_path (str): Path to the input file.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        results = []
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            cleaned = preprocess_text(line)
            analysis = analyze_sentiment(cleaned)
            results.append(f"Line {i}: '{line}' -> Sentiment: {analysis['label']} (Compound: {analysis['compound']:.2f}, Subjectivity: {analysis['subj_label']})")
        
        # Write to output file
        with open('output.txt', 'w', encoding='utf-8') as out:
            out.write("Batch Sentiment Analysis Results\n" + "="*50 + "\n")
            out.write("\n".join(results))
        
        print(f"Batch processing complete! Results saved to 'output.txt'.")
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    """
    Main function: Interactive mode for terminal input, with option to switch to batch mode.
    Allows multiple inputs in a loop.
    """
    print("Welcome to the Advanced Sentiment Analysis Tool!")
    print("Features: Preprocessing (stopwords, lemmatization), VADER sentiment, subjectivity detection, and batch file processing.")
    print("Type 'batch <file_path>' to analyze a file (e.g., 'batch input.txt'). Type 'quit' to exit.\n")
    
    while True:
        user_input = input("Enter text or command: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower().startswith('batch '):
            # Extract file path
            parts = user_input.split(' ', 1)
            if len(parts) < 2:
                print("Usage: batch <file_path>")
                continue
            file_path = parts[1]
            process_batch_file(file_path)
            print("-" * 50 + "\n")
            continue
        
        if not user_input:
            print("Please enter some text or a command.\n")
            continue
        
        # Interactive mode: Preprocess and analyze
        cleaned_text = preprocess_text(user_input)
        analysis = analyze_sentiment(cleaned_text)
        
        # Display detailed results
        print(f"Original: '{user_input}'")
        print(f"Cleaned: '{cleaned_text}'")
        print(f"Sentiment: {analysis['label']}")
        print(f"Compound Score: {analysis['compound']:.2f}")
        print(f"Detailed Scores - Pos: {analysis['pos']:.2f}, Neg: {analysis['neg']:.2f}, Neu: {analysis['neu']:.2f}")
        print(f"Subjectivity: {analysis['subj_label']} (Score: {analysis['subjectivity']:.2f})")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
