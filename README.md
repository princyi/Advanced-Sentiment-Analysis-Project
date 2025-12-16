# Advanced-Sentiment-Analysis-Project
This is an enhanced Python script for sentiment analysis that runs in the VS Code terminal. It builds on basic sentiment classification by adding advanced preprocessing (stopword removal, lemmatization), robust analysis with NLTK's VADER, subjectivity detection, and innovative batch file processing for multiple texts.

How to Run the Project
Install Dependencies:

Ensure Python 3.x is installed.
Install required libraries: Run pip install -r requirements.txt in your terminal.
Run the Script:

Open the script in VS Code.
Open the integrated terminal (View > Terminal).
Run: python advanced_sentiment_analysis.py
Enter text interactively, or use commands like 'batch input.txt' for file analysis. Type 'quit' to exit.
Example Interactive Output:


Copy code
Enter text or command: I absolutely love this innovative project!
Original: 'I absolutely love this innovative project!'
Cleaned: 'absolutely love innovative project'
Sentiment: Positive
Compound Score: 0.85
Detailed Scores - Pos: 0.67, Neg: 0.00, Neu: 0.33
Subjectivity: Subjective (Score: 0.75)
--------------------------------------------------
Enter text or command: batch sample_texts.txt
Batch processing complete! Results saved to 'output.txt'.
--------------------------------------------------
Enter text or command: quit
Goodbye!
Batch Mode:

Create a .txt file (e.g., input.txt) with one sentence per line.
Run the command in the terminal: batch input.txt
Results are saved to output.txt in the same directory.
Library Used for Sentiment Analysis
NLTK (with VADER): The Natural Language Toolkit provides VADER (Valence Aware Dictionary and sEntiment Reasoner), an advanced lexicon-based sentiment analyzer. It outputs compound scores (-1 to 1) and detailed positive/negative/neutral scores, making it more accurate for varied text (e.g., slang, emojis). TextBlob is used lightly for subjectivity detection.
Notes
The script is code-only, with no GUI or web components.
Advanced preprocessing improves analysis by filtering noise.
Innovative features include batch processing (file I/O) and subjectivity insights for richer output.
Handles errors gracefully (e.g., invalid files). Code is modular, commented, and focuses on Python fundamentals. If it doesn't run in the terminal, it doesn't count!
