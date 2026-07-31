import re
import string
from functools import lru_cache
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are downloaded safely
for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except Exception:
        try:
            nltk.download(resource, quiet=True, raise_on_error=False)
        except Exception:
            pass

try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
        "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
        "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
        "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
        "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
        "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
        "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
        "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
        "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
        "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
        "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
        "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
    }

try:
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    LEMMATIZER = None

@lru_cache(maxsize=10000)
def _lemmatize_word(word: str) -> str:
    if LEMMATIZER:
        try:
            return LEMMATIZER.lemmatize(word)
        except Exception:
            pass
    return word

def clean_text(text: str) -> str:
    """
    Cleans raw review text for NLP sentiment classification.
    
    Steps:
    1. Handle NaN / non-string values.
    2. Lowercase text.
    3. Remove HTML tags & URLs.
    4. Remove punctuation, numbers, and special characters.
    5. Tokenize by whitespace.
    6. Remove stopwords and short words.
    7. Fast cached lemmatization.
    8. Rejoin tokens.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)

    # Remove digits and numbers
    text = re.sub(r'\d+', ' ', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenize by whitespace
    tokens = text.split()

    # Filter & lemmatize
    cleaned_tokens = [
        _lemmatize_word(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]

    return " ".join(cleaned_tokens)

if __name__ == "__main__":
    sample = "<p>Love my Echo! Great sound 100% working... http://example.com</p>"
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))
