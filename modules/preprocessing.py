import re
import emoji
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Khởi tạo các đối tượng được gọi trong code:
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

def get_wordnet_pos(treebank_tag):
    """
    Converts NLTK's part-of-speech tags to WordNet's format.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ   # Adjective
    elif treebank_tag.startswith('V'):
        return wordnet.VERB  # Verb
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN  # Noun
    elif treebank_tag.startswith('R'):
        return wordnet.ADV   # Adverb
    else:
        return wordnet.NOUN  # Default to Noun

def emoji_cleaning(text):
    """
    Converts emojis to text with an identifying prefix and removes consecutive spam.
    E.g., 👍👍👍. -> emoji_thumbs_up .
    """
    # 1. Convert emojis to text, automatically adding an 'emoji_' prefix
    # and adding a space afterward to completely separate them from punctuation.
    text = emoji.demojize(text, delimiters=(" emoji_", " "))

    # 2. Split words into a list
    tokens = text.split()
    cleaned_tokens = []

    for t in tokens:
        if not cleaned_tokens:
            cleaned_tokens.append(t)
        else:
            # 3. Filter Spam:
            # ONLY delete if the current word is an emoji AND identical to the immediately preceding word
            if t.startswith('emoji_') and t == cleaned_tokens[-1]:
                continue

            # Normal words or non-repeating emojis will be kept
            cleaned_tokens.append(t)

    return ' '.join(cleaned_tokens)

def delete_repeated_char(text):
    """
    Reduces characters repeated 3 or more times down to 1.
    """
    return re.sub(r'(\w)\1{2,}', r'\1', text)

def preprocess_and_tokenize(text):
    if not isinstance(text, str):
        return ""

    # 1. Lowercase the text
    text = text.lower()

    # 2. Convert emojis to text representations
    text = emoji_cleaning(text)

    # 3. Delete exaggerated repeated characters
    text = delete_repeated_char(text)

    # 4. Remove all punctuation (keeping only lowercase alphabets, numbers, and spaces)
    text = re.sub(r'[^a-z\s_]', ' ', text)

    # 5. Remove extra lines or spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Tokenize the text (split by space)
    tokens = text.split()

    # 7. Lemmatization
    pos_tags = nltk.pos_tag(tokens)
    lemmatized_tokens = []
    for word, tag in pos_tags:
        # If it's an emoji (already tagged with emoji_ from the previous step), we don't need to lemmatize it
        if word.startswith('emoji_'):
            lemmatized_tokens.append(word)
        else:
            # Get the correct part of speech and lemmatize
            wn_tag = get_wordnet_pos(tag)
            lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=wn_tag))

    # 8. Stopword removal
    final_tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    return ' '.join(final_tokens)