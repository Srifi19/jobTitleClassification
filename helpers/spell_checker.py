from textblob import TextBlob

def fix_spelling(text):
    return str(TextBlob(text).correct())