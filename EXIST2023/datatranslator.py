from googletrans import Translator
import string

# spanish punctuations
spanish_punctuation = '¡¿«»'

# punctuations to remove
multilingual_punctuation = string.punctuation + spanish_punctuation
translator = str.maketrans('', '', multilingual_punctuation)

# initialize the translator
google_translator = Translator()


# define a function to translate text to the target language
def translate_text(text, target_language):
    translated_tweet = google_translator.translate(text, dest=target_language).text

    # removing punctuations
    translated_tweet = translated_tweet.translate(translator)

    # convert to lowercase
    translated_tweet = translated_tweet.lower()

    return translated_tweet


