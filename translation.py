from googletrans import Translator
from main_ihm import text

text1 = text

translator = Translator()

translated = translator.translate(text1, src='en', dest='ro')