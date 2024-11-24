import string

import re

import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



STOP_WORDS_EN = set(stopwords.words('english'))
STOP_WORDS_RU = set(stopwords.words('russian'))
LEMMATIZER = nltk.WordNetLemmatizer()

#Функция для предобработки текста
def preproccess_text2emb(seq):
    tokens = word_tokenize(seq) #Разбивка на токены;
    # LEMMATIZER.lemmatize(token.lower()) - для лемматизации
    #нижний регистр, удаление стоп-слов, удаление лишних символов
    filter_tokens = [ tkn.lower() for tkn in tokens if (tkn not in STOP_WORDS_EN) and (tkn not in string.punctuation) and (re.search(r"[A-Za-zА-Яа-я0-9,()./-]", tkn)) ]
    return " ".join(filter_tokens)