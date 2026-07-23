import os
import re
import ast

import requests
from bs4 import BeautifulSoup

import spacy
import nltk
from nltk.corpus import words
from tqdm import tqdm

_OXFORD_HEADERS = {"User-Agent": "Mozilla/5.0"}


# 定义检查单词是否存在的函数
def check_word_in_oxford(word):
    url = f"https://www.oxfordlearnersdictionaries.com/definition/english/{word}"

    # 发送 HTTP 请求获取网页
    response = requests.get(url, headers=_OXFORD_HEADERS)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # 检查页面是否有单词定义
        if soup.find('span', class_='def'):
            return True
        else:
            return False
    else:
        return False

def check_token_validity(tokens):
    new_tokens = []
    for token in tokens:
        if check_word_in_oxford(token):
            new_tokens.append(token)

    return new_tokens

def preprocessing_dataset(documents):
    if os.path.exists(os.path.join(os.getcwd(), "cache", "preprocessed_documents.txt")):
        with open(os.path.join(os.getcwd(), "cache", "preprocessed_documents.txt"), "r") as f:
            new_documents = ast.literal_eval(f.read())
    else:
        nlp = spacy.load("en_core_web_sm")
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        stop_words.update(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                    'v', 'w', 'x', 'y', 'z', 'title', 'abstract', 'metadata', 'paper', 'author', 'year', 'venue', 'conference', '\\'])
        pattern = re.compile(r"\W+", re.I)

        nltk.download('words')
        english_words = set(words.words())

        new_documents = []
        for idx, document in tqdm(enumerate(documents)):
            doc = nlp(re.sub(r'[^a-zA-Z\s]', '', document))
            token = [token.lemma_.lower() for token in doc if token.text.lower() not in stop_words and not token.is_punct and not token.is_digit and token.is_alpha]
            token = check_token_validity(token)
            new_documents.append(token)

        os.makedirs(os.path.join(os.getcwd(), "cache"), exist_ok=True)
        with open(os.path.join(os.getcwd(), "cache", "preprocessed_documents.txt"), "w") as f:
            f.write(str(new_documents))

    return new_documents
