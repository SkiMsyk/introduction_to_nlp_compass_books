import re

from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer
t = Tokenizer()


def clean_html(html, strip=False):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(strip=strip)
    return text


def tokenize(text):
    return t.tokenize(text, wakati=True)


def tokenize_base_form(text):
    tokens = [token.base_form for token in t.tokenize(text)]
    return tokens 


def normalize_number(text, reduce=False):
    if reduce:
        prog = re.compile(r'\d+')
    else:
        prog = re.compile(r'\d')
    return prog.sub('0', text)


def truncate(sequence, maxlen):
    return sequence[:maxlen]


def remove_url(html):
    soup = BeautifulSoup(html, 'html.parser')
    for a in soup.findAll('a'):
        a.replaceWithChildren()
    return str(soup)