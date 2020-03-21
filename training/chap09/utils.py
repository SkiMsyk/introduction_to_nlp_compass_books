import string 
import gensim
import numpy as np  
import pandas as pd  

def filter_by_ascii_rate(text, threshold=0.9):
    ascii_letters = set(string.printable)
    rate = sum(c in ascii_letters for c in text) / len(text)
    return rate <= threshold 

def load_dataset(filename, n=5000, state=6):
    df = pd.read_csv(filename, sep='\t')
    
    mapping = {1:0, 2:0, 4:1, 5:1}
    df = df[df.star_rating != 3]
    df.star_rating = df.star_rating.map(mapping)
    
    is_jp = df.review_body.apply(filter_by_ascii_rate)
    df = df[is_jp]
    
    df = df.sample(frac=1, random_state=state)
    grouped = df.groupby('star_rating')
    df = grouped.head(n=n)
    return df.review_body.values, df.star_rating.values 

def filter_embeddings(embeddings, vocab, num_words, dim=300):
    _embeddings = np.zeros((num_words, dim))
    for word in vocab:
        if word in embeddings:
            word_id = vocab[word]
            if word_id >= num_words:
                continue  
            _embeddings[word_id] = embeddings[word]
    return _embeddings

def load_fasttext(filepath, binary=False):
    model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=binary)
    return model 