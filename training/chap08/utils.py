from tensorflow.keras.preprocessing.text import Tokenizer

def  load_data(filepath, encoding = 'utf-8'):
    with open(filepath, encoding = encoding) as f:
        return f.read()




    