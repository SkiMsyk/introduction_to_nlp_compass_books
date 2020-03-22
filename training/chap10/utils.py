def load_dataset(filename, encoding = 'utf-8'):
    sents, labels = [], [] 
    words, tags = [], [] 
    with open(filename, encoding = encoding) as f:
        for line in f:
            line = line.rstrip() 
            if line:
                word, tag = line.split('\t')
                words.append(word)
                tags.append(tag) 
            else:
                sents.append(words)
                labels.append(tags)
                words, tags = [], [] 
        if words:
            sents.append(words)
            labels.append(tags)
    
    return sents, labels 
