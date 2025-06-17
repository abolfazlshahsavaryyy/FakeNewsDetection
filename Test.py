import numpy as np
def avg_sentences_length(text):
    sentences=text.split('.')
    sentences_length=[len(i) for i in sentences]
    return np.mean(sentences_length)

x=avg_sentences_length("this is.this is an.this is an ea")
print(x)
