import torch
import numpy as np

def read_data(file_name):
    """
    Reads a text file and returns  
    -----------------------------
    Params:
    -------
        file_namr (str) : 
            The directory and filename of the textfile
    Returns:
    --------
        dict with the following items:
            dict["text"] (str): 
                full text in string format
            dict["char2index"] (dict):
                dict mapping each characters to index
            dict["index2char"] (dict):
                dict mapping each index to characters
    """
    text = open(file_name, mode='r').read()
    vocab = sorted(set(text))
    char2index = {char: index for index, char in enumerate(vocab)}
    # index2char = np.array(vocab) needed?
    index2char = {index:char for index, char in enumerate(vocab)} 
    return {"text": text, "char2index": char2index, "index2char": index2char}

if __name__ == '__main__':
    char2index, index2char = read_data("../data/The_Sun_Also_Rises.txt")
    # text = open("../data/The_Sun_Also_Rises.txt", mode='r').read()
    # vocab = sorted(set(text))
    # char2index = {char: index for index, char in enumerate(vocab)}
    # index2char = {index:char for index, char in enumerate(vocab)}     
    # index2char = np.array(vocab) Needed?
