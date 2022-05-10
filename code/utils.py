import torch
import numpy as np

def read_data(train_file_name, test_file_name):
    """
    Reads a text file and returns  
    -----------------------------
    Params:
    -------
        train_file_name (str) : 
            The directory and filename of the train set textfile
        test_file_name (str) : 
            The directory and filename of the test set textfile
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
    train_text = open(train_file_name, mode='r', encoding='utf8').read()
    test_txt = open(test_file_name, mode='r', encoding='utf8').read()
    
    vocab = sorted(set(train_text)) # + test_txt))
    char2index = {char: index for index, char in enumerate(vocab)}
    # index2char = np.array(vocab) needed?
    index2char = {index: char for index, char in enumerate(vocab)} 
    return {"train_text": train_text, "test_txt": test_txt, "char2index": char2index, "index2char": index2char}

# if __name__ == '__main__':
#     train_txt = "../data/The_Sun_Also_Rises.txt"
#     test_txt = "../data/Old_Man_And_The_Sea.txt"
#     data = read_data(train_txt, test_txt)
    # text = open("../data/The_Sun_Also_Rises.txt", mode='r').read()
    # vocab = sorted(set(text))
    # char2index = {char: index for index, char in enumerate(vocab)}
    # index2char = {index:char for index, char in enumerate(vocab)}     
    # index2char = np.array(vocab) Needed?
