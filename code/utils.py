import torch
import numpy as np

def read_data():
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
    train_dir = [
        "../data/train/The_Sun_Also_Rises.txt",
        "../data/train/A_Movable_Feast.txt",
        "../data/train/Death_In_The_Afternoon.txt",
        "../data/train/For_Whom_The_Bell_Tolls.txt",
        "../data/train/Men_Without_Women.txt",
        "../data/train/Three_Stories_And_Ten_poems.txt",
        ]


    test_dir = [
        "../data/test/Across_The_River_And_Into_The_Trees.txt",
        "../data/test/A_Farewell_To_Arms.txt",
        "../data/test/Grenn_Hills_of_Africa.txt",
        "../data/test/In_Our_Time.txt",
        "../data/test/Old_Man_And_The_Sea.txt",
        "../data/test/The_Torrents_Of_Spring.txt", 
    ]
    
    train_text = ""
    for train in train_dir:
        train_text += open(train, mode='r').read()
    
    test_text = ""
    for test in test_dir:
        test_text += open(test, mode='r').read()



    vocab = sorted(set(train_text + test_text)) # + test_text))
    char2index = {char: index for index, char in enumerate(vocab)}
    # index2char = np.array(vocab) needed?
    index2char = {index: char for index, char in enumerate(vocab)} 
    return {"train_text": train_text, "test_text": test_text, "char2index": char2index, "index2char": index2char}

if __name__ == '__main__':
    data = read_data()
