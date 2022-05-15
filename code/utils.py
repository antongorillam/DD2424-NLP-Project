import torch
import numpy as np

def read_data():
    """
    Reads a text file and returns
    -----------------------------
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
        train_text += open(train, mode='r', encoding='utf-8').read()

    test_text = ""
    for test in test_dir:
        test_text += open(test, mode='r', encoding='utf-8').read()



    vocab = sorted(set(train_text + test_text)) # + test_text))
    char2index = {char: index for index, char in enumerate(vocab)}
    # index2char = np.array(vocab) needed?
    index2char = {index: char for index, char in enumerate(vocab)}
    return {"train_text": train_text, "test_text": test_text, "char2index": char2index, "index2char": index2char}

def load_model(dir, hidden_size, num_layers):
    """
    Load a pre-trained LSTM model in a generator object
    (do not work for vanilla RNN for some reason)
    ---------------------------------------------
    params:
        dir (str):
            directory of the model we want to load
        hidden_size (int):
            hidden size of the model to load
        num_layers (int):
            number of layers of the model to load
    """
    import lstm

    data_dict = read_data()
    train_text = data_dict["train_text"]
    test_text = data_dict["test_text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create generator obejct
    lstm_gen = lstm.Generator(
        input_string="train_text",
        test_string="test_text",
        index2char=None,
        char2index=None,
        sequence_length=None,
        batch_size=None
    )
    # initiate empty lstm object
    lstm_model =  lstm.RNN(
        input_size=len(index2char),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=len(index2char),
    )
    # Load an existing lstm parameters to our empty lstm object
    lstm_model.load_state_dict(torch.load(f"{dir}", map_location=device))
    # Put the lstm object in our generator function
    lstm_gen.lstm = lstm_model

    return lstm_gen



def read_data_shakespeare():
    """
    Reads the Shakespeare text and returns a dictonary of it
    This is to validate our dataset
    -------------------------------
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
                dict mappingis each characters to index
            dict["index2char"] (dict):
                dict mapping each index to characters
    """
    DIR = "../data/shakespeare/shakespeare.txt"

    text = open(DIR, mode='r', encoding='utf-8').read()
    vocab = sorted(set(text))
    char2index = {char: index for index, char in enumerate(vocab)}
    index2char = {index: char for index, char in enumerate(vocab)}
    return {"text": text, "char2index": char2index, "index2char": index2char}

if __name__ == '__main__':

    import lstm
    import pandas as pd
    import seaborn as sns
    data_dict = read_data_shakespeare()
    train_text = data_dict["text"]
    # test_text = data_dict["test_text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]

    DIR = "../results/shakespeare"
    SEQUENCE_LENGTH = 100
    BATCH_SIZE = 1
    NUM_EPOCHS = 50000
    HIDDEN_SIZE = 100
    NUM_LAYERS = 2
    TEMPERATURE = 0.8
    LEARNING_RATE = 0.01
    LABEL_SMOOTHING = 0


    lstm_gen = load_model(
        dir=f"{DIR}/lstm_hidden100_epoch50000_lr0.01_nlayer2.pth",
        hidden_size=100,
        num_layers=2,
        )
    lstm_gen.generate()
