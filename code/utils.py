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

def load_model(dir, hidden_size, num_layers, file="/train_shakespeare.txt"):
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

    data_dict = read_data_shakespeare(file=file)
    train_text = data_dict["train_text"]
    test_text = data_dict["test_text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]

    device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")
    # create generator obejct
    lstm_gen = lstm.Generator(
        input_string=train_text,
        test_string=test_text,
        index2char=index2char,
        char2index=char2index,
        sequence_length=None,
        batch_size=None
    )
    # initiate empty lstm object
    lstm_model =  lstm.RNN(
        input_size=len(index2char),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=len(index2char)
    )
    # load an existing lstm parameters to our empty lstm object
    lstm_model.load_state_dict(torch.load(f"{dir}", map_location=device))
    # put the lstm object in our generator function
    lstm_gen.lstm = lstm_model

    return lstm_gen


def read_data_shakespeare(file="/train_shakespeare.txt"):
    """
    Reads the Shakespeare text and returns a dictonary of it
    ---------------------------------------------------------

    Returns:
    --------
        dict with the following items:
            dict["train_set"] (str):
                full text in string format
            dict["test_set"] (str):
                full text in string format
            dict["char2index"] (dict):
                dict mappingis each characters to index
            dict["index2char"] (dict):
                dict mapping each index to characters
    """
    DIR = "../data/shakespeare"
    train_text = open(DIR + file, mode='r', encoding='utf-8').read()
    test_text = open(DIR + file, mode='r', encoding='utf-8').read()

    vocab = sorted(set(train_text + test_text))
    char2index = {char: index for index, char in enumerate(vocab)}
    index2char = {index: char for index, char in enumerate(vocab)}

    return {"train_text": train_text, "test_text": test_text, "char2index": char2index, "index2char": index2char}

def split_shakespeare(test_split=.3, file="/shakespeare.txt"):
    """
    Reads the Shakespeare text and splits the data int train- and test set in a coherent way

    Writes train set to train_shakespeare.txt
    Writes test set to test_shakespeare.txt
    ----------------------------------------------------------------------------------------
    Params:
    -------
        test_split (float) :
            Size (in percentage) of test set
    """
    import re # import regex

    DIR = "../data/shakespeare"
    text = open(DIR + file, mode='r', encoding='utf-8').read()

    text_array = np.array(re.split(r"\n\n", text))
    test_split = int (len(text_array) * test_split)

    test_idx = np.random.choice(np.arange(len(text_array)), test_split, replace=False)
    train_idx = np.array([i for i in range(len(text_array)) if i not in test_idx])

    test_idx.sort()
    train_idx.sort()

    test_set = '\n\n'.join(text_array[test_idx])
    train_set = '\n\n'.join(text_array[train_idx])

    with open(f"{DIR}/train_aug_shakespeare.txt", 'w') as f:
        f.write(train_set)

    with open(f"{DIR}/test_aug_shakespeare.txt", 'w') as f:
        f.write(test_set)
