"""
Example code of how load an LSTM model 
"""
from cgi import test
from utils import read_data
import torch
import pandas as pd
import lstm
import rnn

if __name__ == '__main__':

    train_txt = "../data/The_Sun_Also_Rises.txt"
    test_txt = "../data/Old_Man_And_The_Sea.txt"
    
    data_dict = read_data(train_txt, test_txt)
    text = data_dict["train_text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]

    DIR = "../results"
    SEQUENCE_LENGTH = 25
    BATCH_SIZE = 1
    NUM_EPOCHS = 10000
    HIDDEN_SIZE = 100
    NUM_LAYERS = 2
    TEMPERATURE = 0.28
    LEARNING_RATE = 0.01
    LABEL_SMOOTHING = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    lstm_model =  lstm.RNN(
        input_size=len(index2char), 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        output_size=len(index2char),
    )

    lstm_model.load_state_dict(torch.load(f"{DIR}/lstm_epoch10000_lr0.01_nlayer2.pth", map_location=device))

    lstm_gen = lstm.Generator(
        input_string=text,
        index2char=index2char, 
        char2index=char2index,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )

    lstm_gen.lstm = lstm_model 

    lstm_gen.generate()
    
    """
    Example of how to generate a text, George will have to 
    - modify the function generate() i lstm.py so that it performs nucleaus sampling
    - Also implement Beam Search there is time
    """