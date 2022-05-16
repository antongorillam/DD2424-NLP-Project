"""
Example code of how load an LSTM model 
"""
from utils import read_data, load_model
import torch
import pandas as pd
import lstm
import rnn



if __name__ == '__main__':
    data_dict = read_data()
    train_text = data_dict["train_text"]
    test_text = data_dict["test_text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]

    DIR = "../results/rnn_vs_lstm"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    lstm_gen = load_model(
        dir="../results/rnn_vs_lstm/lstm_hidden100_epoch100000_lr0.01_nlayer2.pth",
        hidden_size=100,
        num_layers=2,
        )
    #test = lstm_gen.generate(temperature=0.9, top_p=.95, top_k=120, generated_seq_length=500)
    test = lstm_gen.generate(beam_search=True, beam_width= 3,generated_seq_length=10)

    print(test)
    """
    Example of how to generate a text, George will have to 
    - modify the function generate() i lstm.py so that it performs nucleaus sampling
    - Also implement Beam Search there is time
    """