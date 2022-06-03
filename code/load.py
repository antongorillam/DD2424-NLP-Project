"""
Example code of how load an LSTM model 
"""
from random import random
from utils import read_data, load_model, read_data_shakespeare
import torch
import pandas as pd
import lstm
import rnn



if __name__ == '__main__':
    data_dict = read_data_shakespeare()
    train_text = data_dict["train_text"]
    test_text = data_dict["test_text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]

    DIR = "../results/rnn_vs_lstm"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    lstm_gen = load_model(
        dir="../results/hidden_vs_loss/learning_rate_0_005/lstm_hidden500_epoch10000_lr0.005_nlayer2.pth",
        hidden_size=500,
        num_layers=2,
        )
    #test = lstm_gen.generate(temperature=0.9, top_p=.95, top_k=120, generated_seq_length=500)
    """ temperature=0.6
    MOND:\nNo know Marcius thy life, prevent,\nWher live your grace made him; and stay\nIn something you that a prince, that hath not death curest made\nTo pray thing my days words answer thrown.
    """
    temperature_gen = lstm_gen.generate(initial_str="MON", gen_type=0, top_k=0, top_p=0, temperature=0.6, beam_width=None, generated_seq_length=200, random_state=10)
    """ top_k=5
    MONE:\nAs thy fit thou welcome, somethim.\n\nDUKE VINCENTIO:\nI would that too myselvess stay of thou dother\nWill that him.\n\nLADY GREY:\nI'll be myself.\n\nLADY GREY:\nAlone! that hath here sweet to my faith mad
    """
    top_k_gen = lstm_gen.generate(initial_str="MON", gen_type=0, top_k=5, top_p=0, temperature=0, beam_width=None, generated_seq_length=200, random_state=10)
    """ top_p=0
    
    """ 
    top_p_gen = lstm_gen.generate(initial_str="MON", gen_type=0, top_k=0, top_p=1, temperature=0, beam_width=None, generated_seq_length=200, random_state=10)
    """
    
    """ 
    combined_gen = lstm_gen.generate(initial_str="MON", gen_type=1, top_k=0, top_p=0, temperature=0.6, beam_width=5, generated_seq_length=200, random_state=10)
    
    
    print(f"temperature_gen:\n\"\"\"\n{temperature_gen}\n\"\"\"")
    print(f"top_k_gen:\n\"\"\"\n{top_k_gen}\n\"\"\"")
    print(f"top_p_gen:\n\"\"\"\n{top_p_gen}\n\"\"\"")
    print(f"combined_gen:\n\"\"\"\n{combined_gen}\n\"\"\"")
    
    
    



    """
    Example of how to generate a text, George will have to 
    - modify the function generate() i lstm.py so that it performs nucleaus sampling
    - Also implement Beam Search there is time
    """
