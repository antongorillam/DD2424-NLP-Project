"""
Example code of how load an LSTM model
"""
from utils import read_data, load_model
import torch
import pandas as pd

class loadModel:

    def __init__(self, modelDir="../results/rnn_vs_lstm/lstm_hidden100_epoch100000_lr0.01_nlayer2", h_size=100, n_layer=2):
        device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")
        self.dir = modelDir+".csv"
        self.lolp = "lolp"
        self.lstm_gen = load_model(
            dir=modelDir+".pth",
            hidden_size=h_size,
            num_layers=n_layer,
        )

    def synthesize(self, initial_input="l", seq_length=500):
        print(self.lstm_gen.generate(temperature=0.9, top_p=.95, top_k=120, generated_seq_length=seq_length, initial_str=initial_input))
        return self.lstm_gen.generate(temperature=0.9, top_p=.95, top_k=120, generated_seq_length=seq_length, initial_str=initial_input)



if __name__ == '__main__':
    model = loadModel()
    model.synthesize(initial_input="l", seq_length=100)
