from utils import read_data
import torch
import pandas as pd
import lstm
import rnn
import seaborn as sns
import hyperParameters
import matplotlib.pyplot as plt


class Run:

    def __init__(self):
        self.data_dict = read_data()
        self.train_text = self.data_dict["train_text"]
        self.test_text = self.data_dict["test_text"]
        self.index2char = self.data_dict["index2char"]
        self.char2index = self.data_dict["char2index"]

    def run_lstm(self, hyper_params, save=True, print_every=500, return_model=False):

        lstm_gen = lstm.Generator(
        input_string=self.train_text,
        test_string=self.test_text,
        index2char=self.index2char,
        char2index=self.char2index,
        sequence_length=hyper_params.SEQUENCE_LENGTH,
        batch_size=hyper_params.BATCH_SIZE
        )

        lstm_model = lstm.RNN(
            input_size=len(self.index2char),
            hidden_size=hyper_params.HIDDEN_SIZE,
            num_layers=hyper_params.NUM_LAYERS,
            output_size=len(self.index2char),
        ).to(lstm_gen.device)

        lstm_gen.train(
            lstm=lstm_model,
            num_epchs=hyper_params.NUM_EPOCHS,
            print_every=500,
            lr=hyper_params.LEARNING_RATE,
            temperature=hyper_params.TEMPERATURE,
            label_smoothing=hyper_params.LABEL_SMOOTHING,
        )

        if (return_model):
            return lstm_gen

        if (save):
            """
            Save LSTM model
            """
            torch.save(lstm_gen.lstm.state_dict(), f"{DIR}/lstm_hidden{HIDDEN_SIZE}_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.pth")
            lstm_df = pd.DataFrame(lstm_gen.history)
            lstm_df.to_csv(f'{DIR}/lstm_hidden{HIDDEN_SIZE}_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.csv', index=False)

            sns.set_style("whitegrid")
            title_string = f"LSTM: Loss vs iterations\nHidden Layers:{HIDDEN_SIZE}, lr:{LEARNING_RATE}"
            lstm_fig = sns.lineplot(data=lstm_df, x="iterations", y="loss")
            lstm_fig.get_figure().savefig(f'{DIR}/lstm_hidden{HIDDEN_SIZE}_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.png')
            """
            To load model to cuda but it was saved on cpu (or vice versa) use:
            device = torch.device("cuda")
            lstm = lstm.RNN(*args, **kwargs)
            lstm.load_state_dict(torch.loadc(PATH, map_location=device))
            """

    def run_rnn(self, hyper_params):
        rnn_gen =  rnn.Generator(
            input_string=self.train_text,
            test_string=self.test_text,
            index2char=self.index2char,
            char2index=self.char2index,
            sequence_length=hyper_params.SEQUENCE_LENGTH,
            batch_size=hyper_params.BATCH_SIZE
            )

        rnn_model =  rnn.RNN(
            input_size=len(self.index2char),
            hidden_size=hyper_params.HIDDEN_SIZE,
            num_layers=hyper_params.NUM_LAYERS,
            output_size=len(self.index2char),
        ).to(rnn_gen.device)

        rnn_gen.train(
            rnn=rnn_model,
            num_epchs=hyper_params.NUM_EPOCHS,
            print_every=500,
            lr=hyper_params.LEARNING_RATE,
            temperature=hyper_params.TEMPERATURE,
            label_smoothing=hyper_params.LABEL_SMOOTHING,
        )

        """
        Save RNN model
        """
        torch.save(rnn_model.rnn.state_dict(), f"{DIR}/rnn_hidden{HIDDEN_SIZE}_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.pth")
        rnn_df = pd.DataFrame(rnn_gen.history)
        rnn_df.to_csv(f"{DIR}/rnn_hidden{HIDDEN_SIZE}_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.csv", index=False)

        sns.set_style("whitegrid")
        title_string = f"RNN: Loss vs iterations\nHidden Layers:{HIDDEN_SIZE}, lr:{LEARNING_RATE}"
        rnn_fig = sns.lineplot(data=rnn_df, x="iterations", y="loss")
        rnn_fig.get_figure().savefig(f'{DIR}/rnn_hidden{HIDDEN_SIZE}_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.png')
