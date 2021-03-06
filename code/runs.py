from turtle import color
from utils import read_data_shakespeare
import torch
import pandas as pd
import lstm
import rnn
import seaborn as sns
import hyperParameters
import matplotlib.pyplot as plt


class Run:

    def __init__(self, training_file="/train_shakespeare.txt"):
        self.data_dict = read_data_shakespeare(file=training_file)
        self.train_text = self.data_dict["train_text"]
        self.test_text = self.data_dict["test_text"]
        self.index2char = self.data_dict["index2char"]
        self.char2index = self.data_dict["char2index"]

    def run_lstm(self, hyper_params, save=True, print_every=500, return_model=False, fileName=""):

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
            print_every=print_every,
            lr=hyper_params.LEARNING_RATE,
            temperature=hyper_params.TEMPERATURE,
            label_smoothing=hyper_params.LABEL_SMOOTHING,
        )

        if (save):
            """
            Save LSTM model
            """
            torch.save(lstm_gen.lstm.state_dict(), f"{hyper_params.DIR}/lstm_hidden{hyper_params.HIDDEN_SIZE}_epoch{hyper_params.NUM_EPOCHS}_lr{hyper_params.LEARNING_RATE}_nlayer{hyper_params.NUM_LAYERS}{fileName}.pth")
            lstm_df = pd.DataFrame(lstm_gen.history)
            lstm_df.to_csv(f'{hyper_params.DIR}/lstm_hidden{hyper_params.HIDDEN_SIZE}_epoch{hyper_params.NUM_EPOCHS}_lr{hyper_params.LEARNING_RATE}_nlayer{hyper_params.NUM_LAYERS}{fileName}.csv', index=False)

            #sns.set_style("whitegrid")
            #title_string = f"LSTM: Loss vs iterations\nHidden Layers:{hyper_params.HIDDEN_SIZE}, lr:{hyper_params.LEARNING_RATE}"
            #lstm_fig = sns.lineplot(data=lstm_df, x="iterations", y="loss").set_title(title_string)
            #lstm_fig.get_figure().savefig(f'{hyper_params.DIR}/lstm_hidden{hyper_params.HIDDEN_SIZE}_epoch{hyper_params.NUM_EPOCHS}_lr{hyper_params.LEARNING_RATE}_nlayer{hyper_params.NUM_LAYERS}.png')
            """
            To load model to cuda but it was saved on cpu (or vice versa) use:
            device = torch.device("cuda")
            lstm = lstm.RNN(*args, **kwargs)
            lstm.load_state_dict(torch.loadc(PATH, map_location=device))
            """
        if (return_model):
            return lstm_gen

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
        torch.save(rnn_model.rnn.state_dict(), f"{hyper_params.DIR}/rnn_hidden{hyper_params.HIDDEN_SIZE}_epoch{hyper_params.NUM_EPOCHS}_lr{hyper_params.LEARNING_RATE}_nlayer{hyper_params.NUM_LAYERS}.pth")
        rnn_df = pd.DataFrame(rnn_gen.history)
        rnn_df.to_csv(f"{hyper_params.DIR}/rnn_hidden{hyper_params.HIDDEN_SIZE}_epoch{hyper_params.NUM_EPOCHS}_lr{hyper_params.LEARNING_RATE}_nlayer{hyper_params.NUM_LAYERS}.csv", index=False)

        sns.set_style("whitegrid")
        title_string = f"RNN: Loss vs iterations\nHidden Layers:{hyper_params.HIDDEN_SIZE}, lr:{hyper_params.LEARNING_RATE}"
        rnn_fig = sns.lineplot(data=rnn_df, x="iterations", y="loss").set_title(title_string)
        rnn_fig.get_figure().savefig(f'{hyper_params.DIR}/rnn_hidden{hyper_params.HIDDEN_SIZE}_epoch{hyper_params.NUM_EPOCHS}_lr{hyper_params.LEARNING_RATE}_nlayer{hyper_params.NUM_LAYERS}.png')

# if __name__=='__main__':
#     h = hyperParameters.Hyper_params(hidden_size=500, num_epochs=10000, learning_rate=0.005, dir="../results/anton_test/learning_rate_0005")
#     run = Run()
#     run.run_lstm(hyper_params=h, save=True, print_every=100)

# if __name__=='__main__':
#     hidden25_top_p01 = pd.read_csv("hidden25_temp0_topp0.1_topk0_check.csv")
#     hidden50_top_p01 = pd.read_csv("hidden50_temp0_topp0.1_topk0_check.csv")
#     hidden250_top_p01 = pd.read_csv("hidden250_temp0_topp0.1_topk0_check.csv")
#     hidden500_top_p01 = pd.read_csv("hidden500_temp0_topp0.1_topk0_check.csv")
    
#     hidden25_top_p03 = pd.read_csv("hidden25_temp0_topp0.3_topk0_check.csv")
#     hidden50_top_p03 = pd.read_csv("hidden50_temp0_topp0.3_topk0_check.csv")
#     hidden250_top_p03 = pd.read_csv("hidden250_temp0_topp0.3_topk0_check.csv")
#     hidden500_top_p03 = pd.read_csv("hidden500_temp0_topp0.3_topk0_check.csv")
    
#     hidden25_top_p05 = pd.read_csv("hidden25_temp0_topp0.5_topk0_check.csv")
#     hidden50_top_p05 = pd.read_csv("hidden50_temp0_topp0.5_topk0_check.csv")
#     hidden250_top_p05 = pd.read_csv("hidden250_temp0_topp0.5_topk0_check.csv")
#     hidden500_top_p05 = pd.read_csv("hidden500_temp0_topp0.5_topk0_check.csv")
    
#     hidden25_top_p07 = pd.read_csv("hidden25_temp0_topp0.7_topk0_check.csv")
#     hidden50_top_p07 = pd.read_csv("hidden50_temp0_topp0.7_topk0_check.csv")
#     hidden250_top_p07 = pd.read_csv("hidden250_temp0_topp0.7_topk0_check.csv")
#     hidden500_top_p07 = pd.read_csv("hidden500_temp0_topp0.7_topk0_check.csv")
    
#     hidden25_top_p09 = pd.read_csv("hidden25_temp0_topp0.9_topk0_check.csv")
#     hidden50_top_p09 = pd.read_csv("hidden50_temp0_topp0.9_topk0_check.csv")
#     hidden250_top_p09 = pd.read_csv("hidden250_temp0_topp0.9_topk0_check.csv")
#     hidden500_top_p09 = pd.read_csv("hidden500_temp0_topp0.9_topk0_check.csv")
    
#     hidden25_top_p11 = pd.read_csv("hidden25_temp0_topp1.1_topk0_check.csv")
#     hidden50_top_p11 = pd.read_csv("hidden50_temp0_topp1.1_topk0_check.csv")
#     hidden250_top_p11 = pd.read_csv("hidden250_temp0_topp1.1_topk0_check.csv")
#     hidden500_top_p11 = pd.read_csv("hidden500_temp0_topp1.1_topk0_check.csv")
    
#     hidden25_top_p13 = pd.read_csv("hidden25_temp0_topp1.3_topk0_check.csv")
#     hidden50_top_p13 = pd.read_csv("hidden50_temp0_topp1.3_topk0_check.csv")
#     hidden250_top_p13 = pd.read_csv("hidden250_temp0_topp1.3_topk0_check.csv")
#     hidden500_top_p13 = pd.read_csv("hidden500_temp0_topp1.3_topk0_check.csv")
    
#     hidden25_top_p15 = pd.read_csv("hidden25_temp0_topp1.5_topk0_check.csv")
#     hidden50_top_p15 = pd.read_csv("hidden50_temp0_topp1.5_topk0_check.csv")
#     hidden250_top_p15 = pd.read_csv("hidden250_temp0_topp1.5_topk0_check.csv")
#     hidden500_top_p15 = pd.read_csv("hidden500_temp0_topp1.5_topk0_check.csv")
    
#     hidden25_top_p17 = pd.read_csv("hidden25_temp0_topp1.7_topk0_check.csv")
#     hidden50_top_p17 = pd.read_csv("hidden50_temp0_topp1.7_topk0_check.csv")
#     hidden250_top_p17 = pd.read_csv("hidden250_temp0_topp1.7_topk0_check.csv")
#     hidden500_top_p17 = pd.read_csv("hidden500_temp0_topp1.7_topk0_check.csv")
    
#     hidden25_top_p19 = pd.read_csv("hidden25_temp0_topp1.9_topk0_check.csv")
#     hidden50_top_p19 = pd.read_csv("hidden50_temp0_topp1.9_topk0_check.csv")
#     hidden250_top_p19 = pd.read_csv("hidden250_temp0_topp1.9_topk0_check.csv")
#     hidden500_top_p19 = pd.read_csv("hidden500_temp0_topp1.9_topk0_check.csv")
    
#     df = pd.concat([
#         hidden25_top_p01,
#         hidden50_top_p01,
#         hidden250_top_p01,
#         hidden500_top_p01,
#         hidden25_top_p03,
#         hidden50_top_p03,
#         hidden250_top_p03,
#         hidden500_top_p03,
#         hidden25_top_p05,
#         hidden50_top_p05,
#         hidden250_top_p05,
#         hidden500_top_p05,
#         hidden25_top_p07,
#         hidden50_top_p07,
#         hidden250_top_p07,
#         hidden500_top_p07,
#         hidden25_top_p09,
#         hidden50_top_p09,
#         hidden250_top_p09,
#         hidden500_top_p09,
#         hidden25_top_p11,
#         hidden50_top_p11,
#         hidden250_top_p11,
#         hidden500_top_p11,
#         hidden25_top_p13,
#         hidden50_top_p13,
#         hidden250_top_p13,
#         hidden500_top_p13,
#         hidden25_top_p15,
#         hidden50_top_p15,
#         hidden250_top_p15,
#         hidden500_top_p15,
#         hidden25_top_p17,
#         hidden50_top_p17,
#         hidden250_top_p17,
#         hidden500_top_p17,
#         hidden25_top_p19,
#         hidden50_top_p19,
#         hidden250_top_p19,
#         hidden500_top_p19,
#     ])
