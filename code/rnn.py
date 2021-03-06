import torch
import torch.nn as nn
import numpy as np
import time 
from utils import read_data
from torch.utils.tensorboard import SummaryWriter


class  RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):    
        super(RNN, self).__init__()
        """
        Class for Reacurrent Neural Network
        -----------------------------------
        Params:
        -------
            input_size (int):
                number of fetures in the input x, eg. number of unique words
            hidden size (int):
                number of feature in the hidden state h (m in assignment 4)
            num_layers (int):
                number of reacurrent layers
            output_size (int):
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(input_size, hidden_size) # embed (nn.Embedding object): Quick lookup table 
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) # fc: Applies linear transformation (y=xA.T+b) that maps hidden states to target space 
    
    def forward(self, x, hidden_prev):
        out = self.embed(x)
        out, hidden = self.rnn(out.unsqueeze(1), hidden_prev)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, hidden

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        # cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden

class Generator():
    def __init__(self, input_string, test_string, index2char, char2index, sequence_length=100, batch_size=100):
        """
        Trains an RNN model that can generate a synthesized text seqence
        ----------------------------------------------------------------
        params:
        -------
            input_string (str):
                the string that represents the train set (Hemingway books)
            test_string (str):
                the string that represents the test set (Hemingway books)
            index2char (dict):
                dictiornay containg index -> unique-characters
            char2index (dict):
                dictiornay unique-characters -> containg index  
        """
        self.input_string = input_string
        self.test_string = test_string 
        self.index2char = index2char
        self.char2index = char2index
        self.sequence_length = sequence_length 
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iteration = 0
        # Keeps track of different metric
        self.history = {
            "iterations":[],
            "loss":[],
            "generated_seq":[]
            }

    def char_tensor(self, string):
        """
        Returns the 1-D tensor representing of a string
        """
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)): 
            tensor[c] = self.char2index[string[c]]
        return tensor

    def get_random_batch(self):
        """
        a random batch of inputs (X) and targets (Y) from self.input_string
        -------------------------------------------------------------------
        Returns:
        --------
        text_input (tensor):
            dim ~ (batch_size, sequence_length)
        text_target (tensor):
            dim ~ (batch_size, sequence_length)
        """
        text_input = torch.zeros(self.batch_size, self.sequence_length)
        text_target = torch.zeros(self.batch_size, self.sequence_length)
        for i in range(self.batch_size):
            # Pick a random chunk of text
            start_idx = np.random.randint(0, len(self.input_string) - self.sequence_length)
            end_idx = start_idx + self.sequence_length + 1
            text_str = self.input_string[start_idx:end_idx]

            text_input[i,:] = self.char_tensor(text_str[:-1])
            text_target[i,:] = self.char_tensor(text_str[1:])

        return text_input.long(), text_target.long()

    def generate(self, generated_seq_length=200, temperature=0.20):
        """
        Generates a synthesized text with the current RNN model  
        -------------------------------------------------------
        Params:
            generated_seq_length (int):
                The  lenght of the string tha we want to generate
            temperature (float between 0 and 1):
                Determines the risk of the synthesized text. For example, 
                if temperature is high, the RNN may generate new words that 
                haven't been seen before. If temperature is low, it takes 
                less risk and picks the most likely next character in the sequence.

        TODO: (Optional) Takes the last x_input and hidden to make an exact sequence prediction 
        """
        initial_str = self.index2char[np.random.randint(len(self.index2char))] # Picks a random word to 
        hidden = self.rnn.init_hidden(batch_size=1, device=self.device)
        initial_input = self.char_tensor(initial_str)
        generated_seq = initial_str #TODO: Should try to generate seq dynamically if there is time
        
        for i in range(len(initial_str) - 1):
            _, hidden = self.rnn(initial_input[i].view(1).to(self.device), hidden)
        
        last_char = initial_input[-1]

        # Generating string
        for i in range(generated_seq_length):
            output, hidden = self.rnn(last_char.view(1).to(self.device), hidden)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            generated_char = self.index2char[top_char.item()]
            generated_seq += generated_char
            last_char = self.char_tensor(generated_char)
        
        return generated_seq 

    def train(self, rnn, num_epchs, temperature=0.2, lr=0.01, print_every=5000, label_smoothing=0.8):
        """
        Trains the RNN model
        --------------------
        params: 
            rnn (rnn object):
                The neural network model to be trained
            num_epochs (int):
                The number of epochs to train the model with
            temperature (float between 0 and 1):
                Determines the risk of the synthesized text
            lr (float between 0 and 1):
                Learning rate aka. eta
            print_every (int):
                How often to print progress. For example if print_every=100, 
                then loss and a synthesized text
        """

        self.rnn = rnn
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=lr)
        compute_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # writer = SummaryWriter(f'Results/name0')

        print("Training starting...")
        toc = time.perf_counter()

        smooth_loss = None 
        for epoch in range(1, num_epchs + 1):
            loss = 0
            hidden = self.rnn.init_hidden(self.batch_size, self.device)
            self.rnn.zero_grad()
            x_input, target = self.get_random_batch()
            hidden = self.rnn.init_hidden(self.batch_size, self.device)

            for c in range(self.sequence_length):
                x_input[:, c]
                output, hidden = self.rnn(x_input[:, c], hidden)
                loss += compute_loss(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() #/ self.sequence_length
            loss /= self.sequence_length
            
            # print(f"before: {smooth_loss}")
            smooth_loss = loss if smooth_loss==None else smooth_loss
            smooth_loss = (0.999 * smooth_loss + 0.001 * loss) 
            # print(f"after: {smooth_loss}\n")
            self.iteration += 1

            if epoch % print_every==0 or epoch==1:
                time_elapsed_sec = time.perf_counter() - toc
                time_elapsed = time.strftime("%Hh:%Mm:%Ss", time.gmtime(time_elapsed_sec))
                generated_seq = self.generate(temperature=temperature)
                print(f"Epoch {epoch}/{num_epchs}, loss: {smooth_loss:.4f}, time elapsed: {time_elapsed}")
                print(generated_seq)
                print()
                self.history["generated_seq"].append(generated_seq)
                self.history["loss"].append(smooth_loss)
                self.history["iterations"].append(self.iteration)

            # writer.add_scalar("Training loss", loss, global_step=loss)       

# if __name__ == '__main__':
#     data_dict = read_data()
#     train_text = data_dict["train_text"]
#     test_text = data_dict["test_text"]
#     index2char = data_dict["index2char"]
#     char2index = data_dict["char2index"]
#     SEQUENCE_LENGTH = 25
#     BATCH_SIZE = 1
#     NUM_EPOCHS = 1000
#     HIDDEN_SIZE = 100
#     NUM_LAYERS = 2
#     TEMPERATURE = 0.28
#     LEARNING_RATE = 0.01

#     generator = Generator(
#         input_string=train_text,
#         test_string=test_text,
#         index2char=index2char, 
#         char2index=char2index,
#         sequence_length=SEQUENCE_LENGTH,
#         batch_size=BATCH_SIZE
#         )
    
#     rnn = RNN(
#         input_size=len(index2char), 
#         hidden_size=HIDDEN_SIZE, 
#         num_layers=NUM_LAYERS, 
#         output_size=len(index2char),
#     ).to(generator.device)

#     generator.train(
#         rnn=rnn,
#         num_epchs=NUM_EPOCHS,
#         print_every=100,
#         lr=LEARNING_RATE,
#         temperature=TEMPERATURE,
#     )