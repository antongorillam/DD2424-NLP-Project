import torch
import torch.nn as nn
import numpy as np
import time 
from utils import read_data
from torch.utils.tensorboard import SummaryWriter


class  RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):    
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
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(input_size, hidden_size) # embed (nn.Embedding object): Quick lookup table 
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) # fc: Applies linear transformation (y=xA.T+b) that maps hidden states to target space 
    
    def forward(self, x, hidden_prev):
        out = self.embed(x)
        # print(f'out: {out.shape}')
        # print(f'hidden_prev: {hidden_prev.shape}')
        out, hidden = self.rnn(out.unsqueeze(1), hidden_prev)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, hidden

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        # cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden

class Generator():
    def __init__(self, input_string, index2char, char2index, sequence_length=100, batch_size=100):
        """
        Trains an RNN model that can generate a synthesize text seqence
        -----------------------------------
        params:
        -------
            input_string (str):
                the whole input string (a full book in our case)
        """
        self.input_string = input_string
        self.index2char = index2char
        self.char2index = char2index
        self.sequence_length = sequence_length 
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def char_tensor(self, string):
        """
        Returns the tensor representing of a string
        """
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)): 
            tensor[c] = char2index[string[c]]
        return tensor

    def get_random_batch(self):

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

    def generate(self, generated_seq_length=200, temperature=0.25):
        #TODO: Should try to randomnize initial_str? (optional)
        initial_str = self.index2char[np.random.randint(len(self.index2char))]
        hidden = self.rnn.init_hidden(batch_size=1, device=self.device)
        initial_input = self.char_tensor(initial_str)
        generated_seq = initial_str #TODO: Should try to generate seq dynamically if there is time
        
        for i in range(len(initial_str) - 1):
            
            _, hidden = self.rnn(initial_input[i].view(1).to(self.device), hidden)
        
        last_char = initial_input[-1]

        for i in range(generated_seq_length):
            output, hidden = self.rnn(last_char.view(1).to(self.device), hidden)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            generated_char = self.index2char[top_char.item()]
            generated_seq += generated_char
            last_char = self.char_tensor(generated_char)
        
        return generated_seq 

    def train(self, hidden_size, num_layers, num_epchs=100, lr=0.01, print_every=100):
        input_size = len(self.char2index)
        output_size = len(self.char2index)
        self.rnn = RNN(input_size, hidden_size, num_layers, output_size).to(self.device)
        
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=lr)
        compute_loss = nn.CrossEntropyLoss(label_smoothing=.8)
        # writer = SummaryWriter(f'Results/name0')

        print("Training starting...")
        loss = 0
        toc = time.perf_counter()

        for epoch in range(1, num_epchs + 1):
            hidden = self.rnn.init_hidden(self.batch_size, self.device)
            self.rnn.zero_grad()
            x_input, target = self.get_random_batch()
            # X_test = ""
            # Y_test = ""
            # for i in x_input[0]:
            #     X_test += index2char[i.item()]
            # for i in target[0]:
            #     Y_test += index2char[i.item()]


            # print(f'X: {X_test}')
            # print(f'Y: {Y_test}\n')

            hidden = self.rnn.init_hidden(self.batch_size, self.device)

            for c in range(self.sequence_length):
                x_input[:, c]
                output, hidden = self.rnn(x_input[:, c], hidden)
                loss += compute_loss(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.sequence_length

            if epoch % print_every==0:
                time_elapsed_sec = time.perf_counter() - toc
                time_elapsed = time.strftime("%Hh:%Mm:%Ss", time.gmtime(time_elapsed_sec))
                print(f"Epoch {epoch}/{num_epchs}, loss: {loss:.2f}, time elapsed: {time_elapsed}")
                print(self.generate())
                print()

            # writer.add_scalar("Training loss", loss, global_step=loss)

            

if __name__ == '__main__':
    data_dict = read_data("../data/The_Sun_Also_Rises.txt")
    text = data_dict["text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]
    SEQUENCE_LENGTH = 25
    BATCH_SIZE = 10
    NUM_EPOCHS = 10000

    generator = Generator(
        input_string=text, 
        index2char=index2char, 
        char2index=char2index,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
        )
    
    generator.train(
        hidden_size=256, 
        num_layers=3,
        num_epchs=NUM_EPOCHS,
        print_every=10
    )