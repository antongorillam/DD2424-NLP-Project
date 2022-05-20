import hyperParameters
import graph
import runs
import numpy as np
import time

class output_data:
    def __init__(self):
        self.list_m = []
        self.list_loss = []

    def add_loss(self, loss):
        self.loss = np.min(loss)
        self.list_loss.append(self.loss)
        print(self.loss)

    def add_hidden_layer(self, m):
        self.list_m.append(m)


class OptimizeHiddenLayerParams:

    def __init__(self):
        #self.test()
        self.findParam()

    def test(self):
        h = hyperParameters.Hyper_params(hidden_size=500, num_epochs=100000)
        run = runs.Run()
        run.run_lstm(hyper_params=h, print_every=5000)

    def findParam(self):
        output =self.gridSearch(search_type="grid")
        plot = graph.Plot()
        plot.add_bar_graph(x=output.list_m, y=output.list_loss, y_label="Loss", x_label="Hidden Layer Size")
        plot.show_graphs()

    def gridSearch(self, search_type):
        output = output_data()
        toc = time.perf_counter()
        if (search_type == "grid"):
            for i in range(9):
                time_elapsed_sec = time.perf_counter() - toc
                print("iter", i+1)
                time_elapsed = time.strftime("%Hh:%Mm:%Ss", time.gmtime(time_elapsed_sec))
                print(f"time elapsed: {time_elapsed}")
                hidden_size = (i+1)*100
                output.add_hidden_layer(m=hidden_size)
                num_epochs = 5000
                run = runs.Run()
                hyper_params = hyperParameters.Hyper_params(num_epochs=num_epochs, hidden_size=hidden_size, learning_rate=0.005)
                lstm = run.run_lstm(hyper_params=hyper_params, save=False, print_every=num_epochs, return_model=True)
                output.add_loss(loss=lstm.history["loss"])
        return output


if __name__ == '__main__':
    OptimizeHiddenLayerParams()
