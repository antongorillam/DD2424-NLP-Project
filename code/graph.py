import matplotlib.pyplot as plt
import numpy as np

class Plot:

    def __init__(self):
        pass

    def add_bar_graph(self, x, y, y_label="", x_label=""):
        y_pos = np.arange(len(y))
        plt.bar(y_pos, y)
        plt.xticks(y_pos, x)
        plt.ylabel(y_label)
        plt.xlabel(x_label)

    def add_graphs(self, y, y_legend="", y_label="", x_label="", color="Blue"):
        plt.plot(y, label=y_legend, color=color)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()

    def show_graphs(self):
        plt.show()
