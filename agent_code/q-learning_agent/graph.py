import matplotlib.pyplot as plt
import numpy as np
import json
import sys
from scipy.stats import linregress

steps = 2000000

def plot_graph_from_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    data = np.array(data)

    if len(data) > steps:
        data = data[::len(data)//steps]

    if len(data) < 2:
        print("Not enough data points after sampling.")
        return
    
    window_size = 50

    if len(data) < window_size:
        print("Not enough data points for moving average.")
        return

    y_moving_avg = np.convolve(data[:, 1], np.ones(window_size) / window_size, mode='valid')

    plt.figure()
    plt.title("Training Performance (epsilon 0.3)")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    #number of points drawn is reduced so that data can be understood more clearly
    plt.plot(data[:, 0][::2000], data[:, 1][::2000], label='Reward', linewidth=1)
    plt.plot(data[:, 0][::2000][:len(y_moving_avg[::2000])], y_moving_avg[::2000], label='Average Reward', color='deeppink', linewidth=1)
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a file path as an argument.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    plot_graph_from_file(file_path)
