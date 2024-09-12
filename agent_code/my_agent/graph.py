import matplotlib.pyplot as plt
import numpy as np
import json
import sys

def plot_graph_from_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    data = np.array(data)
    plt.plot(data[:,0], data[:,1])
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a file path as an argument.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    plot_graph_from_file(file_path)
