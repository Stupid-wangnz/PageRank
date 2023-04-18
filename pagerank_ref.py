import pandas as pd
import numpy as np
import networkx as nx

data_path = "Data/Data.txt"
N = 8297


def nx_pagerank():
    # generate Graph
    G = nx.DiGraph()
    with open(data_path) as f:
        for line in f:
            s, t = [int(x) for x in line.split()]
            G.add_edge(s, t)
    pr = nx.pagerank(G, )
    num = 0
    with open('./result/nx_package.txt', 'w') as f:
        for node, value in sorted(pr.items(), key=lambda x: x[1], reverse=True):
            f.write("{}\t{}\n".format(node, value))
            num += 1
            if num == 100:
                break


if __name__ == "__main__":
    nx_pagerank()
