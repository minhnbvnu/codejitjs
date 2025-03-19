import argparse
import os
import traceback
from os import listdir
from os.path import isfile, join
import torch
import pandas
from graph_embedding.relational_graph import *
import re

def is_path_exist(path):
    return os.path.exists(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # csv
    parser.add_argument('--node_graph_dir', type=str, help='dir of the node files')
    parser.add_argument('--edge_graph_dir', type=str, help='dir of the edge files')
    parser.add_argument('--embedding_graph_dir', type=str, help='dir to save embedding graph')
    parser.add_argument('--label', type=int, help='label of the commits, 1 if the commits are buggy, 0 otherwise')
    args = parser.parse_args()

    node_graph_dir = args.node_graph_dir
    edge_graph_dir = args.edge_graph_dir
    embedding_graph_dir = args.embedding_graph_dir
    if not os.path.isdir(embedding_graph_dir):
        os.makedirs(embedding_graph_dir)
    node_files = [f for f in listdir(node_graph_dir) if isfile(join(node_graph_dir, f))]
    label = int(args.label)
    if label == 1:
        embedding_graph_dir = os.path.join(embedding_graph_dir, "VTC")
    else:
        embedding_graph_dir = os.path.join(embedding_graph_dir, "VFC")
    if not os.path.isdir(embedding_graph_dir):
        os.makedirs(embedding_graph_dir)
    for f in node_files:
        function_name = f.split(".")[0]
        if is_path_exist(os.path.join(embedding_graph_dir, "data_{}.pt".format(function_name))):
            continue
        try:
            function_name = f.split(".")[0]
            print("function_name", function_name)
            node_info = pandas.read_csv(join(node_graph_dir, f))
            edge_info = pandas.read_csv(join(edge_graph_dir, function_name + ".csv"))
            data = embed_graph(function_name, label, node_info, edge_info)
            torch.save(data, os.path.join(embedding_graph_dir, "data_{}.pt".format(function_name)))
        except:
            traceback.print_exc()
            print("exception:" + function_name)
