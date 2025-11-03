from scipy.io import loadmat
from scipy import sparse
import numpy as np
import networkx as nx
import torch

def edge_index(subgraph):
    edge = np.array(subgraph.edges(), dtype=int).T
    if edge.size == 0:
        return torch.empty((2,0), dtype=torch.long)
    edge_index = torch.from_numpy(edge)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index

def train_test_split(node, edge, labels):
    # Convert to CSR (so we can slice)
    if sparse.isspmatrix_coo(edge): 
        edge = edge.tocsr()
    if sparse.isspmatrix_coo(node): 
        node = node.tocsr()

    # Create NetworkX graph
    G = nx.from_scipy_sparse_array(edge)

    # np.random.seed(42)
    nodes = np.array(G.nodes())
    np.random.shuffle(nodes)
    split_point = int(0.8 * len(nodes))
    train_nodes = nodes[:split_point]
    test_nodes = nodes[split_point:]

    # Create induced subgraphs (This is for induction based GNN training)
    G_train = G.subgraph(train_nodes).copy()
    G_test = G.subgraph(test_nodes).copy()

    # Get corresponding features and labels
    train_idx = np.array(list(G_train.nodes()))
    test_idx = np.array(list(G_test.nodes()))

    X_train, X_test = node[train_idx], node[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    train_edge_index = edge_index(G_train)
    test_edge_index = edge_index(G_test)
    print("train_edge_index: ", train_edge_index)
    print("train_edge_index_shape", train_edge_index.shape)

    print("Total nodes:", len(nodes))
    print("Train nodes:", len(train_nodes))
    print("Test nodes:", len(test_nodes))
    print("Train edges:", G_train.number_of_edges())
    print("Test edges:", G_test.number_of_edges())
    print("Anomalies in train:", y_train.sum())
    print("Anomalies in test:", y_test.sum())
    return X_train, train_edge_index, y_train, X_test, test_edge_index, y_test
