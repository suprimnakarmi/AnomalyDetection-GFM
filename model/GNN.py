from torch_geometric.nn import GCNConv, SAGEConv,GATConv, GINConv
import torch.nn.functional as F
import torch
from torch import nn


class GCN(torch.nn.Module):
    def __init__(self,hops,features_in, features_out):
        super().__init__()
        self.hops = hops
        if hops == 1:
            self.conv1 = GCNConv(features_in,64)
        elif hops == 2:
            self.conv1 = GCNConv(features_in,512)
            self.conv2 = GCNConv(512,64)
        elif hops == 3:
            self.conv1 = GCNConv(features_in,512)
            self.conv2 = GCNConv(512,256)
            self.conv3 = GCNConv(256,64)
        else:
            print("Wrong hops!")
            exit()

        self.fc2 = nn.Linear(64,1)
        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()
        # self.activation = nn.LeakyReLU()

    def forward(self, x, edge_index):
        if self.hops == 1:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
        elif self.hops == 2:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
        else:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
            x = self.conv3(x,edge_index)
            x = self.activation(x)

        # emb = x.clone().detach()  # Only when embedding is needed
        x = self.fc2(x)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self,hops,features_in, features_out):
        super().__init__()
        self.hops = hops
        
        # # self.activation = nn.LeakyReLU()
        # # self.activation = nn.ELU()
        if hops == 1:
            self.conv1 = GATConv(features_in,64)
        elif hops == 2:
            self.conv1 = GATConv(features_in,512)
            self.conv2 = GATConv(512,64)
        elif hops == 3:
            self.conv1 = GATConv(features_in,512)
            self.conv2 = GATConv(512,256)
            self.conv3 = GATConv(256,64)
        else:
            print("Wrong hops!")
            exit()

        self.fc2 = nn.Linear(64,1)
        
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        if self.hops == 1:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
        elif self.hops == 2:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
        else:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
            x = self.conv3(x,edge_index)
            x = self.activation(x)

        # emb = x.clone().detach()      # Only when embedding is needed
        x = self.fc2(x)
        return x
    

class GraphSAGE(torch.nn.Module):
    def __init__(self,hops,features_in, features_out):
        super().__init__()
        self.hops = hops
        
        # # self.activation = nn.LeakyReLU()
        # # self.activation = nn.ELU()
        if hops == 1:
            self.conv1 = SAGEConv(features_in,64)
        elif hops == 2:
            self.conv1 = SAGEConv(features_in,512)
            self.conv2 = SAGEConv(512,64)
        elif hops == 3:
            self.conv1 = SAGEConv(features_in,512)
            self.conv2 = SAGEConv(512,256)
            self.conv3 = SAGEConv(256,64)
        else:
            print("Wrong hops!")
            exit()
                
        self.fc2 = nn.Linear(64,1)
        
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_index):
        if self.hops == 1:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
        elif self.hops == 2:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
        else:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
            x = self.conv3(x,edge_index)
            x = self.activation(x)

        # emb = x.clone().detach() # Only when embedding is needed
        x = self.fc2(x)
        return x

    
class GIN(torch.nn.Module):
    def __init__(self, hops, features_in, features_out):
        super().__init__()
        self.hops = hops

        def gin_mlp(in_c, out_c):
            return nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.ReLU(),
                nn.Linear(out_c, out_c),
            )

        # Select layer widths by dataset 
        if hops == 1:
            self.conv1 = GINConv(gin_mlp(features_in, 64), train_eps=True)
        elif hops == 2:
            self.conv1 = GINConv(gin_mlp(features_in, 512), train_eps=True)
            self.conv2 = GINConv(gin_mlp(512, 64), train_eps=True)
        elif hops == 3:
            self.conv1 = GINConv(gin_mlp(features_in, 512), train_eps=True)
            self.conv2 = GINConv(gin_mlp(512, 256), train_eps=True)
            self.conv3 = GINConv(gin_mlp(256, 64), train_eps=True)
        else:
            print("Wrong hops!"); exit()

        # Final classifier head 
        self.fc2 = nn.Linear(64, 1)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        if self.hops == 1:
            x = self.activation(self.conv1(x, edge_index))
        elif self.hops == 2:
            x = self.activation(self.conv1(x, edge_index))
            x = self.activation(self.conv2(x, edge_index))
        else:  
            x = self.activation(self.conv1(x, edge_index))
            x = self.activation(self.conv2(x, edge_index))
            x = self.activation(self.conv3(x, edge_index))

        # emb = x.clone().detach()  # Only when embedding is needed
        x = self.fc2(x)
        return x
