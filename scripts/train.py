'''
https://stackoverflow.com/questions/62067400/understanding-accumulated-gradients-in-pytorch/62076913#62076913
'''

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sys import getsizeof
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.nn import global_mean_pool


from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data

device = torch.device("cuda:0") 
#device = torch.device("cpu")

"""
GCN extends torch.nn.module adding some properties and defying forward method
"""


class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 64) 
        self.conv5 = GCNConv(64, 64)# NOTE: a 4th conv layer!
        self.classifier = Linear(64, 1)
        #self.linear1 = torch.nn.Linear(2, 1)

       

    def forward(self, x, edge_index,batch):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = self.conv4(h, edge_index)
        h = h.tanh()
        h = self.conv5(h, edge_index)
        h = h.tanh()
        h = global_mean_pool(h,batch)
        out = self.classifier(h)
        
        return out#, h
        

"""
Function for training GCN model
"""
model = GCN(1)
model.to(device)
calculate_mse  = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Define optimizer.

def train(data_loader):
    model.train()
    #calculate_mse = torch.nn.MSELoss()
    loss = 0.0
    for idx, data in enumerate(data_loader):
        ref = data.y 
        #optimizer.zero_grad()  
        print(idx)   
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss_for_batch = calculate_mse(out, ref)  # Compute the loss. (LINE 77)
        loss_for_batch.backward()  # Derive gradients. 
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        loss += float(loss_for_batch.detach().cpu().item())
        #loss += float(loss_for_batch.detach().item())
    return loss#, h



if __name__ == "__main__":
    
    #torch.save(global_G,'tensorG.pt')
    global_G = torch.load('tensorG_2000.pt')
    #global_G = torch.load('tensorG.pt')

    print('end loading\n Dataset dimension <nGraphs> = {}'.format(len(global_G.keys())))
    pytg_graph_dict = {k: global_G[k][0] for k in global_G.keys()}

    graph_list = []
    graph_y = []

    for k in pytg_graph_dict.keys():
        graph_list.append(T.ToUndirected()(pytg_graph_dict[k])) #each graph into undirected graph
        graph_y.append(pytg_graph_dict[k].y)
    
    del pytg_graph_dict
    del global_G

    loader = DataLoader(graph_list, batch_size=1,shuffle=True)
    #print(len(graph_list))
    #print(graph_list[0].size())
    #print(type(graph_list[0]))
    
    print("end data loader")
    #torch.cuda.empty_cache()
    loss_values = []

    for epoch in range(401):
        #print(torch.cuda.max_memory_allocated())
        loss_init = train(loader)
        #torch.cuda.empty_cache()
        #print(torch.cuda.max_memory_allocated())
        #torch.cuda.empty_cache()
        loss_values.append(loss_init)
        if epoch % 50 == 0:
            with open("result.txt", 'a') as out:
                out.write(f"Epoch: {epoch:03d}, loss: {loss_init:.4f} \n")
            print(f"Epoch: {epoch:03d}, loss: {loss_init:.4f}")
    #print(loss_values)
    plt.plot(loss_values, label="loss value")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("loss_target_2000_3A_load.png")
    #plt.show()
