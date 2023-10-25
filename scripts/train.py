'''
https://stackoverflow.com/questions/62067400/understanding-accumulated-gradients-in-pytorch/62076913#62076913
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sys import getsizeof
import os
import math
import torch
import torch.nn as nn
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import gc


from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data
from functools import reduce

from dotenv import load_dotenv
load_dotenv()
ROOT_PATH = os.getenv('ROOT_PATH')

device = torch.device("cuda:0") 

calculate_mse  = torch.nn.MSELoss()
optimizer = None 

model = None
"""
GCN extends torch.nn.module adding some properties and defying forward method
"""


class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(num_features, 16)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.conv2 = GCNConv(16, 64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.conv3 = GCNConv(64, 64)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.classifier = Linear(64, 1)
      
   
    def forward(self, x, edge_index,batch):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.batchnorm1(h)
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.batchnorm2(h)
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = global_mean_pool(h,batch)
        out = self.classifier(h)
        
        return out
      

def train(data_loader):
    global model
    
    loss = 0.0
    for idx, data in enumerate(data_loader):
        print(idx)
        data = data.to(device)
        ref = data.y 

        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        

        loss_for_batch = calculate_mse(out, ref)  # Compute the loss. 
        loss_for_batch.backward()  # Derive gradients. 
        
        loss += float(loss_for_batch.detach().cpu().item())
        
        optimizer.step()  # Update parameters based on gradients.
        
        optimizer.zero_grad()

    
        del out
        del loss_for_batch
        del data
        torch.cuda.empty_cache()
        gc.collect()
    
    return loss


def get_dicts():
    
    global_G={}
    files = os.listdir(ROOT_PATH+'/cached_graph')
    files = [ROOT_PATH+'/cached_graph/' + x for x in files if x [-3:]=='.pt'] #select all .pt files

    for f in files:
        local_g= torch.load(f)
        global_G ={**local_g,**global_G} #merge two dicts

    return global_G

def dissect(uni_graph):
    with open('./prova.txt','w') as f:
        f.write(str(uni_graph.edge_index[0].tolist())+',')
        f.write(str(uni_graph.edge_index[1].tolist()))

    

def compress_graph_nodes(graph_list):
    feat_mat = []
    for graph in graph_list:
        print(graph)
        for row in graph.x.tolist():
            feat_mat+=[row]
        
    print(feat_mat[0])
    print(feat_mat[-20])
    feat_mat = np.array(feat_mat)
    feat_mat = feat_mat.transpose()
    column_to_exclude = []
    for idx,row in enumerate(feat_mat):
        if (sum(row)==0):
            column_to_exclude+=[idx]

    print(column_to_exclude)

if __name__ == "__main__":
    
    #global_G = torch.load(ROOT_PATH+'/cached_graph/tensorG_0.pt')
    global_G = get_dicts()
    print('end loading\n Dataset dimension <nGraphs> = {}'.format(len(global_G.keys())))
    
    pytg_graph_dict = {}
    for k in global_G:
        feat_k = global_G[k][1] 
        pytg_graph_dict[k] = global_G[k][0]
        print(k)
        new_x = [ reduce (lambda x,y:x+y,feat_k[int(el.item())]['attributes']) for el in pytg_graph_dict[k].x]

        pytg_graph_dict[k].x = torch.tensor(new_x, dtype = torch.float)# ,device=device)

    
    """
    Function for training GCN model
    """
    k0 = list(pytg_graph_dict.keys())[0]
    num_features = pytg_graph_dict[k0].num_node_features
    model = GCN(pytg_graph_dict[k0].num_node_features)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.008, momentum=0.9) # Define optimizer.
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.00004) # Define optimizer.
  
    graph_list = []
    graph_y = []

    for k in pytg_graph_dict.keys():
        print(pytg_graph_dict[k].y)
        new_g = pytg_graph_dict[k]
        print(new_g)
        #dissect(new_g)
        print(new_g.edge_index.size(dim=1))
        if new_g.edge_index.size(dim=1)<= 4000000:
            graph_list.append(new_g) 
            graph_y.append(pytg_graph_dict[k].y)
    
    batch_size = 6
    print(len(graph_list))
    limit_g = (len(graph_list)//batch_size)*batch_size
    graph_list = graph_list[:limit_g]
    graph_y = graph_y[:limit_g]
    # compress graphs
    #graph_list = compress_graph_nodes(graph_list)
    
    print(limit_g)
    
    del pytg_graph_dict
    del global_G



    loader = DataLoader(graph_list, batch_size=batch_size,shuffle=True)

    print("end data loader")
    loss_values = []

    for epoch in range(6001):
        loss_init = train(loader)
        torch.cuda.empty_cache()

        loss_values.append(loss_init)
        print('##### EPOCH {} #####'.format(epoch))
        with open("result_nn_settings_val_8bs.txt", 'a') as out:
            out.write(f"Epoch: {epoch:03d}, loss: {loss_init:.4f}, min_loss: {min(loss_values):.4f} \n")
        print(f"Epoch: {epoch:03d}, loss: {loss_init:.4f}")
    with open("loss_for_plot_8bs.txt","w") as out:
        out.write(str(loss_values))
                    
    print(loss_values)
    