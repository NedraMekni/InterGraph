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
import gc
import sklearn
import random

from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from functools import reduce

from dotenv import load_dotenv
load_dotenv()
ROOT_PATH = os.getenv('ROOT_PATH')

device = torch.device("cuda:0") 

calculate_mse  = torch.nn.MSELoss()
optimizer = None 

model = None
window_graph = 6
"""
GCN extends torch.nn.module adding some properties and defying forward method
"""

class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(num_features, 16)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.conv2 = GCNConv(16, 32)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.conv3 = GCNConv(32, 32)
        self.batchnorm3 = nn.BatchNorm1d(32)
        self.classifier = Linear(32, 1)

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
    model.train()
   
    # https://stackoverflow.com/questions/71527963/how-can-i-send-a-data-loader-to-the-gpu-in-google-colab
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

def test(data_loader):
    #global model
    model.eval()
    mse = []
    mae = []
    mse_tot = 0.0
    mae_tot = 0.0
    out_g = []
    ref_g = []
    num_examples = 0
    print('IN TEST')
    loss = 0.0
    for idx, data in enumerate(data_loader):
        data = data.to(device)
        ref = data.y
        out = model(data.x, data.edge_index, data.batch)
        loss_for_batch = calculate_mse(out, ref)  # Compute the loss. 
        loss += float(loss_for_batch.detach().cpu().item())
        ref = [yr.item() for yr in ref] # y real
        out = [yp.item() for yp in out] # y predicted
        # Compute mean squared error and mean absolute error
        #print(out, ref)
        mse = sum([(yr - yp) ** 2 for yr,yp in zip(ref,out)])/len(ref)
        mae = sum([abs(yr - yp) for yr,yp in zip(ref,out)])/len(ref)
        mse_tot+=mse
        mae_tot+=mae

       

        #print(mse, mae)
        num_examples += len(out)
        ref_g += ref
        out_g += out

    print("TEST num_examples {}".format(num_examples))
    print("TEST MSE_TOT = {}, LOSS = {}".format(mse_tot,loss))
    #mse = sum(mse)/num_examples
    #mae = sum(mae)/num_examples

    return mse_tot, mae_tot, ref_g, out_g
 

def load_model(num_features,device):
    fname =[name for name in os.listdir("./") if "incremental_training.pt" in name]
    if len(fname) > 0:
        orders = [int(x.split("_")[0]) for x in fname]
        order = max(orders)
        fname = "{}_incremental_training.pt".format(order)
    else:
        fname = ""
        
    model = GCN(num_features) 
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    if os.path.exists(fname):
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
    return model, optimizer

def export_model(model, optimizer, epoch, loss):
    fname =[name for name in os.listdir("./") if "incremental_training.pt" in name]
    version_number = 0
    if len(fname) > 0:
        orders = [int(x.split("_")[0]) for x in fname]
        order = max(orders)
        fname = "{}_incremental_training.pt".format(order)
        version_number = int(fname.split("_")[0]) +1
    fname = "{}_incremental_training.pt".format(version_number)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, fname)
def get_dicts():
    from_graph = 0
    if os.path.exists("./.loaded_graph"):
        with open("./.loaded_graph", "r") as f:
            from_graph = int(f.readline())+1
    global_G={}
    files = os.listdir(ROOT_PATH+'/cached_graph')
    files = [x for x in files if x [-3:]=='.pt']
    files = [x for x in files if int(x.split("_")[1].split(".")[0]) >= from_graph and int(x.split("_")[1].split(".")[0]) < from_graph + window_graph]
    print(files)
    if len(files) == 0:
        print("NO MORE FILES TO LOAD")
    files = [ROOT_PATH+'/cached_graph/' + x for x in files] #select all .pt files
    with open('./.loaded_graph','w') as f:
        f.write(str(from_graph+len(files)-1))

    
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


    graph_list = []
    graph_y = []

    for k in pytg_graph_dict.keys():
        print(pytg_graph_dict[k].y)
        new_g = pytg_graph_dict[k]
        print(new_g)
        print(new_g.edge_index.size(dim=1))
        if new_g.edge_index.size(dim=1)<= 4000000:
            graph_list.append(new_g) 
            graph_y.append(pytg_graph_dict[k].y)
    
    batch_size = 6
    print(len(graph_list))
    limit_g = (len(graph_list)//batch_size)*batch_size
    graph_list = graph_list[:limit_g]
    graph_y = graph_y[:limit_g]
    print(limit_g)

    model_num_feature = pytg_graph_dict[k0].num_node_features
    del pytg_graph_dict
    del global_G


    #seed = 9
    seed = 4
    k = 10
    g10 = math.ceil((len(graph_list)*k)/100)
    random.Random(seed).shuffle(graph_list)
    for i in range(k):
        device = torch.device("cuda:0")
        model, optimizer = load_model(model_num_feature,device) 
        #model.to(device)
        #optimizer = torch.optim.SGD(model.parameters(), lr= 0.004, momentum=0.9) # Define optimizer.
        #optimizer = torch.optim.SGD(model.parameters())
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        #optimizer = torch.optim.AdamW(model.parameters())

        
        #test_set = graph_list[:34] 
        test_set = graph_list[i*g10:(i+1)*g10]
        #alternative
        train_set = [graph for graph in graph_list if graph not in test_set]
        print('n training set {}'.format(len(train_set)))
        #train_set = list(np.setdiff1d(graph_list,test_set))
        
        
        # train the model using train and evaluate it using test
        loader_train = DataLoader(train_set, batch_size=batch_size)#,shuffle=True)
        loader_test=DataLoader(test_set,batch_size=batch_size)

        print("end data loader")
        res = []
        res_t=[]
        loss_values = []
        EPOCH = 1000
        for epoch in range(EPOCH):
            loss_init = train(loader_train)
            
            torch.cuda.empty_cache()
            loss_values.append(loss_init)
            #plot
            mse,mae,out_g,ref_g = test(loader_test) 
            mse_t,mae_t,out_g_t,ref_g_t = test(loader_train)
            res+=[(loss_init,mse,mae)]
            res_t+=[(loss_init,mse_t,mae_t)]
            print(res)
            print(res_t)
            torch.cuda.empty_cache()
        with open('train_adam_ki_external_incremental_0.txt','w') as f:
            f.write(str(res_t))
        # Now use trained model for testing
        #export model
        export_model(model, optimizer, EPOCH, loss_values[-1])
        exit()
        
        if i==1:
            # print the graph
            with open('./test_out_ref_split_90_10_validation.txt', 'a') as t:
                t.write('TRUE VALUE {}, PREDICTED VALUE {} \n'.format(ref_g,out_g))
            

        with open('./k_fold_mse_correction_SGD_2a_b.txt','a') as f:
            f.write('##### FOLD {}, MSE {}, MAE {}, ##### \n'.format(i,mse,mae))

        print('##### FOLD {}, MSE {}, MAE {}, #####'.format(i,mse,mae))
        break
        




        del model
        torch.cuda.empty_cache()
        gc.collect()
        