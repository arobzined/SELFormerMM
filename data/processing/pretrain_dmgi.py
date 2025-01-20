import os

import numpy as np

import torch
from torch.optim import Adam
from torch_geometric import seed_everything

from dmgi_model import load_heterodata, DMGI

from datetime import datetime

# set random seeds
seed_everything(42)
np.random.seed(42)

torch.set_num_threads(5)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='/home/g3bbmproject/main_folder/KG/kg.pt/selformerv2_kg_heterodata_1224.pt')

args = parser.parse_args()


def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    x = data['Compound'].x
    edge_indices = data.edge_index_dict.values()
    pos_hs, neg_hs, summaries = model(x, edge_indices)
    loss = model.loss(pos_hs, neg_hs, summaries)
    loss.backward()
    optimizer.step()
    return float(loss)


def pretrain_dmgi(hps, data, device):
    model = DMGI(data['Compound'].num_nodes,
                data['Compound'].x.size(-1), 
                hps[0], 
                len(data.edge_types))
    
    data, model = data.to(device), model.to(device)
    print(data.node_types)
    # Print available edge types in the dataset
    print("Available edge types in the dataset:", data.edge_types)

    
    optimizer = Adam(model.parameters(), lr=hps[1], weight_decay=hps[2])

    for epoch in range(1, 101):
        epoch_start = datetime.now()
        train_loss = train(data, model, optimizer)
        
        if epoch == 1 or epoch % 25 == 0:
            print(f'\tEpoch: {epoch:03d}, Loss: {train_loss:.4f}, Time: {datetime.now() - epoch_start}')

    return train_loss, model


if __name__ == '__main__':
    data = load_heterodata(args.data)
    print(f'Loaded data: {args.data}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}\n')

    print('Starting training...\n')
    train_start = datetime.now()
    loss, model = pretrain_dmgi([32, 0.01, 0.001], data, device)
    print(f'\nDone. Total training time: {datetime.now() - train_start}')
    
    # save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'data/pretrained_models/kg_dmgi_model.pt')
    print(f'Model saved: data/pretrained_models/kg_dmgi_model.pt\n')