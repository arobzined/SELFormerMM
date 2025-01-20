import torch
import torch.nn.functional as F


import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv


def load_heterodata(path):
    
    data = torch.load(path, map_location=torch.device('cpu'))

    print("Available edge types in the dataset:", data.edge_types)    
    # data['Compound'].train_mask = torch.zeros(data['Compound'].num_nodes, dtype=torch.bool)
    # data['Compound'].val_mask = torch.zeros(data['Compound'].num_nodes, dtype=torch.bool)
    # data['Compound'].test_mask = torch.zeros(data['Compound'].num_nodes, dtype=torch.bool)

    
    # train_indices = np.random.choice(data['Compound'].num_nodes, int(data['Compound'].num_nodes * 0.8), replace=False)
    # val_indices = np.random.choice(np.setdiff1d(np.arange(data['Compound'].num_nodes), train_indices), int(data['Compound'].num_nodes * 0.1), replace=False)
    # test_indices = np.setdiff1d(np.arange(data['Compound'].num_nodes), np.concatenate([train_indices, val_indices]))

    # data['Compound'].train_mask[train_indices] = 1
    # data['Compound'].val_mask[val_indices] = 1
    # data['Compound'].test_mask[test_indices] = 1

    
    # print(f'Train node count: {data["Compound"].train_mask.sum()}')
    # print(f'Val node count: {data["Compound"].val_mask.sum()}')
    # print(f'Test node count: {data["Compound"].test_mask.sum()}')

    metapaths = [
        [('Compound', 'CTI', 'Protein'), ('Protein', 'rev_CTI', 'Compound')],
        [('Drug', 'DTI', 'Protein'), ('Protein', 'rev_DTI', 'Drug')],
        [('Protein', 'PPI', 'Protein'), ('Protein', 'rev_PPI', 'Protein')],
        [('Gene', 'Orthology', 'Gene'), ('Gene', 'rev_Orthology', 'Gene')],
    ]
    print(metapaths)
    
    data = T.AddMetaPaths(metapaths, drop_orig_edge_types=True)(data)
    print('Available edge types in the dataset after adding metapaths:', data.edge_types)

    return data

class DMGI(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, num_relations):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels, out_channels) for _ in range(num_relations)])
        self.M = torch.nn.Bilinear(out_channels, out_channels, 1)
        self.Z = torch.nn.Parameter(torch.empty(num_nodes, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.M.weight)
        self.M.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.Z)

    def forward(self, x, edge_indices):
        pos_hs, neg_hs, summaries = [], [], []
        for conv, edge_index in zip(self.convs, edge_indices):
            pos_h = F.dropout(x, p=0.5, training=self.training)
            pos_h = conv(pos_h, edge_index).relu()
            pos_hs.append(pos_h)

            neg_h = F.dropout(x, p=0.5, training=self.training)
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)]
            neg_h = conv(neg_h, edge_index).relu()
            neg_hs.append(neg_h)

            summaries.append(pos_h.mean(dim=0, keepdim=True))

        return pos_hs, neg_hs, summaries

    def loss(self, pos_hs, neg_hs, summaries):
        loss = 0.
        for pos_h, neg_h, s in zip(pos_hs, neg_hs, summaries):
            s = s.expand_as(pos_h)
            loss += -torch.log(self.M(pos_h, s).sigmoid() + 1e-15).mean()
            loss += -torch.log(1 - self.M(neg_h, s).sigmoid() + 1e-15).mean()

        pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0)
        neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0)

        pos_reg_loss = (self.Z - pos_mean).pow(2).sum()
        neg_reg_loss = (self.Z - neg_mean).pow(2).sum()
        loss += 0.001 * (pos_reg_loss - neg_reg_loss)

        return loss
    

def load_dmgi_model(path, data):
        
        model = DMGI(data['Compound'].num_nodes,
                    data['Compound'].x.size(-1), 
                    64,
                    len(data.edge_types))
        
        model.load_state_dict(torch.load(path))
        
        return model
