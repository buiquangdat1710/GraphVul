import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GatedGraphConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import KeyedVectors
import torch.nn.functional as f

edge_type_map = {
    'IS_AST_PARENT': 1, 'IS_CLASS_OF': 2, 'FLOWS_TO': 3, 'DEF': 4, 'USE': 5,
    'REACHES': 6, 'CONTROLS': 7, 'DECLARES': 8, 'DOM': 9, 'POST_DOM': 10,
    'IS_FUNCTION_OF_AST': 11, 'IS_FUNCTION_OF_CFG': 12, '': 0
}

node_type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5,
    'SizeofOperand': 6, 'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpression': 9,
    'ParameterList': 10, 'IdentifierDeclType': 11, 'SizeofExpression': 12, 'SwitchStatement': 13,
    'IncDec': 14, 'Function': 15, 'BitAndExpression': 16, 'UnaryExpression': 17, 'DoStatement': 18,
    'GotoStatement': 19, 'Callee': 20, 'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23,
    'CFGErrorNode': 24, 'WhileStatement': 25, 'InfiniteForNode': 26, 'RelationalExpression': 27,
    'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30, 'CompoundStatement': 31, 'UnaryOperator': 32,
    'CallExpression': 33, 'CastExpression': 34, 'ConditionalExpression': 35, 'ArrayIndexing': 36,
    'PostIncDecOperationExpression': 37, 'Label': 38, 'ArgumentList': 39, 'EqualityExpression': 40,
    'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44, 'ParameterType': 45,
    'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49, 'CastTarget': 50,
    'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'File': 58, 'UnaryOperationExpression': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67,
    'InitializerList': 68, 'ElseStatement': 69
}

class CustomGatedGraphConv(nn.Module):
    def __init__(self, out_channels, num_layers):
        super(CustomGatedGraphConv, self).__init__()
        self.ggnn = torch_geometric.nn.GatedGraphConv(out_channels=out_channels, num_layers=num_layers)
    def forward(self, x, edge_index, edge_attr):
        return self.ggnn(x, edge_index, edge_attr)
def load_data_from_directory(directory):
    edges_file = os.path.join(directory, 'edges.csv')
    nodes_file = os.path.join(directory, 'nodes.csv')
    if not os.path.exists(edges_file) or not os.path.exists(nodes_file):
        print(f"Missing files in directory: {directory}")
        return None, None
    try:
        edges_data = pd.read_csv(edges_file, sep=None, engine='python')
        nodes_data = pd.read_csv(nodes_file, sep=None, engine='python')
        nodes_data = nodes_data[["key", "type", "code", "isCFGNode"]]
        return edges_data, nodes_data
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, KeyError) as e:
        print(f"Error: {e}")
        return None, None 
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

def get_node_embeddings(nodes_data, wv):
    node_embeddings = {}
    for _, row in nodes_data.iterrows():
        node_key = row["key"]
        node_type = row["type"]
        node_code = row["code"]
        if not isinstance(node_code, str):
            node_code = str(node_code)
        tokens = node_code.split()
        code_embedding = np.zeros(300)
        for token in tokens:
            try:
                embedding = wv[token]
            except KeyError:
                embedding = np.zeros(300)
            code_embedding += embedding
        
        if len(tokens) > 0:
            code_embedding /= len(tokens)
        
        type_embedding = np.zeros(len(node_type_map))
        if node_type in node_type_map:
            type_embedding[node_type_map[node_type] - 1] = 1
        
        full_embedding = np.concatenate([code_embedding, type_embedding])
        node_embeddings[node_key] = full_embedding
    return node_embeddings

def create_grouped_edges(edges_data):
    edges_by_node = {}
    edge_types = []
    for _, row in edges_data.iterrows():
        start = int(row['start'])
        end = int(row['end'])
        edge_type = row['type']
        if start not in edges_by_node:
            edges_by_node[start] = []
        if end not in edges_by_node:
            edges_by_node[end] = []
        edges_by_node[start].append(end)
        if edge_type in edge_type_map:
            edge_types.append(edge_type_map[edge_type])
        else:
            edge_types.append(edge_type_map[''])
    return edges_by_node, edge_types

def create_edge_index(edges_by_node, edge_types):
    start_node_indices = []
    end_node_indices = []
    for start_node, neighbors in edges_by_node.items():
        for end_node in neighbors:
            start_node_indices.append(start_node)
            end_node_indices.append(end_node)
    edge_index = torch.tensor([start_node_indices, end_node_indices], dtype=torch.long)
    edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
    assert edge_index.size(0) == 2, "Edge index should have 2 rows."
    return edge_index, edge_type_tensor

# def adjust_edge_index(edge_index, num_nodes):
#     unique_nodes = torch.unique(edge_index)
#     node_map = {node.item(): idx for idx, node in enumerate(unique_nodes)}
#     start_node_indices_adjusted = [node_map[node.item()] for node in edge_index[0]]
#     end_node_indices_adjusted = [node_map[node.item()] for node in edge_index[1]]
#     edge_index_adjusted = torch.tensor([start_node_indices_adjusted, end_node_indices_adjusted], dtype=torch.long)
#     num_nodes_adjusted = len(unique_nodes)
#     return edge_index_adjusted, num_nodes_adjusted
def prepare_data_for_gcn(node_embeddings, edge_index, edge_type_tensor, labels):
    num_nodes = len(node_embeddings)
    x = np.array(list(node_embeddings.values()))
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type_tensor, num_nodes=num_nodes, y=y)
    return data
all_data = []
all_labels = []
import gensim.downloader as api

wv = api.load("word2vec-google-news-300")
for subdir in glob.glob(os.path.join('output2_pruning_ffmq', '*')):
    subdir_name = os.path.basename(subdir)
    edges_data, nodes_data = load_data_from_directory(subdir)
    if edges_data is None or nodes_data is None:
        continue
    if edges_data.empty or nodes_data.empty:
        continue
    node_embeddings = get_node_embeddings(nodes_data, wv)
    edges_by_node, edge_types = create_grouped_edges(edges_data)
    edge_index, edge_type_tensor = create_edge_index(edges_by_node, edge_types)   
    label = int(subdir_name.split('_')[-1][0])
    data = prepare_data_for_gcn(node_embeddings, edge_index, edge_type_tensor, [label])
    all_labels.append(label)
    all_data.append(data)

from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import DataLoader

def stratified_split_3way(data_list, y_labels, train_size=0.8, valid_size=0.1, test_size=0.1):
    train_indices, rest_indices = train_test_split(
        range(len(data_list)), 
        train_size=train_size, 
        stratify=y_labels
    )
    
    valid_size_adjusted = valid_size / (valid_size + test_size)  
    valid_indices, test_indices = train_test_split(
        rest_indices,
        test_size=1 - valid_size_adjusted,
        stratify=[y_labels[i] for i in rest_indices]  
    )
    
    train_data = [data_list[i] for i in train_indices]
    valid_data = [data_list[i] for i in valid_indices]
    test_data = [data_list[i] for i in test_indices]
    
    return train_data, valid_data, test_data

train_data, valid_data, test_data = stratified_split_3way(all_data, all_labels)

def create_dataloader(data_list, batch_size=1, shuffle=True):
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

train_loader = create_dataloader(train_data, batch_size=1, shuffle=True)
valid_loader = create_dataloader(valid_data, batch_size=1, shuffle=False)
test_loader = create_dataloader(test_data, batch_size=1, shuffle=False)

print(f'Number of training batches: {len(train_loader)}')
print(f'Number of validation batches: {len(valid_loader)}')
print(f'Number of testing batches: {len(test_loader)}')
import torch
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
from dgl.nn import GatedGraphConv
import dgl
from tqdm import tqdm  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GGNN(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, read_out='mean', num_steps=8):
        super(GGNN, self).__init__()
        self.read_out = read_out
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.ggcn = GatedGraphConv(in_feats=output_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=2)

    def forward(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr.long()
        g = dgl.graph((edge_index[0], edge_index[1]))
        num_nodes = g.num_nodes()
        node_features = data.x
        num_features = node_features.shape[0]
        if num_features < num_nodes:
            padding = torch.zeros((num_nodes - num_features, node_features.shape[1]), dtype=node_features.dtype)
            node_features = torch.cat([node_features, padding], dim=0)
        elif num_features > num_nodes:
            node_features = node_features[:num_nodes]
        node_features = self.dropout(self.linear(node_features))
        out = self.ggcn(g, node_features, edge_attr)
        out = self.dropout(out)  
        g.ndata['h'] = out        
        if self.read_out == 'sum':
            feats = dgl.sum_nodes(g, 'h')
        elif self.read_out == 'mean':
            feats = dgl.mean_nodes(g, 'h')
        result = self.classifier(feats)
        return result

num_classes = 2  
model = GGNN(input_dim=369, output_dim=369, max_edge_types=13)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()  

def train(num_epochs, train_loader, test_loader):
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}", unit='batch') as pbar:
            for data in train_loader:
                optimizer.zero_grad()
                out = model(data)  
                loss = criterion(out, data.y) 
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=epoch_loss / (pbar.n + 1)) 
                pbar.update()  
        
        print(f"Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f}")
        
        test(valid_loader)

def test(test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            predicted_labels = torch.argmax(out, dim=1)  
            all_preds.extend(predicted_labels.cpu().numpy()) 
            all_labels.extend(data.y.cpu().numpy())  
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    

num_epochs = 100
train(num_epochs, train_loader, test_loader)
test(test_loader)
