import csv
from operator import itemgetter
import networkx as nx
from networkx.algorithms import community
import numpy as np
import matplotlib.pyplot as plt
import os

# Change the current working directory to parent dir
os.chdir('/')

# Read in the nodelist file
with open('node0.csv', 'r') as nodecsv:
    nodereader = csv.reader(nodecsv)
    nodes = [n for n in nodereader][1:]
#    nd=[n for n in nodereader]

print("Nodes: ",nodes)
#print("Nd: ",nd)

# Get a list of just the node names (the first item in each row)
node_names = [n[1] for n in nodes]
node_id = [int(n[0]) for n in nodes]
print(node_names)
print(node_id)

# Read in the edgelist file
with open('edge0.csv', 'r') as edgecsv:
    next(edgecsv)
    edgereader = csv.reader(edgecsv)
    #ed=[e for e in edgereader]
    #print("ed[0]",ed[0])
    edges=[tuple((node_id.index(int(e[0])),node_id.index(int(e[1])))) for e in edgereader]

# Read in the edgelist file
with open('edge0.csv', 'r') as edgecsv:
    next(edgecsv)
    edgereader = csv.reader(edgecsv)
    ed=[e for e in edgereader]


edge_field1='dist'
edge_field2='ecmp_fwd_uniform'
edge_field3='ecmp_fwd_degree'
edge_field4='ecmp_bwd_uniform'
edge_field5='ecmp_bwd_degree'

#print("Edges: ",ed)
print("Edges: ",edges)

# Print the number of nodes and edges in our two lists
#print("nodes len: ",len(nodes))
#print("node_names len: ",len(node_names))
#print("node_id len: ",len(node_id))
#print("node_edges len: ",len(edges))

H=nx.path_graph(len(nodes))
G = nx.Graph() # Initialize a Graph object
G.add_nodes_from(H)
G.add_edges_from(edges) # Add edges to the Graph
print("G: ",G) # Print information about the Graph

id_dict = {}
name_dict = {}
posx_dict = {}
posy_dict = {}
birth_dict = {}
death_dict = {}
cost_dict = {}
sla_dict = {}

for index,node in enumerate(nodes): # Loop through the list, one row at a time
    id_dict[index] = node[0]
    name_dict[index] = node[1]
#    hist_sig_dict[index] = node[1]
    posx_dict[index] = node[2]
#    gender_dict[index] = node[2]
    posy_dict[index] = node[3]
    cost_dict[index] = int(node[4])
    sla_dict[index] = node[4]
#    birth_dict[index] = int(node[3])
#    death_dict[index] = int(node[4])
#    death_dict[index] = int(node[4])


nx.set_node_attributes(G, posx_dict, 'pos_x')
nx.set_node_attributes(G, posy_dict, 'pos_y')
#nx.set_node_attributes(G, birth_dict, 'birth_year')
#nx.set_node_attributes(G, death_dict, 'death_year')
nx.set_node_attributes(G, name_dict, 'name')
nx.set_node_attributes(G, cost_dict, 'cost')
nx.set_node_attributes(G, sla_dict, 'sla')

print("ed[2]: ",ed[2])

# === 3. AGGIUNTA DEGLI ATTRIBUTI AGLI ARCHI ===
for index,edge in enumerate(G.edges()):
#    print("ed: ",ed[index])
    G[edge[0]][edge[1]]['dist'] =ed[index][2]
    G[edge[0]][edge[1]]['ecmp_fwd_uniform'] =ed[index][3]
#    G[u][v][edge_field3] =ed[4][edge_field3]
    G[edge[0]][edge[1]]['ecmp_fwd_degree'] =ed[index][4]
    G[edge[0]][edge[1]]['ecmp_bwd_uniform'] =ed[index][5]
    G[edge[0]][edge[1]]['ecmp_bwd_degree'] =ed[index][6]

#for n in G.nodes(): # Loop through every node, in our data "n" will be the id of the person
#    print(n, G.nodes[n]['birth_year']) # Access every node by its name, and then by the attribute "birth_year"

density = nx.density(G)
print("Network density:", density)

fell_whitehead_path = nx.shortest_path(G, source=node_id.index(0), target=node_id.index(15))

print("Shortest path between Fell and Whitehead:", fell_whitehead_path)
print("Length of that path:", len(fell_whitehead_path)-1)

# If your Graph has more than one component, this will return False:
print(nx.is_connected(G))

# Next, use nx.connected_components to get the list of components,
# then use the max() command to find the largest one:
components = nx.connected_components(G)
largest_component = max(components, key=len)

# Create a "subgraph" of just the largest component
# Then calculate the diameter of the subgraph, just like you did with density.
#

subgraph = G.subgraph(largest_component)
diameter = nx.diameter(subgraph)
print("Network diameter of largest component:", diameter)

# Visualizzazione del grafo
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)  # Puoi usare altri layout come nx.circular_layout, nx.random_layout, etc.
nx.draw(G, pos, with_labels=True, node_size=20, edge_color='gray', alpha=0.6)
plt.show()

!pip install torch_geometric

import pandas as pd
from scipy.special import expit  # funzione logistica

# === 4. CALCOLO DEL TARGET (is_strategic) ===
def logistic(x): return expit(x)

edges_data = []
for u, v, attrs in G.edges(data=True):
#    print("(u,v):",u," ",v)
#    print(G.edges[u,v])
    ecmp_fw =   float(G.edges[u,v]['ecmp_fwd_uniform'])
    ecmp_bw =   float(G.edges[u,v]['ecmp_bwd_uniform'])
    cost_src =  G.nodes[u]['cost']
    cost_dst =  G.nodes[v]['cost']
    score = 0.02*ecmp_fw+0.001*ecmp_bw+0.005*cost_src+0.001*cost_dst
    #print("score ",score)
    prob = logistic(score)
    utilization = np.random.binomial(1, prob)
    edges_data.append({
        'source': u,
        'target': v,
        **attrs,
        'ecmp_bw' : ecmp_bw,
        'ecmp_fw' : ecmp_fw,
        'utilization': utilization
    })

df_edges = pd.DataFrame(edges_data)

print("Esempio di link strategici simulati:")
print(df_edges.sample(5))

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# === 1. Preprocessing dei nodi ===
node_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
categorical = pd.get_dummies(node_df['sla'])
numerical = node_df[['cost']]
#X_nodes = torch.tensor(np.hstack([numerical.values, categorical.values]), dtype=torch.int16) #dava errore
X_nodes = torch.tensor(np.hstack([numerical.values, categorical.values]), dtype=torch.float)
print(X_nodes)
#print(len(X_nodes))

# === 2. Preprocessing degli archi e target ===
edge_index = torch.tensor(df_edges[['source', 'target']].values.T, dtype=torch.long)
edge_label = torch.tensor(df_edges['utilization'].values, dtype=torch.float)

print("Edge prop: ",df_edges['utilization'].values)

print(edge_index)
print(edge_label)

print(len(edge_index))
print(len(edge_label))

# Split archi in train/val/test
edges_idx = np.arange(edge_index.shape[1])
train_idx, test_idx = train_test_split(edges_idx, test_size=0.3, stratify=edge_label, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, stratify=edge_label[test_idx], random_state=42)

train_idx = torch.tensor(train_idx, dtype=torch.long)
val_idx = torch.tensor(val_idx, dtype=torch.long)
test_idx = torch.tensor(test_idx, dtype=torch.long)

# === 3. Costruzione oggetto Data ===
data = Data(x=X_nodes, edge_index=edge_index)

# === 4. Modello GCN + MLP per edge classification ===
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class EdgeClassifier(torch.nn.Module):
    def __init__(self, encoder, hidden_channels):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_pairs):
        z = self.encoder(x, edge_index)
        src, dst = edge_pairs
        edge_feat = torch.cat([z[src], z[dst]], dim=1)
        return self.classifier(edge_feat).squeeze()

pip install torchviz

from torchviz import make_dot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EdgeClassifier(GCNEncoder(in_channels=X_nodes.shape[1], hidden_channels=32), hidden_channels=32).to(device)

data1 = data.to(device)
input_data = train_idx.to(device)
output = model(data1.x, data1.edge_index, data1.edge_index[:, input_data])

# Visualize the computational graph
make_dot(output, params=dict(model.named_parameters()))

# === 5. Inizializzazione ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EdgeClassifier(GCNEncoder(in_channels=X_nodes.shape[1], hidden_channels=32), hidden_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()

data = data.to(device)
edge_label = edge_label.to(device)
train_idx = train_idx.to(device)
val_idx = val_idx.to(device)
test_idx = test_idx.to(device)

# === 6. Training loop ===
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index, data.edge_index[:, train_idx])
    loss = loss_fn(pred, edge_label[train_idx])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(data.x, data.edge_index, data.edge_index[:, val_idx])
        val_loss = loss_fn(val_pred, edge_label[val_idx])
        val_acc = ((val_pred > 0).float() == edge_label[val_idx]).float().mean()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")

# === 7. Test finale ===
model.eval()
with torch.no_grad():
    test_pred = model(data.x, data.edge_index, data.edge_index[:, test_idx])
    test_acc = ((test_pred > 0).float() == edge_label[test_idx]).float().mean()
print(f"Test Accuracy: {test_acc:.4f}")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    z = model.encoder(data.x, data.edge_index).cpu()

# Estrai embedding degli archi
edge_src = data.edge_index[0].cpu().numpy()
edge_dst = data.edge_index[1].cpu().numpy()
edge_emb = torch.cat([z[edge_src], z[edge_dst]], dim=1).numpy()
labels = edge_label.cpu().numpy()
print("edge_src: ",edge_src)
print("edge_dst: ",edge_dst)
# Riduzione dimensionale con t-SNE
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(edge_emb)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(emb_2d[labels == 0, 0], emb_2d[labels == 0, 1], c='lightgray', label='Non strategico', alpha=0.5)
plt.scatter(emb_2d[labels == 1, 0], emb_2d[labels == 1, 1], c='crimson', label='Strategico', alpha=0.6)
plt.legend()
plt.title("Proiezione t-SNE delle embedding dei link")
plt.xlabel("Dimensione 1")
plt.ylabel("Dimensione 2")
plt.grid(True)
plt.tight_layout()
plt.show()

#######Chatbot
def chatbot_query(user_input: str):
    #conversione del testo utente in embedding
    input_emb = torch.tensor(embedder.encode(user_input)).unsqueeze(0)

    #invio dell'embedding alla GNN per "ragionare" sulle connessioni
    #gnn_output = model(data.x, data.edge_index)
    gnn_output = model(data.x, data.edge_index, data.edge_index[:, val_idx])

    #ricerca del nodo più vicino semanticamente all'input utente
    similarities = torch.cosine_similarity(input_emb, gnn_output)
    best_node_idx = similarities.argmax().item()
    #best_node = idx_to_node[best_node_idx]
    print("best node id: ",best_node_idx)
    #print("best node: ",best_node)

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

promtAI='How can I help you? ---> '
user_input = input(promtAI)
chatbot_query(user_input)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt

# === 1. Wrapper compatibile con Explainer ===
class EdgeExplainerWrapper(torch.nn.Module):
    def __init__(self, model, edge_idx):
        super().__init__()
        self.model = model
        self.edge_idx = edge_idx  # singolo indice dell’arco

    def forward(self, x, edge_index):
        edge_pair = edge_index[:, self.edge_idx].view(2, 1)
        return self.model(x, edge_index, edge_pair)

# === 2. Prepara i dati ===
x_cpu = data.x.cpu()
edge_index_cpu = data.edge_index.cpu()

# === 3. Seleziona un link strategico da spiegare ===
strategic_edges = df_edges[df_edges['utilization'] == 1].reset_index(drop=True)
link_idx = strategic_edges.index[0]  # ad es. il primo arco strategico
source_id = strategic_edges.loc[link_idx, 'source']
target_id = strategic_edges.loc[link_idx, 'target']
print(f"Spiegazione per link strategico: {source_id} → {target_id} (indice {link_idx})")

# === 4. Costruisci il modello wrapper ===
wrapped_model = EdgeExplainerWrapper(model, edge_idx=link_idx)

# === 5. Definisci l’Explainer ===
explainer = Explainer(
    model=wrapped_model,
    algorithm=GNNExplainer(epochs=100),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='edge',
        return_type='raw'
    )
)

# === 6. Ottieni la spiegazione ===
explanation = explainer(x=x_cpu, edge_index=edge_index_cpu)

# === 7. Analizza maschere ===
edge_mask = explanation.get('edge_mask')
node_mask = explanation.get('node_mask')

# === 8. Visualizza top-k archi importanti ===
if edge_mask is not None:
    topk = torch.topk(edge_mask, k=5)
    print("Top 5 archi più influenti:")
    for idx, score in zip(topk.indices, topk.values):
        src, tgt = edge_index_cpu[:, idx]
        print(f"{src.item()} → {tgt.item()} — peso: {score.item():.4f}")

import matplotlib.pyplot as plt
import networkx as nx
import torch
import numpy as np

# === PARAMETRO: numero di archi top da visualizzare ===
k = 5  # puoi modificarlo (es. 20, 50, ecc.)

# === 1. Estrai top-k archi da edge_mask ===
topk = torch.topk(edge_mask, k=k)
top_edge_indices = topk.indices.cpu().numpy()
top_edge_weights = topk.values.cpu().numpy()

# === 2. Costruisci grafo filtrato con top-k archi ===
G_top = nx.DiGraph()

for idx, weight in zip(top_edge_indices, top_edge_weights):
    u = edge_index_cpu[0, idx].item()
    v = edge_index_cpu[1, idx].item()
    G_top.add_edge(u, v, weight=weight)

# === 3. Aggiungi peso ai nodi coinvolti ===
node_mask_np = node_mask.detach().cpu().numpy().flatten()  # <-- fix qui

for node in G_top.nodes():
    raw_value = node_mask_np[node]
    safe_value = float(np.nan_to_num(raw_value, nan=0.0, posinf=0.0, neginf=0.0))
    G_top.nodes[node]['weight'] = safe_value

# === 4. Layout e attributi grafici ===
pos = nx.spring_layout(G_top, seed=42)
nodelist = list(G_top.nodes())
node_sizes = [G_top.nodes[n].get('weight', 0.0) * 1000 for n in nodelist]

edgelist = list(G_top.edges())
edge_widths = [G_top[u][v]['weight'] * 4 for u, v in edgelist]
edge_colors = ['red' if (u == source_id and v == target_id) else 'gray' for u, v in edgelist]

# === 5. Visualizzazione ===
plt.figure(figsize=(10, 7))
nx.draw_networkx_nodes(G_top, pos, nodelist=nodelist, node_size=node_sizes, alpha=0.9)
nx.draw_networkx_edges(G_top, pos, edgelist=edgelist, width=edge_widths, edge_color=edge_colors, alpha=0.7)
nx.draw_networkx_labels(G_top, pos, font_size=8)
plt.title(f"GNNExplainer — Top-{k} archi influenti")
plt.axis("off")
plt.tight_layout()
plt.show()
