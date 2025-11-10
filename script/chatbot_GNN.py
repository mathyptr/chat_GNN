import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import networkx as nx
#from fastembed import TextEmbedding

#retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#from huggingface_hub import snapshot_download
#snapshot_download(repo_id="sentence-transformers/multi-qa-MiniLM-L6-cos-v1", local_dir="./model")

#  CONFIGURAZIONE
#ollama_model="openthinker" #smollm2:135m
ollama_model="smollm2:135m"
openAI_API_key = "ollama"   # required, but unused
client = OpenAI(base_url = 'http://ollama:11434/v1',
    api_key=openAI_API_key,)
#embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

promtAI='How can I help you? ---> '

#  CREAZIONE DEL GRAFO
G = nx.Graph()
G.add_edges_from([
    ("Router_Informatica", "Router_Fisica"),
    ("Router_Informatica", "Router_Ingegneria"),
    ("Router_Matematica", "Router_Informatica"),
    ("Router_Matematica", "Router_Fisica"),
    ("Router_Medicina", "Router_Matematica")
])

# Creiamo dizionari per mappare i nomi con gli indici
node_to_idx = {name: i for i, name in enumerate(G.nodes())}
idx_to_node = {i: name for name, i in node_to_idx.items()}

# Creazione il grafo e conversione degli archi in indici numerici
edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # modifico gli archi bidirezionali

# creazione delle features di base 
x = torch.tensor([embedder.encode(name) for name in G.nodes()], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

#  MODELLO GNN
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GNN(in_channels=x.shape[1], hidden_channels=64, out_channels=x.shape[1])

#   FUNZIONE CHAT
def chatbot_query(user_input: str):
    print(f"\n I'm thinking about it...")

    #conversione del testo utente in embedding
    input_emb = torch.tensor(embedder.encode(user_input)).unsqueeze(0)

    #invio dell'embedding alla GNN per "ragionare" sulle connessioni
    gnn_output = model(data.x, data.edge_index)

    #ricerca del nodo più vicino semanticamente all'input utente
    similarities = torch.cosine_similarity(input_emb, gnn_output)
    best_node_idx = similarities.argmax().item()
    best_node = idx_to_node[best_node_idx]

    #Preparazione del prompt per GPT per generare una risposta coerente
    neighbors = list(G.neighbors(best_node))
    prompt = f"""
I have a graph with these edges: {list(G.edges())}.
the user input is: "{user_input}".
the best node is: "{best_node}".
with neighbors : {neighbors}.

Give me a short response to the user input to explain the link from the request node to the other in the graph.
"""

    # Invio a GPT per generare la risposta
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()
    print(f"\n Chatbot: {answer}")

#  LOOP DI CHAT
if __name__ == "__main__":
    print(" Chatbot GNN on (type 'exit' to quit)\n")
    while True:
        user_input = input(promtAI)
        if user_input.lower() in {"exit", "quit"}:
            break
        chatbot_query(user_input)
