#!/usr/bin/env python3
"""Build script that generates gnn_practical_guide.ipynb (nbformat v4)."""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from build_utils import md, code, write_notebook


def _fix_source(lines):
    """Ensure every line except the last carries a trailing newline (nbformat spec)."""
    if not lines:
        return lines
    return [l if i == len(lines) - 1 else (l if l.endswith("\n") else l + "\n")
            for i, l in enumerate(lines)]


cells = []

# ── Cell 1: Title Banner ──────────────────────────────────────────────────
cells.append(md([
    "# Graph Neural Networks: Practical Guide",
    "",
    "![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)",
    "![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)",
    "![PyG](https://img.shields.io/badge/PyG-2.4+-3C2179?logo=pyg&logoColor=white)",
    "![License](https://img.shields.io/badge/License-MIT-green)",
    "",
    "A hands-on, competition-ready walkthrough of Graph Neural Networks (GNNs) ",
    "covering theory, from-scratch implementations, and PyTorch Geometric workflows.",
]))

# ── Cell 2: TL;DR ─────────────────────────────────────────────────────────
cells.append(md([
    "## TL;DR",
    "",
    "Graphs are everywhere: molecules, social networks, knowledge bases, program ASTs. ",
    "This notebook walks through **GCN**, **GAT**, and **GraphSAGE** with code you can ",
    "copy straight into a Kaggle kernel. We build layers from scratch, then show how ",
    "**PyTorch Geometric** makes it production-ready.",
]))

# ── Cell 3: Table of Contents ─────────────────────────────────────────────
cells.append(md([
    "## Table of Contents",
    "",
    "1. [Setup & Imports](#1-setup--imports)",
    "2. [Why Graphs?](#2-why-graphs)",
    "3. [Graph Basics with NetworkX](#3-graph-basics-with-networkx)",
    "4. [Message Passing Framework](#4-message-passing-framework)",
    "5. [Graph Convolutional Network (GCN)](#5-graph-convolutional-network-gcn)",
    "6. [Graph Attention Network (GAT)](#6-graph-attention-network-gat)",
    "7. [GraphSAGE](#7-graphsage)",
    "8. [Graph-Level Predictions](#8-graph-level-predictions)",
    "9. [PyTorch Geometric Workflow](#9-pytorch-geometric-workflow)",
    "10. [Applications & Competition Tips](#10-applications--competition-tips)",
    "11. [Further Reading](#11-further-reading)",
]))

cells.append(md([
    "## Objective & Evaluation Strategy",
    "",
    "**Objective:** build a practical graph-learning workflow that translates from concept demos to Kaggle-ready node and graph prediction tasks.",
    "",
    "**Evaluation:** use validation accuracy, task-specific metrics, and downstream error analysis before reading too much into leaderboard movement.",
    "",
    "**Hypothesis:** architectures that respect neighborhood structure should outperform flat tabular baselines because relational context carries the strongest signal.",
]))

# ── Cell 4: Key Takeaway box ──────────────────────────────────────────────
cells.append(md([
    "> **Key Takeaway** -- GNNs learn representations by *message passing*: each ",
    "> node aggregates features from its neighbors, then updates its own state. ",
    "> Different architectures (GCN, GAT, GraphSAGE) differ mainly in *how* they ",
    "> aggregate.",
]))

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 -- Setup & Imports
# ══════════════════════════════════════════════════════════════════════════

cells.append(md([
    "---",
    "## 1. Setup & Imports <a id='1-setup--imports'></a>",
]))

cells.append(code([
    "import warnings",
    "warnings.filterwarnings('ignore')",
    "",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import networkx as nx",
    "",
    "import torch",
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "",
    "from sklearn.metrics import accuracy_score, classification_report",
    "",
    "# PyTorch Geometric (optional -- graceful fallback)",
    "try:",
    "    import torch_geometric",
    "    from torch_geometric.data import Data, Batch",
    "    from torch_geometric.loader import DataLoader as PyGDataLoader",
    "    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool",
    "    HAS_PYG = True",
    "    print(f'torch_geometric {torch_geometric.__version__} loaded')",
    "except ImportError:",
    "    HAS_PYG = False",
    "    print('PyTorch Geometric not installed -- from-scratch cells still work')",
    "",
    "print(f'PyTorch {torch.__version__}')",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "print(f'Device: {DEVICE}')",
]))

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 -- Why Graphs?
# ══════════════════════════════════════════════════════════════════════════

cells.append(md([
    "---",
    "## 2. Why Graphs? <a id='2-why-graphs'></a>",
    "",
    "Many real-world datasets are naturally **graph-structured**:",
    "",
    "| Domain | Nodes | Edges |",
    "|--------|-------|-------|",
    "| Social networks | Users | Friendships / follows |",
    "| Molecules | Atoms | Chemical bonds |",
    "| Citation graphs | Papers | Citations |",
    "| Protein interactions | Proteins | Physical interactions |",
    "| Knowledge graphs | Entities | Relations |",
    "",
    "Traditional ML flattens this structure into fixed-size vectors, losing ",
    "relational information. GNNs operate **directly on the graph**.",
]))

cells.append(md([
    "### Adjacency Matrix Representation",
    "",
    "A graph with *N* nodes can be represented by an *N x N* adjacency matrix **A** ",
    "where A[i][j] = 1 if there is an edge from node *i* to node *j*.",
]))

cells.append(code([
    "# Build a small example graph and its adjacency matrix",
    "G_example = nx.karate_club_graph()",
    "A = nx.adjacency_matrix(G_example).todense()",
    "",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
    "",
    "axes[0].set_title('Zachary Karate Club Graph')",
    "nx.draw_kamada_kawai(G_example, ax=axes[0], node_size=120, node_color='steelblue', ",
    "                     edge_color='#cccccc', width=0.8)",
    "",
    "axes[1].set_title('Adjacency Matrix')",
    "axes[1].imshow(A, cmap='Blues', interpolation='nearest')",
    "axes[1].set_xlabel('Node index')",
    "axes[1].set_ylabel('Node index')",
    "",
    "plt.tight_layout()",
    "plt.show()",
    "print(f'Nodes: {G_example.number_of_nodes()}, Edges: {G_example.number_of_edges()}')",
]))

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 -- Graph Basics with NetworkX
# ══════════════════════════════════════════════════════════════════════════

cells.append(md([
    "---",
    "## 3. Graph Basics with NetworkX <a id='3-graph-basics-with-networkx'></a>",
    "",
    "NetworkX is the go-to library for graph creation, manipulation, and analysis in Python.",
]))

cells.append(code([
    "# Create a graph from scratch",
    "G = nx.Graph()",
    "G.add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,4),(3,4),(3,5),(4,5),(4,6),(5,6)])",
    "",
    "# Assign random features to nodes",
    "for node in G.nodes():",
    "    G.nodes[node]['feature'] = np.random.randn(4)",
    "",
    "print('Nodes:', list(G.nodes()))",
    "print('Edges:', list(G.edges()))",
    "print('Degree of node 4:', G.degree(4))",
]))

cells.append(code([
    "# Visualize with centrality-based coloring",
    "centrality = nx.betweenness_centrality(G)",
    "node_colors = [centrality[n] for n in G.nodes()]",
    "",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))",
    "",
    "# Betweenness centrality",
    "pos = nx.spring_layout(G, seed=42)",
    "nx.draw(G, pos, ax=axes[0], node_color=node_colors, cmap='YlOrRd',",
    "        with_labels=True, node_size=500, font_weight='bold')",
    "axes[0].set_title('Betweenness Centrality')",
    "",
    "# Degree centrality",
    "deg_cent = nx.degree_centrality(G)",
    "deg_colors = [deg_cent[n] for n in G.nodes()]",
    "nx.draw(G, pos, ax=axes[1], node_color=deg_colors, cmap='YlOrRd',",
    "        with_labels=True, node_size=500, font_weight='bold')",
    "axes[1].set_title('Degree Centrality')",
    "",
    "# Closeness centrality",
    "close_cent = nx.closeness_centrality(G)",
    "close_colors = [close_cent[n] for n in G.nodes()]",
    "nx.draw(G, pos, ax=axes[2], node_color=close_colors, cmap='YlOrRd',",
    "        with_labels=True, node_size=500, font_weight='bold')",
    "axes[2].set_title('Closeness Centrality')",
    "",
    "plt.tight_layout()",
    "plt.show()",
]))

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 -- Message Passing Framework
# ══════════════════════════════════════════════════════════════════════════

cells.append(md([
    "---",
    "## 4. Message Passing Framework <a id='4-message-passing-framework'></a>",
    "",
    "Almost every modern GNN follows the **message-passing** paradigm:",
    "",
    "```",
    "for each node v:",
    "    messages = [MESSAGE(h_u) for u in neighbors(v)]",
    "    aggregated = AGGREGATE(messages)        # e.g. sum, mean, max",
    "    h_v_new    = UPDATE(h_v, aggregated)    # e.g. MLP, linear + activation",
    "```",
    "",
    "Different GNN variants differ in their choice of MESSAGE, AGGREGATE, and UPDATE.",
]))

cells.append(md([
    "> **Key Takeaway** -- Think of message passing as *each node sending a letter ",
    "> to its neighbors every round*. After K rounds a node knows about its ",
    "> K-hop neighborhood.",
]))

cells.append(code([
    "class BasicGNNLayer(nn.Module):",
    "    \"\"\"Minimal message-passing GNN layer from scratch.\"\"\"",
    "",
    "    def __init__(self, in_dim, out_dim):",
    "        super().__init__()",
    "        self.linear = nn.Linear(in_dim, out_dim)",
    "",
    "    def forward(self, x, edge_index):",
    "        # x: (N, in_dim)  edge_index: (2, E)",
    "        src, dst = edge_index",
    "        N = x.size(0)",
    "",
    "        # 1. MESSAGE -- transform neighbor features",
    "        messages = self.linear(x)  # (N, out_dim)",
    "",
    "        # 2. AGGREGATE -- sum messages from neighbors",
    "        aggr = torch.zeros(N, messages.size(1), device=x.device)",
    "        aggr.index_add_(0, dst, messages[src])",
    "",
    "        # 3. UPDATE -- add self-loop + activation",
    "        out = aggr + self.linear(x)",
    "        return F.relu(out)",
    "",
    "",
    "# Quick test",
    "torch.manual_seed(42)",
    "layer = BasicGNNLayer(4, 8)",
    "x_test = torch.randn(7, 4)",
    "edges = torch.tensor([[0,0,1,1,2,3,3,4,4,5],",
    "                       [1,2,2,3,4,4,5,5,6,6]], dtype=torch.long)",
    "out = layer(x_test, edges)",
    "print('Input shape: ', x_test.shape)",
    "print('Output shape:', out.shape)",
]))

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 -- GCN
# ══════════════════════════════════════════════════════════════════════════

cells.append(md([
    "---",
    "## 5. Graph Convolutional Network (GCN) <a id='5-graph-convolutional-network-gcn'></a>",
    "",
    "### Spectral vs. Spatial Convolutions",
    "",
    "| Approach | Idea | Pro | Con |",
    "|----------|------|-----|-----|",
    "| **Spectral** | Convolution via graph Fourier transform | Mathematically grounded | Needs eigen-decomposition; not inductive |",
    "| **Spatial** | Aggregate neighbors directly | Fast; inductive | Less theoretically motivated |",
    "",
    "Kipf & Welling (2017) bridged the gap with a first-order Chebyshev approximation ",
    "that reduces to a spatial neighborhood aggregation with symmetric normalization.",
]))

cells.append(code([
    "class GCNLayer(nn.Module):",
    "    \"\"\"Graph Convolutional Layer (Kipf & Welling, 2017).\"\"\"",
    "",
    "    def __init__(self, in_dim, out_dim):",
    "        super().__init__()",
    "        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))",
    "        nn.init.xavier_uniform_(self.weight)",
    "",
    "    def forward(self, x, edge_index):",
    "        src, dst = edge_index",
    "        N = x.size(0)",
    "",
    "        # Compute degree for normalization  D^{-1/2}",
    "        deg = torch.zeros(N, device=x.device)",
    "        ones = torch.ones(src.size(0), device=x.device)",
    "        deg.index_add_(0, dst, ones)",
    "        deg_inv_sqrt = (deg + 1).pow(-0.5)  # +1 for self-loop",
    "",
    "        # Symmetric normalization coefficients",
    "        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]",
    "",
    "        # Transform features",
    "        h = x @ self.weight  # (N, out_dim)",
    "",
    "        # Aggregate with normalization",
    "        aggr = torch.zeros(N, h.size(1), device=x.device)",
    "        aggr.index_add_(0, dst, h[src] * norm.unsqueeze(1))",
    "",
    "        # Add self-loop contribution",
    "        self_norm = deg_inv_sqrt * deg_inv_sqrt",
    "        aggr = aggr + h * self_norm.unsqueeze(1)",
    "",
    "        return aggr",
    "",
    "",
    "print('GCNLayer defined.')",
]))

cells.append(md([
    "### Training GCN on Synthetic Cora-like Data",
    "",
    "We generate a synthetic citation-like graph for node classification. ",
    "Each node has a feature vector and belongs to one of several classes.",
]))

cells.append(code([
    "# ── Generate synthetic Cora-like data ──",
    "np.random.seed(0)",
    "torch.manual_seed(0)",
    "",
    "NUM_NODES = 200",
    "NUM_FEATURES = 32",
    "NUM_CLASSES = 5",
    "NUM_EDGES = 1200",
    "",
    "# Random features and labels",
    "X_syn = torch.randn(NUM_NODES, NUM_FEATURES)",
    "Y_syn = torch.randint(0, NUM_CLASSES, (NUM_NODES,))",
    "",
    "# Random edges (undirected)",
    "src = torch.randint(0, NUM_NODES, (NUM_EDGES,))",
    "dst = torch.randint(0, NUM_NODES, (NUM_EDGES,))",
    "edge_index_syn = torch.stack([",
    "    torch.cat([src, dst]),",
    "    torch.cat([dst, src])",
    "])",
    "",
    "# Train / val / test masks",
    "perm = torch.randperm(NUM_NODES)",
    "train_mask = torch.zeros(NUM_NODES, dtype=torch.bool)",
    "val_mask   = torch.zeros(NUM_NODES, dtype=torch.bool)",
    "test_mask  = torch.zeros(NUM_NODES, dtype=torch.bool)",
    "train_mask[perm[:120]] = True",
    "val_mask[perm[120:160]] = True",
    "test_mask[perm[160:]] = True",
    "",
    "print(f'Nodes: {NUM_NODES}, Edges: {edge_index_syn.size(1)}, Classes: {NUM_CLASSES}')",
    "print(f'Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}')",
]))

cells.append(code([
    "class GCNModel(nn.Module):",
    "    def __init__(self, in_dim, hidden_dim, out_dim):",
    "        super().__init__()",
    "        self.gcn1 = GCNLayer(in_dim, hidden_dim)",
    "        self.gcn2 = GCNLayer(hidden_dim, out_dim)",
    "",
    "    def forward(self, x, edge_index):",
    "        x = F.relu(self.gcn1(x, edge_index))",
    "        x = F.dropout(x, p=0.5, training=self.training)",
    "        x = self.gcn2(x, edge_index)",
    "        return F.log_softmax(x, dim=1)",
    "",
    "",
    "model_gcn = GCNModel(NUM_FEATURES, 64, NUM_CLASSES)",
    "optimizer = torch.optim.Adam(model_gcn.parameters(), lr=0.01, weight_decay=5e-4)",
    "",
    "# Training loop",
    "losses = []",
    "for epoch in range(200):",
    "    model_gcn.train()",
    "    optimizer.zero_grad()",
    "    out = model_gcn(X_syn, edge_index_syn)",
    "    loss = F.nll_loss(out[train_mask], Y_syn[train_mask])",
    "    loss.backward()",
    "    optimizer.step()",
    "    losses.append(loss.item())",
    "",
    "# Evaluate",
    "model_gcn.eval()",
    "with torch.no_grad():",
    "    pred = model_gcn(X_syn, edge_index_syn).argmax(dim=1)",
    "    test_acc = accuracy_score(Y_syn[test_mask].numpy(), pred[test_mask].numpy())",
    "",
    "plt.figure(figsize=(8, 3))",
    "plt.plot(losses)",
    "plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'GCN Training  --  Test Acc: {test_acc:.2%}')",
    "plt.tight_layout(); plt.show()",
]))

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 -- GAT
# ══════════════════════════════════════════════════════════════════════════

cells.append(md([
    "---",
    "## 6. Graph Attention Network (GAT) <a id='6-graph-attention-network-gat'></a>",
    "",
    "GCN treats all neighbors equally. **GAT** (Velickovic et al., 2018) learns ",
    "**attention coefficients** so that more informative neighbors contribute more.",
    "",
    "The attention coefficient between nodes *i* and *j* is:",
    "",
    "```",
    "e_ij = LeakyReLU( a^T [Wh_i || Wh_j] )",
    "alpha_ij = softmax_j(e_ij)",
    "```",
]))

cells.append(code([
    "class GATLayer(nn.Module):",
    "    \"\"\"Single-head Graph Attention Layer.\"\"\"",
    "",
    "    def __init__(self, in_dim, out_dim, negative_slope=0.2):",
    "        super().__init__()",
    "        self.W = nn.Linear(in_dim, out_dim, bias=False)",
    "        self.a_src = nn.Parameter(torch.empty(out_dim, 1))",
    "        self.a_dst = nn.Parameter(torch.empty(out_dim, 1))",
    "        nn.init.xavier_uniform_(self.a_src)",
    "        nn.init.xavier_uniform_(self.a_dst)",
    "        self.leaky_relu = nn.LeakyReLU(negative_slope)",
    "",
    "    def forward(self, x, edge_index):",
    "        src, dst = edge_index",
    "        N = x.size(0)",
    "        h = self.W(x)  # (N, out_dim)",
    "",
    "        # Attention logits",
    "        e_src = (h @ self.a_src).squeeze()  # (N,)",
    "        e_dst = (h @ self.a_dst).squeeze()  # (N,)",
    "        e = self.leaky_relu(e_src[src] + e_dst[dst])  # (E,)",
    "",
    "        # Softmax per destination node",
    "        e_max = torch.zeros(N, device=x.device)",
    "        e_max.index_reduce_(0, dst, e, 'amax', include_self=False)",
    "        e_exp = torch.exp(e - e_max[dst])",
    "        e_sum = torch.zeros(N, device=x.device)",
    "        e_sum.index_add_(0, dst, e_exp)",
    "        alpha = e_exp / (e_sum[dst] + 1e-9)",
    "",
    "        # Aggregate",
    "        aggr = torch.zeros(N, h.size(1), device=x.device)",
    "        aggr.index_add_(0, dst, h[src] * alpha.unsqueeze(1))",
    "        return F.elu(aggr)",
    "",
    "",
    "print('GATLayer defined.')",
]))

cells.append(code([
    "class MultiHeadGAT(nn.Module):",
    "    \"\"\"Multi-head GAT -- concatenates K attention heads.\"\"\"",
    "",
    "    def __init__(self, in_dim, out_dim, num_heads=4):",
    "        super().__init__()",
    "        self.heads = nn.ModuleList([",
    "            GATLayer(in_dim, out_dim) for _ in range(num_heads)",
    "        ])",
    "",
    "    def forward(self, x, edge_index):",
    "        head_outs = [head(x, edge_index) for head in self.heads]",
    "        return torch.cat(head_outs, dim=-1)  # (N, out_dim * num_heads)",
    "",
    "",
    "# Quick test",
    "mh_gat = MultiHeadGAT(NUM_FEATURES, 16, num_heads=4)",
    "out_gat = mh_gat(X_syn, edge_index_syn)",
    "print(f'Multi-head GAT output: {out_gat.shape}  (expect [200, 64])')",
]))

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 -- GraphSAGE
# ══════════════════════════════════════════════════════════════════════════

cells.append(md([
    "---",
    "## 7. GraphSAGE <a id='7-graphsage'></a>",
    "",
    "**GraphSAGE** (Hamilton et al., 2017) introduced two key ideas:",
    "",
    "1. **Sampling**: Instead of using all neighbors, sample a fixed-size subset ",
    "   to keep computation bounded.",
    "2. **Inductive learning**: Learn an *aggregation function* rather than ",
    "   per-node embeddings, enabling generalization to unseen nodes / graphs.",
    "",
    "| Property | Transductive (GCN) | Inductive (GraphSAGE) |",
    "|----------|-------------------|-----------------------|",
    "| Training | Full graph needed | Mini-batch on subgraphs |",
    "| New nodes | Retrain required | Forward pass suffices |",
    "| Scalability | O(N) memory | O(batch) memory |",
]))

cells.append(code([
    "class GraphSAGELayer(nn.Module):",
    "    \"\"\"GraphSAGE layer with mean aggregation.\"\"\"",
    "",
    "    def __init__(self, in_dim, out_dim):",
    "        super().__init__()",
    "        self.linear_neigh = nn.Linear(in_dim, out_dim, bias=False)",
    "        self.linear_self  = nn.Linear(in_dim, out_dim, bias=False)",
    "",
    "    def forward(self, x, edge_index):",
    "        src, dst = edge_index",
    "        N = x.size(0)",
    "",
    "        # Mean aggregation of neighbor features",
    "        aggr = torch.zeros(N, x.size(1), device=x.device)",
    "        aggr.index_add_(0, dst, x[src])",
    "        deg = torch.zeros(N, device=x.device)",
    "        deg.index_add_(0, dst, torch.ones(src.size(0), device=x.device))",
    "        deg = deg.clamp(min=1)",
    "        aggr = aggr / deg.unsqueeze(1)",
    "",
    "        # Combine self + neighbor",
    "        out = self.linear_self(x) + self.linear_neigh(aggr)",
    "        out = F.normalize(out, p=2, dim=1)  # L2 normalization",
    "        return F.relu(out)",
    "",
    "",
    "# Quick test",
    "sage_layer = GraphSAGELayer(NUM_FEATURES, 32)",
    "out_sage = sage_layer(X_syn, edge_index_syn)",
    "print(f'GraphSAGE output: {out_sage.shape}')",
]))

# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 -- Graph-Level Predictions
# ══════════════════════════════════════════════════════════════════════════

cells.append(md([
    "---",
    "## 8. Graph-Level Predictions <a id='8-graph-level-predictions'></a>",
    "",
    "For tasks like molecular property prediction, we need a **single vector per graph**. ",
    "This requires a **readout / pooling** step.",
    "",
    "| Pooling | Formula | Notes |",
    "|---------|---------|-------|",
    "| Mean    | mean(h_v) | Simple; loses magnitude info |",
    "| Sum     | sum(h_v) | Preserves graph size info |",
    "| Max     | max(h_v) | Captures salient features |",
    "| Hierarchical | DiffPool, SAGPool | Learns coarsened graph |",
]))

cells.append(code([
    "class GraphClassifier(nn.Module):",
    "    \"\"\"Simple graph classification model with global pooling.\"\"\"",
    "",
    "    def __init__(self, in_dim, hidden_dim, out_dim, pool='mean'):",
    "        super().__init__()",
    "        self.conv1 = GCNLayer(in_dim, hidden_dim)",
    "        self.conv2 = GCNLayer(hidden_dim, hidden_dim)",
    "        self.classifier = nn.Linear(hidden_dim, out_dim)",
    "        self.pool = pool",
    "",
    "    def forward(self, x, edge_index, batch=None):",
    "        x = F.relu(self.conv1(x, edge_index))",
    "        x = F.relu(self.conv2(x, edge_index))",
    "",
    "        # Global pooling",
    "        if batch is None:",
    "            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)",
    "",
    "        num_graphs = batch.max().item() + 1",
    "        graph_embs = torch.zeros(num_graphs, x.size(1), device=x.device)",
    "",
    "        if self.pool == 'sum':",
    "            graph_embs.index_add_(0, batch, x)",
    "        elif self.pool == 'max':",
    "            for g in range(num_graphs):",
    "                mask = (batch == g)",
    "                graph_embs[g] = x[mask].max(dim=0).values",
    "        else:  # mean",
    "            graph_embs.index_add_(0, batch, x)",
    "            counts = torch.zeros(num_graphs, device=x.device)",
    "            counts.index_add_(0, batch, torch.ones(x.size(0), device=x.device))",
    "            graph_embs = graph_embs / counts.unsqueeze(1)",
    "",
    "        return self.classifier(graph_embs)",
    "",
    "",
    "# Test with a single graph",
    "clf = GraphClassifier(NUM_FEATURES, 32, 3)",
    "logits = clf(X_syn, edge_index_syn)",
    "print(f'Graph-level logits shape: {logits.shape}')",
]))

# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 -- PyTorch Geometric Workflow
# ══════════════════════════════════════════════════════════════════════════

cells.append(md([
    "---",
    "## 9. PyTorch Geometric Workflow <a id='9-pytorch-geometric-workflow'></a>",
    "",
    "PyTorch Geometric (PyG) provides optimized implementations of GNN layers, ",
    "data structures, and dataloaders. Here is the typical workflow.",
]))

cells.append(md([
    "### Core Abstractions",
    "",
    "- **`Data`** -- holds a single graph: `x`, `edge_index`, `y`, and any extra attributes.",
    "- **`Batch`** -- stacks multiple `Data` objects into a single disconnected graph with a `batch` vector.",
    "- **`DataLoader`** -- yields `Batch` objects for mini-batch training.",
    "- **Built-in layers**: `GCNConv`, `GATConv`, `SAGEConv`, `GINConv`, `TransformerConv`, etc.",
]))

cells.append(code([
    "if HAS_PYG:",
    "    # Wrap our synthetic data into a PyG Data object",
    "    data = Data(",
    "        x=X_syn,",
    "        edge_index=edge_index_syn,",
    "        y=Y_syn,",
    "        train_mask=train_mask,",
    "        val_mask=val_mask,",
    "        test_mask=test_mask",
    "    )",
    "    print(data)",
    "    print(f'Is undirected: {data.is_undirected()}')",
    "    print(f'Has self-loops: {data.has_self_loops()}')",
    "else:",
    "    print('Skipped -- PyTorch Geometric not installed')",
]))

cells.append(code([
    "if HAS_PYG:",
    "    class PyGModel(nn.Module):",
    "        def __init__(self):",
    "            super().__init__()",
    "            self.conv1 = GCNConv(NUM_FEATURES, 64)",
    "            self.conv2 = GCNConv(64, NUM_CLASSES)",
    "",
    "        def forward(self, data):",
    "            x, edge_index = data.x, data.edge_index",
    "            x = F.relu(self.conv1(x, edge_index))",
    "            x = F.dropout(x, p=0.5, training=self.training)",
    "            x = self.conv2(x, edge_index)",
    "            return F.log_softmax(x, dim=1)",
    "",
    "    pyg_model = PyGModel()",
    "    pyg_opt = torch.optim.Adam(pyg_model.parameters(), lr=0.01, weight_decay=5e-4)",
    "",
    "    pyg_model.train()",
    "    for epoch in range(200):",
    "        pyg_opt.zero_grad()",
    "        out = pyg_model(data)",
    "        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])",
    "        loss.backward()",
    "        pyg_opt.step()",
    "",
    "    pyg_model.eval()",
    "    with torch.no_grad():",
    "        pred = pyg_model(data).argmax(dim=1)",
    "        acc = accuracy_score(data.y[data.test_mask].numpy(), pred[data.test_mask].numpy())",
    "    print(f'PyG GCN Test Accuracy: {acc:.2%}')",
    "else:",
    "    print('Skipped -- PyTorch Geometric not installed')",
]))

cells.append(md([
    "### Batching Multiple Graphs (Graph Classification)",
]))

cells.append(code([
    "if HAS_PYG:",
    "    # Create a few toy graphs",
    "    graphs = []",
    "    for i in range(16):",
    "        n = np.random.randint(10, 30)",
    "        e = np.random.randint(20, 60)",
    "        ei = torch.randint(0, n, (2, e))",
    "        g = Data(x=torch.randn(n, 8), edge_index=ei, y=torch.tensor([i % 3]))",
    "        graphs.append(g)",
    "",
    "    loader = PyGDataLoader(graphs, batch_size=4, shuffle=True)",
    "    batch = next(iter(loader))",
    "    print(f'Batch: {batch}')",
    "    print(f'batch.batch: {batch.batch.shape}  (maps each node to its graph index)')",
    "    print(f'Number of graphs in batch: {batch.num_graphs}')",
    "else:",
    "    print('Skipped -- PyTorch Geometric not installed')",
]))

# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 -- Applications & Competition Tips
# ══════════════════════════════════════════════════════════════════════════

cells.append(md([
    "---",
    "## 10. Applications & Competition Tips <a id='10-applications--competition-tips'></a>",
]))

cells.append(md([
    "### Where GNNs Shine on Kaggle",
    "",
    "| Competition | GNN Use Case |",
    "|-------------|--------------|",
    "| **CAFA 6 -- Protein Function Prediction** | Protein contact / interaction graphs; predict GO-term labels per protein |",
    "| **Stanford RNA 3D Folding** | RNA nucleotide graphs with 3D coordinates; predict structural properties |",
    "| Drug discovery challenges | Molecular graphs: atoms = nodes, bonds = edges |",
    "| Recommendation systems | User-item bipartite graphs |",
    "| Fraud detection | Transaction graphs; link prediction |",
]))

cells.append(md([
    "### Competition Tips",
    "",
    "> **Key Takeaway** -- GNNs are most impactful when the *graph structure itself* ",
    "> carries predictive signal. If edges are random or uninformative, a standard MLP ",
    "> on node features may perform just as well.",
    "",
    "1. **Start simple**: GCN with 2-3 layers is a strong baseline.",
    "2. **Feature engineering matters**: Edge features, positional encodings (Laplacian eigenvectors), and node degree are cheap to add.",
    "3. **Over-smoothing**: Deep GNNs (>5 layers) make all node embeddings converge. Use residual connections or jumping knowledge.",
    "4. **Ensemble**: Combine GCN + GAT + GraphSAGE predictions.",
    "5. **PyG ecosystem**: Leverage `torch_geometric.transforms` for feature augmentation.",
]))

cells.append(code([
    "# ── Quick link-prediction example ──",
    "# Predict missing edges in a graph (common in knowledge graphs / rec systems)",
    "",
    "def link_prediction_score(z, edge_index):",
    "    \"\"\"Simple dot-product decoder for link prediction.\"\"\"",
    "    src, dst = edge_index",
    "    return (z[src] * z[dst]).sum(dim=1)",
    "",
    "",
    "# Generate node embeddings via our GCN",
    "model_gcn.eval()",
    "with torch.no_grad():",
    "    node_emb = model_gcn(X_syn, edge_index_syn)  # log-softmax outputs",
    "",
    "# Score existing edges (positive) and random pairs (negative)",
    "pos_scores = link_prediction_score(node_emb, edge_index_syn[:, :50])",
    "neg_edges = torch.randint(0, NUM_NODES, (2, 50))",
    "neg_scores = link_prediction_score(node_emb, neg_edges)",
    "",
    "print(f'Positive edge scores (mean): {pos_scores.mean().item():.4f}')",
    "print(f'Negative edge scores (mean): {neg_scores.mean().item():.4f}')",
    "print(f'Score gap (higher = better):  {(pos_scores.mean() - neg_scores.mean()).item():.4f}')",
]))

cells.append(md([
    "### Drug Discovery -- Molecular Graphs",
]))

cells.append(code([
    "# Visualize a molecule as a graph (caffeine)",
    "# Node = atom, Edge = bond",
    "caffeine = nx.Graph()",
    "atoms = ['C','C','N','C','N','C','C','N','C','N','O','O','C','C']",
    "for i, a in enumerate(atoms):",
    "    caffeine.add_node(i, element=a)",
    "",
    "bonds = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(5,6),(6,7),(7,8),(8,9),",
    "         (9,3),(6,10),(8,11),(0,12),(2,13)]",
    "caffeine.add_edges_from(bonds)",
    "",
    "color_map = {'C': '#404040', 'N': '#3050F8', 'O': '#FF0D0D'}",
    "node_colors = [color_map.get(atoms[n], 'gray') for n in caffeine.nodes()]",
    "labels = {n: atoms[n] for n in caffeine.nodes()}",
    "",
    "plt.figure(figsize=(6, 5))",
    "pos = nx.spring_layout(caffeine, seed=7)",
    "nx.draw(caffeine, pos, node_color=node_colors, labels=labels,",
    "        node_size=600, font_color='white', font_weight='bold',",
    "        edge_color='#888888', width=2)",
    "plt.title('Caffeine Molecular Graph')",
    "plt.show()",
]))

# ══════════════════════════════════════════════════════════════════════════
# SECTION 11 -- Further Reading
# ══════════════════════════════════════════════════════════════════════════

cells.append(md([
    "## Interpretation, Trade-offs, and Limitations",
    "",
    "- **Observation:** graph structure helps most when neighborhood information carries signal that ordinary row-wise models cannot capture.",
    "- **Interpretation:** deeper GNN stacks can improve receptive field size, but they often oversmooth node states and blur class boundaries.",
    "- **Trade-off:** attention-based models are expressive, yet they usually cost more memory and latency than simple message-passing layers.",
    "- **Limitation:** synthetic demos validate mechanics, not benchmark dominance, so real-world validation should test sparsity, noise, and scale.",
]))

cells.append(md([
    "---",
    "## 11. Further Reading <a id='11-further-reading'></a>",
    "",
    "- Kipf & Welling (2017) -- [Semi-Supervised Classification with GCNs](https://arxiv.org/abs/1609.02907)",
    "- Velickovic et al. (2018) -- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)",
    "- Hamilton et al. (2017) -- [Inductive Representation Learning (GraphSAGE)](https://arxiv.org/abs/1706.02216)",
    "- Xu et al. (2019) -- [How Powerful are GNNs?](https://arxiv.org/abs/1810.00826)",
    "- PyTorch Geometric docs -- [pyg.org](https://pyg.org/)",
    "- Stanford CS224W -- [Graph ML Course](http://web.stanford.edu/class/cs224w/)",
]))

# ── CTA ────────────────────────────────────────────────────────────────────
cells.append(md([
    "---",
    "## Try It Yourself!",
    "",
    "1. **Fork** this notebook and swap the synthetic data for a real dataset (`Planetoid('Cora')`).",
    "2. Add **edge features** to the GAT implementation.",
    "3. Build a **GIN (Graph Isomorphism Network)** layer -- it is the most expressive simple GNN.",
    "4. Submit to a graph-related Kaggle competition: ",
    "   - [CAFA 6 -- Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)",
    "   - [Stanford RNA 3D Folding](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding)",
    "",
    "If you found this useful, please **upvote** and leave a comment!",
]))

# ---------------------------------------------------------------------------
# Assemble the notebook
# ---------------------------------------------------------------------------

# Fix sources
for cell in cells:
    cell["source"] = _fix_source(cell["source"])


write_notebook(cells, __file__, "gnn_practical_guide.ipynb")
