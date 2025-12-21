import matplotlib.pyplot as plt
import networkx as nx


def draw_bayesian_network(save_path="bayesian_network.png"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    G = nx.DiGraph()
    nodes = ["M", "S", "I", "F", "T", "D"]
    node_labels = {
        "M": "Immune System\n(M)",
        "S": "Season\n(S)",
        "I": "Disease Severity\n(I)",
        "F": "Financial Capability\n(F)",
        "T": "Treatment Type\n(T)",
        "D": "Death Probability\n(D)",
    }
    edges = [("M", "I"), ("S", "I"), ("I", "D"), ("I", "T"), ("F", "T"), ("T", "D")]
    G.add_edges_from(edges)
    pos = {"M": (0, 2), "S": (2, 2), "I": (1, 1), "F": (3, 1), "T": (2, 0), "D": (1, -1)}
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=3000, ax=ax)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, arrowstyle="->", width=2, ax=ax)
    ax.set_title("Bayesian Network: Disease Model", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return G


def draw_markov_network(save_path="markov_network.png"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    G = nx.Graph()
    edges = [("C", "O"), ("O", "A"), ("O", "S"), ("A", "T"), ("S", "T"), ("T", "B"), ("T", "M")]
    G.add_edges_from(edges)
    pos = {"C": (1, 3), "O": (1, 2), "A": (2, 1.5), "S": (0, 1), "T": (1, 0.5), "B": (2, -0.5), "M": (0, -0.5)}
    nx.draw_networkx_nodes(G, pos, node_color="lightcoral", node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=2, ax=ax)
    ax.set_title("Markov Network (Undirected)", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return G
