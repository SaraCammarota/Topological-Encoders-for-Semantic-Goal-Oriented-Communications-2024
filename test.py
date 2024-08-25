import torch
import torch.nn as nn
from layers import DGM_d

# Example embed_f function for testing
class SimpleEmbed(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleEmbed, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x, edge_index):
        # For simplicity, we ignore edge_index here
        return self.fc(x)

# Create a simple test function
def test_dgm_d():
    # Parameters
    num_graphs = 3
    num_nodes_per_graph = [4, 3, 5]  # Different number of nodes in each graph
    feature_dim = 5
    output_dim = 3
    k = 2

    # Generate random features for each node in the batch
    x = torch.randn(sum(num_nodes_per_graph), feature_dim)

    # Generate batch tensor indicating graph membership for each node
    batch = torch.cat([torch.full((n,), i) for i, n in enumerate(num_nodes_per_graph)], dim=0)

    # Create a dummy adjacency matrix (not really used in this test)
    A = torch.eye(sum(num_nodes_per_graph))

    # Create the model
    embed_f = SimpleEmbed(feature_dim, output_dim)
    model = DGM_d(embed_f, k=k, distance='euclidean', sparse=False)

    # Run the model forward
    x_embedded, edges_hat, logprobs = model(x, A, batch=batch)

    # Print the outputs
    print("Embedded node features:")
    print(x_embedded)
    print("\nSampled edges (global indices):")
    print(edges_hat)
    print("\nLog probabilities of sampled edges:")
    print(logprobs)

    # Verify that the sampled edges are correctly associated with each graph
    for i, num_nodes in enumerate(num_nodes_per_graph):
        node_indices = (batch == i).nonzero(as_tuple=True)[0]
        edge_indices = edges_hat[(edges_hat[:, 0] >= node_indices[0]) & (edges_hat[:, 0] < node_indices[-1])]
        print(f"\nGraph {i}:")
        print("Node indices:", node_indices)
        print("Sampled edge indices:", edge_indices)


if __name__ == "__main__":
    
    #train_and_plot()
    test_dgm_d()
    # trainer, channel, train_loader, val_loader = setup_training()
    # trainer.fit(channel, train_dataloaders=train_loader, val_dataloaders=val_loader)

