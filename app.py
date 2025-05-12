from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
from pypf.pfnet import PFnet
from pypf.proximity import Proximity

app = Flask(__name__)
CORS(app)

network = None

def load_network_data():
    global network
    filepath = input("Please enter the filepath of your data file (or leave blank to skip): ")
    if filepath:
        print(f"Attempting to load data from: {filepath}")
        prx = Proximity(filepath=filepath)
        if prx.terms:
            print("Creating PFnet object...")
            net = PFnet(prx)
            print(f"Number of links: {net.nlinks}")
            network = net
        else:
            print("Failed to load proximity data.")
    else:
        print("Skipping data loading.")

# Load network data before starting the Flask app
load_network_data()

@app.route('/network_data', methods=['GET'])
def get_network_data():
    global network

    if network and hasattr(network, 'terms'):
        print("Serving network data.")
        nodes = []
        for i, term in enumerate(network.terms):
            nodes.append({'data': {'id': str(i), 'label': term}})

        edges = []
        # adjacency_matrix = network.adjmat
        # n_nodes = len(network.terms)
        # for i in range(n_nodes):
        #     for j in range(i + 1, n_nodes):  # Avoid duplicates for undirected graphs
        #         if adjacency_matrix[i, j] != 0 and not np.isnan(adjacency_matrix[i, j]):
        #             n_edges.append({
        #                 'data': {
        #                     'source': str(i),
        #                     'target': str(j),
        #                     'weight': adjacency_matrix[i, j]  # You can include weights if needed
        #                 }
        #             })
        #     if network.isdirected:  # Handle directed n_edges
        #         for j in range(n_nodes):
        #             if i != j and adjacency_matrix[i, j] != 0 and not np.isnan(adjacency_matrix[i, j]):
        #                 n_edges.append({
        #                     'data': {
        #                         'source': str(i),
        #                         'target': str(j),
        #                         'weight': adjacency_matrix[i, j]
        #                     }
        #                 })

        network_data = {'n_nodes': nodes, 'n_edges': edges}
        return jsonify(network_data)
    else:
        return jsonify({'error': 'Network data not loaded'}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)