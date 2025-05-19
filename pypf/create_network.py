from pypf.pfnet import PFnet
from pypf.proximity import Proximity
import pickle  # To save the PFnet object

def create_and_save_network(output_filepath="network.pkl"):
    print("Creating Proximity object...")
    prx = Proximity()
    if prx.terms:
        print("Creating PFnet object...")
        net = PFnet(prx)
        print(f"Number of links: {net.nlinks}")
        with open(output_filepath, 'wb') as f:
            pickle.dump(net, f)
        print(f"PFnet object saved to {output_filepath}")
    else:
        print("Failed to load proximity data, PFnet not created.")

if __name__ == "__main__":
    create_and_save_network()