#pypf/pfnet.py
import numpy as np

class PFnet:
    def __init__(self, proximity, q=np.inf, r=np.inf):
        self.q = q # Inf yields nterms-1
        self.r = r
        self.terms = []  # node labels
        self.nnodes = 0
        self.nlinks = 0
        self.dismat = np.array([])  # distance matrix
        self.mindis = np.array([])  # minimum distances
        self.adjmat = np.array([])  # adjacency matrix with link weights
        self.isdirected = False  # true for directed graph
        self.name = "default"  # Initialize the name property

        if hasattr(proximity, 'name'):
            self.name = proximity.name

        if hasattr(proximity, 'terms'):
            self.terms = proximity.terms
            self.nnodes = len(self.terms)
            if self.q >= self.nnodes -1:
                self.q = np.inf
            if hasattr(proximity, 'dismat'):
                self.dismat = np.array(proximity.dismat)
                self.isdirected = not np.array_equal(self.dismat, self.dismat.T)
                self.mindis = self._getmindis(self.q, self.r, self.dismat)
                links = (self.dismat < np.inf) & (self.dismat == self.mindis)
                self.adjmat = np.copy(self.dismat)
                self.adjmat[~links] = 0
                self.nlinks = np.count_nonzero(self.adjmat)
                if not self.isdirected:
                    self.nlinks = self.nlinks / 2
            else:
                print("Warning: proximity object missing 'dismat' property.")
        else:
            print("Error: Input 'proximity' object missing 'terms' property.")

    def _getmindis(self, q, r, dis):
        if q >= (self.nnodes - 1):
            mindis = self._floyd(dis, r)
        else:
            mindis = self._dijkstra(dis, q, r)
        return mindis

    def _floyd(self, dis, r):
        dis = np.array(dis, dtype=float)
        n = len(dis)
        mindis = np.copy(dis)
        for ind in range(n):
            for row in range(n):
                for col in range(n):
                    indirect = self._minkowski(mindis[row, ind], mindis[ind, col], r)
                    if indirect < mindis[row, col]:
                        mindis[row, col] = indirect
        return mindis

    def _dijkstra(self, dis, q, r):
        dis = np.array(dis, dtype=float)
        n = len(dis)
        mindis = np.copy(dis)
        pass_count = 1
        changed = True
        while changed and (pass_count < q):
            pass_count += 1
            changed = False
            m = np.copy(mindis)
            for row in range(n):
                for col in range(n):
                    for ind in range(n):
                        indirect1 = self._minkowski(dis[row, ind], m[ind, col], r)
                        indirect2 = self._minkowski(m[row, ind], dis[ind, col], r)
                        indirect = min(indirect1, indirect2)
                        if indirect < mindis[row, col]:
                            mindis[row, col] = indirect
                            changed = True
            # Check for convergence (no changes in this pass)
            if np.array_equal(m, mindis):
                changed = False
        return mindis

    def netprint(self, note=None):
        print("\nPFnet Object:", note)
        print(f"name: {self.name}")
        print(f"terms[0]: {self.terms[0]}")
        print(f"nnodes: {self.nnodes}")
        print(f"q: {self.q}")
        print(f"r: {self.r}")
        print(f"nlinks: {self.nlinks}")
        print(f"isdirected?: {self.isdirected}")

    @staticmethod
    def _minkowski(dis1, dis2, r):
        if r == 1:
            dis = dis1 + dis2
        elif np.isinf(r):
            dis = max(dis1, dis2)
        else:
            dis = (dis1**r + dis2**r)**(1/r)
            # Remove rounding errors
            dis = round(dis, 12)
        return dis

if __name__ == '__main__':
    # Example usage (assuming you have a Proximity class defined)
    from proximity import Proximity  # Adjust import if needed

    # Create a Proximity object (this will trigger the file dialog)
    proximity_obj = Proximity()
    if proximity_obj.terms:
        # Create a PFnet object using the Proximity object
        network = PFnet(proximity_obj, q=2)
        if network.name:
            print(f"PFnet Name: {network.name}")
        print(f"Number of Nodes: {network.nnodes}")
        print(f"Number of Links: {network.nlinks}")

        # Example with a specific file
        proximity_obj_from_file = Proximity("psy.self.xlsx") # Replace with a valid path
        if proximity_obj_from_file.terms:
            network_from_file = PFnet(proximity_obj_from_file, q=2)
            if network_from_file.name:
                print(f"PFnet Name (from file): {network_from_file.name}")