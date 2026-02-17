PyPathfinder is a Python-based application built with Streamlit designed for creating, analyzing, and visualizing Pathfinder Networks from proximity data. The app's architecture is modular, separating the user interface from the core data processing and network analysis logic.
App Architecture
The application is structured into several key components:
  User Interface (Streamlit):
  stapp.py: The main entry point that sets up the application layout and navigation. It includes a sidebar for navigating between pages like "Build Nets," "Select Display," and "Interactive Display".
  Page Modules: Specific functionalities are organized into pages such as display_network.py for visualization settings, move_nodes.py for interactive graph adjustment, and help.py for documentation.
  Core Data Classes (pypf package):
    Proximity: Represents the raw proximity data, including terms (items) and their pairwise distances.
    PFnet: Manages the generated Pathfinder networks, handling properties like adjacency matrices, nodes, links, and network metrics.
    Collection: A management class that stores multiple proximity objects and networks, allowing for bulk operations like averaging or merging.
    Netpic: A dedicated class for generating and managing network plots using Matplotlib.
Main Functionality
  Network Generation: The app derives Pathfinder networks ($q, r$ parameters) from proximity data. It also supports threshold and nearest-neighbor networks.
  Data Management: Users can load their own data files (.xlsx or .csv) or use legacy file formats. The app includes tools to average compatible proximity datasets or merge existing networks.
Visualization & Interaction:
  Static Display: Uses various layout algorithms (e.g., Kamada-Kawai) to visualize networks with adjustable font and node sizes.
  Interactive Display: Utilizes Pyvis (vis.js) to allow users to manually drag nodes to customize the layout, which can then be saved as an image.
Analysis: Provides detailed network properties, including eccentricity, radius, diameter, and link lists.

