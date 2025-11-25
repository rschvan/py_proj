# pages/move_nodes.py
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from pypf import Netpic

st.set_page_config(
    page_title="PyPF Adjust",
    page_icon="🌐")

def create_visjs_html(pic: Netpic, physics=False, font_size = 23) -> tuple[str, Network]:
    """
    Generates an HTML string containing the pyvis/vis.js graph for Streamlit embedding.
    pic is a Netpic object.
    """
    # 1. Initialize Pyvis network
    # Note: Use 'in_line' to ensure all JS/CSS is included, making it portable.
    g = Network(height="600px", width="100%",
                directed=pic.is_directed,
                cdn_resources='in_line',
                notebook=True
                )

    # 2. Add Nodes and Edges using pic
    for i, node_name in enumerate(pic.nodes):
        label = pic.nodelabels[i]  # Use your split label
        x, y = pic.net.coords[i] #if pic.net.coords is not None else (None, None)
        x = 400*float(x) #if x is not None else None
        y = -400*float(y) #if y is not None else None

        # Add node with specific options for draggability and text-as-node appearance
        g.add_node(node_name,
                   label=label,
                   x=x, y=y,
                   title=f"Node: {node_name}",
                   shape='box',
                   color='white',
                   size=12,
                   )

    # 3. Add Edges
    for i in range(len(pic.nodes)):
        for j in range(len(pic.nodes)):
            weight = float(pic.adjmat[i, j])
            if weight > 0:
                s_node = pic.nodes[i]
                t_node = pic.nodes[j]

                g.add_edge(s_node, t_node,
                           #value=.5,
                           # title=f"Weight: {weight:.4g}",
                           # label=f"{weight:.4g}",
                           width=2,
                           color='blue',
                           smooth=False,
                           arrowStrikethrough=False,
                           physics=True)

    # 4. Enable manipulation and configure

    g.options = {
        "physics": {
            "enabled": physics,
            "solver": 'hierarchicalRepulsion', # Specify the solver
            "forceAtlas2Based": {         # Configuration for the solver
                "gravitationalConstant": -250, # Adjust for tighter/looser clusters
                "centralGravity": 0.03,      # Adjust central pull
                "springLength": 100,
                "springConstant": 0.10
            },
            "minVelocity": 0.75, # Prevents small, annoying jitters
            "stabilization": {
                "enabled": True,
                "iterations": 1000 # Increase for better initial layout stability
            }
        },
        "autoResize": False,
        "nodes": {
            "font": {
                "size": font_size,
                "face": "arial",
                "align": "center"
            },
            "margin": 3  # Increase margin slightly to account for larger font
        },
        # --- MODIFIED MANIPULATION BLOCK ---
        "manipulation": {
            "enabled": True,  # Keep core manipulation (dragging) enabled
            "initiallyActive": True,  # Ensure dragging works immediately
            "showNodeProperties": False,  # <-- Hides the "Edit Node" button/dialog
            "editEdge": False,  # <-- Hides the "Edit Edge" button/dialog
            "addNode": False,
            "addEdge": False,
            "deleteNode": False,
            "deleteEdge": False
        },
        "configure": {
            "enabled": physics,
            "filter": "physics"
        }
        # -----------------------------------
    }

    # Generate the HTML content string from the pyvis object
    # notebook=True is necessary here to inline all JavaScript/CSS
    html_content = g.generate_html(notebook=True)

    # Return the HTML string directly for st.components.v1.html and the pyvis graph
    return html_content, g

pic = st.session_state.pic # alias for easy reference
if "text_size" in st.session_state:
    font_size = st.session_state.text_size
else:
    font_size = 23

with st.sidebar:
    display_type = st.radio("**Select Display Type**",options=["Static", "Dynamic"])
    if display_type == "Static":
        interactive_html, pyvis_net = create_visjs_html(pic=pic, physics=False, font_size=font_size)
    else:
        interactive_html, pyvis_net = create_visjs_html(pic=pic, physics=True, font_size=font_size+6)

    st.number_input("**node size**", min_value=2, max_value=44, value=23, step=1, key="text_size", )
    st.write("Set node size before moving nodes")
    st.write("""You can drag the nodes to new locations
     and can capture the image with a screenshot.""")

st.write(f"**Network: {pic.net.name}**")
st.write(f"**{pic.net.nnodes} nodes and {pic.net.nlinks} links**")

# Embed the HTML directly into the Streamlit app
# The height must be set to ensure the visualization is visible
if interactive_html:
    components.html(
        interactive_html,
        height=650,  # Set height to match the '600px' defined in pyvis + buffer
        scrolling=True
    )
else:
    st.error("Failed to generate interactive network HTML.")