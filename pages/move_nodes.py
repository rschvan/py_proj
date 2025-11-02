# pages/move_nodes.py
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from pypf import Netpic

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
                notebook=True)

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
                   size=12)

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
        "physics": {"enabled": physics},
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
        }
        # -----------------------------------
    }

    # Generate the HTML content string from the pyvis object
    # notebook=True is necessary here to inline all JavaScript/CSS
    html_content = g.generate_html(notebook=True)

    # Return the HTML string directly for st.components.v1.html and the pyvis graph
    return html_content, g

def get_new_coords(net:Network):
    for node in net.nodes:
        x, y = net.nodes[node]['position']['x'], net.nodes[node]['position']['y']
        x = float(x) / 400
        y = -float(y) / 400
        st.session_state.pic.net.coords[int(node)] = (x, y)

pic = st.session_state.pic # alias for easy reference
if "text_size" in st.session_state:
    font_size = st.session_state.text_size
else:
    font_size = 23

with st.sidebar:
    display_type = st.radio("Select Display Type",options=["Static", "Dynamic"])
    if display_type == "Static":
        interactive_html, pyvis_net = create_visjs_html(pic, physics=False, font_size=font_size)
    else:
        interactive_html, pyvis_net = create_visjs_html(pic, physics=True, font_size=font_size)

    st.number_input("node size", min_value=12, max_value=34, value=23, step=1, key="text_size",
                    )
    st.write("Set node size before moving nodes")
    st.write("""You can drag the nodes to new locations
     and can download the image with a screenshot.""")
    # save_coords = st.button("Save Coordinates")
    # if save_coords:
    #     get_new_coords(pyvis_net)
st.subheader(f"{pic.net.name}: {pic.net.nnodes} nodes and {pic.net.nlinks} links")

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