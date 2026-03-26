# pages/view_net.py
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from pypf import PFnet
from pypf.stutils import init_pf_session_state
import numpy as np
import math
import time

# alias for st.system_state
stss = st.session_state

st.set_page_config(
    page_title="Net View",
    page_icon="🌐")

if "col" not in stss:
    init_pf_session_state()

focus_net = stss.col.focus_net # alias for easy reference
warning_message = None

def do_rotation():
    rotation = stss.rotation
    coords = focus_net.coords
    # rotate code
    theta_rad = math.radians(rotation)
    # Create the 2x2 rotation matrix
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])
    # Perform the rotation by matrix multiplication
    # (n x 2) @ (2 x 2) -> (n x 2)
    focus_net.coords = coords @ rotation_matrix
    time.sleep(.5)
    stss.rotation = 0

def weight_string(weight, is_merge=False):
    if is_merge:
        # Convert integer bitmask to binary string (e.g., 3 -> '11')
        # [2:] removes the '0b' prefix from Python's bin() output
        ws = bin(int(weight))[2:]
    else:
        # Standard Pathfinder distance formatting
        if (weight > 100) or (weight < 1.0e-10):
            # Show as integer if it's larger than 100 or the 1e-20 placeholder
            ws = f"{weight:.0f}"
        else:
            # Otherwise use 4 significant digits
            ws = f"{weight:.4g}"
    return ws


def create_visjs_html(fnet: PFnet, physics=False, font_size=23, plot_weights=False) -> tuple[str, Network]:
    """
    Generates an HTML string for pyvis with proportional scaling
    and restricted manipulation UI.
    """
    g = Network(height="590px", width="100%",
                directed=fnet.isdirected,
                cdn_resources='in_line',
                notebook=True,
                )

    # Unified scale factor based on your base font (default 23)
    global_scale = font_size / 23.0
    arrow_scale = max(0.5, global_scale)

    # 1. Add Nodes
    for i, node_name in enumerate(fnet.nodes):
        label = fnet.nodes[i]
        x, y = fnet.coords[i]

        g.add_node(node_name,
                   label=label,
                   x=400 * float(x), y=-400 * float(y),
                   title=f"Node: {node_name}",
                   shape='box',
                   color='white',
                   font={'size': font_size},
                   margin=5 * global_scale,
                   borderWidth=1 * global_scale)

    # 2. Add Edges
    for i in range(fnet.nnodes):
        for j in range(fnet.nnodes):
            weight = float(fnet.adjmat[i, j])
            if weight > 0:
                s_node = fnet.nodes[i]
                t_node = fnet.nodes[j]

                edge_label = None
                if plot_weights:
                    edge_label = weight_string(weight, is_merge=fnet.type == "merge")

                g.add_edge(s_node, t_node,
                           label=edge_label,
                           width=1.5 * global_scale,
                           arrows={'to': {'enabled': fnet.isdirected,
                                          'scaleFactor': arrow_scale}},
                           color='blue',
                           font={'size': max(8, font_size - 8),
                                 'color': 'blue', 'align': 'top'},
                           smooth={'enabled': fnet.isdirected,
                                   'type': 'curvedCW',
                                   'roundness': 0.15} if fnet.isdirected else False,
                           arrowStrikethrough=False,
                           physics=True)

    # 3. Configure Options with your UI Guardrails
    g.options = {
        "physics": {
            "enabled": physics,
            "solver": 'forceAtlas2Based',
            "forceAtlas2Based": {
                "gravitationalConstant": -250,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.10
            },
            "minVelocity": 0.75,
            "stabilization": {"enabled": True, "iterations": 1000}
        },
        "autoResize": False,
        "nodes": {"font": {"face": "arial", "align": "center"}},
        "manipulation": {
            "enabled": True,  # Keep dragging active
            "initiallyActive": True,
            "showNodeProperties": False,
            "editEdge": False,
            "addNode": False,
            "addEdge": False,
            "deleteNode": False,
            "deleteEdge": False
        },
        "configure": {
            "enabled": physics,  # Only show settings for Dynamic view
            "filter": "physics"
        }
    }

    return g.generate_html(notebook=True), g

base_font = stss.get("text_size", 23)

with st.sidebar:
    st.number_input("**Node Size**", min_value=2, max_value=44, value=23, step=1, key="text_size", )
    #st.write("Set node size before moving nodes")

    display_type = st.radio("**Select Display Type**",
                            options=["Static", "Dynamic"])
    if display_type == "Static":
        st.write("""**Drag nodes as last step**""")
    else:
        st.write("""**Drag nodes to affect layout**""")

    # 1. PRE-CHECK: Intercept and revert the state BEFORE the widget is created
    if stss.get("layout_selector") == "saved":
        tid = focus_net.termsid
        if tid not in stss.col.saved_layouts:
            # Revert the session state key to the last valid layout
            stss.layout_selector = stss.layout
            warning_message = "No saved layout for these terms. Layout unchanged."

    # 2. THE WIDGET:
    layout_choice = st.selectbox(
        "**Layout Method**",
        options=stss.layouts,
        key="layout_selector"  # Streamlit uses this key to set the value
    )

    # 3. COORDINATE INJECTION
    if layout_choice == "saved":
        tid = focus_net.termsid
        focus_net.coords = stss.col.saved_layouts[tid]
        stss.layout = "saved"

    layout_changed = layout_choice != stss.layout

    c1, c2 = st.columns(2)
    with c1:
        if st.button("flip x"):
            focus_net.coords[:, 0] *= -1
        if st.button("flip y"):
            focus_net.coords[:, 1] *= -1

    with c2:
        if st.button("x <> y"):
            focus_net.coords[:, :] = focus_net.coords[:, [1, 0]]
        redraw_pushed = st.button("redraw")

    rotate = st.slider("**Degrees of Clockwise Rotation**", 0, 360, key="rotation", on_change=do_rotation)

    # Save current focus_net coords to the global collection
    if st.button("📌 Save Terms Layout", use_container_width=False):
        # Ensure coords exist and aren't empty
        if hasattr(stss.col.focus_net, 'coords') and stss.col.focus_net.coords is not None:
            tid = stss.col.focus_net.termsid
            # Update the Collection's global layout dictionary
            stss.col.saved_layouts[tid] = stss.col.focus_net.coords
            # Flag that the project has unsaved changes
            stss.col.changed = True
            st.success(f"Layout saved.")
        else:
            st.warning("No coordinates available to pin.")

    show_weights = st.checkbox("Show Link Weights")

    if redraw_pushed or layout_changed:
        if layout_choice == "saved":
            focus_net.coords = stss.col.saved_layouts[focus_net.termsid]
        else:
            focus_net.get_layout(method=layout_choice)
        stss.layout = layout_choice

st.write(f"**Network: {focus_net.name}**: Layout {layout_choice}")
st.write(f"**{focus_net.nnodes} nodes and {focus_net.nlinks} links**")
if warning_message:
    st.warning(warning_message)

if display_type == "Static":
    final_font = base_font
    interactive_html, pyvis_net = create_visjs_html(fnet=focus_net, physics=False,
                        font_size=final_font, plot_weights=show_weights)
else:
    nnodes = focus_net.nnodes
    scale = np.sqrt(nnodes) / np.sqrt(9)
    scale = np.clip(scale, 0.7, 2.0)
    interactive_html, pyvis_net = create_visjs_html(fnet=focus_net, physics=True,
                        font_size=base_font* scale, plot_weights=show_weights)

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