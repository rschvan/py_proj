# pages/view_net.py
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from pypf import PFnet
from pypf.stutils import init_pf_session_state, tight_divider
import numpy as np
import math
import time

# alias for st.session_state
stss = st.session_state

st.set_page_config(
    page_title="Net View",
    page_icon="🌐")

# set best scrolling position
components.html("""
    <script>
        function scrollStreamlit() {
            const doc = window.parent.document;
            // Target Streamlit's specific scrollable section containers
            const mainContainer = doc.querySelector('section.main') 
                               || doc.querySelector('[data-testid="stMain"]')
                               || doc.querySelector('[data-testid="stAppViewContainer"]');

            if (mainContainer) {
                mainContainer.scrollTo({ top: 50, behavior: 'instant' });
            }
        }

        // Execute immediately and after a short delay to override iframe/layout autofocus
        scrollStreamlit();
        setTimeout(scrollStreamlit, 100);
        setTimeout(scrollStreamlit, 300);
    </script>
    """,
    height=0)

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

def weight_string(weight):
    if focus_net.type == "mg":
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

def show_link(wt) -> bool:
    if focus_net.type == "mg":
        nn = int(wt).bit_count()
        return nn >= cutoff
    else:
        return wt <= cutoff

def create_visjs_html(fnet: PFnet, font_size=16) -> tuple[str, Network]:
    """Generates an HTML string for pyvis with proportional scaling,

    persistent edge weight labels, and client-side font & visibility controls.
    """
    g = Network(
        height="600px",
        width="100%",
        directed=fnet.isdirected,
        cdn_resources="in_line",
        notebook=True,
    )

    global_scale = font_size / 28.0
    arrow_scale = max(0.5, global_scale)
    initial_edge_font_size = max(8, font_size - 8)

    # 1. Add Nodes
    for i, node_name in enumerate(fnet.nodes):
        label = fnet.nodes[i]
        x, y = fnet.coords[i]
        x = 650 * float(x)
        y = -450 * float(y)

        g.add_node(
            node_name,
            label=label,
            x=x,
            y=y,
            title=f"Node: {node_name}",
            shape="box",
            color="white",
            font={"size": font_size},
            margin=5 * global_scale,
            borderWidth=0 * global_scale,
        )

    # 2. Add Edges (Always include formatted weight in originalLabel; start label as None)
    for i in range(fnet.nnodes):
        for j in range(fnet.nnodes):
            weight = float(fnet.adjmat[i, j])
            if 0 < weight and show_link(weight):
                s_node = fnet.nodes[i]
                t_node = fnet.nodes[j]
                wt_str = weight_string(weight)

                g.add_edge(
                    s_node,
                    t_node,
                    label=None,  # Initial state: invisible
                    originalLabel=wt_str,  # Preserved in Vis.js dataset
                    width=1.5 * global_scale,
                    arrows={
                        "to": {
                            "enabled": fnet.isdirected,
                            "scaleFactor": arrow_scale,
                        }
                    },
                    color="blue",
                    font={
                        "size": initial_edge_font_size,
                        "color": "blue",
                        "align": "top",
                    },
                    smooth={
                        "enabled": fnet.isdirected,
                        "type": "curvedCW",
                        "roundness": 0.15,
                    }
                    if fnet.isdirected
                    else False,
                    arrowStrikethrough=False,
                    physics=True,
                )

    # 3. Configure Options
    g.options = {
        "physics": {
            "enabled": False,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -120,
                "centralGravity": 0.015,
                "springLength": 100,
                "springConstant": 0.1,
                "damping": 0.6,
                "avoidOverlap": 0.8,
            },
            "minVelocity": 0.5,
            "stabilization": {"enabled": True, "iterations": 150, "fit": True},
        },
        "autoResize": True,
        "nodes": {"font": {"face": "arial", "align": "center"}},
        "manipulation": {
            "enabled": True,
            "initiallyActive": True,
            "showNodeProperties": False,
            "editEdge": False,
            "addNode": False,
            "addEdge": False,
            "deleteNode": False,
            "deleteEdge": False,
        },
        "configure": {"enabled": True, "filter": "physics"},
        "interaction": {"zoomView": True, "min": 0.8, "max": 1.2},
    }
    raw_html = g.generate_html(notebook=True)

    # Floating Controls: A-, A+, and Weights Toggle
    overlay_script = """
        <!-- Floating Font & Visibility Controls -->
        <div id="font-controls" style="
            position: absolute;
            top: 15px;
            right: 15px;
            z-index: 9999;
            display: flex;
            gap: 6px;
            background: rgba(255, 255, 255, 0.85);
            padding: 5px 8px;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            font-family: sans-serif;
        ">
            <button id="font-dec" title="Decrease Font" style="
                cursor: pointer;
                padding: 4px 10px;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: #f8f9fa;
            ">A-</button>
            <button id="font-inc" title="Increase Font" style="
                cursor: pointer;
                padding: 4px 10px;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: #f8f9fa;
            ">A+</button>
            <button id="toggle-weights" title="Toggle Edge Weights" style="
                cursor: pointer;
                padding: 4px 10px;
                font-size: 13px;
                font-weight: 500;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: #f8f9fa;
            ">Weights: Off</button>
        </div>

        <script type="text/javascript">
            window.addEventListener("load", function() {
                if (typeof network !== "undefined") {
                    let weightsVisible = false;

                    const fitOptions = {
                        animation: { duration: 350, easingFunction: "easeInOutQuad" }
                    };

                    network.fit(fitOptions);
                    network.on("stabilizationIterationsDone", function () { network.fit(fitOptions); });
                    network.on("stabilized", function () { network.fit(fitOptions); });

                    // Synchronized Font Scaling for Nodes & Edges
                    function adjustFontSize(delta) {
                        // 1. Scale Nodes
                        const nodesData = network.body.data.nodes;
                        const allNodes = nodesData.get();
                        const nodeUpdates = [];

                        allNodes.forEach(function(node) {
                            let curSize = 16;
                            if (typeof node.font === "object" && node.font && node.font.size) {
                                curSize = node.font.size;
                            } else if (typeof node.font === "number") {
                                curSize = node.font;
                            }
                            nodeUpdates.push({
                                id: node.id,
                                font: { size: Math.max(6, Math.min(48, curSize + delta)) }
                            });
                        });
                        nodesData.update(nodeUpdates);

                        // 2. Scale Edge Weights
                        const edgesData = network.body.data.edges;
                        const allEdges = edgesData.get();
                        const edgeUpdates = [];

                        allEdges.forEach(function(edge) {
                            let curEdgeSize = 10;
                            if (typeof edge.font === "object" && edge.font && edge.font.size) {
                                curEdgeSize = edge.font.size;
                            } else if (typeof edge.font === "number") {
                                curEdgeSize = edge.font;
                            }
                            edgeUpdates.push({
                                id: edge.id,
                                font: { size: Math.max(6, Math.min(40, curEdgeSize + delta)) }
                            });
                        });
                        edgesData.update(edgeUpdates);
                    }

                    // Toggle Weight Labels Visibility
                    function toggleWeightVisibility() {
                        weightsVisible = !weightsVisible;
                        const btn = document.getElementById("toggle-weights");
                        btn.innerText = weightsVisible ? "Weights: On" : "Weights: Off";
                        btn.style.background = weightsVisible ? "#e2e6ea" : "#f8f9fa";
                    
                        const edgesData = network.body.data.edges;
                        const allEdges = edgesData.get();
                        const updates = [];
                    
                        allEdges.forEach(function(edge) {
                            updates.push({
                                id: edge.id,
                                // A space " " reliably clears the Vis.js canvas text renderer without being ignored as empty
                                label: weightsVisible ? (edge.originalLabel || "") : " "
                            });
                        });
                        
                        edgesData.update(updates);
                    }

                    document.getElementById("font-inc").addEventListener("click", function(e) {
                        e.preventDefault();
                        adjustFontSize(+2);
                    });

                    document.getElementById("font-dec").addEventListener("click", function(e) {
                        e.preventDefault();
                        adjustFontSize(-2);
                    });

                    document.getElementById("toggle-weights").addEventListener("click", function(e) {
                        e.preventDefault();
                        toggleWeightVisibility();
                    });
                }
            });
        </script>
        """
    final_html = raw_html.replace("</body>", f"{overlay_script}</body>")
    return final_html, g

base_font = stss.get("text_size", 18)

with st.sidebar:
    # LAYOUT:
    clicked = st.button("**Layout Tools**",help="Use tools to get a 1st approximation before adjusting nodes.")
    if clicked:
        st.write("**Use tools to get a 1st approximation before adjusting nodes.**",)

    layout_choice = st.selectbox(
        label="**Select Algorithm**",
        options=stss.layouts,
        key="layout_selector"  # Streamlit uses this key to set the value
    )

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
    rotate = st.slider("**Rotate Clockwise (degrees)**", 0, 360, key="rotation", on_change=do_rotation)
    tight_divider()
    # show_weights = st.checkbox("**Show Link Weights**")
    # tight_divider()
    mx = max(focus_net.unique_weights)
    if focus_net.type == "mg":
        val = 1
        lab = "**Links in at least __ nets**"
    else:
        val = mx
        lab = "**Hide links by distance**"

    cutoff = st.select_slider(
            label=lab,
            options=focus_net.unique_weights,
            value=val,
            format_func=lambda x: "{:.4g}".format(x)
    )

    if focus_net.type == "mg":
        srces = f"bit code for nets in merge:  \n"
        for i, src in enumerate(focus_net.sources):
            b = f"{int(2 ** i):b}"
            srces += f"{src}: {b}  \n"
        st.write(srces)

    if redraw_pushed or layout_changed:
        focus_net.get_layout(method=layout_choice)
        stss.layout = layout_choice

st.write(f"**{focus_net.name}: {focus_net.nnodes} nodes, {focus_net.nlinks} links**: Layout {layout_choice}")

scale = 30.0/focus_net.nnodes
scale = np.clip(scale, a_min=1.2, a_max=2.0)
interactive_html, pyvis_net = create_visjs_html(fnet=focus_net,
                    font_size=round(scale*base_font), )

# Embed the HTML directly into the Streamlit app
# The height must be set to ensure the visualization is visible

if interactive_html:
    components.html(
        interactive_html,
        width="stretch",
        height=1200,  # Set height to match the '600px' defined in pyvis + physics
        scrolling=True
    )
else:
    st.error("Failed to generate interactive network HTML.")
