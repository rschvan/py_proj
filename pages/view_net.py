# pages/view_net.py
import streamlit as st
from pyvis.network import Network
from pypf import PFnet
from pypf.stutils import init_pf_session_state, tight_divider
from pypf.utility import split_long_term, sorted_keys
import numpy as np
import math
import time

# alias for st.session_state
stss = st.session_state

st.set_page_config(
    page_title="Net View",
    page_icon="🌐")

# set best scrolling position
# st.html("""
#     <script>
#         function scrollToTop() {
#             const doc = window.parent.document;
#             const container = doc.querySelector('section.main')
#                            || doc.querySelector('[data-testid="stMain"]')
#                            || doc.querySelector('[data-testid="stAppViewContainer"]')
#                            || window.parent;
#             if (container) {
#                 container.scrollTo({ top: 0, behavior: 'instant' });
#             }
#         }
#
#         // Run immediately and across the render lifecycle
#         scrollToTop();
#         [100, 300, 600, 1000, 1500].forEach(delay => setTimeout(scrollToTop, delay));
#     </script>
#     """)

if "col" not in stss:
    init_pf_session_state()

# focus_net = stss.col.focus_net # alias for easy reference

@st.cache_data(max_entries=20, show_spinner=False)
def cached_net_html(_fnet: PFnet, name: str, termsid: str, links_key: int,
                    coords_key: bytes, font_size: int) -> str:
    # Only _fnet is used; the other arguments are the cache key.
    html, _ = create_visjs_html(_fnet, font_size=font_size)
    return html

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

def create_visjs_html(fnet: PFnet, font_size=16) -> tuple[str, Network]:
    """Generates an HTML string for pyvis with proportional scaling,

    persistent edge weight labels, and client-side font & rank visibility
    controls.
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
    for i, term in enumerate(fnet.terms):
        label = split_long_term(term)
        x, y = fnet.coords[i]
        x = 650 * float(x)
        y = -450 * float(y)

        g.add_node(
            term,
            label=label,
            x=x,
            y=y,
            title=f"Node: {term}",
            shape="box",
            color="white",
            font={"size": font_size},
            margin=5 * global_scale,
            borderWidth=0 * global_scale,
        )

    # 2. Determine max rank safely
    max_rank = max(fnet.links.values()) if fnet.links else 1

    # 3. Add Edges (storing rank in originalLabel for client-side filtering)
    for (i, j), rank in fnet.links.items():
        s_node = fnet.terms[i]
        t_node = fnet.terms[j]
        wt_str = str(rank)

        g.add_edge(
            s_node,
            t_node,
            label=None,  # Initial state: invisible label
            originalLabel=wt_str,  # Preserved in Vis.js edge object
            width=1.5 * global_scale,
            arrows={"to": {"enabled": fnet.isdirected, "scaleFactor": arrow_scale}},
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

    # 4. Configure Vis.js Options
    g.options = {
        "physics": {
            "enabled": False,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -120,
                "centralGravity": 0.015,
                "springLength": 75,
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

    # 5. Pure string overlay (no f-string syntax conflict)
    overlay_script = """
        <!-- Floating Font, Visibility, & Rank Slider Controls -->
        <div id="font-controls" style="
            position: absolute;
            top: 15px;
            right: 15px;
            z-index: 9999;
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.92);
            padding: 6px 12px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.18);
            font-family: sans-serif;
            font-size: 13px;
        ">
            <button id="font-dec" title="Decrease Font" style="
                cursor: pointer;
                padding: 4px 8px;
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: #f8f9fa;
            ">A-</button>
            <button id="font-inc" title="Increase Font" style="
                cursor: pointer;
                padding: 4px 8px;
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: #f8f9fa;
            ">A+</button>
            <button id="toggle-weights" title="Toggle Edge Weights" style="
                cursor: pointer;
                padding: 4px 8px;
                font-size: 12px;
                font-weight: 500;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: #f8f9fa;
            ">Ranks: Off</button>

            <!-- Rank Threshold Slider -->
            <div style="display: flex; align-items: center; gap: 6px; margin-left: 6px; border-left: 1px solid #ddd; padding-left: 8px;">
                <label for="rank-slider" style="font-weight: 600; color: #333;">Max Rank:</label>
                <input type="range" id="rank-slider" min="1" max="__MAX_RANK__" value="__MAX_RANK__" step="1" style="cursor: pointer; width: 100px;">
                <span id="rank-display" style="font-weight: bold; min-width: 22px; color: #0066cc;">__MAX_RANK__</span>
            </div>
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

                    try {
                        const parentDoc = window.parent.document;
                        const scrollContainer = parentDoc.querySelector('section.main') 
                                             || parentDoc.querySelector('[data-testid="stMain"]')
                                             || parentDoc.querySelector('[data-testid="stAppViewContainer"]')
                                             || window.parent;
                        if (scrollContainer) {
                            scrollContainer.scrollTo({ top: 0, behavior: 'instant' });
                        }
                    } catch (e) {}

                    // Synchronized Font Scaling for Nodes & Edges
                    function adjustFontSize(delta) {
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
                        btn.innerText = weightsVisible ? "Ranks: On" : "Ranks: Off";
                        btn.style.background = weightsVisible ? "#e2e6ea" : "#f8f9fa";

                        const edgesData = network.body.data.edges;
                        const allEdges = edgesData.get();
                        const updates = [];

                        allEdges.forEach(function(edge) {
                            updates.push({
                                id: edge.id,
                                label: weightsVisible ? (edge.originalLabel || "") : " "
                            });
                        });
                        edgesData.update(updates);
                    }

                    // Filter Edge Visibility by Rank using originalLabel
                    function updateEdgeRankVisibility(maxAllowedRank) {
                        const edgesData = network.body.data.edges;
                        const allEdges = edgesData.get();
                        const updates = [];

                        allEdges.forEach(function(edge) {
                            const edgeRank = parseInt(edge.originalLabel, 10);
                            const shouldHide = !isNaN(edgeRank) && (edgeRank > maxAllowedRank);
                            updates.push({
                                id: edge.id,
                                hidden: shouldHide
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

                    const rankSlider = document.getElementById("rank-slider");
                    const rankDisplay = document.getElementById("rank-display");
                    rankSlider.addEventListener("input", function(e) {
                        const currentVal = parseInt(e.target.value, 10);
                        rankDisplay.innerText = currentVal;
                        updateEdgeRankVisibility(currentVal);
                    });
                }
            });
        </script>
        """

    # Safely replace placeholders without touching JavaScript braces
    overlay_script = overlay_script.replace("__MAX_RANK__", str(max_rank))

    final_html = raw_html.replace("</body>", f"{overlay_script}</body>")
    return final_html, g

base_font = stss.get("text_size", 18)

def new_net():
    new = stss.selected_net_name
    stss.col.focus_net = stss.col.pfnets[new]
    stss.layout_version += 1

if not stss.col.pfnets:
    st.info("No networks yet. Add proximity data on the **Build Nets** page.")
    st.stop()

#---------------------------------------------------------------------------------------------------------sidebar
with st.sidebar:
    # nets
    pfnet_names = sorted_keys(dic=stss.col.pfnets, )
    nnets = len(pfnet_names)
    if nnets == 1:
        stss.col.focus_net = stss.col.pfnets[pfnet_names[0]]
    elif nnets > 1:
        if stss.col.focus_net is None or stss.col.focus_net.name not in stss.col.pfnets:
            stss.col.focus_net = stss.col.pfnets[pfnet_names[0]]
        index = pfnet_names.index(stss.col.focus_net.name)
        st.selectbox(label="**Select Network** (searchable)",
                     options=pfnet_names,
                     index=index,
                     filter_mode="fuzzy",
                     key="selected_net_name",
                     on_change=new_net,
                     )
    tight_divider()
    focus_net = stss.col.focus_net

    # LAYOUT:
    st.write("**Layout Tools**")

    layout_choice = st.selectbox(
        label="**Select Algorithm**",
        options=stss.layouts,
        index=0,
        filter_mode="prefix",
        key=f"layout_selector_{stss.layout_version}"
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
            focus_net.coords[:, 0] *= -1

        if layout_choice in ['gravity', 'kamada_kawai', "davidson_harel", "spring"]:
            redraw_pushed = st.button("redraw")
        else:
            redraw_pushed = False
    rotate = st.slider("**Rotate Clockwise (degrees)**", 0, 360,
                       key="rotation", on_change=do_rotation)
    tight_divider()

    if redraw_pushed or layout_changed:
        focus_net.get_layout(method=layout_choice)
        stss.layout = layout_choice

#-------------------------------------------------------------------------------------------  MAIN PANEL

planarity = "planar" if focus_net.is_planar == True else "nonplanar"
st.write(f"**{focus_net.name}: {focus_net.nnodes} nodes, "
         f"{focus_net.nlinks} links** -- {planarity} -- Layout: {focus_net.layout_method} ")

scale = 30.0 / focus_net.nnodes
scale = np.clip(scale, a_min=1.2, a_max=2.0)
# interactive_html, pyvis_net = create_visjs_html(fnet=focus_net,
#                                                 font_size=round(scale * base_font), )
font_size = round(scale * base_font)
interactive_html = cached_net_html(
    focus_net,
    name=focus_net.name,
    termsid=focus_net.termsid,
    links_key=hash(tuple(sorted(focus_net.links.items()))),
    coords_key=focus_net.coords.tobytes(),
    font_size=font_size,
)

# Embed the HTML directly into the Streamlit app
# The height must be set to ensure the visualization is visible
if interactive_html:
    st.iframe(
        interactive_html,
        width="stretch",
        height=1200,  # room for the 600px network plus the physics settings panel
    )
else:
    st.error("Failed to generate interactive network HTML.")
