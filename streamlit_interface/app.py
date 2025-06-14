# streamlit_interface/app.py
import sys
import os

# Get the absolute path of the directory containing 'app.py'
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (which should be 'py_proj')
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Streamlit Application Layout (MUST BE FIRST Streamlit commands) ---
import streamlit as st  # Keep this at the top

st.set_page_config(layout="wide")
st.title("PyPathfinder Web")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib
import networkx as nx  # Import networkx

# Adjust this import based on your actual project structure.
# If 'pypf' is a package, and 'collection.py' is inside it, this should work.
# Ensure your project's root directory (py_proj) is accessible to Python.
try:
    from pypf.collection import Collection
    from pypf.proximity import Proximity
    from pypf.pfnet import PFnet
    from pypf.utility import mypfnets_dir  # This function should correctly point to pypf/data
except ImportError as e:
    st.error(f"Could not import pypf modules. Please ensure your project structure is correct and "
             f"the pypf package is accessible. Error: {e}")
    st.stop()  # Stop the app if crucial imports fail

# --- Global Initialization (runs once per user session) ---
@st.cache_resource
def get_collection_instance():
    """Initializes and returns a Collection instance.
    Cached to persist across reruns for the same session.
    """
    # Initialize Collection with to_csv=False if you don't want automatic CSV output
    col = Collection(to_csv=False)  # Changed to_csv based on user's new class property

    # Example: Add some initial data if available for testing
    try:
        data_dir = col.mypfnets_dir

        psy_prx_path = os.path.join(data_dir, "psy.prx.xlsx")
        bio_prx_path = os.path.join(data_dir, "bio.prx.xlsx")

        if os.path.exists(psy_prx_path):
            #st.info(f"Loading {os.path.basename(psy_prx_path)}...")
            px = Proximity(psy_prx_path)
            col.add_proximity(px)
        else:
            st.warning(f"Psychology proximity data not found at: {psy_prx_path}")

        if os.path.exists(bio_prx_path):
            #st.info(f"Loading {os.path.basename(bio_prx_path)}...")
            bx = Proximity(bio_prx_path)
            col.add_proximity(bx)
        else:
            st.warning(f"Biology proximity data not found at: {bio_prx_path}")

        # Add initial PFnets if desired, for testing
        if col.proximities:
            for prx_name, prx_obj in col.proximities.items():
                # Only derive PFnet if it doesn't already exist to avoid errors on rerun
                if prx_name not in col.pfnets:
                    pf = PFnet(prx_obj)
                    col.add_pfnet(pf)
                    #st.info(f"Derived PFnet '{pf.name}' from '{prx_name}'.")

    except Exception as e:
        st.warning(f"Could not load example data or derive initial PFnets: {e}")
    return col

col = get_collection_instance()
r_param = np.inf
q_param = np.inf

# --- Sidebar for Navigation/Global Actions ---
with st.sidebar:
    st.subheader("Display?")
    show_corrs = st.checkbox("Proximity Correlations", value=False, key="show_corrs")
    show_net_info = st.checkbox("Network Info", value=False, key="show_net_info")
    show_netsim = st.checkbox("Network Similarity", value=False, key="show_netsim")
    #st.markdown("---")
    ave_method = st.radio("Averaging Method",["mean","median"], index=0)
    #st.markdown("---")
    net_type = st.radio("Network Type",["Pathfinder","Threshold", "Nearest Neighbor"], index=0)
    if net_type == "Pathfinder":
        qval = st.text_input("q-value", value="inf", key="qval")
        rval = st.text_input("r-value", value="inf", key="rval")
        # qval = st.number_input("q-value", min_value=2, max_value=10000, value=10000, step=1, key="qval")
        # rval = st.number_input("r-value", min_value=1.0, max_value=10000.0, value=10000.0, step=0.1, key="rval")
        qval = qval.lower()
        if qval == "" or qval == "inf" or qval.__contains__("n"):
            q_param = np.inf
        else:
            try:
                q_param = int(qval)
            except:
                q_param = np.inf
                qval = "inf"
        rval = rval.lower()
        if rval == "" or rval == "inf":
            r_param = np.inf
        else:
            try:
                r_param = float(rval)
            except:
                r_param = np.inf
                rval = "inf"
    #st.markdown("---")
    layout_method = st.selectbox("Layout Method", ["gravity", "dot", "kamada_kawai", "spring"], index=0)

# --- Main Content Area ---

uploaded_file = st.file_uploader("Upload Proximity Excel (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    try:
        # Save uploaded file temporarily for processing
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        new_prx = Proximity(temp_file_path)
        col.add_proximity(new_prx)
        st.success(f"Proximity '{new_prx.name}' added successfully!")
        os.remove(temp_file_path)
    except Exception as e:
        st.error(f"Error processing file: {e}")

st.subheader("Proximity Information")
prx_info_df = col.get_proximity_info()
if not prx_info_df.empty:
    st.dataframe(prx_info_df, use_container_width=False, hide_index=True)
else:
    st.info("No Proximity objects loaded yet.")

if show_corrs:
    st.subheader("Proximity Correlations")
    prx_corrs_df = col.get_proximity_correlations()
    if not prx_corrs_df.empty:
        st.dataframe(prx_corrs_df, use_container_width=False, hide_index=False)

if show_net_info:
    st.subheader("Network Information")
    pfnet_info_df = col.get_pfnet_info()
    if not pfnet_info_df.empty:
        st.dataframe(pfnet_info_df, use_container_width=False, hide_index=True)
    else:
        st.info("No PFnet objects loaded yet.")

if show_netsim:
    st.subheader("Network Similarity")
    sim_df = col.network_similarity()
    if not sim_df.empty:
        st.dataframe(sim_df, use_container_width=False, hide_index=False)
    else:
        st.info("No PFnet objects loaded yet.")

prxcol, netcol = st.columns(2)
with prxcol:  # Proximity List
    st.subheader("Proximity List")
    prx_names = list(col.proximities.keys())
    if not prx_names:
        st.info("No Proximity objects loaded yet. Upload one from the sidebar or ensure example data loads.")
    else:
        st.info("Click in first column to select:")
        prx_df = pd.DataFrame({"Name": prx_names})

        selected_prxs_data = st.dataframe(
            prx_df,
            use_container_width=True,
            height=400,  # Set a fixed height to make it look more like a listbox
            hide_index=True,  # Hide index as 'Name' column is present
            selection_mode=["multi-row"],  # Enable multi-row selection, using working syntax
            key="prx_selection_df",
            on_select="rerun"  # Crucial for getting selection updates
        )

        # Direct retrieval of selected names now that on_select is used
        if selected_prxs_data.selection.rows:
            col.selected_prxs = [prx_names[i] for i in selected_prxs_data.selection.rows]
        else:
            col.selected_prxs = []

        delprx, dernet, aveprx = st.columns(3)
        with delprx:
            if st.button("Delete Proximities", key="delete_prx_btn"):
                if col.selected_prxs:
                    for prx_name in col.selected_prxs:
                        col.proximities.pop(prx_name)
                    col.selected_prxs = []  # Clear selection after deletion
                    st.success("Selected proximities deleted.")
                    st.rerun()  # Rerun to update the display
                else:
                    st.warning("Please select at least one proximity to delete.")
        with dernet:
            if st.button("Derive Network", key="derive_pfnet_btn"):
                if len(col.selected_prxs) == 1:
                    prx_name = col.selected_prxs[0]
                    # Check if PFnet with this name already exists before deriving
                    if prx_name in col.pfnets:
                        st.warning(f"PFnet '{prx_name}' already exists. Not re-deriving.")
                    else:
                        prx = col.proximities[prx_name]
                        match net_type:
                            case "Pathfinder":
                                pf = PFnet(prx, q=q_param, r=r_param)
                            case "Threshold":
                                pf = PFnet(prx, type="th")
                            case "Nearest Neighbor":
                                pf = PFnet(prx, type="nn")
                        col.add_pfnet(pf)
                        st.success(f"PFnet '{pf.name}' derived from '{prx_name}'.")
                    st.rerun()  # Rerun
                else:
                    st.warning("Please select exactly one proximity to derive a PFnet.")
        with aveprx:
            if st.button("Average Proximities", key="average_prx_btn"):
                if len(col.selected_prxs) >= 2:
                    try:
                        col.average_proximities(choice=ave_method)
                        st.success("Proximities averaged.")
                        st.rerun()  # Rerun
                    except Exception as e:
                        st.error(f"Error averaging proximities: {e}")
                else:
                    st.warning("Please select at least two proximities to average.") ##

with netcol:
    st.subheader("Network List")

    pfnet_names = list(col.pfnets.keys())
    if not pfnet_names:
        st.info("No PFnet objects loaded yet. Derive one from Proximities above.")
    else:
        st.info("Click in first column to select:")
        pfnet_df = pd.DataFrame({"Name": pfnet_names})

        selected_nets_data = st.dataframe(
            pfnet_df,
            use_container_width=True,
            height=400,  # Set a fixed height
            hide_index=True,  # Hide index
            selection_mode=["multi-row"],  # Enable multi-row selection, using working syntax
            key="pfnet_selection_df",
            on_select="rerun"  # Crucial for getting selection updates
        )

        # Direct retrieval of selected names now that on_select is used
        if selected_nets_data.selection.rows:
            col.selected_nets = [pfnet_names[i] for i in selected_nets_data.selection.rows]
        else:
            col.selected_nets = []

        delnet, mrgnet, dispnet = st.columns(3)
        with delnet:
            if st.button("Delete Networks", key="delete_pfnet_btn"):
                if col.selected_nets:
                    for net_name in col.selected_nets:
                        col.pfnets.pop(net_name)
                    col.selected_nets = []  # Clear selection after deletion
                    st.success("Selected PFnets deleted.")
                    st.rerun()  # Rerun
                else:
                    st.warning("Please select at least one PFnet to delete.")
        with mrgnet:
            if st.button("Merge Networks", key="merge_nets_btn"):
                if len(col.selected_nets) >= 2:
                    try:
                        # Assuming col.merge_networks() handles adding the merged net to col.pfnets
                        col.merge_networks()
                        st.success("Selected PFnets merged.")
                        st.rerun()  # Rerun
                    except Exception as e:
                        st.error(f"Error merging networks: {e}")
                else:
                    st.warning("Please select at least two PFnets to merge.")
        with dispnet:
            # Move the "Get Network Info" logic here as the button is in col_c
            if st.button("Display Network", key="display_net_btn"):
                if len(col.selected_nets) == 1:
                    pf_name = col.selected_nets[0]
                    pf = col.pfnets[pf_name]
                else:
                    st.warning("Please select exactly one PFnet to display.")

# --- net Visualization
# This block will execute if Display Net button was clicked AND a single net was selected
if (col.selected_nets and len(col.selected_nets) == 1 and
        st.session_state.get("display_net_btn", False)):
    pf_name = col.selected_nets[0]
    pf = col.pfnets[pf_name]

    st.info(f"Display of {pf_name} using {layout_method}:  {pf.nnodes} nodes  {pf.nlinks} links")
    #st.info(f"Layout Method: {layout_method}")
    fig, ax = plt.subplots(figsize=(10, 8))  # Increased figure size for better visibility
    seed = np.random.randint(1000000)
    randpos = nx.random_layout(pf.graph, seed=seed)
    ecc = pf.eccentricity if pf.eccentricity else None
    root = ecc["center"] if ecc else None
    #st.info(f"Root node(s): {root}")

    match layout_method:
        case "gravity":
            pos = nx.forceatlas2_layout(pf.graph, strong_gravity=True, seed=seed, pos=randpos)
        case "dot":
            pos = nx.drawing.nx_pydot.graphviz_layout(pf.graph, prog='dot', root=root)
        case "kamada_kawai":
            pos = nx.kamada_kawai_layout(pf.graph, pos=randpos)
        case "spring":
            pos = nx.spring_layout(pf.graph, pos=randpos, seed=seed)
        case _:
            pos = nx.forceatlas2_layout(pf.graph, strong_gravity=True, seed=seed, pos=randpos)
    #st.info(f"layout: {pos}")

    # Draw the graph
    nx.draw(pf.graph, pos, ax=ax, with_labels=True, node_color='white',
            node_size=600, edge_color='blue', font_size=9, font_weight='normal',)

    # Optional: Draw edge labels if you have weights (e.g., if edges are tuples (u, v, {'weight': w}))
    # edge_labels = nx.get_edge_attributes(pf.graph, 'weight')
    # nx.draw_networkx_edge_labels(pf.graph, pos, edge_labels=edge_labels, ax=ax)

    st.pyplot(fig)  # Display the matplotlib figure in Streamlit
    plt.close(fig)  # Close the figure to free up memory