# stapp.py: streamlit interface
import sys
import os
import streamlit as st
# --- Streamlit Application Layout (MUST BE FIRST Streamlit command) ---
st.set_page_config(layout="wide")

from pypf.netpic import Netpic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
from pypf.collection import Collection
from pypf.proximity import Proximity
from pypf.pfnet import PFnet
from pypf.utility import sample_data, file_format
import time

# Get the absolute path of the directory containing 'stapp.py' (which should be 'py_proj')
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Global Initialization (runs once per user session) ---
@st.cache_resource
def get_collection_instance():
    """Initializes and returns a Collection instance and sample data.
    Cached to persist across reruns for the same session.
    """
    sample = sample_data()
    sample_col:Collection = sample["collection"] # Collection with sample Proximitys and PFnets
    sample_sources:pd.DataFrame = sample["sources"] # DataFrame with sources of sample Proximities
    prx_file_format:pd.DataFrame = file_format() # DataFrame with sample .xlsx proximity file contents
    return sample_col, sample_sources, prx_file_format

def make_fig(font_size=10):
    pic = st.session_state.pic
    fig = pic.create_view(font_size=font_size)
    fig.canvas.mpl_connect('button_press_event', pic._on_press)
    fig.canvas.mpl_connect('motion_notify_event', pic._on_motion)
    fig.canvas.mpl_connect('button_release_event', pic._on_release)
    st.session_state.fig = fig

def add_demo(col):
    for px in sample_col.proximities.values():
        col.add_proximity(px)
    for pf in sample_col.pfnets.values():
        col.add_pfnet(pf)

def del_demo(col):
    for demoprx in sample_col.proximities.keys():
        if demoprx in col.proximities:
            col.proximities.pop(demoprx)
    for demopf in sample_col.pfnets.keys():
        if demopf in col.pfnets:
            del col.pfnets[demopf]

# get app data
sample_col, sample_sources, prx_file_format = get_collection_instance()

# initialize session state (runs once per user session)
if 'pf_name' not in st.session_state:
    st.session_state.col = copy.deepcopy(sample_col)
    col = st.session_state.col
    st.session_state.pf_name = col.pfnets["bank6_pf"].name
    st.session_state.layout = "kamada_kawai"
    st.session_state.font_size = 10

    # Initial figure creation
    pf = col.pfnets[st.session_state.pf_name]
    pf.get_layout(method=st.session_state.layout)
    st.session_state.pic = Netpic(pf)
    make_fig(font_size=st.session_state.font_size)
    st.session_state.lastlo = st.session_state.layout
    st.session_state.new_layout = True
    st.session_state.count = 0
    st.session_state.q_param = np.inf
    st.session_state.r_param = np.inf

col = st.session_state.col # alias for easy reference

# --- Sidebar for Global Actions ---
with st.sidebar:
    st.subheader("App Controls")
    show_intro_info = st.checkbox("Intro Info", value=True, key="show_intro_info")
    show_corrs = st.checkbox("Proximity Correlations", value=False, key="show_corrs")
    show_net_info = st.checkbox("Network Info", value=False, key="show_net_info")
    show_netsim = st.checkbox("Network Similarity", value=False, key="show_netsim")
    ave_method = st.radio("Averaging Method",["mean","median"], index=0, )
    #st.markdown("")
    net_type = st.radio("Network Type",["Pathfinder","Threshold", "Nearest Neighbor"],
                        index=0, )
    if net_type == "Pathfinder":
        qv, rv = st.columns(2)
        with qv:
            qval = st.text_input("q-value", value="inf", key="qval")
        with rv:

            rval = st.text_input("r-value", value="inf", key="rval")
        qval = qval.lower()
        if qval == "" or qval == "inf" or qval.__contains__("n"):
            st.session_state.q_param = np.inf
        else:
            try:
                st.session_state.q_param = int(qval)
            except:
                st.session_state.q_param = np.inf
                qval = "inf"
        rval = rval.lower()
        if rval == "" or rval == "inf":
            st.session_state.r_param = np.inf
        else:
            try:
                st.session_state.r_param = float(rval)
            except:
                st.session_state.r_param = np.inf
                rval = "inf"
    st.write("layout:")

    fx, fy, sxy = st.columns(3)
    with fx:
        if st.button("flipx"):
            st.session_state.fig.axes[0].invert_xaxis()
    with fy:
        if st.button("flipy"):
            st.session_state.fig.axes[0].invert_yaxis()
    with sxy:
        if st.button("swapxy"):
            st.session_state.pic.net.coords[:, [0, 1]] = st.session_state.pic.net.coords[:, [1, 0]]
            st.session_state.fig = st.session_state.pic.create_view(font_size=st.session_state.font_size)

    #@st.fragment
    def toggle_weights():
        st.session_state.pic.toggle_weights()

    #@st.fragment
    def change_font_size():
        st.session_state.pic.change_font_size(st.session_state.font_size)

    tw, red = st.columns(2)
    with tw:
        if st.button('toggle weights'):
            toggle_weights()

        st.number_input("font size", min_value=5, max_value=20,step=1, key="font_size",
                        on_change=change_font_size)
    with red:
        if st.button("redraw net"):
            st.session_state.new_layout = True

        st.selectbox("layout method", ["kamada_kawai", "dot", "neato", "circo", "gravity", "spring",
                                       "distance"], index=0, key="layout",)

        if st.session_state.lastlo != st.session_state.layout:
            st.session_state.lastlo = st.session_state.layout
            st.session_state.new_layout = True

# --- Main Content Area ---

st.title("PyPathfinder")

# Intro Info
if show_intro_info:
    st.subheader("Welcome to PyPathfinder!")
    st.link_button("Wikipedia Page", "https://en.wikipedia.org/wiki/Pathfinder_network")
    st.info("Intro info adds an example of the required Proximity file format and sample "
            "Proximities and PFnets for demo purposes.  The examples will be deleted "
            "when you uncheck the Intro Info box.  "
            "The examples will be added back when you check the Intro Info box again. "
            'Anything you add will be preserved when you uncheck the Intro Info box. '
            "Use the examples to explore the functionality of the application.  Load your own "
            "data files whenever you like.  Enjoy!"
    )

# ---Upload Files and Create Proximities---
uploaded_files =  None
st.subheader("Add Proximity Files")
uploaded_files = st.file_uploader("Upload Proximity Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True,
                        help="Upload one or more Proximity Excel files (.xlsx) to add to the collection.")
for uploaded_file in uploaded_files:
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
            os.remove(temp_file_path)
        except Exception as e:
            st.error(f"Error processing file: {e}")
uploaded_files = []

# ---show intro if checked---
if show_intro_info:
    add_demo(col)
    st.subheader("Required .xlsx Proximity File Format")
    st.info("Example file ( bank6.prx.xlsx ): terms on Rows and Columns.  Distance values in Matrix.")
    st.dataframe(prx_file_format, use_container_width=False, hide_index=False)
else:
    del_demo(col)

st.subheader("Proximity Information")
prx_info_df = col.get_proximity_info()
if not prx_info_df.empty:
    st.dataframe(prx_info_df, use_container_width=False, hide_index=True)
else:
    st.info("No Proximity objects loaded yet.")

# ---conditional displays---
if show_intro_info:
    st.dataframe(sample_sources, use_container_width=False, hide_index=True)

if show_corrs and len(col.proximities) > 1:
    st.subheader("Proximity Correlations")
    prx_corrs_df = col.get_proximity_correlations()
    if not prx_corrs_df.empty:
        st.dataframe(prx_corrs_df, use_container_width=False, hide_index=False)

if show_net_info and len(col.pfnets) > 0:
    st.subheader("Network Information")
    pfnet_info_df = col.get_pfnet_info()
    if not pfnet_info_df.empty:
        st.dataframe(pfnet_info_df, use_container_width=False, hide_index=True)
    else:
        st.info("No PFnet objects loaded yet.")

if show_netsim and len(col.pfnets) > 1:
    st.subheader("Network Similarity")
    sim_df = col.network_similarity()
    if not sim_df.empty:
        st.dataframe(sim_df, use_container_width=False, hide_index=False)
    else:
        st.info("No PFnet objects loaded yet.")

# ---Proximity and Network Lists
prxlist, netlist = st.columns(2)
with prxlist:
    st.subheader("Proximity List")
    prx_names = list(col.proximities.keys())
    if not prx_names:
        st.info("No Proximity objects loaded yet.")
    else:
        st.info("Click in first column (not Name) to select:")
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
        if selected_prxs_data.selection.rows:
            col.selected_prxs = [prx_names[i] for i in selected_prxs_data.selection.rows]
            # st.write(f"Selected Proximities: {col.selected_prxs}")
        else:
            col.selected_prxs = []

        delprx, aveprx, dernet = st.columns(3)
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
            if st.button("Derive Networks", key="derive_pfnet_btn"):
                if len(col.selected_prxs) > 0:
                    for prx_name in col.selected_prxs:
                        # Check if PFnet with this name already exists before deriving
                        if prx_name in col.pfnets:
                            st.warning(f"PFnet '{prx_name}' already exists. Not re-deriving.")
                        else:
                            prx = col.proximities[prx_name]
                            match net_type:
                                case "Pathfinder":
                                    pf = PFnet(prx, q=st.session_state.q_param, r=st.session_state.r_param)
                                case "Threshold":
                                    pf = PFnet(prx, type="th")
                                case "Nearest Neighbor":
                                    pf = PFnet(prx, type="nn")
                            col.add_pfnet(pf)
                else:
                    st.warning("Please select at least one proximity to derive networks.")
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

with netlist:
    st.subheader("Network List")
    pfnet_names = list(col.pfnets.keys())
    if not pfnet_names:
        st.info("No PFnet objects loaded yet.")
    else:
        st.info("Clicking at the top selects or deselects all:")
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
                    #st.success("Selected PFnets deleted.")
                    st.rerun()  # Rerun
                else:
                    st.warning("Please select at least one network to delete.")
        with mrgnet:
            if st.button("Merge Networks", key="merge_nets_btn"):
                if len(col.selected_nets) >= 2:
                    try:
                        # Assuming col.merge_networks() handles adding the merged net to col.pfnets
                        col.merge_networks()
                        #st.success("Selected PFnets merged.")
                        st.rerun()  # Rerun
                    except Exception as e:
                        st.error(f"Error merging networks: {e}")
                else:
                    st.warning("Please select at least two PFnets to merge.")
        with dispnet:
            if st.button("Display Network", key="display_net_btn"):
                if len(col.selected_nets) == 1:
                    st.session_state.new_layout = True
                    st.session_state.pf_name = col.selected_nets[0]
                else:
                    st.warning("Please select one network to display.")

# --- net Visualization
# This block will execute if pf_name is present
if st.session_state.pf_name and st.session_state.pf_name in col.pfnets:
    pf = col.pfnets[st.session_state.pf_name]
    if st.session_state.new_layout:
        pf.get_layout(method=st.session_state.layout)
        st.session_state.pic = Netpic(net=pf)
        make_fig(font_size=st.session_state.font_size)

    st.write(f"{pf.name}: {pf.nnodes} nodes {pf.nlinks} links, "
              f"using {st.session_state.layout} layout {st.session_state.count}")

    st.session_state.view = st.pyplot(fig=st.session_state.fig, use_container_width=False)

    st.session_state.new_layout = False
    st.session_state.count += 1
else:
    st.warning("No network created.")
