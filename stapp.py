# stapp.py: streamlit interface
import math
import sys
import os
import streamlit as st
from pypf.netpic import Netpic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
from pypf.collection import Collection
from pypf.proximity import Proximity
from pypf.pfnet import PFnet
from pypf.read_legacy_files import legacy_prx_files
import time

def sample_data() -> dict:
    files = ["bank6","bio", "psy", "bank", "cities", "statecaps"]
    srcs = ["coocurrences in Longman's dictionary",
        "relatedness ratings by biology grad students",
         "relatedness ratings by psychology 101 students",
         "coocurrences in Longman's dictionary",
         "distances between cities in the US",
         "distances between state capitals in the US"]
    sources = pd.DataFrame(data={"file": files, "source": srcs})
    col = Collection()
    for file in files:
        fn = os.path.join("pypf","data", file + ".prx.xlsx")
        px = Proximity(fn)
        col.add_proximity(px)
        pf = PFnet(px)
        col.add_pfnet(pf)
        if file in ["bio", "psy"]:
            pf = PFnet(px, q=2)
            col.add_pfnet(pf)
        if file in ["statecaps"]:
            pf = PFnet(px, q=2, r=2)
            col.add_pfnet(pf)
    dic = {"collection": col, "sources": sources, }
    return dic

#
# --- Streamlit Application Layout (MUST BE FIRST Streamlit command) ---
st.set_page_config(layout="wide", page_title="PyPathfinder", page_icon="🏠")

home_dir = os.path.dirname(os.path.abspath(__file__))
home_page = st.Page(page=os.path.join(home_dir,"stapp.py"), title="Build Nets", icon="🏠")
help_page = st.Page(page=os.path.join(home_dir,"pages","help.py"), title="Help", icon="❓")
display_page = st.Page(page=os.path.join(home_dir,"pages","display_network.py"), title="Select Display", icon="🌐")
move_page = st.Page(page=os.path.join(home_dir,"pages","move_nodes.py"), title="Interactive Display", icon="🌐")
pages = [home_page, display_page, move_page, help_page]
page_selected = st.navigation(pages=pages, position="sidebar")
if page_selected != home_page:
    page_selected.run()
else:
    st.title("PyPathfinder")

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
        return sample_col, sample_sources

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

    sample_col, sample_sources = get_collection_instance()

    # initialize session state (runs once per user session)
    if 'pf_name' not in st.session_state:
        st.session_state.home_dir = os.path.dirname(os.path.abspath(__file__))
        st.session_state.col = copy.deepcopy(sample_col)
        col = st.session_state.col
        st.session_state.pf_name = col.pfnets["bank6_pf"].name
        st.session_state.layouts = ["kamada_kawai", "neato", "force", "gravity", "spring", "spiral", "circle", "MDS"]
        st.session_state.layout_index = 0
        st.session_state.layout = st.session_state.layouts[st.session_state.layout_index]
        st.session_state.last_layout = st.session_state.layout
        st.session_state.font_size = 10
        st.session_state.rotation = 0
        st.session_state.intro_info = True
        pf = col.pfnets[st.session_state.pf_name]
        pf.get_layout(method=st.session_state.layout)
        st.session_state.pic = Netpic(net=pf)
        st.session_state.pic.create_view(font_size=st.session_state.font_size)
        st.session_state.count = 0
        st.session_state.q_param = np.inf
        st.session_state.r_param = np.inf
        st.session_state.current_intro_value = True
        st.session_state.current_file_type_index = 0
        st.session_state.last_leg_files = []
        st.session_state.last_ss_files = []

    col = st.session_state.col # alias for easy reference
    st.session_state.count += 1
    # --- Sidebar for Global Actions ---
    with st.sidebar:

        show_intro_info = st.checkbox("**Intro Info**", value= st.session_state.current_intro_value,
                                      key="show_intro_info",)
        st.session_state.current_intro_value = show_intro_info
        file_type = st.radio("**Proximity Data File Type**",["Spreadsheet","Legacy Text"],
                             index=st.session_state.current_file_type_index, )
        st.session_state.current_file_type_index = ["Spreadsheet","Legacy Text"].index(file_type)
        st.write("**Optional Information:**")
        show_corrs = st.checkbox("Proximity Correlations", value=False, key="show_corrs")
        show_net_info = st.checkbox("Network Info", value=False, key="show_net_info")
        show_netsim = st.checkbox("Network Similarity", value=False, key="show_netsim")
        ave_method = st.radio("**Proximity Averaging Method**",["mean","median"], index=0, )

        type, params = st.columns([6, 4])
        with type:
            net_type = st.radio("**Network Type**",["Pathfinder","Threshold", "Nearest Neighbor"],
                            index=0, )
        with params:
            if net_type == "Pathfinder":
                qval = st.text_input("q-value (2 - inf)", value="inf", key="qval")
                rval = st.text_input("r-value (1.0 - inf)", value="inf", key="rval")
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

    # --- Main Content Area ---

    # Intro Info

    if show_intro_info:
        if "bank6_pf" not in col.pfnets and "statecaps" not in col.proximities:
            add_demo(col)

        if st.button("PyPathfinder Demo Video", key="vid_button"):
            st.video("pypf/data/video_demo.mp4")
        st.write('''**Welcome!**  
PyPathfinder is a tool for creating, exploring, and visualizing Pathfinder networks. 
The app contains four pages, **Build Nets** (this page), **Select Display**, **Interactive Display**
and **Help**. You can navigate to a page by clicking the page name in the sidebar.
  
The **Build Nets** page is where you add Proximity data 
and Derive Networks. You can choose to load either spreadsheet Proximity files or text files 
using the legacy formatting (Help page gives details). You can also view Proximity Correlations, 
Network Information and Network Similarity 
by checking the boxes in the sidebar. Compatible Proximity Data sets can be averaged, and different
types of networks can be derived. Pathfinder networks require q and r values to be specified. The default
values of infinity yield the network with the minimum number of links. It will be a minimal spanning
tree (MST) if the network is connected and there is a unique MST. Ties in link distances may lead to 
cycles in the minimal network.

The **Select Display** page allows you to try different layouts and view the network in various ways. When 
you have a layout that is close to what you want, you can go to the **Interactive Display** page to adjust
the node positions.


The **Intro Info checkbox** makes starting information available including access to a demo video 
and several sample Proximities and Networks to illustrate the app and to allow you to explore 
the app's functionality.  Unchecking this checkbox removes the introductory information.  Checking it
again will add the introductory information back.

Go to the **Help** page for more information.  
        ''')
    else:
        del_demo(col)

    # ---Upload Files and Create Proximities---

    match file_type:
        case "Spreadsheet":
            st.subheader("Add Proximity Spreadsheet Files")
            ss_files = []
            ss_files = st.file_uploader("Upload Proximity Spreadsheets (.xlsx or .csv)", type=["xlsx","csv"],
                                accept_multiple_files=True,
                                help="Upload one or more Proximity files (.xlsx or .csv) to add to the app.",
                                )
            #st.write(f"nssfiles: {len(ss_files)}")
            if len(ss_files) > 0 and ss_files != st.session_state.last_ss_files:
                st.session_state.last_ss_files = ss_files
                ss_dir = "temp_ss"
                os.makedirs(ss_dir, exist_ok=True)
                for ss_file in ss_files:
                    if ss_file is not None:
                        try:
                            # Save uploaded file temporarily for processing
                            temp_file_path = os.path.join(ss_dir, ss_file.name)
                            with open(temp_file_path, "wb") as f:
                                f.write(ss_file.getbuffer())
                        except Exception as e:
                            st.error(f"Error processing file: {e}")
                files = os.listdir(path=ss_dir)
                for file in files:
                    file_path = os.path.join(ss_dir, file)
                    new_prx = Proximity(file_path)
                    col.add_proximity(new_prx)
                    os.remove(file_path)
                os.rmdir(ss_dir)
            else:
                ss_files.clear()

        case "Legacy Text":
            st.subheader("Add Legacy Proximity and Terms Text Files")
            leg_files = st.file_uploader("Upload Legacy Proximity and Terms Text Files (.txt)", type=["txt"],
                                        accept_multiple_files=True,
                                        help="Upload one or more files (.txt) to add to the app.",
                                        key="leg_files_key",)
            if len(leg_files) > 0 and leg_files != st.session_state.last_leg_files:
                st.session_state.last_leg_files = leg_files
                leg_dir = "temp_leg"
                os.makedirs(leg_dir, exist_ok=True)
                for leg_file in leg_files:
                    if leg_file is not None:
                        file = os.path.join(leg_dir, leg_file.name)
                        with open(file, "wb") as f:
                            f.write(leg_file.getbuffer())
                all_leg_files = os.listdir(leg_dir)
                file_dat = legacy_prx_files(path=leg_dir, files=all_leg_files)
                for dat in file_dat:
                    if dat["error"]:  # does not contain valid proximity data
                        continue
                    new_prx = Proximity(terms=dat["terms"], dismat=dat["dismat"], name=dat["name"])
                    col.add_proximity(new_prx)
                for file in all_leg_files:
                    os.remove(os.path.join(leg_dir, file))
                os.rmdir(leg_dir)
            else:
                leg_files.clear()

    st.subheader("Proximity Information")
    prx_info_df = col.get_proximity_info()
    height = min(600, 40 + 35 * len(col.proximities))
    if not prx_info_df.empty:
        st.dataframe(prx_info_df, width='content', height=height, hide_index=True)
    else:
        st.info("No Proximity objects.")

    # ---conditional displays---
    if show_intro_info:
        current_sources = sample_sources.copy(deep=True)
        allowed_keys = col.proximities.keys()
        rows_to_keep_mask = current_sources["file"].isin(allowed_keys)
        current_sources = current_sources[rows_to_keep_mask]
        if  not current_sources.empty:
            st.subheader("Sources of sample Proximities")
            st.dataframe(current_sources, width='content', hide_index=True)

    if show_corrs and len(col.proximities) > 1:
        st.subheader("Proximity Correlations")
        st.info("""Correlations of the distance data in pairs of equal sized data sets""")
        prx_corrs_df = col.get_proximity_correlations()
        if not prx_corrs_df.empty:
            st.dataframe(prx_corrs_df, width='content', hide_index=False)

    if show_net_info and len(col.pfnets) > 0:
        st.subheader("Network Information")
        pfnet_info_df = col.get_pfnet_info()
        if not pfnet_info_df.empty:
            st.dataframe(pfnet_info_df, width='content', hide_index=True)
        else:
            st.info("No PFnet objects.")

    if show_netsim and len(col.pfnets) > 1:
        st.subheader("Network Similarity")
        st.info("""Similarity is the number of shared links 
            divided by the number of unique links in two networks.""")
        sim_df = col.network_similarity()
        if not sim_df.empty:
            st.dataframe(sim_df, width='content', hide_index=False)
        else:
            st.info("No PFnet objects.")

    # ---Proximity and Network Lists
    prxlist, netlist = st.columns(2)
    with prxlist:
        st.subheader("Proximity List")
        prx_names = list(col.proximities.keys())
        if not prx_names:
            st.info("No Proximity objects.")
        else:
            st.write("Click in the box left of Name column to select:")
            height = min(600, 40 + 35*len(col.proximities))
            prx_df = pd.DataFrame({"Name": prx_names})
            selected_prxs_data = st.dataframe(
                prx_df,
                width='stretch',
                height=height,  # Set a height to reflect contents
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
                        st.rerun()  # Rerun to update the display
                    else:
                        st.warning("Click one or more proximity check boxes to delete.")
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
                        st.warning("Click one or more proximity check boxes to derive networks.")
            with aveprx:
                if st.button("Average Proximities", key="average_prx_btn"):
                    if len(col.selected_prxs) >= 2:
                        try:
                            col.average_proximities(method=ave_method)
                            st.success("Proximities averaged.")
                            st.rerun()  # Rerun
                        except Exception as e:
                            st.error(f"Error averaging proximities: {e}")
                    else:
                        st.warning("Click two or more proximity check boxes to average.") ##

    with netlist:
        st.subheader("Network List")
        pfnet_names = list(col.pfnets.keys())
        if not pfnet_names:
            st.info("No PFnet objects.")
        else:
            st.write("Clicking at the top selects or deselects all:")
            height = min(600, 40 + 35 * len(col.pfnets))
            pfnet_df = pd.DataFrame({"Name": pfnet_names})

            selected_nets_data = st.dataframe(
                pfnet_df,
                width='stretch',
                height=height,  # Set a fixed height
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
                            #st.session_state.deleted_pfnets.append(net_name)
                        col.selected_nets = []  # Clear selection after deletion
                        #st.success("Selected PFnets deleted.")
                        st.rerun()  # Rerun
                    else:
                        st.warning("Click one or more network check boxes to delete.")
            with mrgnet:
                if st.button("Merge Networks", key="merge_nets_btn"):
                    if len(col.selected_nets) >= 2:
                        try:
                            # Assuming col.merge_networks() handles adding the merged net to col.pfnets
                            col.merge_networks()
                            st.rerun()  # Rerun
                        except Exception as e:
                            st.error(f"Error merging networks: {e}")
                    else:
                        st.warning("Click two or more network check boxes to merge.")
            with dispnet:
                if st.button("Display Network", key="display_net_btn"):
                    if len(col.selected_nets) == 1:
                        st.session_state.pf_name = col.selected_nets[0]
                        pf = col.pfnets[st.session_state.pf_name]
                        st.session_state.pic = Netpic(net=pf)
                        st.session_state.pic.net.get_layout(method=st.session_state.layout)
                        st.session_state.pic.create_view(font_size=st.session_state.font_size)
                        # open the Display page
                        st.switch_page(display_page)
                        col.selected_nets = []
                    else:
                        st.warning("Click one network check box to display.")

        st.write(f"**{st.session_state.pf_name}** is currently displayed.")

    st.session_state.count += 1
    #st.subheader(f"Current Network: {st.session_state.pic.ax.title.get_text()}")