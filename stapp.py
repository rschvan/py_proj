# stapp.py: streamlit interface
import math
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
from streamlit import fragment

from pypf.collection import Collection
from pypf.proximity import Proximity
from pypf.pfnet import PFnet
from pypf.legacy_files import legacy_txt_files
import time
from pypf.stutils import (init_pf_session_state, tight_divider, add_proximity_files,
                          get_selected_values, save_project, load_project,)
from pypf.utility import sorted_keys, get_terms

import logging

# Intercept and drop the st.components.v1.html deprecation notice
_orig_logger_handle = logging.Logger.handle

def _suppress_html_deprecation(self, record: logging.LogRecord) -> bool:
  try:
    msg = record.getMessage()
    if "st.components.v1.html" in msg or "st.iframe" in msg:
      return False
  except Exception:
    pass
  return _orig_logger_handle(self, record)

logging.Logger.handle = _suppress_html_deprecation

# alias for st.system_state
stss = st.session_state

if "col" not in stss:
    init_pf_session_state()

#aliases
sample_col = stss.sample_col
sample_sources = stss.sample_sources
col = stss.col

# --- Streamlit Application Layout (MUST BE FIRST Streamlit command) ---
st.set_page_config(layout="wide", page_title="PyPathfinder", page_icon="🏠")

home_dir = os.path.dirname(os.path.abspath(__file__))
home_page = st.Page(page=os.path.join(home_dir,"stapp.py"), title="Build Nets", icon="🏠")
help_page = st.Page(page=os.path.join(home_dir,"pages","help.py"), title="Help", icon="❓")
view_page = st.Page(page=os.path.join(home_dir,"pages","view_net.py"), title="View Network", icon="🌐")
pages = [home_page, view_page, help_page]
page_selected = st.navigation(pages=pages, position="sidebar")

if page_selected != home_page:
    page_selected.run()
    st.stop()

st.title("PyPathfinder")

# Get the absolute path of the directory containing 'stapp.py' (which should be 'py_proj')
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_idx(val, opt_list):
    try:
        return opt_list.index(val)
    except ValueError:
        return 0  # Default to the first item if something goes wrong

def add_demo(col):
    for px in sample_col.proximities.values():
        if not px.name in col.proximities:
            col.add_proximity(px)
    for pfn in sample_col.pfnets.values():
        if not pfn.name in col.pfnets:
            col.add_pfnet(pfn)

def del_demo(col):
    for demoprx in sample_col.proximities.keys():
        if demoprx in col.proximities:
            col.proximities.pop(demoprx)
    for demopf in sample_col.pfnets.keys():
        if demopf in col.pfnets:
            del col.pfnets[demopf]

def mergeable(col:Collection) -> str:
    nets = col.selected_nets
    state = "ok"
    if len(nets) < 2:
        return "Click check boxes of 2 or more networks to merge"
    tid = col.pfnets[nets[0]].termsid
    for net in nets[1:]:
        if col.pfnets[net].termsid != tid:
            state = "All networks must have the same terms to be merged."
    return state

def averageable(col:Collection) -> str:
    prxs = col.selected_prxs
    state = "ok"
    if len(prxs) < 2:
        return "Click check boxes of 2 or more proximities with the same terms to average"
    tid = col.proximities[prxs[0]].termsid
    for prx in prxs[1:]:
        if col.proximities[prx].termsid != tid:
            state = "All proximities must have same terms to average."
    return state

stss.count += 1

# --- Sidebar for Global Actions ---

#----------------------------------------------------------------------------------------------------sidebar
def hide_sample():
    stss.sample_visible = False
    del_demo(col)
    stss.intro_visible = False


with (st.sidebar):
    if stss.sample_visible == True:
        if stss.count < 2:
            add_demo(col)
        st.button(label="**Remove Sample Data**", on_click=hide_sample)
        tight_divider()

    st.text_input(label="**Project Name**",
                  value="pypf_project",
                  key="project_name")

    tight_divider()
    st.write("**Comparison Tables**")
    show_corrs = st.checkbox("**Proximity Correlations**", value=False, key="show_corrs")
    show_netsim = st.checkbox("**Network Similarities**", value=False, key="show_netsim")

    tight_divider()

    st.info(
    """Selecting Proximities or Networks in the Lists will enable permissible action buttons for the selections.
    """
    )

    st.write(f"**In View: {stss.col.focus_net.name}**")

# ------------------------------------------------------------------------------------ Main Panel
# ------------------------------------------------------------------------------------ Project Save and Load

st.subheader("**Manage Data Files**")
stss.data_option = st.radio(label="**Select File Operation**",
         index=0,
         key=f"manage_data_{stss.manage_data_version}",
         options=["**None**", "**Add Proximity Data**",
                  "**Load Project File**",
                  "**Save Project File**",],
        on_change=st.rerun
                            )

if stss.data_option is not "**None**":
    with st.container(border=True, width="content", height="content", gap="small"):
        st.write(f"**{stss.data_option}**")
        match stss.data_option:
            case "**Save Project File**":
                save_project(col, name=stss.project_name)

            case "**Load Project File**":
                load_project(col)

            case "**Add Proximity Data**":
                add_proximity_files(col)

def toggle_intro():
    stss.intro_visible = not stss.intro_visible

with st.expander("**Introductory Information -- click to hide/reveal**",
                 on_change=toggle_intro, expanded=stss.intro_visible, ):
    #st.write("**Click to access YouTube videos about Pathfinder Nets and the app**")
    st.link_button("Click to open PFNets YouTube Channel", "https://www.youtube.com/@PFNets")

    # Intro
    st.write('''**Introduction**  
PyPathfinder is a tool for creating, exploring, and visualizing Pathfinder networks. 
The app contains three pages, **Build Nets** (this page), **View Network**, 
and **Help**. You can navigate to a page by clicking the page name in the sidebar.

The **Build Nets** page is where you add Proximity data and Derive Networks.  The minimum 
Pathfinder Network (q and r equal infinity) is generated when you load proximity data.
You can choose to add either spreadsheet Proximity files or Legacy text Proximity files.
(Help page gives details). You can also view Proximity Correlations and Network Similarities
by checking the boxes in the sidebar. Compatible Proximity Data sets can be averaged, and different
types of networks can be derived. Pathfinder networks require q and r values to be specified. The default
values of infinity yield the network with the minimum number of links. It will be a minimum spanning
tree (MST) if the network is connected and there is a unique MST. Ties in link distances may lead to 
cycles in the minimum network.  

The **Proximity and Network Lists** allow you to select Proximities or Networks to perform various operations. 
You select items by clicking the check box left of the Name column, and possible operations are invoked by 
clicking the buttons below the lists. You control the type of network generated with the controls in the sidebar.

The **View Network** page allows you to try different layouts and view the network in various ways. The Layout Tools 
allow you to use different algorithms to generate node positions, and the other widgets allow you to adjust the 
positions.  The **redraw** button will generate a new layout which will usually be different because a random 
layout is used as a starting point.  You can drag nodes to improve the layout, and you can change the font size
with the buttons in the display. Checking **enabled** checkbox below the display will engage an active layout 
engine which will further adjust the nodes. Dragging nodes with it engaged will influence the adjustments, but the
engine will continue to adjust the nodes until a stable point is reached. Disenabling the engine will allow you
to move only individual nodes.

The **Add Sample Data** checkbox controls the inclusion of 
the sample Proximities and Networks to allow you to explore 
the app's functionality.  Remove these resources by unchecking the checkbox.

Load your own proximity data using the **Add Proximity Data Files** utility. The sparsest Pathfinder Network 
is also created for each file you add.

You can save the state of a project to a json file on your computer and later reload it using the following buttons.

Go to the **Help** page for more information including required formats for Proximity files.

This Introductory Information can be hidden by closing the expander.  
    ''')


# sample sources
if stss.sample_visible:
    st.subheader("Sample Proximity Data and Derived Networks")
    st.write("""Example Proximity Data Spreadsheet (bank6 data)  
            Terms in first row and column with 
            distances or dissimilarities in the matrix""")
    st.dataframe(stss.example_file_format, width='content', hide_index=False, )
    st.write("Sample Proximity Sources")
    st.dataframe(sample_sources, width='content', hide_index=True)

# ---Proximity and Network Lists

# --------------------------------------------------------------------------------------------------Proximity List
st.subheader("Proximity List")
prx_names = sorted_keys(col.proximities)
if not prx_names:
    st.info("No Proximity files added.")
else:
    st.write("Click in the box left of Name column to select:")
    prx_df = col.get_proximity_info()
    prxs_event = st.dataframe(
        prx_df,
        width='content',
        height='content',  # Set a height to reflect contents
        hide_index=True,  # Hide index as 'Name' column is present
        selection_mode=["multi-row", ],
        key=f"prx_selection_df{stss.prx_version}",
        on_select="rerun"  # Crucial for getting selection updates
    )
    col.selected_prxs = get_selected_values(df=prx_df, event=prxs_event, column="name")

    dernet, aveprx, dispprx, delprx = st.columns(4)

    with delprx:
        if len(col.selected_prxs) > 0:
            count = len(col.selected_prxs)
            label = f"**Delete Selected ({count})**"

            with st.expander(label, expanded=False,
                             key=f"delete_selection_{stss.count}"):

                if st.button("Confirm", key="confirm_delete_btn", type="primary"):
                    for name in col.selected_prxs:
                        col.proximities.pop(name, None)

                    col.selected_prxs = []
                    stss.prx_version += 1
                    st.success(f"Deleted {count} proximities.")
                    time.sleep(1)
                    st.rerun()

                if st.button("Cancel", key="cancel_delete_btn"):
                    stss.prx_version -= 1
                    st.rerun()
        else:
            st.button(
                "Delete Proximities",
                disabled=True,
                help="Delete selected proximities.",
            )

    with dernet:
        if len(col.selected_prxs) > 0:
            with st.expander(label="Derive Networks", expanded=False):
                stss.perm_net_type = st.radio("**Network Type**", ["Pathfinder", "Threshold", "Nearest Neighbor"],
                                              key="network_type", help="Derive networks from selected proximities.")
                if stss.perm_net_type == "Pathfinder":
                    # Get raw text input
                    q_raw = st.text_input("**q:** (2-inf)", value="inf", key="q_input").lower().strip()
                    r_raw = st.text_input("**r:** (1.0-inf)", value="inf", key="r_input").lower().strip()
                    # Sanitize q: Force to float(inf) or int
                    try:
                        stss.q_param = int(float(q_raw))
                    except:
                        stss.q_param = np.inf
                    # Sanitize r: Force to float
                    if r_raw in ["", "inf", "none"]:
                        stss.r_param = np.inf
                    else:
                        try:
                            stss.r_param = float(r_raw)
                        except ValueError:
                            stss.r_param = np.inf

                if st.button("Derive Now", key="derive_pfnet_btn",
                             #disabled=len(col.selected_prxs) == 0,
                             help="Select proximities to derive nets"):
                    if len(col.selected_prxs) > 0:
                        for prx_name in col.selected_prxs:
                            # Check if PFnet with this name already exists before deriving
                            if prx_name in col.pfnets:
                                st.warning(f"PFnet '{prx_name}' already exists. Not re-deriving.")
                            else:
                                prx = col.proximities[prx_name]
                                match stss.perm_net_type:
                                    case "Pathfinder":
                                        pf = PFnet(prx, q=stss.q_param, r=stss.r_param)
                                    case "Threshold":
                                        pf = PFnet(prx, type="th")
                                    case "Nearest Neighbor":
                                        pf = PFnet(prx, type="nn")
                                col.add_pfnet(pf)
                        col.selected_prxs = []
                        stss.prx_version += 1
                        st.rerun()
                    else:
                        st.warning("Click one or more proximity check boxes to derive networks.")
        else:
            st.button("Derive Ntworks", disabled=True, help="Derive Ntworks from selected proximities.")
    with aveprx:
        state = averageable(col)
        if state == "ok":
            with st.expander("Average Proximities", expanded=False, ):

                ave_name = st.text_input(
                    "**Average Name**",
                    value= "", #stss.get("perm_ave_name", ""),
                    key=f"ave_name_{stss.ave_version}",
                )

                ave_method = st.radio(
                    "**Method**",
                    options=["mean", "median"],
                    index=0,
                    key=f"ave_method_radio_{stss.ave_version}",
                )

                if st.button("Compute Average", ):
                    try:
                        ave_prx = col.average_proximities(
                            ave_name=ave_name, method=ave_method
                        )
                        stss.ave_version += 1

                        col.add_proximity(ave_prx)
                        col.selected_prxs = []
                        stss.prx_version += 1

                        st.info(f"Average Proximity: {ave_prx.name} created")
                        time.sleep(3)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error averaging proximities: {e}")
        else:
            st.button("Average Proximities", disabled=True,
                      help="Create new proximity with average distance matrices of selected proximities.")

    with dispprx:
        display_matrix = st.button("Display Matrix",
                                      disabled=len(col.selected_prxs) != 1,
                                      help=f"Show the distance matrix of a selected proximity.")
        if display_matrix:
            if len(col.selected_prxs) != 1:
                st.warning("Show the distance matrix of a selected proximity.")

#---------------------------------------------------------------------------------------------------prx details
    if len(col.selected_prxs) == 1 and display_matrix:
        prx = col.proximities[col.selected_prxs[0]]
        st.subheader(f"Distance Matrix for {prx.name}")
        if hasattr(prx,"sources") and len(prx.sources) > 0:
            st.write(f"Sources: {prx.sources}")
        #create dataframe with terms and dismat
        dismat = prx.dismat
        terms = prx.terms
        df = pd.DataFrame(dismat, index=terms, columns=terms)
        st.dataframe(df, width='content', height='content')
        col.selected_prxs = []
        # stss.prx_version += 1

if len(col.proximities) > 1 and show_corrs:
    st.subheader("Proximity Correlations")
    st.info("""Correlations of the distance data in pairs of equal sized data sets""")
    prx_corrs_df = col.get_proximity_correlations()
    if not prx_corrs_df.empty:
        st.dataframe(prx_corrs_df, width='content', hide_index=False, height='content')

# ----------------------------------------------------------------------------------------------------Network List
st.subheader("Network List")
pfnet_names = sorted_keys(dic=col.pfnets, attrs=["termsid","name"])
if not pfnet_names:
    st.info("No networks created.")
else:
    st.write("Clicking at the top of the boxes selects or deselects all:")
    pfnet_df = col.get_pfnet_info()
    nets_event = st.dataframe(
        pfnet_df,
        width='content',
        height='content',
        hide_index=True,
        selection_mode=["multi-row", ],
        key=f"pfnet_selection_df{stss.net_version}",
        on_select="rerun"  # Crucial for getting selection updates
    )
    col.selected_nets = get_selected_values(pfnet_df, nets_event)

    dispnet, properties, links, mrgnet, delnet = st.columns(5)

    with delnet:
        if len(col.selected_nets) > 0:
            count = len(col.selected_nets)
            net_str = "net" if count == 1 else "nets"
            label = f"Delete {count} {net_str}"
            with st.expander(label, expanded=False, key=f"delete_net_selection_{stss.count}"):

                if st.button("Confirm", key="confirm_delete_net_btn", type="primary"):
                    for name in col.selected_nets:
                        col.pfnets.pop(name, None)

                    col.selected_nets = []
                    stss.net_version += 1
                    st.success(f"Deleted {count} {'networks' if count > 1 else 'network'}.")
                    time.sleep(1)
                    st.rerun()

                if st.button("Cancel", key="cancel_delete_net_btn"):
                    st.rerun()
        else:
            st.button(
                "Delete Networks",
                disabled=True,
                help="Delete selected networks.",
            )

    with mrgnet:
        state = mergeable(col)
        if state == 'ok':
            with st.expander(f"Merge {len(col.selected_nets)} nets", expanded=False):
                mrg_name = st.text_input("**Merged Net Name**", key=f"mrg_name_{stss.mrg_version}")
                if st.button("Merge Now", key="mrg_nets_btn", ):
                    state = mergeable(col)
                    if state == "ok":
                        try:
                            # Assuming col.merge_networks() handles adding the merged net to col.pfnets
                            col.merge_networks(mrg_name=mrg_name)
                            col.selected_nets = []
                            stss.net_version += 1
                            stss.mrg_version += 1
                            st.info("Networks Merged")
                            time.sleep(2)
                            st.rerun()  # Rerun
                        except Exception as e:
                            st.error(f"Error merging networks: {e}")
                    else:
                        st.warning(f"Error: {state}.")
                        time.sleep(2)
        else:
            st.button("Merge Networks", disabled=True, help="Select 2 or more networks with the same terms to merge them")
    with dispnet:
        if st.button("Display Network", key="display_net_btn", disabled=len(col.selected_nets) != 1,):
            if len(col.selected_nets) == 1:
                sel = col.pfnets[col.selected_nets[0]]
                stss.col.focus_net = sel
                col.selected_nets = []
                st.switch_page(view_page)
            else:
                st.warning("Error: Select one network to display.")
    with links:
        show_link_list = False
        if st.button("Network Link List", key="link_list_btn", disabled=len(col.selected_nets) != 1):
            if len(col.selected_nets) == 1:
                show_link_list = True
            else:
                st.warning("Error: Select one network to display link list.")
    with properties:
        show_props = False
        if st.button("Network Properties", key="net_props_btn", disabled=len(col.selected_nets) != 1):
            if len(col.selected_nets) == 1:
                show_props = True
            else:
                st.warning("Error: Select one network to display properties.")

#------------------------------------------------------------------------------------------------net details
if len(col.pfnets) > 0 and show_link_list:
    net = col.pfnets[col.selected_nets[0]]
    st.write("**Link List**")
    st.write(f"{net.name} : {net.nnodes} nodes -- {net.nlinks} links -- net.isdirected={net.isdirected}")
    if net.type == "mg":
        st.write(f"Sources: {net.sources}")
        for i, source in enumerate(net.sources):
            b = f"{int(2**i):b}"
            st.write(f"{b} {source}")
        st.write("""**weight** is a binary code using bits shown above   
            **nets** shows which nets using bits shown above  
            **count** shows the number of nets that contain the link
            """)
    llist = net.get_link_list()
    st.dataframe(data=llist, width="content", hide_index=False, height='content')
    col.selected_nets = []
    stss.net_version += 1

if len(col.pfnets) > 0 and show_props:
    st.write("**Network Properties**")
    net = col.pfnets[col.selected_nets[0]]
    if net.type == "mg":
        st.write(f"Sources: {net.sources}")
    ecc = net.eccentricity
    st.write(f"{net.name}")
    st.write(
        f"{net.nnodes} nodes and {net.nlinks} links -- radius = {ecc['radius']} -- diameter = {ecc['diameter']}")
    st.write(f"")
    tab = pd.DataFrame(net.get_network_properties())
    st.dataframe(data=tab, width="content", hide_index=True, height='content')
    col.selected_nets = []
    stss.net_version += 1

if show_netsim and len(col.pfnets) > 1:
    st.subheader("Network Similarity")
    st.info("""Similarity is the number of shared links 
        divided by the number of unique links in two networks.""")
    sim_df = col.network_similarity()
    if not sim_df.empty:
        st.dataframe(sim_df, width='content', hide_index=False, height='content')
    else:
        st.info("No PFnet objects.")

stss.count += 1