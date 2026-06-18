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
from pypf.read_legacy_files import legacy_prx_files
import time
from pypf.stutils import init_pf_session_state

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
            state = "Merged networks must have same terms"
    return state

def averageable(col:Collection) -> str:
    prxs = col.selected_prxs
    state = "ok"
    if len(prxs) < 2:
        return "Click check boxes of 2 or more proximities to average"
    nterms = col.proximities[prxs[0]].nterms
    for prx in prxs[1:]:
        if col.proximities[prx].nterms != nterms:
            state = "Proximities must have same number of terms (nodes) to average"
    return state

stss.count += 1

# --- Sidebar for Global Actions ---
def toggle_intro():
    stss.intro_visible = not stss.intro_visible

def hide_sample():
    stss.sample_visible = False
    del_demo(col)
    if len(col.proximities) == 0 and len(col.pfnets) == 0:
        col.changed = False

with st.sidebar:
    if stss.sample_visible == True:
        add_demo(col)
        st.checkbox("**Add Sample Data**", value=True,
                    on_change=hide_sample,)

    st.write("**Information Tables**")
    show_corrs = st.checkbox("Proximity Correlations", value=False, key="show_corrs")
    show_netsim = st.checkbox("Network Similarities", value=False, key="show_netsim")

    stss.perm_ave_type =st.radio("**Proximity Averaging Method**",["mean","median"],
             index=get_idx(stss.perm_ave_type, ["mean", "median"]),
             key="averaging", )

    type, params = st.columns([6, 3])
    with type:
        stss.perm_net_type = st.radio("**Network Type**",["Pathfinder","Threshold", "Nearest Neighbor"],
                        index=get_idx(stss.perm_net_type, ["Pathfinder", "Threshold", "Nearest Neighbor"]),
                        key="network_type", )

    with params:
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

    #st.write("**currently selected net:**")
    st.write(f"**In View: {stss.col.focus_net.name}**")


# --- Main Panel ---

def make_symmetric_if_force(prx:Proximity) -> Proximity:
    if force_symmetric and not prx.issymmetric:
        prx.dismat = np.minimum(prx.dismat, prx.dismat.T)
        prx.name += "_fs"
        prx.issymmetric = True
    return prx

with st.expander("**Introductory Information**", on_change=toggle_intro, expanded=stss.intro_visible):
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

The **View Network** page allows you to try different layouts and view the network in various ways. The **Static** 
display shows the layout from the selected **Layout Method**, and you can move the nodes around as desired.  The 
**Dynamic** display starts from the layout generated positions, but it uses an active algorithm to adust the node
positions.  You can interact with the algorithm by dragging nodes, but the algorithm will continue to control
the layout.  Unchecking the **enabled** checkbox at the bottom will stop the algorithm allowing you 
to adjust node positions without the influence of the algorithm.

The **Add Sample Data** checkbox
loads several sample Proximities and Networks to illustrate the app and to allow you to explore 
the app's functionality.  Remove these resources by unchecking the checkbox.

You can save the state of a project to a json file on your computer and later reload it using the following buttons.

Go to the **Help** page for more information including required formats for Proximity files.

This Introductory Information can be hidden by closing the expander.  
    ''')

# --- Project Save and Load ---
savep, loadp = st.columns([1, 1])

with savep:
    # Step 1: User clicks to "prepare" the data
    if st.button("**📝 Save Project File**", use_container_width=True):
        if col.changed:
            stss.pending_json = col.get_project_state()
            stss.json_ready = True
        else:
            stss.pending_json = None
            stss.json_ready = False
            st.warning("No changes in the project.")
            time.sleep(.6)
            st.rerun()

    # Step 2: The actual download button appears only when the JSON exists
    if stss.get("json_ready"):
        st.download_button(
            label="📥 Download Now",
            data=stss.pending_json,
            file_name="pypf_project.json",
            mime="application/json",
            use_container_width=True
        )
        # Step 3: Clear the flags so the button disappears on the NEXT interaction
        stss.json_ready = False
        stss.pending_json = None
        col.changed = False

with loadp:
    # Step 1: User clicks to "trigger" the loading process
    if st.button("**📂 Load Project File**", use_container_width=True):
        stss.load_ready = True

    # Step 2: The uploader and Confirm button appear only when triggered
    if stss.get("load_ready"):
        uploaded_file = st.file_uploader(
            "Select project file (.json)",
            type="json",
            key="main_project_loader",
        )

        if uploaded_file is not None:
            if st.button("🚀 Confirm and Load", use_container_width=True):
                try:
                    project_data = uploaded_file.read().decode("utf-8")
                    col.load_project_state(project_data)
                    st.success("Project Merged Successfully!")

                    # Step 3: Reset flags and refresh the UI
                    stss.load_ready = False
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        # Optional: A cancel button to hide the loader if you change your mind
        if st.button("❌ Cancel Load", use_container_width=True):
            stss.load_ready = False
            st.rerun()

# --- Upload Files and Create Proximities (Now in an Expander) ---
if hasattr(stss,"xlerror") and stss.xlerror:
    st.error(stss.xlerror)
    stss.xlerror = None
with st.expander(label="📥 **Add Proximity Data Files**", expanded=False, width=480):
    stss.perm_file_type = st.radio("**Proximity Data File Type**", ["Spreadsheet", "Legacy Text"],
             index=get_idx(stss.perm_file_type, ["Spreadsheet", "Legacy Text"]),
             key="prx_file_type", )

    tog, val, fil = st.columns([5, 2, 2])
    with tog:
        force_symmetric = st.toggle("Force Proximity Symmetric  = ", value=False,
                                    help="When True, uses minimum of dij and dji if asymmetric",)
    with val:
        " "
        force_symmetric

    # File loaders
    match stss.perm_file_type:
        case "Spreadsheet":
            #st.write("**Add Proximity Spreadsheet Files**")
            ss_files = st.file_uploader("**Upload Proximity Spreadsheets (.xlsx or .csv)**", type=["xlsx","csv"],
                                key = f"ss_files{stss.file_version}", accept_multiple_files=True,
                                help="Upload one or more Proximity files (.xlsx or .csv) to add to the app.",
                                )
            if len(ss_files) > 0:
                for ss_file in ss_files:
                    if ss_file is not None:
                        try:
                            new_prx = Proximity(filepath=ss_file)
                            new_prx = make_symmetric_if_force(new_prx)
                            col.add_proximity(new_prx)
                        except Exception as e:
                            stss.xlerror = "Error: an invalid proximity file was loaded."
                stss.file_version += 1
                st.rerun()

        case "Legacy Text":
            #st.subheader("Add Legacy Proximity and Terms Text Files")
            leg_files = st.file_uploader("**Upload Legacy Proximity and Terms Text Files (.txt)**", type=["txt"],
                                        accept_multiple_files=True,
                                        help="Upload one or more files (.txt) to add to the app.",
                                        key=f"leg_files_key{stss.file_version}",)
            if len(leg_files) > 0:
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
                    new_prx = make_symmetric_if_force(new_prx)
                    col.add_proximity(new_prx)
                for file in all_leg_files:
                    os.remove(os.path.join(leg_dir, file))
                os.rmdir(leg_dir)
                stss.file_version += 1
                st.rerun()


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

def get_selected_rows(df, names) -> list:
    sel_rows = df.selection.rows
    sel_cells = df.selection.cells
    if len(sel_rows) > 0:
        sel = sel_rows
    elif len(sel_cells) > 0:
        sel = [sel_cells[0][0]]
    else:
        return []
    return [key for i, key in enumerate(names) if i in sel]

# Proximity List
st.subheader("Proximity List")
prx_names = list(col.proximities.keys())

if not prx_names:
    st.info("No Proximity objects loaded yet.")
else:
    st.write("Click in the box left of Name column to select:")
    height = min(600, 40 + 35*len(col.proximities))
    prx_df = col.get_proximity_info()
    selected_prxs_data = st.dataframe(
        prx_df,
        width='content',
        height=height,  # Set a height to reflect contents
        hide_index=True,  # Hide index as 'Name' column is present
        selection_mode=["multi-row","single-cell"],
        key=f"prx_selection_df{stss.prx_version}",
        on_select="rerun"  # Crucial for getting selection updates
    )
    col.selected_prxs = get_selected_rows(selected_prxs_data, prx_names)

    dernet, aveprx, dispprx, delprx, f2, f3 = st.columns(6)
    with delprx:
        if st.button("Delete Proximities", key="delete_prx_btn"):
            if col.selected_prxs:
                for prx_name in col.selected_prxs:
                    col.proximities.pop(prx_name)
                col.selected_prxs = []  # Clear selection after deletion
                stss.prx_version += 1
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
    with aveprx:
        if st.button("Average Proximities", key="average_prx_btn"):
            state = averageable(col)
            if state == "ok":
                try:
                    # with st.form(key="ave_prx_form"):
                    #     ave_name = st.text_input("Name for average", value="default")
                    #     submitted = st.form_submit_button(key="ave_prx_submit")
                    # if submitted:
                    #     ave_prx = col.average_proximities(method=stss.averaging)
                    #     if ave_name.strip() != "default" and ave_name.strip() != "":
                    #         ave_prx.name = ave_name.strip()
                    ave_prx = col.average_proximities(method=stss.averaging)
                    col.add_proximity(ave_prx)
                    col.selected_prxs = []
                    stss.prx_version += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Error averaging proximities: {e}")
            else:
                st.warning(f"Error: {state}.")
    with dispprx:
        display_proximity = st.button("Display Distances", )
        if display_proximity:
            if len(col.selected_prxs) != 1:
                st.warning("Select one proximity to display.")

if display_proximity:
    if len(col.selected_prxs) == 1:
        st.subheader(f"Distance Matrix for {col.selected_prxs[0]}")
        #create dataframe with terms and dismat
        prx = col.proximities[col.selected_prxs[0]]
        dismat = prx.dismat
        terms = prx.terms
        df = pd.DataFrame(dismat, index=terms, columns=terms)
        st.dataframe(df, width='content')
        col.selected_prxs = []
        stss.prx_version += 1

if show_corrs and len(col.proximities) > 1:
    st.subheader("Proximity Correlations")
    st.info("""Correlations of the distance data in pairs of equal sized data sets""")
    prx_corrs_df = col.get_proximity_correlations()
    if not prx_corrs_df.empty:
        st.dataframe(prx_corrs_df, width='content', hide_index=False)

# Network List
st.subheader("Network List")
pfnet_names = list(col.pfnets.keys())
if not pfnet_names:
    st.info("No networks created yet.")
else:
    st.write("Clicking at the top of the boxes selects or deselects all:")
    height = min(600, 40 + 35 * len(col.pfnets))
    pfnet_df = col.get_pfnet_info()

    selected_nets_data = st.dataframe(
        pfnet_df,
        width='content',
        height=height,  # Set a fixed height
        hide_index=True,  # Hide index
        selection_mode=["multi-row", "single-cell"],
        key=f"pfnet_selection_df{stss.net_version}",
        on_select="rerun"  # Crucial for getting selection updates
    )
    col.selected_nets = get_selected_rows(selected_nets_data, pfnet_names)

    dispnet, properties, links, mrgnet, delnet, c3 = st.columns(6)
    with delnet:
        if st.button("Delete Networks", key="delete_pfnet_btn"):
            if col.selected_nets:
                for net_name in col.selected_nets:
                    col.pfnets.pop(net_name)
                col.selected_nets = []  # Clear selection after deletion
                stss.net_version += 1
                st.rerun()  # Rerun
            else:
                st.warning("Click one or more network check boxes to delete.")
    with mrgnet:
        if st.button("Merge Networks", key="merge_nets_btn"):
            state = mergeable(col)
            if state == "ok":
                try:
                    # Assuming col.merge_networks() handles adding the merged net to col.pfnets
                    col.merge_networks()
                    col.selected_nets = []
                    stss.net_version += 1
                    st.rerun()  # Rerun
                except Exception as e:
                    st.error(f"Error merging networks: {e}")
            else:
                st.warning(f"Error: {state}.")
    with dispnet:
        if st.button("Display Network", key="display_net_btn"):
            if len(col.selected_nets) == 1:
                sel = col.pfnets[col.selected_nets[0]]
                stss.col.focus_net = sel
                col.selected_nets = []
                st.switch_page(view_page)
            else:
                st.warning("Error: Select one network to display.")
    with links:
        show_link_list = False
        if st.button("Network Link List", key="link_list_btn"):
            if len(col.selected_nets) == 1:
                show_link_list = True
            else:
                st.warning("Error: Select one network to display link list.")
    with properties:
        show_props = False
        if st.button("Network Properties", key="net_props_btn"):
            if len(col.selected_nets) == 1:
                show_props = True
            else:
                st.warning("Error: Select one network to display properties.")

if show_link_list:
    net = col.pfnets[col.selected_nets[0]]
    st.write("**Link List**")
    st.write(f"{net.name} : {net.nnodes} nodes -- {net.nlinks} links ")
    if net.type == "mg":
        st.write("""**weight** is a binary code  
            **nets** shows which nets contain the link (reading right to left) -- e.g., 110 means in nets 2 & 3  
            **count** shows the number of nets that contain the link -- e.g. 110 count would be 2
            """)
    llist = net.get_link_list()
    st.dataframe(data=llist, width="content", hide_index=False, )
    col.selected_nets = []
    stss.net_version += 1

if show_props:
    st.write("**Network Properties**")
    net = col.pfnets[col.selected_nets[0]]
    ecc = net.eccentricity
    st.write(f"{net.name}")
    st.write(
        f"{net.nnodes} nodes and {net.nlinks} links -- radius = {ecc['radius']} -- diameter = {ecc['diameter']}")
    st.write(f"")
    tab = pd.DataFrame(net.get_network_properties())
    st.dataframe(data=tab, width="content", hide_index=True, )
    col.selected_nets = []
    stss.net_version += 1


if show_netsim and len(col.pfnets) > 1:
    st.subheader("Network Similarity")
    st.info("""Similarity is the number of shared links 
        divided by the number of unique links in two networks.""")
    sim_df = col.network_similarity()
    if not sim_df.empty:
        st.dataframe(sim_df, width='content', hide_index=False)
    else:
        st.info("No PFnet objects.")

stss.count += 1