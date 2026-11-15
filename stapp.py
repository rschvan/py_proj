# stapp.py: streamlit interface
import os
import streamlit as st
import pandas as pd
import numpy as np
import time

from pypf.pfnet import PFnet, parse_q, parse_r
from stutils import (init_pf_session_state, tight_divider, add_proximity_files,
                     get_selected_values, save_project, load_project, get_focus_net,
                     proximity_table, network_table, properties_table,
                     keep_panels_in_view, REVEAL_KEY)
from pypf.utility import sorted_keys

# alias for st.system_state
stss = st.session_state

if "col" not in stss:
    init_pf_session_state()

#aliases
sample_col = stss.sample_col
sample_sources = stss.sample_sources
col = stss.col

st.set_page_config(layout="wide", page_title="PyPathfinder", page_icon="🏠")

home_dir = os.path.dirname(os.path.abspath(__file__))
home_page = st.Page(page=os.path.join(home_dir,"stapp.py"), title="Build Nets", icon="🏠")
help_page = st.Page(page=os.path.join(home_dir,"app_pages","help.py"), title="Help", icon="❓")
view_page = st.Page(page=os.path.join(home_dir,"app_pages","view_net.py"), title="View Network", icon="🌐")
pages = [home_page, view_page, help_page]
page_selected = st.navigation(pages=pages, position="sidebar")

if page_selected != home_page:
    page_selected.run()
    st.stop()

st.title("PyPathfinder")
keep_panels_in_view()

# --- Sidebar for Global Actions ---

#----------------------------------------------------------------------------------------------------sidebar
def hide_sample():
    """Callback for Clear Demo: remove the sample proximities and networks."""
    stss.sample_visible = False
    col.delete_proximities(sample_col.proximities.keys())
    col.delete_pfnets(sample_col.pfnets.keys())
    stss.intro_visible = False


with (st.sidebar):
    if stss.sample_visible == True:
        #st.info("**Click to Bypass Demo**")
        st.button(label="**Clear Demo**", on_click=hide_sample)
        tight_divider()

    stss.project_name =st.text_input(label="**Project Name**",
                  value=stss.project_name,
                  key=f"proj_name_{stss.proj_name_ver}",)

    tight_divider()
    st.write("**Comparison Tables**")
    show_corrs = st.checkbox("**Proximity Correlations**", value=False, key="show_corrs")
    show_netsim = st.checkbox("**Network Similarities**", value=False, key="show_netsim")

    tight_divider()

    st.info(
    """Selecting Proximities or Networks in the Lists will enable permissible action buttons for the selections.
    """
    )

    if get_focus_net() is not None:
        st.write(f"**In View: {stss.focus_net_name}**")

# ------------------------------------------------------------------------------------ Main Panel
# ------------------------------------------------------------------------------------ Project Save and Load

manage, act = st.columns([0.30,0.70])

with manage:
    st.subheader("**Manage Data Files**")
    stss.data_option = st.radio(label="**Select File Operation**",
             index=None,
             key=f"manage_data_{stss.manage_data_version}",
             options=[ "**Add Proximity Data Files**",
                      "**Load Project File**",
                      "**Save Project File**",],
                                )
with act:
    if stss.data_option:
        with st.container(border=True, width="content", height="content", gap="small"):
            st.write(f"**{stss.data_option}**")
            match stss.data_option:
                case "**Save Project File**":
                    save_project(col, name=stss.project_name)

                case "**Load Project File**":
                    load_project(col)

                case "**Add Proximity Data Files**":
                    add_proximity_files(col)

# Report files that could not be added (shown once, after the upload rerun)
if stss.get("upload_errors"):
    st.error("**Some files could not be added:**  \n" + "  \n".join(stss.upload_errors))
    stss.upload_errors = []

def toggle_intro():
    stss.intro_visible = not stss.intro_visible

with st.expander("**Getting Started** (click to hide or show)",
                 on_change=toggle_intro, expanded=stss.intro_visible, ):
    intro = ("**PyPathfinder** builds and displays Pathfinder networks from proximity data: "
             "distances or relatedness values between every pair of a set of terms.")
    if stss.sample_visible:
        intro += (" Sample data is loaded so you can explore; click **Clear Demo** in the "
                  "sidebar to remove it.")
    st.markdown(intro)
    st.markdown(
        "1. **Add data**: in **Manage Data Files**, choose **Add Proximity Data Files**. "
        "The minimum Pathfinder network (q = r = ∞) is derived for each file.\n"
        "2. **Work with the lists**: check the boxes left of the Name column to select items. "
        "The buttons below each list let you derive networks with other q and r values, "
        "average proximities, merge networks, and show details.\n"
        "3. **View a network**: select one and click **Display Network**. On **View Network**, "
        "try layouts, use the toolbar to orient and adjust it, and click **Save Layout** to keep it.\n"
        "4. **Save your project** in **Manage Data Files**."
    )
    st.warning("The app loses its data when idle for a while. "
               "Save your project before taking a break.", icon="⚠️")
    help_link, video_link = st.columns(2)
    with help_link:
        st.page_link(help_page, label="Help page: full details",
                     icon="❓")
    with video_link:
        st.link_button("PFNets YouTube channel", "https://www.youtube.com/@PFNets", icon="▶️")


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
    prx_df = proximity_table(col)
    prxs_event = st.dataframe(
        prx_df,
        width='content',
        height='content',  # Set a height to reflect contents
        hide_index=True,  # Hide index as 'Name' column is present
        selection_mode=["multi-row", ],
        key=f"prx_selection_df{stss.prx_version}",
        on_select="rerun"  # Crucial for getting selection updates
    )
    sel_prxs = get_selected_values(df=prx_df, event=prxs_event, column="name")

    dernet, aveprx, dispprx, delprx = st.columns(4)

    with delprx:
        if len(sel_prxs) > 0:
            count = len(sel_prxs)
            label = f"**Delete Selected ({count})**"

            with st.expander(label, expanded=False,
                             key=f"delete_selection_{stss.prx_version}"):

                if st.button("Confirm", key="confirm_delete_btn", type="primary"):
                    col.delete_proximities(sel_prxs)

                    stss.prx_version += 1
                    st.info(f"Deleted {count} proximities.")
                    time.sleep(2)
                    st.rerun()

                if st.button("Cancel", key="cancel_delete_btn"):
                    stss.prx_version += 1
                    st.rerun()
        else:
            st.button(
                "Delete Proximities",
                disabled=True,
                help="Delete selected proximities.",
            )

    with dernet:
        if len(sel_prxs) > 0:
            with st.expander(label=f"Derive Networks ({len(sel_prxs)})", expanded=False):
                stss.perm_net_type = st.radio("**Network Type**", ["Pathfinder", "Nearest Neighbor"],
                                              key="network_type", help="Derive networks from selected proximities.")
                if stss.perm_net_type == "Pathfinder":
                    stss.q_param = parse_q(st.text_input("**q:** (1-inf)", value="inf", key="q_input"))
                    stss.r_param = parse_r(st.text_input("**r:** (1.0-inf)", value="inf", key="r_input"))

                if st.button("Derive Now", key="derive_pfnet_btn",
                             #disabled=len(sel_prxs) == 0,
                             help="Select proximities to derive nets"):
                    if len(sel_prxs) > 0:
                        for prx_name in sel_prxs:
                            prx = col.proximities[prx_name]
                            match stss.perm_net_type:
                                case "Pathfinder":
                                    pf = PFnet(prx, q=stss.q_param, r=stss.r_param)
                                case "Nearest Neighbor":
                                    pf = PFnet(prx, type="nn")
                            col.add_pfnet(pf)
                        stss.prx_version += 1
                        st.rerun()
                    else:
                        st.warning("Click one or more proximity check boxes to derive networks.")
        else:
            st.button("Derive Networks", disabled=True, help="Derive Networks from selected proximities.")
    with aveprx:
        ave_problem = col.average_problem(sel_prxs)
        if ave_problem is None:
            with st.expander(f"Average Proximities ({len(sel_prxs)})", expanded=False, ):

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
                            sel_prxs, ave_name=ave_name, method=ave_method
                        )
                        stss.ave_version += 1

                        col.add_proximity(ave_prx)
                        stss.prx_version += 1

                        st.info(f"Average Proximity: {ave_prx.name} created")
                        time.sleep(3)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error averaging proximities: {e}")
        else:
            st.button("Average Proximities", disabled=True,
                      help="Create new proximity with average distance matrices of selected proximities. "
                           + ave_problem)

    with dispprx:
        display_matrix = st.button("Display Matrix",
                                      disabled=len(sel_prxs) != 1,
                                      help=f"Show the distance matrix of a selected proximity.")
        if display_matrix:
            if len(sel_prxs) != 1:
                st.warning("Show the distance matrix of a selected proximity.")

#---------------------------------------------------------------------------------------------------prx details
    if len(sel_prxs) == 1 and display_matrix:
        with st.container(key=f"{REVEAL_KEY}_matrix"):
            prx = col.proximities[sel_prxs[0]]
            st.subheader(f"Distance Matrix for {prx.name}")
            if hasattr(prx, "sources") and len(prx.sources) > 0:
                st.write(f"Sources: {prx.sources}")
            df = pd.DataFrame(prx.dismat, index=prx.terms, columns=prx.terms)
            st.dataframe(df, width='content', height='content')
        # stss.prx_version += 1

if len(col.proximities) > 1 and show_corrs:
    st.subheader("Proximity Correlations")
    st.info("""Correlations of the distance data in pairs of data sets with the same terms""")
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
    pfnet_df = network_table(col)
    nets_event = st.dataframe(
        pfnet_df,
        width='content',
        height='content',
        hide_index=True,
        selection_mode=["multi-row", ],
        key=f"pfnet_selection_df{stss.net_version}",
        on_select="rerun"  # Crucial for getting selection updates
    )
    sel_nets = get_selected_values(pfnet_df, nets_event)

    dispnet, properties, links, mrgnet, delnet = st.columns(5)

    with delnet:
        if len(sel_nets) > 0:
            count = len(sel_nets)
            net_str = "net" if count == 1 else "nets"
            label = f"Delete {count} {net_str}"
            with st.expander(label, expanded=False, key=f"delete_net_selection_{stss.net_version}"):

                if st.button("Confirm", key="confirm_delete_net_btn", type="primary"):
                    col.delete_pfnets(sel_nets)

                    stss.net_version += 1
                    st.info(f"Deleted {count} {'networks' if count > 1 else 'network'}.")
                    time.sleep(1)
                    st.rerun()

                if st.button("Cancel", key="cancel_delete_net_btn"):
                    stss.net_version += 1
                    st.rerun()
        else:
            st.button(
                "Delete Networks",
                disabled=True,
                help="Delete selected networks.",
            )

    with mrgnet:
        mrg_problem = col.merge_problem(sel_nets)
        if mrg_problem is None:
            with st.expander(f"Merge {len(sel_nets)} nets", expanded=False):
                mrg_name = st.text_input("**Merged Net Name**", key=f"mrg_name_{stss.mrg_version}")
                if st.button("Merge Now", key="mrg_nets_btn", ):
                    try:
                        col.merge_networks(sel_nets, mrg_name=mrg_name)  # adds the merged net
                        stss.net_version += 1
                        stss.mrg_version += 1
                        st.info("Networks Merged")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error merging networks: {e}")
        else:
            st.button("Merge Networks", disabled=True, help=mrg_problem)
    with dispnet:
        if st.button("Display Network", key="display_net_btn", disabled=len(sel_nets) != 1,):
            if len(sel_nets) == 1:
                stss.focus_net_name = sel_nets[0]
                st.switch_page(view_page)
            else:
                st.warning("Error: Select one network to display.")
    with links:
        show_link_list = False
        if st.button("Network Link List", key="link_list_btn", disabled=len(sel_nets) != 1):
            if len(sel_nets) == 1:
                show_link_list = True
            else:
                st.warning("Error: Select one network to display link list.")
    with properties:
        show_props = False
        if st.button("Network Properties", key="net_props_btn", disabled=len(sel_nets) != 1):
            if len(sel_nets) == 1:
                show_props = True
            else:
                st.warning("Error: Select one network to display properties.")

#------------------------------------------------------------------------------------------------net details
if len(col.pfnets) > 0 and show_link_list:
    with st.container(key=f"{REVEAL_KEY}_links"):
        net = col.pfnets[sel_nets[0]]
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
    stss.net_version += 1

if len(col.pfnets) > 0 and show_props:
    with st.container(key=f"{REVEAL_KEY}_props"):
        st.write("**Network Properties**")
        net = col.pfnets[sel_nets[0]]
        if net.type == "mg":
            st.write(f"Sources: {net.sources}")
        ecc = net.eccentricity
        st.write(f"{net.name}")
        st.write(
            f"{net.nnodes} nodes and {net.nlinks} links -- radius = {ecc['radius']} -- diameter = {ecc['diameter']}")
        tab = properties_table(net)
        st.dataframe(data=tab, width="content", hide_index=True, height='content')
    stss.net_version += 1

if show_netsim and len(col.pfnets) > 1:
    st.subheader("Network Similarity")
    st.info("""Similarity: The number of common links 
        divided by the number of unique links in two networks with the same terms""")
    sim_df = col.network_similarity()
    if not sim_df.empty:
        st.dataframe(sim_df, width='content', hide_index=False, height='content')
    else:
        st.info("No PFnet objects.")
