# stapp.py: streamlit interface
import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np


@st.cache_resource
def _module_mtimes() -> dict:
    """File times of the project modules as last seen; kept for the life of the server."""
    return {}


def _drop_stale_modules() -> None:
    """Make the imports below load the current code of any project module whose file changed.

    Streamlit reloads changed modules only for sessions that are open when the files change.
    When the code is updated on a server with no open sessions (Streamlit Cloud after a push),
    new sessions would run this new script with the old pypf, stutils, etc. still in memory.
    """
    root = Path(__file__).resolve().parent
    seen = _module_mtimes()
    project = {}
    for name, mod in list(sys.modules.items()):
        file = getattr(mod, "__file__", None)
        if name == "__main__" or not file or "site-packages" in file:
            continue
        path = Path(file).resolve()
        if root in path.parents and path.parts[len(root.parts)] not in ("venv", ".venv"):
            project[name] = path
    stale = False
    for name, path in project.items():
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None
        stale |= seen.get(name, mtime) != mtime
        seen[name] = mtime
    if stale:
        for name in project:
            sys.modules.pop(name, None)
        seen.clear()


_drop_stale_modules()

from pypf.pfnet import PFnet, parse_q, parse_r
from stutils import (init_pf_session_state, tight_divider, add_files_dialog,
                     get_selected_values, save_project_button, load_project_dialog, get_focus_net,
                     proximity_table, network_table, properties_table,
                     keep_panels_in_view, REVEAL_KEY,
                     load_demo, clear_demo, demo_loaded, is_demo, mark_demo)
from pypf.utility import sorted_keys

# alias for st.system_state
stss = st.session_state

if "col" not in stss:
    init_pf_session_state()

#aliases
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
if stss.get("toast"):  # message from an action window, shown after its rerun
    st.toast(stss.toast)
    stss.toast = None
keep_panels_in_view()

# --- Sidebar for Global Actions ---

#----------------------------------------------------------------------------------------------------sidebar
def demo_button_clicked(action) -> None:
    """Callback for Clear Demo and Load Demo."""
    action()
    if action is clear_demo:
        stss.intro_visible = False  # the user is ready for their own data
    stss.prx_version += 1  # clears the selections
    stss.net_version += 1


def also_deleted(names: list[str]) -> None:
    """In a Delete window: list the items that depend on the selection and go with it."""
    if names:
        st.warning("**Also deletes** (made from the selection):  \n" + "  \n".join(f"`{n}`" for n in names))


# ----------------------------------------------------------------- action windows
# Each opens over the page for the current selection and closes (with a rerun) when its
# action is done; closing it with ✕ cancels. Messages are shown as a toast after the rerun.

def selected_line(names: list[str]) -> None:
    st.caption("Selected: " + ", ".join(f"`{n}`" for n in names))


def done(message: str) -> None:
    stss.toast = message
    st.rerun()


@st.dialog("Derive Networks", width="small")
def derive_dialog(names: list[str]) -> None:
    selected_line(names)
    stss.perm_net_type = st.radio("**Network Type**", ["Pathfinder", "Nearest Neighbor"],
                                  key="network_type", horizontal=True)
    if stss.perm_net_type == "Pathfinder":
        q_col, r_col = st.columns(2)
        stss.q_param = parse_q(q_col.text_input("**q** (1 – inf)", value="inf", key="q_input"))
        stss.r_param = parse_r(r_col.text_input("**r** (1 – inf)", value="inf", key="r_input"))
    if st.button("Derive", type="primary"):
        for prx_name in names:
            prx = col.proximities[prx_name]
            if stss.perm_net_type == "Pathfinder":
                col.add_pfnet(PFnet(prx, q=stss.q_param, r=stss.r_param))
            else:
                col.add_pfnet(PFnet(prx, type="nn"))
        stss.prx_version += 1  # clears the selection
        stss.net_version += 1
        done(f"Derived {len(names)} network{'s' if len(names) > 1 else ''}.")


@st.dialog("Average Proximities", width="small")
def average_dialog(names: list[str]) -> None:
    selected_line(names)
    ave_name = st.text_input("**Average Name** (optional)", key=f"ave_name_{stss.ave_version}")
    ave_method = st.radio("**Method**", options=["mean", "median"], horizontal=True,
                          key=f"ave_method_radio_{stss.ave_version}")
    if st.button("Compute Average", type="primary"):
        try:
            ave_prx = col.average_proximities(names, ave_name=ave_name, method=ave_method)
        except ValueError as e:
            st.error(f"Could not average: {e}")
            return
        if all(is_demo(col.proximities[n]) for n in names):
            mark_demo(ave_prx)  # made only from demo data: Clear Demo removes it
        col.add_proximity(ave_prx)
        stss.ave_version += 1
        stss.prx_version += 1
        stss.net_version += 1
        done(f"Created `{ave_prx.name}`.")


@st.dialog("Delete Proximities", width="small")
def delete_proximities_dialog(names: list[str]) -> None:
    selected_line(names)
    also_prxs, also_nets = col.dependents(prx_names=names)
    also_deleted(also_prxs + also_nets)
    if st.button("Delete", type="primary"):
        col.delete_proximities(names)
        stss.prx_version += 1
        stss.net_version += 1
        done(f"Deleted {len(names) + len(also_prxs)} proximities and {len(also_nets)} networks.")


@st.dialog("Merge Networks", width="small")
def merge_dialog(names: list[str]) -> None:
    selected_line(names)
    mrg_name = st.text_input("**Merged Network Name** (optional)", key=f"mrg_name_{stss.mrg_version}")
    if st.button("Merge", type="primary"):
        try:
            merged = col.merge_networks(names, mrg_name=mrg_name)  # adds the merged net
        except ValueError as e:
            st.error(f"Could not merge: {e}")
            return
        stss.net_version += 1
        stss.mrg_version += 1
        done(f"Created `{merged.name}`.")


@st.dialog("Delete Networks", width="small")
def delete_networks_dialog(names: list[str]) -> None:
    selected_line(names)
    _, also_nets = col.dependents(net_names=names)
    also_deleted(also_nets)
    if st.button("Delete", type="primary"):
        col.delete_pfnets(names)
        stss.net_version += 1
        count = len(names) + len(also_nets)
        done(f"Deleted {count} network{'s' if count > 1 else ''}.")


# ----------------------------------------------------------------- table windows
# Tables open in a window over the page; close it with ✕ or Esc. The selection stays,
# so another table for the same item is one click away.

DOWNLOAD_TIP = "Hover over the table and click ⤓ at its top right to download it as a .csv file."


@st.dialog("Distance Matrix", width="large")
def matrix_dialog(name: str) -> None:
    prx = col.proximities[name]
    st.markdown(f"**{prx.name}**: {len(prx.terms)} terms")
    if prx.sources:
        st.caption("Average of: " + ", ".join(f"`{s}`" for s in prx.sources))
    st.dataframe(pd.DataFrame(prx.dismat, index=prx.terms, columns=prx.terms), width="content")
    st.caption(DOWNLOAD_TIP)


def fitted_columns(df: pd.DataFrame, px_per_char: float = 6.5, padding: int = 18) -> dict:
    """Column widths (pixels) that fit each column's longest value or header."""
    def longest(col) -> int:
        values = [col] + [f"{v:g}" if isinstance(v, float) else str(v) for v in df[col]]
        return max(len(v) for v in values)
    return {col: st.column_config.Column(width=round(longest(col) * px_per_char) + padding,
                                         alignment="center")
            for col in df.columns}


@st.dialog("Network Properties", width="large")
def properties_dialog(name: str) -> None:
    net = col.pfnets[name]
    ecc = net.eccentricity
    st.markdown(f"**{net.name}**: {net.nnodes} nodes, {net.nlinks} links, "
                f"radius {ecc['radius']}, diameter {ecc['diameter']}")
    if net.type == "mg":
        st.caption("Merged from: " + ", ".join(f"`{s}`" for s in net.sources))
    tab = properties_table(net)
    columns = fitted_columns(tab)
    table_width = sum(c["width"] for c in columns.values()) + 20  # + the vertical scrollbar
    st.dataframe(tab, width=table_width, hide_index=True, column_config=columns)
    st.caption(DOWNLOAD_TIP)


@st.dialog("Network Link List", width="large")
def link_list_dialog(name: str) -> None:
    net = col.pfnets[name]
    st.markdown(f"**{net.name}**: {net.nnodes} nodes, {net.nlinks} links, "
                f"{'directed' if net.isdirected else 'undirected'}")
    if net.type == "mg":
        n = len(net.sources)
        codes = [f"`{1 << i:0{n}b}` {source}" for i, source in enumerate(net.sources)]
        st.markdown("Codes of the merged networks:  \n" + "  \n".join(codes))
        st.caption("**weight** and **nets** show which networks have each link, using these codes; "
                   "**count** is how many networks have it.")
    st.dataframe(net.get_link_list(), width="content", hide_index=False)
    st.caption(DOWNLOAD_TIP)


with (st.sidebar):
    if demo_loaded():
        st.button(label="**Clear Demo**", on_click=demo_button_clicked, args=(clear_demo,),
                  help="Remove the sample data and anything made only from it.")
    else:
        st.button(label="Load Demo", on_click=demo_button_clicked, args=(load_demo,),
                  help="Add the sample proximities and networks to explore.")
    tight_divider()

    stss.project_name =st.text_input(label="**Project Name**",
                  value=stss.project_name,
                  key=f"proj_name_{stss.proj_name_ver}",)
    save_project_button(col, name=stss.project_name)
    if st.button("📂 Load Project…", width="stretch",
                 help="Replace the current project with a saved one, or add it."):
        load_project_dialog()
    if st.button("➕ Add Data Files…", width="stretch",
                 help="Add proximity files (spreadsheets or legacy text files)."):
        add_files_dialog()

    tight_divider()

    if get_focus_net() is not None:
        st.write(f"**In View: {stss.focus_net_name}**")

# ------------------------------------------------------------------------------------ Main Panel

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
    if demo_loaded():
        intro += (" Sample data is loaded so you can explore; click **Clear Demo** in the "
                  "sidebar to remove it.")
    else:
        intro += " To explore with sample data, click **Load Demo** in the sidebar."
    st.markdown(intro)
    st.markdown(
        "1. **Add data**: click **➕ Add Data Files…** in the sidebar. "
        "The minimum Pathfinder network (q = r = ∞) is derived for each file.\n"
        "2. **Work with the lists**: check the boxes left of the Name column to select items. "
        "The buttons below each list become available for the selection: derive networks "
        "with other q and r values, average proximities, merge networks, and show details.\n"
        "3. **View a network**: select one and click **Display Network**. On **View Network**, "
        "try layouts, use the toolbar to orient and adjust it, and click **Save Layout** to keep it.\n"
        "4. **Save your project** with **💾 Save Project** in the sidebar, and bring it back "
        "later with **📂 Load Project**."
    )
    with st.expander("Spreadsheet layout example (bank6)"):
        st.caption("Terms in the first row and column, distances or dissimilarities in the matrix.")
        st.dataframe(stss.example_file_format, width="content", hide_index=False)
    st.warning("The app loses its data when idle for a while. "
               "Save your project before taking a break.", icon="⚠️")
    help_link, video_link = st.columns(2)
    with help_link:
        st.page_link(help_page, label="Help page: full details",
                     icon="❓")
    with video_link:
        st.link_button("PFNets YouTube channel", "https://www.youtube.com/@PFNets", icon="▶️")


# where the sample data came from
if demo_loaded():
    with st.expander("**Demo data sources**"):
        st.dataframe(stss.sample_sources, width="content", hide_index=True)

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

    n_prx = len(sel_prxs)
    ave_problem = col.average_problem(sel_prxs)
    with st.container(horizontal=True, gap="small"):
        if st.button(f"Derive Networks… ({n_prx})" if n_prx else "Derive Networks…",
                     disabled=n_prx == 0, help="Derive networks with other q and r values, "
                     "or Nearest Neighbor networks, from the selected proximities."):
            derive_dialog(sel_prxs)
        if st.button("Average Proximities…", disabled=ave_problem is not None,
                     help=ave_problem or "Average the distances of the selected proximities."):
            average_dialog(sel_prxs)
        if st.button("Display Matrix", disabled=n_prx != 1,
                     help="Show the distance matrix of the selected proximity."):
            matrix_dialog(sel_prxs[0])
        if st.button(f"Delete Proximities… ({n_prx})" if n_prx else "Delete Proximities…",
                     disabled=n_prx == 0, help="Delete the selected proximities and what was made from them."):
            delete_proximities_dialog(sel_prxs)


if len(col.proximities) > 1 and st.toggle(
        "**Show Proximity Correlations**", key="show_corrs",
        help="Correlations of the distance data in pairs of proximities with the same terms"):
    with st.container(key=f"{REVEAL_KEY}_corrs"):
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

    n_net = len(sel_nets)
    mrg_problem = col.merge_problem(sel_nets)
    with st.container(horizontal=True, gap="small"):
        if st.button("Display Network", key="display_net_btn", disabled=n_net != 1,
                     help="Show the selected network on View Network."):
            stss.focus_net_name = sel_nets[0]
            st.switch_page(view_page)
        if st.button("Network Properties", key="net_props_btn", disabled=n_net != 1,
                     help="Degree, eccentricity, center and median of each node."):
            properties_dialog(sel_nets[0])
        if st.button("Network Link List", key="link_list_btn", disabled=n_net != 1,
                     help="The links of the selected network with their weights."):
            link_list_dialog(sel_nets[0])
        if st.button("Merge Networks…", disabled=mrg_problem is not None,
                     help=mrg_problem or "Merge the selected networks into one."):
            merge_dialog(sel_nets)
        if st.button(f"Delete Networks… ({n_net})" if n_net else "Delete Networks…",
                     disabled=n_net == 0, help="Delete the selected networks and merges made from them."):
            delete_networks_dialog(sel_nets)

if len(col.pfnets) > 1 and st.toggle(
        "**Show Network Similarities**", key="show_netsim",
        help="Shared links in pairs of networks with the same terms"):
    with st.container(key=f"{REVEAL_KEY}_netsim"):
        st.subheader("Network Similarity")
        st.info("""Similarity: The number of common links
            divided by the number of unique links in two networks with the same terms""")
        sim_df = col.network_similarity()
        if not sim_df.empty:
            st.dataframe(sim_df, width='content', hide_index=False, height='content')
        else:
            st.info("No PFnet objects.")
