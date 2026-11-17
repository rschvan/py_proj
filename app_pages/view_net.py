# app_pages/view_net.py
import numpy as np
import pandas as pd
import streamlit as st

from netview import network_view
from pypf.utility import sorted_keys
from stutils import init_pf_session_state, tight_divider, get_focus_net

# alias for st.session_state
stss = st.session_state

st.set_page_config(
    page_title="Net View",
    page_icon="🌐")

if "col" not in stss:
    init_pf_session_state()
stss.setdefault("layout_seq", 0)
stss.setdefault("compare_nets", [])   # the comparison set: PFnet objects, in the order added
stss.setdefault("compare_version", 0)  # bumped to reset the comparison table's checkboxes

# layouts that start from random positions, so Redraw gives a different result
REDRAWABLE = ["gravity", "kamada_kawai", "davidson_harel", "spring"]


def show_net(name: str) -> None:
    """Make the named net the focus net; bump layout_version so layout widgets reset."""
    stss.focus_net_name = name
    stss.layout_version += 1
    stss.layout_seq += 1


def new_net():
    """Callback for the network selectbox."""
    show_net(stss.selected_net_name)


# ----------------------------------------------------------------- comparison set
# Kept as the network objects, not names, so a deleted network (or a replaced project)
# drops out by itself and a later network that reuses a name is not taken for it.

def compare_nets() -> list:
    """The comparison set's networks that are still in the project."""
    nets = [net for net in stss.compare_nets if stss.col.pfnets.get(net.name) is net]
    stss.compare_nets = nets
    return nets


def compare_add_problem(net, nets: list) -> str | None:
    """Why the net cannot join the set, or None if it can. The set holds only networks
    with the same terms, so they can be compared with the nodes in the same places."""
    if any(net is n for n in nets):
        return "The network in view is already in the set."
    if nets and net.termsid != nets[0].termsid:
        return (f"The set holds networks with the terms of {nets[0].name}; "
                f"{net.name} has different terms.")
    return None


def add_to_compare():
    net = get_focus_net()
    if net is not None and compare_add_problem(net, compare_nets()) is None:
        stss.compare_nets.append(net)
        stss.compare_version += 1


def clear_compare():
    stss.compare_nets = []
    stss.compare_version += 1


def compare_table_changed(key: str, nets: list):
    """Callback for the comparison table: a ticked view box shows that network,
    a ticked delete box removes it from the set."""
    edits = stss[key]["edited_rows"]
    for row, change in edits.items():
        if change.get("view"):
            show_net(nets[row].name)
            if "selected_net_name" in stss:
                stss.selected_net_name = nets[row].name  # keep the selector in step
    removed = {row for row, change in edits.items() if change.get("delete")}
    stss.compare_nets = [net for i, net in enumerate(nets) if i not in removed]
    stss.compare_version += 1


def redraw():
    """Callback for the Redraw button in the network view."""
    net = get_focus_net()
    if net is not None:
        net.get_layout(method=getattr(net, "layout_request", "auto"))
        stss.layout_seq += 1


def save_layout():
    """Callback for the Save Layout button: store the focus net's coords for its terms and
    switch this net to the "saved" layout."""
    net = get_focus_net()
    if net is not None:
        stss.col.save_layout(net)  # also sets net.layout_request = "saved"
        stss.layout_version += 1   # new selector widget, so it shows "saved"
        stss.layout_seq += 1       # view's Reset now returns to the saved positions
        st.toast(f"Layout saved for all networks with the terms of {net.name}", icon="💾")


if not stss.col.pfnets:
    st.info("No networks yet. Add proximity data on the **Build Nets** page.")
    st.stop()

#---------------------------------------------------------------------------------------------------------sidebar
# Sidebar contents:
#   - Network selector: chooses the focus net (shown only when more than one net exists;
#     with a single net it is selected automatically). Changing it calls new_net().
#   - Select Algorithm: picks the layout method; changing it recomputes the layout.
# Orientation, redraw, physics, text size and rank tools are in the network view toolbar.
with st.sidebar:
    pfnet_names = sorted_keys(dic=stss.col.pfnets, )
    nnets = len(pfnet_names)
    if get_focus_net() is None:  # first visit, or the shown net was deleted
        stss.focus_net_name = pfnet_names[0]
        stss.layout_seq += 1
    if nnets > 1:
        index = pfnet_names.index(stss.focus_net_name)
        st.selectbox(label="**Select Network** (searchable)",
                     options=pfnet_names,
                     index=index,
                     filter_mode="fuzzy",
                     key="selected_net_name",
                     on_change=new_net,
                     )
    tight_divider()
    focus_net = get_focus_net()

    # each net remembers the algorithm chosen for it, so returning to this page
    # (or switching back to a net) keeps its layout instead of recomputing it
    has_saved = stss.col.has_saved_layout(focus_net.termsid)
    layout_options = (["saved"] if has_saved else []) + stss.layouts
    current = getattr(focus_net, "layout_request", "auto")
    if current not in layout_options:
        current = "auto"
    layout_choice = st.selectbox(
        label="**Select Layout Algorithm**",
        options=layout_options,
        index=layout_options.index(current),
        filter_mode="prefix",
        key=f"layout_selector_{stss.layout_version}",
        help="**saved** uses the layout saved with the button below for networks "
             "with these terms.",
    )
    if layout_choice != current:
        if layout_choice == "saved":
            stss.col.apply_saved_layout(focus_net)
        else:
            focus_net.get_layout(method=layout_choice)
        stss.layout_seq += 1

    st.button("💾 Save Layout", on_click=save_layout, width="stretch",
              help="Save the current node positions. Every network with the same terms "
                   "can then use them by choosing **saved** above. Saved layouts are "
                   "stored in the project file.")

    tight_divider()
    st.markdown("**Comparison Set**")
    nets = compare_nets()
    add_problem = compare_add_problem(focus_net, nets)
    with st.container(horizontal=True, gap="small"):
        st.button("➕ Add", on_click=add_to_compare, disabled=add_problem is not None,
                  help=add_problem or "Add the network in view to the comparison set.")
        st.button("Clear All", on_click=clear_compare, disabled=not nets,
                  help="Empty the comparison set.")
    if nets:
        key = f"compare_table_{stss.compare_version}"
        st.data_editor(
            pd.DataFrame({"view": [net is focus_net for net in nets],
                          "network": [net.name for net in nets],
                          "delete": [False] * len(nets)}),
            key=key, on_change=compare_table_changed, args=(key, nets),
            hide_index=True, width="stretch", disabled=["network"],
            column_config={
                "view": st.column_config.CheckboxColumn("view", width="small",
                                                        help="Tick to show this network."),
                "delete": st.column_config.CheckboxColumn("delete", width="small",
                                                          help="Tick to remove it from the set."),
            },
        )

#-------------------------------------------------------------------------------------------  MAIN PANEL

planarity = "planar" if focus_net.is_planar else "nonplanar"
st.write(f"**{focus_net.name}: {focus_net.nnodes} nodes, "
         f"{focus_net.nlinks} links** -- {planarity} -- Layout: {focus_net.layout_method} ")

base_font = stss.get("text_size", 18)
scale = np.clip(30.0 / focus_net.nnodes, a_min=1.2, a_max=2.0)

network_view(
    focus_net,
    layout_seq=stss.layout_seq,
    font_size=round(scale * base_font),
    can_redraw=layout_choice in REDRAWABLE,
    on_redraw=redraw,
)
