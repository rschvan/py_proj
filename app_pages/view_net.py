# app_pages/view_net.py
import numpy as np
import streamlit as st

from netview import network_view
from pypf.utility import sorted_keys
from stutils import init_pf_session_state, tight_divider

# alias for st.session_state
stss = st.session_state

st.set_page_config(
    page_title="Net View",
    page_icon="🌐")

if "col" not in stss:
    init_pf_session_state()
stss.setdefault("layout_seq", 0)

# layouts that start from random positions, so Redraw gives a different result
REDRAWABLE = ["gravity", "kamada_kawai", "davidson_harel", "spring"]


def new_net():
    """Callback for the network selectbox: make the selected net the focus net
    and bump layout_version so layout widgets reset for the new network."""
    new = stss.selected_net_name
    stss.col.focus_net = stss.col.pfnets[new]
    stss.layout_version += 1
    stss.layout_seq += 1


def redraw():
    """Callback for the Redraw button in the network view."""
    net = stss.col.focus_net
    if net is not None:
        net.get_layout(method=getattr(net, "layout_request", "auto"))
        stss.layout_seq += 1


def save_layout():
    """Callback for the Save Layout button: store the focus net's coords for its terms."""
    net = stss.col.focus_net
    if net is not None:
        stss.col.save_layout(net)
        if getattr(net, "layout_request", None) == "saved":
            stss.col.apply_saved_layout(net)  # Reset now returns to the newly saved layout
            stss.layout_seq += 1
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
    if stss.col.focus_net is None or stss.col.focus_net.name not in stss.col.pfnets:
        stss.col.focus_net = stss.col.pfnets[pfnet_names[0]]
        stss.layout_seq += 1
    if nnets > 1:
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
    st.caption("Use the toolbar above the network to orient, redraw, "
               "and adjust text. Drag nodes to move them.")

    st.button("💾 Save Layout", on_click=save_layout, width="stretch",
              help="Save the current node positions. Every network with the same terms "
                   "can then use them by choosing **saved** above. Saved layouts are "
                   "stored in the project file.")
    if has_saved:
        st.caption("A saved layout exists for these terms.")

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
