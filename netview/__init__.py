# netview: two-way Streamlit component (CCv2) that draws a PFnet with vis-network.
# The browser owns orientation, physics, text size and rank display; it reports node
# positions back so PFnet.coords stays in sync, and asks Python to redraw the layout.
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyvis
import streamlit as st

from pypf.utility import split_long_term

_HERE = Path(__file__).parent
# vis-network ships inside pyvis; its UMD bundle sets globalThis.vis, so it can be
# prepended to the component's ES module.
_VIS_JS = (Path(pyvis.__file__).parent / "lib" / "vis-9.1.2" / "vis-network.min.js").read_text(
    encoding="utf-8")

# canonical (-1..1, y up) coords <-> vis canvas units (y down)
_SCALE = np.array([650.0, -450.0])

_COMPONENT = st.components.v2.component(
    "pypf_netview",
    html=(_HERE / "netview.html").read_text(encoding="utf-8"),
    css=(_HERE / "netview.css").read_text(encoding="utf-8"),
    js=_VIS_JS + "\n;\n" + (_HERE / "netview.js").read_text(encoding="utf-8"),
)

_TOKEN_KEY = "_netview_token"  # session_state: token of the net currently shown


def _to_canvas(coords: np.ndarray) -> list[list[float]]:
    return (np.asarray(coords, dtype=float) * _SCALE).round(3).tolist()


def network_view(net, *, layout_seq: int, font_size: int, can_redraw: bool,
                 on_redraw: Callable[[], None], key: str = "netview"):
    """Mounts the network view for `net`.

    layout_seq: bump it whenever Python computes a new layout (or shows a different
        net); the browser then redraws from net.coords instead of keeping its own positions.
    on_redraw: called (before the script body) when the user presses Redraw.
    """
    token = f"{id(net)}-{layout_seq}"
    st.session_state[_TOKEN_KEY] = token

    def _apply_coords():
        msg = (st.session_state.get(key) or {}).get("coords") or {}
        current = st.session_state.get("col").focus_net if "col" in st.session_state else None
        # ignore positions reported for a net or layout that is no longer shown
        if current is None or msg.get("token") != st.session_state.get(_TOKEN_KEY):
            return
        xy = np.asarray(msg.get("xy", []), dtype=float)
        if xy.shape == (current.nnodes, 2):
            current.coords = xy / _SCALE

    base = getattr(net, "layout_coords", None)
    if base is None:
        base = net.coords
    nodes = [
        {"id": i, "label": split_long_term(term), "title": f"Node: {term}", "x": x, "y": y}
        for i, (term, (x, y)) in enumerate(zip(net.terms, _to_canvas(net.coords)))
    ]
    edges = [{"from": int(i), "to": int(j), "rank": int(rank)}
             for (i, j), rank in net.links.items()]
    # Streamlit drops a component's own state when its page is left, so the text/label
    # settings are copied to a plain session key that survives page switches.
    view_key = f"_{key}_view"

    def _keep_view():
        st.session_state[view_key] = (st.session_state.get(key) or {}).get("view")

    view = st.session_state.get(view_key)

    return _COMPONENT(
        key=key,
        data={
            "token": token,
            "name": net.name,
            "nodes": nodes,
            "edges": edges,
            "base": _to_canvas(base),
            "directed": bool(net.isdirected),
            "maxRank": max(net.links.values()) if net.links else 1,
            "canRedraw": can_redraw,
            "fontSize": font_size,
            "view": view,
        },
        on_coords_change=_apply_coords,
        on_view_change=_keep_view,
        on_redraw_change=on_redraw,
    )
