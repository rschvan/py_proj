# stutils.py: streamlit helpers for the app (project root, kept out of the pypf package)
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import pypf
from pypf.collection import Collection
from pypf.readers import read_spreadsheet

# alias for st.session_state
stss = st.session_state

# data files ship inside the pypf package
DATA_DIR = pypf.sample_path()

@st.cache_data
def get_cached_help_html():
    """Loads and caches the help content from disk."""
    # app content, kept with the app (static/ is also served at app/static/help.html)
    help_path = Path(__file__).parent / "static" / "help.html"
    if help_path.exists():
        return help_path.read_text(encoding="utf-8", errors="replace")
    return "<h3>Help content not found.</h3>"

@st.cache_data
def get_cached_sample_json():
    """Loads and caches the raw sample project JSON string."""
    jf_path = DATA_DIR / "sample_proj.json"
    if jf_path.exists():
        return jf_path.read_text(encoding="utf-8")
    return None

@st.cache_data
def get_sample_sources():
    """Descriptions of the sample data sets (pypf/data/samples.csv)."""
    return pd.read_csv(DATA_DIR / "samples.csv")

@st.cache_data
def get_example_file_format():
    """The bank6 sample as a table, to show the spreadsheet layout."""
    prx = read_spreadsheet("bank6.prx.xlsx", (DATA_DIR / "bank6.prx.xlsx").read_bytes())
    dis = prx.dismat.astype(int) if np.all(prx.dismat == prx.dismat.round()) else prx.dismat
    return pd.DataFrame(dis, index=list(prx.terms), columns=list(prx.terms))

def pypf_code_id() -> str:
    """A fingerprint of the pypf source files; it changes whenever pypf's code changes."""
    import hashlib
    h = hashlib.sha1()
    for path in sorted(Path(pypf.__file__).parent.glob("*.py")):
        h.update(path.read_bytes())
    return h.hexdigest()


@st.cache_resource(show_spinner="Loading sample data…", max_entries=1)
def get_sample_collection(code_id: str) -> Collection:
    """Builds the sample Collection once per server process.
    Shared by all sessions: never modify it, always deepcopy it.

    code_id (from pypf_code_id) is part of the cache key. Streamlit keeps cached objects
    when the code is updated in place (as Streamlit Cloud does after a push), and a
    Collection built by the old code lacks any new methods; the new code_id rebuilds it."""
    col = Collection()
    col.load_project_state(get_cached_sample_json())
    return col


# ----------------------------------------------------------------- demo data
# Demo items are tracked by object, not by name, so a file of the user's that has a sample's
# name is never taken for demo data. stss.demo_items maps id -> object; holding the objects
# also keeps their ids from being reused while they are tracked.

def load_demo() -> None:
    """Adds a copy of the sample data to the project and tracks it as the demo."""
    prxs, nets = stss.col.add_copy_of(stss.sample_col)
    stss.demo_items = {id(item): item for item in [*prxs, *nets]}


def mark_demo(item) -> None:
    """Tracks an item made only from demo data (an average), so Clear Demo removes it."""
    stss.demo_items[id(item)] = item


def is_demo(item) -> bool:
    return id(item) in stss.demo_items


def demo_names() -> tuple[list[str], list[str]]:
    """Names of the demo proximities and networks still in the project."""
    col = stss.col
    return ([name for name, prx in col.proximities.items() if is_demo(prx)],
            [name for name, net in col.pfnets.items() if is_demo(net)])


def demo_loaded() -> bool:
    return any(demo_names())


def clear_demo() -> None:
    """Removes the demo items and what depends on them. Averages that also used the
    user's own proximities are kept."""
    prxs, nets = demo_names()
    stss.col.delete_proximities(prxs)
    stss.col.delete_pfnets(nets)
    stss.demo_items = {}

# ----------------------------------------------------------------- table formatting
# pypf returns unrounded numbers and flags; these helpers shape them for display.

def sig_figs(x, n: int = 3):
    """x rounded to n significant figures (whole numbers without .0); non-finite x unchanged."""
    if x is None or not np.isfinite(x) or x == 0:
        return x
    mag = int(np.floor(np.log10(abs(x))))
    if mag > n - 1:
        return int(round(x))
    rounded = round(x, n - 1 - mag)
    return int(rounded) if rounded == int(rounded) else rounded

def _param_text(v) -> str:
    """q or r for display: "inf", a whole number without .0, or blank (nn and merged nets)."""
    if v is None or pd.isna(v):
        return ""
    if np.isinf(v):
        return "inf"
    return f"{v:g}"

def proximity_table(col: Collection) -> pd.DataFrame | None:
    df = col.get_proximity_info()
    if df is not None:
        for c in ("mean", "sd", "min", "max"):
            df[c] = df[c].map(sig_figs)
    return df

def network_table(col: Collection) -> pd.DataFrame | None:
    df = col.get_pfnet_info()
    if df is not None:
        for c in ("q", "r"):
            df[c] = df[c].map(_param_text)
    return df

def properties_table(net) -> pd.DataFrame:
    """Network properties with the flag columns showing the node's name (blank if not)."""
    df = pd.DataFrame(net.get_network_properties())
    for c in ("max degree", "center", "median"):
        if c in df:
            df[c] = [node if flag else "" for node, flag in zip(df["node"], df[c])]
    return df

def get_focus_net():
    """The network shown on the View Network page, or None if it no longer exists."""
    return stss.col.pfnets.get(stss.get("focus_net_name"))

def get_idx(val, opt_list):
    try:
        return opt_list.index(val)
    except ValueError:
        return 0  # Default to the first item if something goes wrong

def tight_divider(space_px=2, color="green", thickness=2):
    """
    Renders a thin horizontal line with minimal vertical padding.
    """
    st.markdown(
        f"<hr style='margin-top: {space_px}px; margin-bottom: {space_px}px; "
        f"border: 0; border-top: {thickness}px solid {color};'>",
        unsafe_allow_html=True
    )

REVEAL_KEY = "pf_reveal"  # prefix for st.container keys whose contents should be scrolled into view

_REVEAL_JS = """
<script>
(() => {
  if (window.__pfReveal) return;
  window.__pfReveal = true;
  const TOP = 70, BOTTOM = 16;  // keep clear of the app header and the window edge
  function scroller(el) {
    for (let p = el.parentElement; p; p = p.parentElement) {
      const s = getComputedStyle(p).overflowY;
      if ((s === "auto" || s === "scroll") && p.scrollHeight > p.clientHeight) return p;
    }
    return document.scrollingElement;
  }
  // Scroll just enough to show the bottom of el, without pushing its top out of sight.
  function reveal(el) {
    const r = el.getBoundingClientRect();
    const dy = Math.min(r.bottom - window.innerHeight + BOTTOM, r.top - TOP);
    if (dy > 0) scroller(el).scrollBy({top: dy, behavior: "smooth"});
  }
  const later = (el, ms) => setTimeout(() => reveal(el), ms);
  // Expanders open in the browser, with no rerun.
  document.addEventListener("toggle", (e) => {
    const d = e.target;
    if (d.tagName === "DETAILS" && d.open) later(d.closest('[data-testid="stExpander"]') || d, 350);
  }, true);
  // Results shown after a button click arrive in a keyed container.
  const sel = '[class*="st-key-""" + REVEAL_KEY + """"]';
  new MutationObserver((muts) => {
    for (const m of muts) for (const n of m.addedNodes) {
      if (!(n instanceof Element)) continue;
      const el = n.matches(sel) ? n : n.querySelector(sel);
      if (el) { later(el, 400); return; }
    }
  }).observe(document.body, {childList: true, subtree: true});
})();
</script>
"""


def keep_panels_in_view():
    """Scroll opened expanders and newly shown REVEAL_KEY containers into view."""
    st.html(_REVEAL_JS, unsafe_allow_javascript=True)


def get_selected_values(df, event, column: str = "name") -> list:
    # gets the values in column for selected rows and/or rows of selected cells
    sel = event.selection
    row_indices = set(sel.rows)
    for cell in sel.cells:
        row_indices.add(cell[0])
    col = df[column]
    return [col.iat[i] for i in row_indices if 0 <= i < len(df)]

def save_project_button(col: Collection, name: str = "pypf_project") -> None:
    """Sidebar button that downloads the project as <name>.json."""
    st.download_button(
        label="💾 Save Project",
        data=col.get_project_state,  # called only when clicked
        file_name=f"{name}.json",
        mime="application/json",
        on_click="ignore",  # no rerun needed
        width="stretch",
        help=f"Download the project as {name}.json, to load again later.",
    )


LOAD_MODES = ["Replace current project", "Add to current project"]


@st.dialog("Load Project", width="medium")
def load_project_dialog() -> None:
    """Choose a saved project file and replace the current project with it or add it."""
    col = stss.col
    uploaded_file = st.file_uploader(
        "**Browse for or drag and drop a saved project (.json)**",
        type="json",
        key=f"main_project_loader_{stss.manage_data_version}",
    )
    if uploaded_file is None:
        return

    mode = st.radio("**Load into**", LOAD_MODES, horizontal=True,
                    key=f"load_mode_{stss.manage_data_version}",
                    help="**Replace** starts fresh with only the project in the file "
                         "(including removing the demo data). **Add** keeps what is "
                         "loaded now; names already in use get a number added.")

    if st.button("📂 Load", type="primary", key="confirm_load_btn"):
        try:
            project_data = uploaded_file.read().decode("utf-8")
            if mode == LOAD_MODES[0]:
                # load into a new collection first, so a bad file leaves the current one intact
                stss.col = Collection.from_project_state(project_data)
                stss.focus_net_name = None    # View Network shows the first network
                stss.layout_seq += 1
            else:
                Collection.from_project_state(project_data)  # check it loads before adding
                col.load_project_state(project_data)

            fname = os.path.basename(uploaded_file.name)
            stss.project_name = fname.split(".")[0]
            stss.proj_name_ver += 1
            stss.manage_data_version += 1
            st.rerun()  # closes the dialog
        except Exception as e:
            st.error(f"{uploaded_file.name} could not be loaded; it does not look like a "
                     f"PyPathfinder project file. Nothing was changed. (Details: {e})")


FILE_TYPES = {"Spreadsheet": "spreadsheet", "Legacy Text": "legacy"}


@st.dialog("Add Data Files", width="medium")
def add_files_dialog() -> None:
    """Upload proximity files; the reading and adding is done by col.add_files."""
    stss.perm_file_type = st.radio("**Proximity Data File Type**", list(FILE_TYPES),
                                   index=get_idx(stss.perm_file_type, list(FILE_TYPES)),
                                   key="prx_file_type", horizontal=True)
    force_symmetric = st.toggle("Force Proximity Symmetric", value=False,
                                help="When enabled, uses minimum of dij and dji if asymmetric",
                                key=f"force_symmetric_{stss.manage_data_version}", )
    if stss.perm_file_type == "Spreadsheet":
        uploaded = st.file_uploader(
            "**Browse for or drag and drop proximity spreadsheets (.xlsx or .csv)**",
            type=["xlsx", "csv"], accept_multiple_files=True,
            help="Upload one or more Proximity files (.xlsx or .csv) to add to the app.",
            key=f"ss_files{stss.file_version}", )
        with st.expander("Spreadsheet layout example (bank6)"):
            st.caption("Terms in the first row and column, distances or dissimilarities in the matrix.")
            st.dataframe(stss.example_file_format, width="content", hide_index=False)
    else:
        uploaded = st.file_uploader(
            "**Browse for or drag and drop legacy proximity and terms text files (.txt)**",
            type=["txt"], accept_multiple_files=True,
            help="Upload data files together with their terms files (.trm.txt or terms.txt).",
            key=f"leg_files_key{stss.file_version}", )

    if uploaded:
        files = {f.name: f.getvalue() for f in uploaded}
        _, errors = stss.col.add_files(files, kind=FILE_TYPES[stss.perm_file_type],
                                       force_symmetric=force_symmetric)
        stss.upload_errors = errors   # shown on the page after the dialog closes
        stss.file_version += 1
        stss.manage_data_version += 1
        st.rerun()  # closes the dialog


def init_pf_session_state():
    stss.home_dir = os.path.dirname(os.path.abspath(__file__))
    stss.sample_col = get_sample_collection(pypf_code_id())  # shared, read-only
    stss.sample_sources = get_sample_sources()
    stss.example_file_format = get_example_file_format()
    stss.col = Collection()
    load_demo()                                          # this session's own copy
    stss.focus_net_name = "bio_pf"                       # network shown on View Network
    stss.layouts = ["auto",
                    "gravity",
                    "kamada_kawai",
                    "tree",
                    "radial_tree",
                    "mds_distance",
                    "mds_graph",
                    #"fruchterman_reingold",
                    #"davidson_harel",
                    "spring",
                    "spiral",
                    "planar",
                    "sugiyama",
                    #"circle",
                    ]
    stss.layout = "auto"
    stss.layout_seq = 0   # bumped when Python computes a new layout for the view
    stss.count = 0
    stss.q_param = np.inf
    stss.r_param = np.inf
    stss.intro_visible = True
    stss.perm_file_type = "Spreadsheet"
    stss.perm_net_type = "Pathfinder"
    stss.project_name = "pypf_project"
    stss.proj_name_ver = 0
    stss.prx_version = 0
    stss.net_version = 0
    stss.file_version = 0
    stss.ave_version = 0
    stss.mrg_version = 0
    stss.layout_version = 0
    stss.manage_data_version = 0
