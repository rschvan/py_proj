# stutils.py: streamlit helpers for the app (project root, kept out of the pypf package)
import copy
import os
import time
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

@st.cache_resource(show_spinner="Loading sample data…")
def get_sample_collection() -> Collection:
    """Builds the sample Collection once per server process.
    Shared by all sessions: never modify it, always deepcopy it."""
    col = Collection()
    col.load_project_state(get_cached_sample_json())
    return col

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

def get_selected_values(df, event, column: str = "name") -> list:
    # gets the values in column for selected rows and/or rows of selected cells
    sel = event.selection
    row_indices = set(sel.rows)
    for cell in sel.cells:
        row_indices.add(cell[0])
    col = df[column]
    return [col.iat[i] for i in row_indices if 0 <= i < len(df)]

def save_project(col: Collection, name: str = "pypf_project") -> None:
    # Lazy download evaluation
    st.download_button(
        label=f"**📥 Save project {name} in .json file**",
        data=col.get_project_state,  # Callable deferred until click
        file_name=f"{name}.json",
        mime="application/json",
        use_container_width=False, )
    st.caption("Merged networks are not saved; merge them again after loading.")
    stss.manage_data_version += 1

    # Scoped cancellation gate
    if st.button("**❌ Cancel Save**", ):
        stss.manage_data_version += 1
        st.rerun()

LOAD_MODES = ["Replace current project", "Add to current project"]

def load_project(col: Collection) -> None:
    uploaded_file = st.file_uploader(
        "**Browse for or drag and drop saved .json file**",
        type="json",
        key=f"main_project_loader_{stss.manage_data_version}",
        width=300,
    )

    # No file selected yet (waiting on user or cancelled in OS dialog)
    if uploaded_file is None:
        if st.button("❌ Cancel Load", key=f"abort_empty_load_{stss.manage_data_version}"):
            stss.manage_data_version += 1 # Resets the uploader widget key
            st.rerun()
        return

    # File has been selected
    mode = st.radio("**Load into**", LOAD_MODES, horizontal=True,
                    key=f"load_mode_{stss.manage_data_version}",
                    help="**Replace** starts fresh with only the project in the file "
                         "(including removing the sample data). **Add** keeps what is "
                         "loaded now; names already in use get a number added.")

    if st.button("🚀 Confirm and Load", type="primary", key="confirm_load_btn"):
        try:
            project_data = uploaded_file.read().decode("utf-8")
            if mode == LOAD_MODES[0]:
                # load into a new collection first, so a bad file leaves the current one intact
                stss.col = Collection.from_project_state(project_data)
                stss.sample_visible = False   # the sample data is gone
                stss.focus_net_name = None    # View Network shows the first network
                stss.layout_seq += 1
                done = "Project loaded"
            else:
                Collection.from_project_state(project_data)  # check it loads before adding
                col.load_project_state(project_data)
                done = "Project added"

            fname = os.path.basename(uploaded_file.name)
            stss.project_name = fname.split(".")[0]
            stss.proj_name_ver += 1
            st.success(f"{done} from {fname}")

            stss.manage_data_version += 1
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"{uploaded_file.name} could not be loaded; it does not look like a "
                     f"PyPathfinder project file. Nothing was changed. (Details: {e})")

    if st.button("❌ Cancel", key="cancel_staged_load_btn"):
        stss.manage_data_version += 1 # Wipes the staged file key
        st.rerun()

FILE_TYPES = {"Spreadsheet": "spreadsheet", "Legacy Text": "legacy"}

def add_proximity_files(col: Collection) -> None:
    """Upload widgets for proximity files; the reading and adding is done by col.add_files."""
    stss.perm_file_type = st.radio("**Proximity Data File Type**", list(FILE_TYPES),
                                   index=get_idx(stss.perm_file_type, list(FILE_TYPES)),
                                   key="prx_file_type", )
    force_symmetric = st.toggle("Force Proximity Symmetric", value=False,
                                help="When enabled, uses minimum of dij and dji if asymmetric",
                                key=f"force_symmetric_{stss.manage_data_version}", )
    if stss.perm_file_type == "Spreadsheet":
        uploaded = st.file_uploader(
            "**Browse for or Drag and Drop Proximity Spreadsheets (.xlsx or .csv)**",
            type=["xlsx", "csv"], accept_multiple_files=True,
            help="Upload one or more Proximity files (.xlsx or .csv) to add to the app.",
            key=f"ss_files{stss.file_version}", )
    else:
        uploaded = st.file_uploader(
            "**Browse for or Drag and Drop Legacy Proximity and Terms Text Files (.txt)**",
            type=["txt"], accept_multiple_files=True,
            help="Upload data files together with their terms files (.trm.txt or terms.txt).",
            key=f"leg_files_key{stss.file_version}", )

    if uploaded:
        files = {f.name: f.getvalue() for f in uploaded}
        _, errors = col.add_files(files, kind=FILE_TYPES[stss.perm_file_type],
                                  force_symmetric=force_symmetric)
        stss.upload_errors = errors
        stss.file_version += 1
        stss.manage_data_version += 1
        st.rerun()

    if st.button("❌ Cancel Add Proximity", ):
        stss.manage_data_version += 1
        st.rerun()

def init_pf_session_state():
    stss.home_dir = os.path.dirname(os.path.abspath(__file__))
    stss.sample_col = get_sample_collection()          # shared, read-only
    stss.sample_sources = get_sample_sources()
    stss.example_file_format = get_example_file_format()
    stss.col = copy.deepcopy(stss.sample_col)          # this session's own copy
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
    stss.sample_visible = True
    stss.perm_file_type = "Spreadsheet"
    stss.perm_net_type = "Pathfinder"
    stss.data_option = "None"
    stss.project_name = "pypf_project"
    stss.proj_name_ver = 0
    stss.prx_version = 0
    stss.net_version = 0
    stss.file_version = 0
    stss.ave_version = 0
    stss.mrg_version = 0
    stss.layout_version = 0
    stss.manage_data_version = 0
