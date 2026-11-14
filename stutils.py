# stutils.py: streamlit helpers for the app (project root, kept out of the pypf package)
import time
import streamlit as st
from pathlib import Path
from typing import Literal
import pypf
from pypf.collection import Collection
from pypf.proximity import Proximity
from pypf.legacy_files import legacy_txt_files
import pandas as pd
import numpy as np
import os
import tempfile
import copy
from pypf.utility import get_terms

# alias for st.session_state
stss = st.session_state

# data files ship inside the pypf package
DATA_DIR = Path(pypf.__file__).parent / "data"

@st.cache_data
def get_cached_help_html():
    """Loads and caches the help content from disk."""
    help_path = DATA_DIR / "help.html"
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
    files = ["bank6", "bio", "psy", "bank", "cities", "statecaps"]
    srcs = ["coocurrences in Longman's dictionary",
            "relatedness ratings by biology grad students",
            "relatedness ratings by psychology 101 students",
            "coocurrences in Longman's dictionary",
            "distances (km) between cities in the US",
            "distances (miles) between state capitals in the US"]
    return pd.DataFrame(data={"file": files, "source": srcs})

@st.cache_data
def get_example_file_format():
    tb6 = ["account", "bank", "flood", "money", "river", "save"]
    bank6 = pd.DataFrame(np.array([[0, 3, 32, 26, 32, 32], [3, 0, 27, 21, 14, 30],
                                   [32, 27, 0, 31, 23, 29], [26, 21, 31, 0, 31, 23],
                                   [32, 14, 23, 31, 0, 31], [32, 30, 29, 23, 31, 0]]),
                         index=tb6, columns=tb6)
    return bank6

@st.cache_resource(show_spinner="Loading sample data…")
def get_sample_collection() -> Collection:
    """Builds the sample Collection once per server process.
    Shared by all sessions: never modify it, always deepcopy it."""
    col = Collection()
    col.load_project_state(get_cached_sample_json())
    return col

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

def make_symmetric_if_force(prx:Proximity) -> Proximity:
    if stss.force_symmetric and not prx.issymmetric:
        prx.dismat = np.minimum(prx.dismat, prx.dismat.T)
        prx.name += "_fs"
        prx.issymmetric = True
    return prx

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

    if st.button("🚀 Confirm and Load", type="primary", key="confirm_load_btn"):
        try:
            project_data = uploaded_file.read().decode("utf-8")
            col.load_project_state(project_data)

            fname = os.path.basename(uploaded_file.name)
            stss.project_name = fname.split(".")[0]
            stss.proj_name_ver += 1
            st.success("Project Added Successfully!")

            stss.manage_data_version += 1
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("❌ Cancel", key="cancel_staged_load_btn"):
        stss.manage_data_version += 1 # Wipes the staged file key
        st.rerun()

def _is_valid_prx(prx: Proximity) -> bool:
    """Proximity() reports read problems by printing, so check what it produced."""
    return prx.nterms >= 2 and prx.dismat.shape == (prx.nterms, prx.nterms)

def add_proximity_files(col: Collection) -> None:
    errors = []
    stss.perm_file_type = st.radio("**Proximity Data File Type**", ["Spreadsheet", "Legacy Text"],
                                   index=get_idx(stss.perm_file_type, ["Spreadsheet", "Legacy Text"]),
                                   key="prx_file_type", )

    stss.force_symmetric = st.toggle("Force Proximity Symmetric", value=False,
                                    help="When enabled, uses minimum of dij and dji if asymmetric",
                                     key=f"force_symmetric_{stss.manage_data_version}", )
    # Proximity File loaders
    match stss.perm_file_type:
        case "Spreadsheet":
            # st.write("**Add Proximity Spreadsheet Files**")
            ss_files = st.file_uploader("**Browse for or Drag and Drop Proximity Spreadsheets (.xlsx or .csv)**", type=["xlsx", "csv"],
                                        key=f"ss_files{stss.file_version}", accept_multiple_files=True,
                                        help="Upload one or more Proximity files (.xlsx or .csv) to add to the app.",
                                        )
            if len(ss_files) > 0:
                # Sort the uploaded files alphabetically by their filename before processing
                sorted_ss_files = sorted(ss_files, key=lambda f: f.name)
                for ss_file in sorted_ss_files:
                    if ss_file is not None:
                        try:
                            new_prx = Proximity(filepath=ss_file)
                            if not _is_valid_prx(new_prx):
                                errors.append(f"{ss_file.name}: not a valid proximity spreadsheet "
                                              "(see Help for the required format)")
                                continue
                            new_prx = make_symmetric_if_force(new_prx)
                            col.add_proximity(new_prx)
                        except Exception:
                            errors.append(f"{ss_file.name}: not a valid proximity spreadsheet "
                                          "(see Help for the required format)")
                stss.upload_errors = errors
                stss.file_version += 1
                stss.manage_data_version += 1
                st.rerun()

        case "Legacy Text":
            # st.subheader("Add Legacy Proximity and Terms Text Files")
            leg_files = st.file_uploader("**Browse for or Drag and Drop Legacy Proximity and Terms Text Files (.txt)**", type=["txt"],
                                         accept_multiple_files=True,
                                         help="Upload one or more files (.txt) to add to the app.",
                                         key=f"leg_files_key{stss.file_version}", )
            if len(leg_files) > 0:
                with tempfile.TemporaryDirectory() as leg_dir:
                    #os.makedirs(leg_dir, exist_ok=True)
                    for leg_file in leg_files:
                        if leg_file is not None:
                            file = os.path.join(leg_dir, os.path.basename(leg_file.name))
                            with open(file, "wb") as f:
                                f.write(leg_file.getbuffer())
                    all_leg_files = sorted(os.listdir(leg_dir))
                    for leg_file in all_leg_files:
                        # terms files are read by get_terms() below, not as data
                        if leg_file.lower().endswith(".trm.txt") or leg_file.lower() == "terms.txt":
                            continue
                        fname = os.path.join(leg_dir, leg_file)
                        try:
                            with open(fname, errors="replace") as f:
                                raw_lines = f.readlines()
                            pdict = legacy_txt_files(prx_file=fname, raw_lines=raw_lines)
                            if pdict["error"]:  # error occurred on this file
                                errors.append(f"{leg_file}: {pdict['error']}")
                                continue
                            pdict["terms"] = get_terms(name=pdict["name"], n=pdict["nnodes"], fdir=leg_dir)
                            new_prx = Proximity(terms=pdict["terms"], dismat=pdict["dismat"], name=pdict["name"])
                            if not _is_valid_prx(new_prx):
                                errors.append(f"{leg_file}: not a valid proximity file")
                                continue
                            new_prx = make_symmetric_if_force(new_prx)
                            col.add_proximity(new_prx)
                        except Exception as e:
                            errors.append(f"{leg_file}: {e}")
                    # the with-block deletes leg_dir and its files on exit
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
    stss.col.focus_net = stss.col.pfnets["bio_pf"]
    stss.help_html = get_cached_help_html()
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
