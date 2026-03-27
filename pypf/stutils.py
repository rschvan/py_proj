# stutils.py
import streamlit as st
from pypf.collection import Collection
import pandas as pd
import numpy as np
import os
import copy

# alias for st.system_state
stss = st.session_state

@st.cache_data
def get_cached_help_html():
    """Loads and caches the help content from disk."""
    help_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pypf", "data", "help.html")
    if os.path.exists(help_path):
        with open(help_path, "r", encoding="utf-8", errors='replace') as f:
            return f.read()
    return "<h3>Help content not found.</h3>"

@st.cache_data
def get_cached_sample_json():
    """Loads and caches the raw sample project JSON string."""
    jf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pypf", "data", "sample_proj.json")
    if os.path.exists(jf_path):
        with open(jf_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

@st.cache_data
def get_sample_sources():
    files = ["bank6","bio", "psy", "bank", "cities", "statecaps"]
    srcs = ["coocurrences in Longman's dictionary",
        "relatedness ratings by biology grad students",
         "relatedness ratings by psychology 101 students",
         "coocurrences in Longman's dictionary",
         "distances (km) between cities in the US",
         "distances (miles) between state capitals in the US"]
    return pd.DataFrame(data={"file": files, "source": srcs})

@st.cache_data
def get_example_file_format():
    tb6 = ["account","bank","flood","money","river", "save"]
    bank6 = pd.DataFrame(np.array([[0, 3,32,26,32,32],[3,0,27,21,14,30],
                                           [32,27,0,31,23,29],[26,21,31,0,31,23],
                                           [32,14,23,31,0,31],[32,30,29,23,31,0]]),
                                 index=tb6, columns=tb6)
    return bank6

def init_pf_session_state():
    stss.home_dir = os.path.dirname(os.path.abspath(__file__))
    col = Collection()
    js = get_cached_sample_json()
    col.load_project_state(js)
    stss.sample_col = col # Collection with sample Proximitys and PFnets
    stss.sample_sources = get_sample_sources() # DataFrame with sources of sample Proximities
    stss.example_file_format = get_example_file_format() # DataFrame showing file format
    stss.col = copy.deepcopy(stss.sample_col) # collection for manipulation
    stss.col.focus_net = stss.col.pfnets["bio_pf"]
    stss.help_html = get_cached_help_html()
    stss.col.changed = False
    stss.layouts = ["kamada_kawai", "saved", "gravity", "force", "spring", "MDS", "spiral", "circle"]
    stss.layout = "kamada_kawai"
    stss.layout_selector = stss.layout
    stss.font_size = 10
    stss.rotation = 0
    stss.count = 0
    stss.q_param = np.inf
    stss.r_param = np.inf
    stss.intro_visible = True
    stss.sample_visible = True
    stss.perm_ave_type = "mean"
    stss.perm_file_type = "Spreadsheet"
    stss.perm_net_type = "Pathfinder"
    stss.last_leg_files = []
    stss.last_ss_files = []

