# app_pages/help.py
import streamlit as st

from stutils import get_cached_help_html

st.set_page_config(
    page_title="PyPF Help",
    page_icon="❓", # Use an emoji
)

stss = st.session_state

# Sidebar table of contents: links jump to the section ids in static/help.html
st.sidebar.subheader("📖 Contents")
st.sidebar.markdown(
    """
<style>
    .help-toc { font-size: 15px; line-height: 1.9; }
    .help-toc a { text-decoration: none; color: inherit; }
    .help-toc a:hover { text-decoration: underline; }
</style>
<div class="help-toc">
    <a href="#start" target="_self">🚀 Getting Started</a><br>
    <a href="#build" target="_self">🏠 Build Nets Page</a><br>
    <a href="#view" target="_self">🌐 View Network Page</a><br>
    <a href="#measures" target="_self">📐 Measures</a><br>
    <a href="#projects" target="_self">💾 Saving &amp; Loading Projects</a><br>
    <a href="#format" target="_self">📄 Spreadsheet Data Format</a><br>
    <a href="#legacy" target="_self">📜 Legacy Text Data Format</a><br>
    <a href="#resources" target="_self">🔗 Other Resources</a>
</div>
""",
    unsafe_allow_html=True,
)

# Main content
st.subheader("PyPathfinder Help")
st.html(get_cached_help_html())
