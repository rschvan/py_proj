# pages/help.py
import streamlit as st

st.set_page_config(
    page_title="PyPF Help",
    page_icon="❓", # Use an emoji
)

stss = st.session_state

# 1. Sidebar Navigation Menu
st.sidebar.subheader("📖 Select Contents")

st.sidebar.markdown("""
<style>
    /* Prevent the top header from being cut off when jumping */
    :target::before {
        content: "";
        display: block;
        height: 70px; /* Adjust based on Streamlit's header height */
        margin: -70px 0 0;
    }
    .toc { font-size: 16px; line-height: 1.8; }
    .toc a { text-decoration: none; color: #007bff; }
</style>
<div class="toc">
    <a href="#top" target="_self">🔗 Other Resources</a><br>
    <a href="#welcome" target="_self">🏠 About PyPathfinder</a><br>
    <a href="#lists" target="_self">📋 Proximity & Network Lists</a><br>
    <a href="#display" target="_self">🖥️ Displaying Networks</a><br>
    <a href="#saving" target="_self">💾 Saving & Loading Projects</a><br>
    <a href="#tips" target="_self">💡 Tips about the App</a><br>
    <a href="#format" target="_self">📄 Spreadsheet Data Format</a><br>
    <a href="#legacy" target="_self">📜 Legacy Text Data Format</a>
</div>
""", unsafe_allow_html=True)

# Main content

st.subheader("PyPathfinder Help")
st.html(stss.help_html)
#     /* <a href="#links" target="_self">🔗 Useful Links</a><br> */
# <a href="" onclick="window.scrollTo(0,0); return false;" target="_self">🔗 Useful Links</a><br>