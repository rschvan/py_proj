# pages/display_network.py
import math
import time
import numpy as np
import pandas as pd
import streamlit as st
# import streamlit.components.v1 as components
# from pypf.create_visjs_html import create_visjs_html
# import webbrowser # Although components.html is often better for new tabs

st.set_page_config(
    page_title="PyPF Display",
    page_icon="🌐")

pic = st.session_state.pic # alias for easy reference

def do_rotation():
    rotation = st.session_state.rotation
    coords = pic.net.coords
    # rotate code
    theta_rad = math.radians(rotation)
    # Create the 2x2 rotation matrix
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])
    # Perform the rotation by matrix multiplication
    # (n x 2) @ (2 x 2) -> (n x 2)
    pic.net.coords = coords @ rotation_matrix
    pic.create_view(font_size=st.session_state.font_size)
    time.sleep(0.5)
    st.session_state.rotation = 0

def update_layout():
    pic.net.get_layout(method=st.session_state.layout)
    pic.create_view(font_size=st.session_state.font_size)

def do_toggle_weights():
    pic.toggle_link_weights()

def update_font_size():
    pic.change_font_size(st.session_state.font_size)

with st.sidebar:
    st.subheader("Layout Controls")
    fx, fy, red = st.columns([5,5,6], gap="small")
    with fx:
        if st.button("flipx"):
            pic.net.coords[:,0] *= -1
            pic.create_view(font_size=st.session_state.font_size)
    with fy:
        if st.button("flipy"):
            pic.net.coords[:,1] *= -1
            pic.create_view(font_size=st.session_state.font_size)
    with red:
        if st.button("redraw"):
            update_layout()

    # rotation
    rotate = st.slider("**degrees of clockwise rotation**", 0, 360, key="rotation", on_change=do_rotation)

    st.session_state.layout = st.selectbox("**Layout Method**", st.session_state.layouts,
                 index=st.session_state.layout_index ,)
    if st.session_state.layout != st.session_state.last_layout:
        update_layout()
        st.session_state.layout_index = st.session_state.layouts.index(st.session_state.layout)
        st.session_state.last_layout = st.session_state.layout

    tw, fs = st.columns([3, 3])
    with tw:
        if st.button(label='toggle weights', key="weights"):
            do_toggle_weights()

    with fs:
        st.number_input("**font size**", min_value=5, max_value=20, value=10, step=1, key="font_size",
                        on_change=update_font_size())

    st.write("""Use the Interactive Display if you wish to modify the image.  
        Right click on the image to save it. """)

    show_properties = st.checkbox("**Show Net Properties**", value=False, key="show_properties")

    show_link_list = st.checkbox("**Show Link List**", value=False, key="show_link_list")

st.write(f"**Network: {pic.net.name}**")
st.write(f"**{pic.net.nnodes} nodes - {pic.net.nlinks} links - {st.session_state.layout} layout**")

st.pyplot(fig=pic.fig, width=750)

if show_properties:
    ecc = pic.net.eccentricity
    st.write("**Network Properties**")
    st.write(f"{pic.net.name}")
    st.write(f"{pic.net.nnodes} nodes and {pic.net.nlinks} links -- radius = {ecc['radius']} -- diameter = {ecc['diameter']}")
    st.write(f"")

    tab = pd.DataFrame(pic.net.get_network_properties())
    st.dataframe(data=tab, width="content", hide_index=True,)

if show_link_list:
    st.write("**Link List**")
    st.write(f"{pic.net.name} : {pic.net.nnodes} nodes -- {pic.net.nlinks} links ")
    if pic.net.type == "merge":
        st.write("""**weight** is a binary code  
        **nets** shows which nets contain the link (reading right to left) -- e.g., 110 means in nets 2 & 3  
        **count** shows the number nets that contain the link -- e.g. 110 count would be 2
        """)

    lprops = pic.net.get_link_list()
    st.dataframe(data=lprops, width="content", hide_index=False,)
