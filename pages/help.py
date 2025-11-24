# pages/help.py
import streamlit as st
import os
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="PyPF Help",
    page_icon="❓", # Use an emoji
)

def file_format()->pd.DataFrame:
    tb6 = ["account","bank","flood","money","river", "save"]
    bank6 = pd.DataFrame(np.array([[0, 3,32,26,32,32],[3,0,27,21,14,30],
                                           [32,27,0,31,23,29],[26,21,31,0,31,23],
                                           [32,14,23,31,0,31],[32,30,29,23,31,0]]),
                                 index=tb6, columns=tb6)
    return bank6

st.subheader("Welcome to PyPathfinder!")
st.link_button("Pathfinder Wikipedia Page", "https://en.wikipedia.org/wiki/Pathfinder_network")

with open(os.path.join(st.session_state.home_dir,
                       "pypf","data","Schvaneveldt_Durso_Dearholt_1989.pdf"), "rb") as f:
    st.download_button(label="Download Introductory Pathfinder Paper",
                      data=f,
                      file_name="Schvaneveldt_Durso_Dearholt_1989.pdf",
                      mime=None)
if st.button("PyPathfinder Demo Video", key="vid_button"):
    path = os.path.dirname(__file__)
    parent = os.path.dirname(path)
    video_path = os.path.join(parent, "pypf/data/video_demo.mp4")
    st.video(video_path)

st.write("""
        PyPathfinder is a Python app for creating and analyzing Pathfinder Networks.
        It allows you to load Proximity data files and create Pathfinder Networks as well as
        Threshold networks and Nearest Neighbor networks.
        It provides information about the networks and data files.
        It also provides tools for viewing information about proximities and networks and for 
        averaging compatible data sets. Additional pages allow you to explore options for 
        visualizing and saving images of networks.
        
        Use the examples to explore the functions in the app. Play with the various tools to learn about
        their effects.  
        
        Load your own data files whenever you like.      
        """)
st.markdown("**Sample Proximity files and Networks**")
st.write('''Intro info adds access to information about Pathfinder and the app 
        including a video demo. 
        It also loads an example of the required Proximity file formats and sample 
        Proximities and PFnets for demo purposes.  These elements will be deleted 
        when you uncheck the Intro Info box. 
        Anything you add will be preserved when you uncheck the Intro Info box. 
        Your files are not saved over sessions. 
        ''')
st.subheader("**Tips about the app**")
st.write('''
        In the Proximity and Network Lists, you can select items by checking the box to the left
        of the name. A single item can be selected by clicking on its name, but selecting mutiple
        items requires checking the select boxes. All items can be selected or deselected by 
        checking or unchecking the select box at the top of the list.   
        
        Download any table into a csv file by hovering at the top of the table and
        clicking on the download icon. 
        
        Download a network figure into a png file by right clicking on it and selecting "Save image as".
        
        The q-value is the maximum number of links in paths examined in searching for the minimum-length path. 
        The default value of infinity means that all possible paths are examined. The paths with the most 
        possible links have n-1 links where n is the number of nodes. With q = 2, only two-link
        paths are considered.
        
        Threshold networks show the links with smaller distances.  
        
        Links in Nearest neighbor networks point from a node to the nearest other node or nodes 
        if there are ties.  
        
        Networks with the same nodes can be merged into a single network which includes all 
        of the links in the merged networks. The link list will show which networks include each link.  
        
        The MDS layout uses the complete distance matrix to find a 2-D Multidimensional Scaling Fit for locating
        the nodes. It is particularly useful when all of the distances are considered meaningful.  
        
        Eccentricity is the maximum number of links in paths between a node and nodes that can be 
        reached from that node. The measure may be misleading if the network is not completly connected. 
''')
st.subheader("**Spreadsheet Data File Format (.xlsx or .csv files)**")
st.write("""The spreadsheet format is the simplest way to arrange your data for analysis by the app.
    Here is an example of the required Proximity file format for the sample "bank6" data set.
    The file could be named "bank6.xlsx" or "bank6.csv" or "bank6.prx.xlsx" or "bank6.prx.csv". 
    The part of the filename before the first period is the name of the proximity.  The optional ".prx" helps to
    mark the file as a proximity data file.
""")

st.dataframe(file_format(), width='content', hide_index=False)
st.write(
    """
    Organize your data files just as shown in the example.  With n terms, the spreadsheet must consist 
    of  n+1 rows and n+1 columns.  The first column must contain the n terms starting in row 2, 
    the first row must contain the n terms starting in column 2.  The remaining cells contain the distances 
    with 0’s on the diagonal, representing zero distance between an item and itself.  If the off-diagonal 
    distances are symmetric around the diagonal, Pathfinder networks will be undirected.  Non-symmetric 
    distances will result in directed Pathfinder networks where row items point to column items.
    \nThe values in the data matrix must 
    be distances or dissimilarities where lower values mean smaller distances.  If your original data 
    measure similarity, invert them to produce the data for PyPathfinder.  For example, invert measures 
    like correlation, cosine, or probability using the formula ( d = 1 – c ) where d is distance and c 
    is correlation, cosine, or probability.
    If your data are similarities with higher values meaning more similar or more related, 
    they can be transformed  using ( d = min + max – s ) where d is dissimilarity and s = similarity, 
    min is the minimum s and max is the maximum s.  The minimum and maximum will remain the same with 
    the order of values inverted. 
    \nAn infinite distance results with anything other than a finite positive real number in a distance cell. 
    The cell can be empty or have the string “nan” or “inf” or “na” etc. Infinite distances will 
    never become links in the networks created.
    """)

st.subheader("**Displaying a Network**")
st.write("""The **Select Display** page provides tools for exploring various layouts and characteristics of the 
    display of networks. Use the tools to get a satisfactory view of the network. If you would like to adjust
    the view by moving the nodes around, go to the **Interactive Display** page.""")

st.subheader("**Legacy Text File Format (.txt files)**")
st.write("""A text file format is used in several earlier versions of Pathfinder software and in various methods
 for creating proximity data so PyPathfinder 
allows you to load these files. With legacy text files,
terms are stored in separate text files, with one term on each line.  These term files must be loaded along
with the text data files to create complete Proximities in PyPathfinder. The term files can be named in two ways. 
For data files with the same terms, a single file named "terms.txt" can be used for all the data files. 
To associate a term file with a specific data file, the term file name must include the data file name followed by 
".trm". So, for example, if the data file is named "data.txt" or "data.prx.txt" the associated terms file must be named 
"data.trm.txt".    

The legacy text files are more complex than the spreadsheet format, so we encourage the use of the newer spreadsheet
 format for newly created data files.  Details about the format of legacy proximity .txt files can be seen by clicking 
 the button below. 
""")

# Legacy details

if st.button("Click for Details on Legacy Text Proximity Files"):
    st.write("""The data may be in the form of similarities, dissimilarities, probabilities, distances, 
    coordinates, or features.  With dissimilarities or distances, smaller numbers represent pairs of entities 
    that are close or similar or related and larger numbers represent pairs of entities that are distant 
    or dissimilar or unrelated.  The opposite is true of similarities, probabilities, or relatedness 
    i.e., smaller numbers represent entities that are distant or dissimilar or unrelated and larger 
    numbers represent pairs of entities that are close or similar or related. With distance measures, 
    the distance between an entity and itself (the major diagonal entries in a data matrix) is usually 0 (zero).
    All entries in the data must be positive or zero. Negative numbers are not allowed. Values outside the 
    minimum – maximum range (see below) will never produce links in the networks generated. A strictly formatted 
    text file is required for proximity data.  Here is a small example of such a file:

data  
similarity  
 5 nodes  
comment  
10 minimum value  
90 maximum value  
lower triangle  
 32  
 40 49  
 32 38 53  
 73 63 77 18   
 
Required Data file format.  / indicates alternatives:  
Line 1: Identification as data file = Data/DATA/data (must have the word 'data' or 'DATA' or 'Data')  
Line 2: Type of data: either dissimilarity or distance or dis or similarity or sim  
Line 3: Number of nodes = integer   
Line 4: comment : short description   
Line 5: Minimum data value = positive real number  
Line 6: Maximum data value = positive real number   
Line 7: Order of data values : either matrix or upper or lower or list or coord or featur or attrib   
Line 8: Data     
Line 9: Data    
.  
.    
Line ?: Data   

The lines in the file must be organized as shown above.  For Lines 3, 5, and 6, the program uses only 
the first number on the line.  Some following text helps to indicate what the number represents.
 
Details on the required input are as follows:    
**Line 1.** "Data," "DATA," or "data" is used to identify the file type   
**Line 2.** "similarity," "dissimilarity," "probability," or "distance," (or "sim," "dis," or "prob,") 
This line indicates data direction. With similarity data, larger values represent greater similarity. 
With distance data, smaller numbers mean closer (or more similar).  
**Line 3.**  The number of nodes (or terms) in the data set.  The word nodes is optional  
**Line 4.**  Comment: a short description of the data  
**Line 5.**  the minimum value in the data set.   Words are optional.  
**Line 6.**  the maximum data value.   Words are optional.  
The minimum and maximum values are used as cutoffs in handling the data.  Any value in the data outside 
the minimum - maximum range will never become a link in networks.  In other words, two nodes with a 
proximity value outside the range can never be linked in any network generated by the program. 
Missing data can be handled by using values outside the range, or by using the "list" format for your data.  
**Line 7.**  "matrix" or "upper" or "lower" or "list" or "coord" or "featur."  This line specifies the 
nature of the data following this line.  Various ways of supplying proximities are possible based on a 
full matrix, an upper triangle, a lower triangle, a list, or vectors of features, attributes, or coordinates. 
The upper and lower (half-matrix) methods do not include the major diagonal (the proximity of an item with itself). 
The lines of data in the file do not have to have these shapes, but the data must be in the same order as 
they would if the lines did have those shapes when we characterize order as reading across each line in turn. 
The following examples may be of help.
    
matrix:	  
0 1 3 2 3  
1 0 1 4 6  
3 1 0 5 5  
2 4 5 0 4  
3 6 5 4 0    

lower:  
1   
3 1   
2 4 5   
3 6 5 4	  

upper:  
1 3 2 3  
1 4 6  
5 5  
4  	

list:  
10 pairs  
symmetric  
2 1 1  
3 1 3  
3 2 1  
4 1 2  
4 2 4  
4 3 5  
5 1 3  
5 2 6  
5 3 5  
5 4 4  

These four sets of data produce the same distance matrix. Of course, if your data are asymmetric, 
they must be input as a matrix or a list with "nonsymmetric" or “asymmetric” specified. 
If the data are symmetric, any of the four shapes is acceptable. 
With the list format, the number of pairs in the list must be specified on the line following "list." 
The next line specifies whether the pairs define symmetric or nonsymmetric data.  With the list format, 
missing pairs will never be linked. Following the header lines, the data must occur as discussed above.  

With n nodes (or entities), a matrix 
must contain n x n data elements, upper or lower triangles must contain n(n-1)/2 data elements. 

The list data must contain 3 numbers for each pair listed, the number of the source term, the number of the 
destination term, and the proximity value.

**Coordinates, Features, Vectors, or Attributes**

When data are in the form of vectors, the appropriate format starting at Line 7 is as follows:  
Line 7. “coord” or “feature” or “attrib” indicating that the data to follow are to be interpreted as 
vectors of numbers, one vector for each item or node.  
Line 8. Integer = The number of dimensions, attributes, or features in the vector for each item (vector length)   
Line 9. “euclidean” or “city block” or “dominance” or “hamming” or “cosine” plus optional: “standardize”  
Line 10: Vector for item 1  
Line 11: Vector for item 2  
Etc. (one vector for each item)

Here is a small example proximity data file for 5 nodes in 2 dimensions:  
data  
distance  
5 nodes  
Euclidean Distances  
0 minimum data value  
16 maximum data value  
coord:    
2 dimensions  
Euclidean Distance   
9  1  
8  4  
2  6  
5  4  
1  8  
In this case, each of 5 items (nodes) has values on each of two dimensions as in representing points 
in 2-dimensional space.  The number of items or nodes is specified in Line 3 of the data file. 
The number of dimensions or features must be given following the “coord” or “feature” or “attribute” line. 
There can be any number of dimensions.  With coordinates, we often think of the data as coming from the 
spatial layout of items.  With features or attributes, the data may be the ratings of each item on 
several different Likert-type scales so the scales can be thought of as features.  Features could also 
identify the presence or absence of features using 1’s and 0’s.   The line following the number of 
dimensions or features must have "Euclidean," "City Block," "Dominance," “Hamming,” or “Cosine.” 
This determines how distance data are computed from the coordinates or features.  Distances are 
computed for all pairs of items.  With Euclidean, we find the straight line distance between 
the items (nodes) in multidimensional space.  With City Block, the distances are determined by  
summing the distance between items (nodes) on each dimension or feature.  City Block is usually the 
most appropriate method for Likert-type rating scale data.  With Dominance, the distance between items 
is the maximum difference for the items across all the dimensions or features.  Hamming distances are the 
number of features on which items (nodes) differ.  Cosine bases the distances on the cosine of the angle 
between the vectors which varies from 1 to -1.  Because cosine is a measure of similarity rather than 
distance, a distance value is obtained by: 1 – cosine.  This varies from 0 to 2 with 0 meaning vectors 
pointing in the same direction, and 2 meaning vectors pointing in opposite directions. You can also 
determine whether to standardize the vectors before computing distances.  When “standard” is included on 
the line, the vectors are normalized to have a standard length 1.  This is often appropriate when the vectors 
vary greatly in magnitude.  For example, with vectors from Latent Semantic Analysis (LSA), computing 
Euclidean distances with standardized vectors provides a measure which is very similar to the cosine 
measure usually used in LSA work.

Here is a complete sample data file with the coordinate/feature format:  
data  
distances  
5 nodes   
City Block Distances   
1 minimum data value   
6 maximum data value   
features:  
4 features  
City Block Metric    
2 1 3 2     
1 5 1 4    
3 1 3 5    
2 4 1 2    
3 6 5 4  

For an example of how distances are created from the data, the distance between items 1 and 2 
(rows 1 and 2) is computed as follows:

|2-1| + |1-5| + |3-1| + |2-4| = 9   where |x-y| is the absolute value of the difference between x and y.

the distance between items 2 and 4 is:

|1-2| + |5-4| + |1-1| + |4-2| = 4

Distance is computed for all pairs of items in this way.

Following the header lines, the data must occur as discussed above.  The coord data must have a value 
on each dimension or feature for each of the nodes.  The data elements must be separated by one or 
more spaces and/or line breaks.  

    """)



