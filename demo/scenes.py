# demo/scenes.py: the PyPathfinder demo, scene by scene.
#
# Each scene has a title (a YouTube chapter), the narration to read over it, and the
# actions the recorder performs. The recorder holds each scene until its narration would
# have been read (about 145 words a minute), so edit the narration and the timing follows.
# Actions use the helper methods of record_demo.Demo (d).

from dataclasses import dataclass
from typing import Callable


@dataclass
class Scene:
    title: str
    narration: str
    actions: Callable  # actions(d) performs the scene's steps


def intro(d):
    d.page_top()
    d.caption("PyPathfinder")
    d.pause(3)
    d.point_at(d.text("Build Nets", sidebar=True))
    d.pause(1.5)
    d.point_at(d.text("View Network", sidebar=True))
    d.pause(1.5)
    d.point_at(d.text("Help", sidebar=True))
    d.pause(1.5)
    d.point_at(d.text("Getting Started (click to hide or show)"))


def sample_data(d):
    d.scroll_to_heading("Sample Proximity Data and Derived Networks")
    d.pause(2)
    d.point_at(d.dataframe(0))
    d.pause(5)
    d.point_at(d.dataframe(1))


def proximity_list(d):
    d.scroll_to_heading("Proximity List")
    d.pause(2)
    d.point_at_table_cell(table=2, row="bio", frac=0.12)   # name
    d.pause(3)
    d.point_at_table_cell(table=2, row="bio", frac=0.42)   # coherence
    d.pause(3)
    d.point_at_table_cell(table=2, row="bio", frac=0.88)   # termsid


def network_list(d):
    d.scroll_to_heading("Network List")
    d.pause(2)
    d.point_at_table_cell(table=3, row="bio_pf", frac=0.3)


def derive(d):
    d.scroll_to_heading("Proximity List")
    d.select_rows(table=2, names=["bio"])
    d.click(d.text("Derive Networks (1)"))
    d.type_into("q:", "2")
    d.type_into("r:", "1")
    d.pause(1)
    d.click(d.button("Derive Now"))
    d.pause(2)
    d.scroll_to_heading("Network List")
    d.point_at_table_cell(table=3, row="bio_pf_q2_r1", frac=0.12)


def properties(d):
    d.scroll_to_heading("Network List")
    d.select_rows(table=3, names=["bio_pf"])
    d.click(d.button("Network Properties"))
    d.pause(1.5)
    d.scroll_to_text("radius =")


def average(d):
    d.scroll_to_heading("Proximity List")
    d.select_rows(table=2, names=["bio", "psy"])
    d.click(d.text("Average Proximities (2)"))
    d.pause(1.5)
    d.click(d.button("Compute Average"))
    d.pause(4)
    d.scroll_to_heading("Proximity List")
    d.point_at_table_cell(table=2, row_index=0, frac=0.12)


def merge(d):
    d.scroll_to_heading("Network List")
    d.select_rows(table=3, names=["bio_pf", "psy_pf"])
    d.click(d.text("Merge 2 nets"))
    d.pause(1.5)
    d.click(d.button("Merge Now"))
    d.pause(3)
    d.scroll_to_heading("Network List")
    d.point_at_table_cell(table=3, row_startswith="~~mg2", frac=0.12)


def compare(d):
    d.click(d.text("Proximity Correlations", sidebar=True))
    d.pause(1)
    d.scroll_to_heading("Proximity Correlations")
    d.pause(5)
    d.click(d.text("Network Similarities", sidebar=True))
    d.pause(1)
    d.scroll_to_heading("Network Similarity")
    d.pause(5)
    d.click(d.text("Proximity Correlations", sidebar=True))
    d.click(d.text("Network Similarities", sidebar=True))


def view_network(d):
    d.scroll_to_heading("Network List")
    d.select_rows(table=3, names=["bio_pf"])
    d.click(d.button("Display Network"))
    d.pause(3)
    d.point_at(d.toolbar("flipx"))
    d.pause(2)
    d.point_at(d.toolbar("physics"))
    d.pause(2)
    d.point_at(d.toolbar("fontinc"))
    d.pause(2)
    d.point_at(d.toolbar("png"))


def orient(d):
    d.click(d.toolbar("flipx"))
    d.pause(2)
    d.click(d.toolbar("flipy"))
    d.pause(2)
    d.click(d.toolbar("rot90"))
    d.pause(2)
    d.drag_slider(".nv-rotate", fraction=0.65)
    d.pause(1.5)
    d.click(d.toolbar("reset"))


def labels(d):
    d.click(d.toolbar("fontinc"))
    d.pause(1)
    d.click(d.toolbar("fontinc"))
    d.pause(1.5)
    d.click(d.toolbar("weights"))
    d.pause(3)
    d.drag_slider(".nv-maxrank", fraction=0.25)
    d.pause(4)
    d.drag_slider(".nv-maxrank", fraction=1.0)
    d.pause(1)
    d.click(d.toolbar("weights"))
    d.click(d.toolbar("fontdec"))
    d.click(d.toolbar("fontdec"))


def layouts(d):
    d.choose_layout("tree")
    d.pause(4)
    d.choose_layout("radial_tree")
    d.pause(4)
    d.choose_layout("kamada_kawai")
    d.pause(2)
    d.click(d.toolbar("redraw"))
    d.pause(3)
    d.click(d.toolbar("redraw"))
    d.pause(3)
    d.choose_network("cities_pf")
    d.pause(2)
    d.choose_layout("mds_distance")
    d.pause(4)


def physics(d):
    d.choose_network("bio_pf")
    d.pause(2)
    d.click(d.toolbar("physics"))
    d.pause(7)
    d.click(d.toolbar("physics"))
    d.pause(1)
    d.click(d.toolbar("fit"))


def saved_layout(d):
    d.click(d.button("💾 Save Layout"))
    d.pause(3)
    d.choose_network("psy_pf")
    d.pause(3)
    d.choose_layout("saved")
    d.pause(3)


def projects(d):
    d.go_page("Build Nets")
    d.page_top()
    d.click(d.text("Save Project File"))
    d.pause(4)
    d.click(d.button("❌ Cancel Save"))
    d.pause(1)
    d.click(d.text("Load Project File"))
    d.pause(4)
    d.click(d.button("❌ Cancel Load"))


def help_page(d):
    d.go_page("Help")
    d.pause(2)
    for anchor in ("view", "measures", "format", "legacy"):
        d.help_jump(anchor)
        d.pause(3)
    d.help_jump("start")


def closing(d):
    d.go_page("View Network")
    d.pause(2)
    d.caption("PyPathfinder")


SCENES = [
    Scene("Introduction", """
Welcome to PyPathfinder, an app for creating, exploring, and visualizing Pathfinder
networks. Pathfinder networks reveal the structure in proximity data: the distances,
dissimilarities, or relatedness values between every pair of a set of terms. Each
network keeps only the most direct links between the terms, so the essential structure
stands out. The app has three pages, listed in the sidebar: Build Nets, where you add
data and derive networks; View Network, where you lay out and explore a network; and
Help, with full details. A short Getting Started panel at the top of Build Nets
summarizes the main steps.
""", intro),

    Scene("Proximity data and the samples", """
The app starts with sample data so you can try everything right away. Here is the
layout of a proximity spreadsheet, using the small bank6 data set: the terms run across
the first row and down the first column, and each cell holds the distance between a
pair of terms. Smaller numbers mean the terms are closer or more related. You can load
spreadsheets in Excel or CSV format, and the older Pathfinder text formats are also
supported. Below the example, the sample data sets are described: word associations
from a dictionary, relatedness ratings by biology and psychology students, and
distances between American cities and state capitals.
""", sample_data),

    Scene("The Proximity List", """
Each proximity you load appears in the Proximity List. The columns show its name,
the number of terms, and whether the distances are symmetric. Coherence measures the
internal consistency of the data: terms that are close to each other should have
similar distances to the other terms. Ratings from people who know a subject well
tend to be more coherent. The list also shows the mean, spread, and range of the
distances, and an identifier for the list of terms. Proximities with the same terms
can be averaged, correlated, and compared.
""", proximity_list),

    Scene("Minimum Pathfinder networks", """
When a proximity is loaded, the app immediately derives its minimum Pathfinder
network, with the parameters q and r both set to infinity. This network has the fewest
links of any Pathfinder network for the data, and its links appear in every other
Pathfinder network you derive from it. The Network List shows each network's nodes and
links, whether it is connected, whether it can be drawn without crossing links, and
whether it is directed.
""", network_list),

    Scene("Deriving networks with q and r", """
To derive other networks, select one or more proximities with the check boxes and open
Derive Networks. The q parameter limits how many links a path may have when the app
searches for a shorter route between two terms, and r controls how link distances
combine along a path. With r equal to one, a path's length is the sum of its links;
with r at infinity, it is the longest single link. Smaller values of q and r keep more
links. Here we derive the biology network with q equal to two and r equal to one. The
new network appears in the list with its parameters in its name.
""", derive),

    Scene("Network properties", """
Selecting a single network and clicking Network Properties shows measures for every
node. Degree is the number of links at a node. Eccentricity is the largest number of
links needed to reach any other node, and the nodes with the smallest eccentricity form
the center of the network. The median nodes are closest to all the others on average.
The radius and diameter of the network are shown above the table.
""", properties),

    Scene("Averaging proximities", """
Proximities with the same terms can be averaged. Here we select the biology and
psychology ratings, which use the same twenty-five terms, open Average Proximities, and
compute the mean. You can also choose the median. The average appears at the top of the
Proximity List, along with its minimum network, ready to compare with the individual
data sets.
""", average),

    Scene("Merging networks", """
Networks with the same terms can also be merged. The merged network contains every link
found in any of the selected networks, and each link's weight is a code showing which
networks contain it. The Link List of a merged network explains the code and counts how
many of the networks include each link, which makes it easy to see where two groups
agree and where they differ.
""", merge),

    Scene("Comparing data sets", """
The sidebar has two comparison tables. Proximity Correlations shows how closely the
distances of each pair of data sets agree. Network Similarities shows the proportion of
links that each pair of networks share. Both are computed for data sets with the same
terms; other pairs are left blank. Here the biology and psychology ratings are highly
correlated, but their networks share only part of their links.
""", compare),

    Scene("Viewing a network", """
To look at a network, select it and click Display Network. The View Network page shows
the network with a toolbar above it. The Orient tools flip and rotate the picture. The
Layout tools redraw the layout, run a physics engine, fit the network to the view, and
reset your changes. The Text tools change the label size and show link weights. The PNG
button saves the picture as an image. You can also drag individual nodes, drag the
background to move the picture, and scroll to zoom.
""", view_network),

    Scene("Orienting the picture", """
Layouts have no natural up or down, so it often helps to turn them. The first two
buttons flip the picture left to right and top to bottom, and the next one rotates it
by ninety degrees. The rotate slider turns the picture smoothly to any angle; release
it to keep the rotation. Reset returns to the layout as it was computed.
""", orient),

    Scene("Labels, weights and ranks", """
The A buttons make the labels larger or smaller. Weights shows each link's distance.
Links are also ranked, where rank one is the shortest distance, the strongest link.
The Max rank slider hides links with higher ranks, so you can watch the network build
up from its strongest links as you move the slider back to the right.
""", labels),

    Scene("Layout algorithms", """
The sidebar offers several layout algorithms. The tree layouts grow from the center of
the network, which suits Pathfinder networks with few cycles. Kamada Kawai is a good
general choice, and the Redraw button computes a new layout from a random start, so you
can try a few and keep the one you like. For geographic data, the MDS distance layout
uses all of the distances, not just the links. Here the network of American cities
comes out looking much like a map.
""", layouts),

    Scene("The physics engine", """
The Physics button turns on a layout engine that keeps moving the nodes until they
settle. While it runs, you can drag nodes to help untangle crossed links. Turn Physics
off to freeze the picture, and then dragged nodes stay exactly where you put them.
""", physics),

    Scene("Saving and sharing a layout", """
When a layout looks right, click Save Layout. The layout is saved for the network's
terms, not just for this network, so any network with the same terms can use it. Here
we switch to the psychology network and choose saved. Now both networks have their
nodes in the same places, which makes it easy to compare which links they share and
which they don't.
""", saved_layout),

    Scene("Saving and loading projects", """
Back on Build Nets, the Manage Data Files section saves your whole project to a file on
your computer: the proximities, how to rebuild each network, the chosen layouts, and
saved layouts. When you load a project, you can replace what is loaded or add to it.
The app forgets its data when left idle for a while, so save your project before taking
a break.
""", projects),

    Scene("Help", """
The Help page explains every part of the app: the Build Nets tools, the View Network
toolbar and layout algorithms, the measures such as coherence and eccentricity, and the
spreadsheet and legacy text file formats, with examples. The contents in the sidebar
jump straight to each section.
""", help_page),

    Scene("Wrapping up", """
That is a quick tour of PyPathfinder. Load your own proximity data, derive networks
with different values of q and r, compare data sets, and arrange the networks until
their structure is clear. The Help page and the other videos on this channel go into
more detail. Thanks for watching.
""", closing),
]
