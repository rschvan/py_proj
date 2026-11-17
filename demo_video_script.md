# PyPathfinder demo video script

Nov 17, 2026 · @Roger Schvaneveldt

Eight scenes walk through the app in about 8 to 10 minutes in all. Each step is an action to take on screen, with what to say in {curly brackets} where it fits. Bold names are buttons and controls in the app. Start from a fresh session, so the demo data is loaded. Times are rough targets; cut any scene or (Optional) step you don't need.

## Scene 1: Opening (30 sec)

1. Open the app on the Build Nets page. \
{The PyPathfinder app builds Pathfinder networks from proximity data: matrices of distances or relatedness values for pairs of a set of terms.}
2. Point at Build Nets, View Network and Help in the sidebar. \
{The app has three pages: Build Nets, View Network and Help. Navigate among the pages here.}
3. Point at the Proximity and Network Lists. \
{When started, the app loads sample data and networks to help you become acquainted with it.}

## Scene 2: Build Nets and demo data (1 min)

1. Scroll through **Getting Started**. \
{Getting Started summarizes a typical session: add data, work with the lists, view a network, save your project.}
2. Open the **Spreadsheet layout example (bank6)** expander. \
{Data files are simple: terms across the first row and down the first column, distances in the matrix. Later, we'll cover adding data.}
3. Point at the Proximity List and Network List, already filled. \
{Sample data loads when the app starts, so you can try everything before using your own data.}
4. Open **Demo data sources**. \
{bio and psy are relatedness ratings of the same 25 terms by biology grad students and psychology 101 students; cities and statecaps are real distances; bank and bank6 come from word co-occurrences in a dictionary.}

## Scene 3: Working with proximities (2 min)

1. Point at the Proximity List columns. \
{Each row shows the number of terms, whether the data are symmetric, coherence, and summary statistics. Coherence measures how internally consistent the distances are. Here bio is a little higher than psy, as you might expect from the more expert raters.}
2. Select psy. \
{Check boxes to select items; the buttons light up when the selection allows them.}
3. Click **Display Matrix**. \
{Here is a copy of the distance matrix.} \
Close the window with ✕ or Esc.
4. Click **Derive Networks…**. \
{Choose Pathfinder or Nearest Neighbor networks.} \
Set q = 2 \
{Let's set q = 2} \
and click **Derive**.
5. Point out psy\_pf\_q2 in the Network List. \
{See the new net.}
6. Select bio and psy, click **Average Proximities…**, choose mean, and click **Compute Average**. \
{This creates an average proximity and its basic pfnet by averaging distances cell by cell, using means or medians.}
7. Turn on **Show Proximity Correlations** under the list. \
{This shows correlations of the distances of every pair of proximities with the same terms.}
8. (Optional) Point at **Delete Proximities…**. \
{Delete Proximities removes a proximity and its networks. The window lists everything that goes with it.}

## Scene 4: Working with networks (1.5 min)

1. Point at the Network List columns. \
{Names show how each network was made: \_pf for Pathfinder with its q and r, if not infinity, \_nn for Nearest Neighbor. The columns give nodes and links, and whether the network is connected, planar and directed.  The Help page gives details.}
2. Select bio\_pf and click **Network Properties**. \
{Properties give degree, eccentricity, and the center and median nodes.}\
 Close it with ✕ or Esc.
3. Click **Network Link List**. \
{The Link List shows every link and its distance. The selection stays, so the next table is one click away. To save a table as a .csv file, hover over it and click the download icon at its top right.}\
 Close the window.
4. Turn on **Show Network Similarities** under the list. \
{Similarity is the share of links two networks have in common: 1 for identical networks, 0 for none in common.}
5. Select bio\_pf and psy\_pf, click **Merge Networks…**, then **Merge**. \
{A merged network holds every link from either network. Each link's weight is a code for which networks have it; we'll see it on the next page.}
6. (Optional) Point at **Delete Networks…**. \
{Delete Networks also removes any merged networks built from the selection.}

## Scene 5: Adding your own data (1 min)

1. Click **Add Data Files…** in the sidebar. \
{Choose Spreadsheet for .xlsx or .csv files, or Legacy Text for users with data in that format. The **Force Proximity Symmetric** switch makes asymmetric data symmetric by keeping the smaller distance of each pair, which gives undirected networks.}
2. Browse for aviation.prx.xlsx\
{The proximity is named from the file name, up to the first period. The aviation data are derived from average similarity ratings of experienced pilots.}
3. Point at the new proximity and its network in the lists. \
{Each loaded file gets its minimum links Pathfinder network automatically, with q and r both infinite.}

## Scene 6: Viewing a network (2.5 min)

1. On Build Nets, select the merged network and click **Display Network**
2. On View Network, point at the legend in the corner of the merged network. \
{The legend lists each source network with its code.}\
 Turn on **Weights**. \
{Every link now shows its code: 11 means both networks have the link, 01 or 10 only one.}
3. Switch to bio\_pf with **Select Network** in the sidebar.
4. Try auto, kamada\_kawai, tree and gravity in **Select Layout Algorithm**, clicking **Redraw** for gravity. \
{Layouts vary a lot. Try several, use Redraw for the ones that start at random, then finish by dragging nodes.}
5. Drag a node, drag the background, scroll to zoom, and hover over a node.
6. Use the Orient tools: flip left-right, flip top-bottom, rotate 90°, and the Rotate slider. Then click **Fit** and **Reset**.
7. With **Weights** on, step **Max rank** with ‹ and ›, watching the link count. \
{Max rank shows which links appear first as the distance limit grows: the strongest links come first.}
8. (Optional) Turn on **Physics**, drag a node while it settles, and turn it off.\
{With Physics enabled an algorithm seeks a layout.  You can interact by dragging and holding nodes, but the algorithm still works.  Turn it off to refine the positions by hand.}
9. Click **Save Layout** in the sidebar, then switch to psy\_pf\_q2 and choose **saved**. \
{A saved layout belongs to the terms, so every network with the same terms can use it. That makes networks easy to compare with the nodes in the same places.}
10. Under **Comparison Set** at the bottom of the sidebar, click **Add** for psy\_pf\_q2. Switch to bio\_pf and psy\_pf and add each, choosing **saved** as the layout for psy\_pf. \
{The comparison set keeps networks ready to show again. Tick view to switch to one, tick delete to take one out, and Clear All empties the set.}
11. Tick **view** on each in turn. \
{The nodes stay in place while the links change. The set holds networks with the same terms, so they can share the saved layout. Add is greyed out for a network with other terms, such as statecaps\_pf.}
12. (Optional) Switch to cities\_pf; it opens with its **saved** layout. \
{This places the cities much like a map. That layout was made and saved with the sample data.}\
 Click **PNG**. \
{PNG saves the picture.}

## Scene 7: Saving and loading projects (1 min)

1. Go back to Build Nets, type a **Project Name**, and click **Save Project**. \
{The app loses its data when it's idle for a while, so save often. Loading it again brings you back to the last save.  You can also load your project in a later session.}
2. Click **Load Project…**, choose the file, and point at the two options. \
{Replace starts fresh with only the project in the file. Add keeps what's loaded and adds the file's items; names already in use get a number added.}
3. Load with **Replace**; the lists come back.

## Scene 8: Closing (about 30 seconds)

1. Open the Help page and scroll past the sections. \
{Help covers every feature, the measures, and both file formats in detail.}
2. Point at the PFNets YouTube link in Getting Started. \
{More videos on Pathfinder and on PyPathfinder are on the PFNets channel.}
3. End on the app. \
{To recap: add your data, derive and compare networks, lay them out, and save your project.}
