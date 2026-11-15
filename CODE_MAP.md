# PyPathfinder code map

The project has two layers:

- **The science (`pypf/`):** the plain Python package. It holds the proximity data, the networks, layouts and statistics. It has no Streamlit, so it also works in a notebook or a script.
- **The app (everything else):** Streamlit pages that show widgets and call `pypf`.

Line numbers are approximate. They're a starting point, so search for the name if one has moved.

```
py_proj/
├── stapp.py            App entry point and the Build Nets page
├── stutils.py          App helpers: session state, project load/save, file upload, tables
├── app_pages/
│   ├── view_net.py     View Network page (sidebar: network, algorithm, Save Layout)
│   └── help.py         Help page (sidebar contents + static/help.html)
├── netview/            Network drawing component (vis-network in the browser, two-way)
│   ├── __init__.py     Python side: network_view(), coords in and out
│   ├── netview.js      Browser side: drawing, toolbar actions, PNG export
│   ├── netview.html    Toolbar and canvas markup
│   └── netview.css     Toolbar and canvas styles
├── pypf/               The science package (no Streamlit)
│   ├── __init__.py     Exports Proximity, PFnet, Collection; sample_path(), load_sample()
│   ├── proximity.py    Proximity: terms + distance matrix, stats
│   ├── pfnet.py        PFnet: Pathfinder / nearest-neighbor networks; parse_q, parse_r
│   ├── collection.py   Collection: all proximities, networks, saved layouts; project JSON
│   ├── readers.py      Reading spreadsheet/CSV and legacy text files into Proximity objects
│   ├── legacy_files.py Parser for old-format .prx.txt files
│   ├── layouts.py      Layout algorithms (igraph and networkx) and "auto" choice
│   ├── utility.py      Math helpers: Floyd/Dijkstra, correlations, similarity, link ranks
│   └── data/           Sample data, samples.csv (descriptions), sample_proj.json
├── static/help.html    The help text (sections: start, build, view, measures, projects, …)
├── tests/              pytest suite for pypf and the table helpers
├── demo/               Scripted demo-video recorder (Playwright) and its scenes
├── pyproject.toml      Package definition: pypf deps; extras [app] and [dev]
└── requirements*.txt   What the app venv / dev tools install
```

## Where to find…

| You're looking for | Go to |
|---|---|
| Page list and navigation | [stapp.py:33](stapp.py:33) `st.navigation` |
| Build Nets sidebar (Clear Demo, project name, save/load) | [stapp.py:53](stapp.py:53) |
| Manage Data Files (add files, load/save project) | [stapp.py:79](stapp.py:79) → [stutils.py](stutils.py) `add_proximity_files`, `load_project`, `save_project` |
| Getting Started box | [stapp.py:114](stapp.py:114) |
| Sample data table | [stapp.py:144](stapp.py:144) |
| Proximity List and its buttons (Derive, Average, Display Matrix, Delete) | [stapp.py:155](stapp.py:155) |
| Distance matrix display | [stapp.py:274](stapp.py:274) |
| Proximity correlations | [stapp.py:286](stapp.py:286) → `Collection.get_proximity_correlations` |
| Network List and its buttons (Display, Properties, Link List, Merge, Delete) | [stapp.py:293](stapp.py:293) |
| Link list / properties display | [stapp.py:377](stapp.py:377) |
| Network similarity | [stapp.py:411](stapp.py:411) → `Collection.network_similarity` |
| Columns shown in the Proximity and Network tables | [stutils.py:79](stutils.py:79) `proximity_table`, `network_table`, `properties_table` |
| Number formatting (significant figures) | [stutils.py:61](stutils.py:61) `sig_figs` |
| Panels scrolling into view | [stutils.py](stutils.py) `keep_panels_in_view`, `REVEAL_KEY` |
| Session state defaults (versions, layout list, sample) | [stutils.py:283](stutils.py:283) `init_pf_session_state` |
| View Network sidebar (Select Network, Select Algorithm, Save Layout) | [app_pages/view_net.py:61](app_pages/view_net.py:61) |
| Network drawing call | [app_pages/view_net.py:120](app_pages/view_net.py:120) → [netview/__init__.py:52](netview/__init__.py:52) |
| Toolbar buttons (flip, rotate, redraw, physics, fit, reset, text, ranks, PNG) | [netview/netview.html](netview/netview.html) (markup), [netview/netview.js:147](netview/netview.js:147) (`actions`) |
| How dragged/rotated positions get back into `net.coords` | [netview/__init__.py:63](netview/__init__.py:63) `_apply_coords` |
| Long node labels split over lines | [netview/__init__.py:31](netview/__init__.py:31) `split_long_term` |
| Help text | [static/help.html](static/help.html); sidebar contents in [app_pages/help.py:14](app_pages/help.py:14) |
| Pathfinder algorithm (q, r) | [pypf/pfnet.py:179](pypf/pfnet.py:179) `_getmindis` → [pypf/utility.py](pypf/utility.py) `floyd`, `dijkstra`, `minkowski` |
| Network naming (`_pf`, `_q2`, `_r1`, `_nn`) | [pypf/pfnet.py:92](pypf/pfnet.py:92) |
| Network properties (degree, center, median, eccentricity) | [pypf/pfnet.py:152](pypf/pfnet.py:152), [pypf/pfnet.py:271](pypf/pfnet.py:271) |
| Link list and link ranks | [pypf/pfnet.py:243](pypf/pfnet.py:243), [pypf/utility.py](pypf/utility.py) `get_link_ranks` |
| Coherence and proximity stats | [pypf/proximity.py:98](pypf/proximity.py:98) → [pypf/utility.py](pypf/utility.py) `coherence` |
| Layout algorithms and "auto" | [pypf/layouts.py](pypf/layouts.py) `get_coords`, `resolve_auto_algorithm` |
| Saved layouts | [pypf/collection.py:60](pypf/collection.py:60) `save_layout`, `apply_saved_layout` |
| Average proximities / merge networks | [pypf/collection.py:194](pypf/collection.py:194), [pypf/collection.py:259](pypf/collection.py:259) |
| Project JSON (what's saved, load replace/add) | [pypf/collection.py:308](pypf/collection.py:308) `get_project_state`, `from_project_state`, `load_project_state` |
| Reading uploaded files | [pypf/readers.py](pypf/readers.py) `read_files` (via `Collection.add_files`) |
| Accepted spreadsheet layouts | [pypf/readers.py:34](pypf/readers.py:34) `table_to_matrix` |
| Legacy `.prx.txt` + `.trm.txt` files | [pypf/readers.py:101](pypf/readers.py:101) `read_legacy_files`, [pypf/legacy_files.py](pypf/legacy_files.py) |
| Sample files and their descriptions | [pypf/data/](pypf/data), [pypf/data/samples.csv](pypf/data/samples.csv) |

## How a click flows

- **Adding files:** Manage Data Files in `stapp.py` → `stutils.add_proximity_files` → `Collection.add_files` → `readers.read_files` → `Proximity` objects. Each one is added with a default PF network.
- **Derive Networks:** the Derive expander in `stapp.py` → `PFnet(prx, q, r)` → `col.add_pfnet`.
- **Display Network:** sets `focus_net_name` and switches to `view_net.py`. That page lays the network out with the algorithm chosen in Select Algorithm (`net.get_layout`), or with "saved" (`col.apply_saved_layout`), then calls `network_view`. The toolbar works in the browser and sends positions back, which become `net.coords`. Redraw asks Python for a new layout.
- **Save Layout:** `view_net.save_layout` → `Collection.save_layout`. It's stored by `termsid`, so it applies to every network with the same terms.

## Running things

```bash
venv/Scripts/python.exe -m streamlit run stapp.py
```

```bash
venv/Scripts/python.exe -m pytest -q
```

```bash
venv/Scripts/python.exe demo/record_demo.py
```
