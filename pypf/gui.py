import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
import numpy as np

from pypf.proximity import Proximity
from pypf.pfnet import PFnet
from pypf.collection import Collection



from pypf.shownet import shownet
from pypf.utility import mypfnets_dir  # directory for sample data and help

class PyPathfinder(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PyPathfinder")
        self.geometry("800x600")

        # Configure columns and rows for the main window to be resizable
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # Internal data storage (dictionaries to store objects by name) in Collection col
        self.col: Collection = Collection()

        self._create_widgets()
        # Initialize the project path combobox
        self.project_path_var.set(self.col.project_dirs[0])

    def _create_widgets(self):

        # --- Left Panel (Proximity Management) ---
        left_panel = ttk.Frame(self, padding="10")
        left_panel.grid(row=0, column=0, rowspan=5, sticky="nsew")
        left_panel.grid_rowconfigure(2, weight=1)  # Allow proximity listbox to expand
        for i in range(3):
            left_panel.grid_columnconfigure(i, weight=1)

        # Project Directory Section
        project_dir_frame = ttk.LabelFrame(left_panel, text="Project Directory", padding="10")
        project_dir_frame.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        project_dir_frame.grid_columnconfigure(0, weight=1)

        # Project Directory Combobox (Dropdown)
        self.project_path_var = tk.StringVar()  # Initialize as blank
        self.project_path_combobox = ttk.Combobox(
            project_dir_frame,
            textvariable=self.project_path_var,
            values=self.col.mypfnets_dir,  # Set initial dropdown value
            state="readonly")
        self.project_path_combobox.grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        self.project_path_combobox.bind("<<ComboboxSelected>>", self._on_directory_select)
        # Bind for when user types and leaves the entry (focus out)
        self.project_path_combobox.bind("<FocusOut>", self._on_directory_select)

        new_dir_button = tk.Button(project_dir_frame, text="New Directory", command=self._on_new_directory)
        new_dir_button.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        open_dir_button = tk.Button(project_dir_frame, text="Open Directory", command=self._on_open_directory)
        open_dir_button.grid(row=1, column=2, padx=5, pady=5, sticky="e")

        # Add Proximity Data Button
        add_proximity_button = tk.Button(left_panel, text="Add Proximity Data", command=self._on_add_proximity_data)
        add_proximity_button.grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

        # Proximity Data Display (Listbox for selection)
        # selectmode set to tk.EXTENDED for multi-select
        self.proximity_listbox = tk.Listbox(left_panel, selectmode=tk.EXTENDED, bg="white", relief="sunken")
        self.proximity_listbox.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.proximity_listbox.bind("<<ListboxSelect>>", self._on_proximity_select)  # Callback for selection

        proximity_scroll = ttk.Scrollbar(left_panel, command=self.proximity_listbox.yview)
        proximity_scroll.grid(row=2, column=2, sticky="ns")
        self.proximity_listbox.config(yscrollcommand=proximity_scroll.set)

        # Get Proximity Info / Data Correlations Buttons
        info_correlations_frame = ttk.Frame(left_panel)
        info_correlations_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")  # Placed below listbox
        info_correlations_frame.grid_columnconfigure(0, weight=1)
        info_correlations_frame.grid_columnconfigure(1, weight=1)

        get_proximity_info_button = tk.Button(info_correlations_frame, text="Get Proximity Info",
                                              command=self._on_get_proximity_info)
        get_proximity_info_button.grid(row=1, column=0, columnspan=1, padx=5, pady=5, sticky="ew")
        data_correlations_button = tk.Button(info_correlations_frame, text="Data Correlations",
                                             command=self._on_data_correlations)
        data_correlations_button.grid(row=0, column=1, columnspan=1, padx=5, pady=5, sticky="ew")

        # Average Proximities Button
        avg_proximities_button = tk.Button(info_correlations_frame, text="Average Proximities", command=self._on_average_proximities)
        avg_proximities_button.grid(row=0, column=0, columnspan=1, padx=5, pady=10, sticky="ew")

        # Delete Proximity Button
        delete_proximity_button = tk.Button(left_panel, text="Delete Proximity", command=self._on_delete_proximity)
        delete_proximity_button.grid(row=5, column=1, columnspan=1, padx=5, pady=5, sticky="ew")

        # --- Right Panel (Network Management) ---
        right_panel = ttk.Frame(self, padding="10")
        right_panel.grid(row=0, column=1, rowspan=5, columnspan=2, sticky="nsew")
        right_panel.grid_rowconfigure(2, weight=1)  # Allow network listbox to expand
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_columnconfigure(1, weight=1)
        right_panel.grid_columnconfigure(2, weight=1)

        # Network Type Section
        network_type_frame = ttk.LabelFrame(right_panel, text="Network Type", padding="10")
        network_type_frame.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.network_type_var = tk.StringVar(value="Pathfinder")  # Default selection
        pathfinder_radio = ttk.Radiobutton(network_type_frame, text="Pathfinder", variable=self.network_type_var,
                                           value="Pathfinder", command=self._on_network_type_change)
        pathfinder_radio.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        threshold_radio = ttk.Radiobutton(network_type_frame, text="Threshold", variable=self.network_type_var,
                                          value="Threshold", command=self._on_network_type_change)
        threshold_radio.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        nearest_neighbor_radio = ttk.Radiobutton(network_type_frame, text="Nearest Neighbor",
                                                 variable=self.network_type_var, value="Nearest Neighbor",
                                                 command=self._on_network_type_change)
        nearest_neighbor_radio.grid(row=2, column=0, sticky="w", padx=5, pady=2)

        # --- Help Button (Top Right) ---
        help_button = tk.Button(network_type_frame, text="Help", command=self._on_help_click)
        help_button.grid(row=2, column=4, padx=20, pady=5, sticky="ne")

        # q and r inputs
        q_label = tk.Label(network_type_frame, text="q =")
        q_label.grid(row=0, column=1, sticky="w", padx=5)
        self.q_entry = ttk.Entry(network_type_frame, width=5)
        self.q_entry.grid(row=0, column=2, sticky="ew", padx=5)
        self.q_entry.insert(0, "inf")  # Default value from image

        r_label = tk.Label(network_type_frame, text="r =")
        r_label.grid(row=1, column=1, sticky="w", padx=5)
        self.r_entry = ttk.Entry(network_type_frame, width=5)
        self.r_entry.grid(row=1, column=2, sticky="ew", padx=5)
        self.r_entry.insert(0, "inf")  # Default value from image

        # Derive Network Button
        derive_network_button = tk.Button(right_panel, text="Derive Network", command=self._on_derive_network)
        derive_network_button.grid(row=1, column=0, columnspan=1, padx=5, pady=10, sticky="ew")

        # Layout Method Dropdown
        self.layout_method = tk.StringVar(value="gravity")  # Initialize
        self.layout_combobox = ttk.Combobox(
            right_panel,
            textvariable=self.layout_method,
            values=["gravity", "neato", "dot", "kamada_kawai" ,"MDS"],  # Set dropdown values
            state="readonly")
        self.layout_combobox.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        # Network Display (Listbox for selection)
        # selectmode set to tk.EXTENDED for multi-select
        self.pfnet_listbox = tk.Listbox(right_panel, selectmode=tk.EXTENDED, bg="white", relief="sunken")
        self.pfnet_listbox.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.pfnet_listbox.bind("<<ListboxSelect>>", self._on_pfnet_select)  # Callback for selection

        pfnet_scroll = ttk.Scrollbar(right_panel, command=self.pfnet_listbox.yview)
        pfnet_scroll.grid(row=2, column=2, sticky="ns")
        self.pfnet_listbox.config(yscrollcommand=pfnet_scroll.set)

        # Network Info Buttons (Stacked vertically on the right)
        network_info_frame = ttk.Frame(right_panel)
        network_info_frame.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky="ew")  # Placed below listbox
        network_info_frame.grid_columnconfigure(0, weight=1)
        network_info_frame.grid_columnconfigure(1, weight=1)
        network_info_frame.grid_columnconfigure(2, weight=1)

        get_network_info_button = tk.Button(network_info_frame, text="Get Network Info",
                                            command=self._on_get_network_info)
        get_network_info_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        merge_networks_button = tk.Button(network_info_frame, text="Merge Networks", command=self._on_merge_networks)
        merge_networks_button.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        network_link_list_button = tk.Button(network_info_frame, text="Network Link List",
                                             command=self._on_network_link_list)
        network_link_list_button.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        network_similarity_button = tk.Button(network_info_frame, text="Network Similarity",
                                              command=self._on_network_similarity)
        network_similarity_button.grid(row=2, column=0, columnspan=1, padx=5, pady=5, sticky="ew")

        # Display Network Button
        display_network_button = tk.Button(network_info_frame, text="Display Network", command=self._on_display_network)
        display_network_button.grid(row=1, column=0, columnspan=1, padx=5, pady=10, sticky="ew")

        # Delete Network Button
        delete_network_button = tk.Button(network_info_frame, text="Delete Network", command=self._on_delete_network)
        delete_network_button.grid(row=3, column=1, columnspan=1, padx=5, pady=5, sticky="ew")

    # --- Utility methods for updating listboxes ---
    def _update_proximity_listbox(self):
        self.proximity_listbox.delete(0, tk.END)  # Clear current items
        for name in self.col.proximities.keys():
            self.proximity_listbox.insert(tk.END, name)

    def _update_pfnet_listbox(self):
        self.pfnet_listbox.delete(0, tk.END)  # Clear current items
        for name in self.col.pfnets.keys():
            self.pfnet_listbox.insert(tk.END, name)

    # --- Project Directory Callbacks ---
    def _on_directory_select(self, event):
        """Called when a directory is selected from the combobox dropdown or typed in."""
        selected_dir = self.project_path_var.get()
        if selected_dir:
            # Normalize path (e.g., remove trailing slashes) for consistent storage
            selected_dir = os.path.normpath(selected_dir)
            if os.path.isdir(selected_dir) and selected_dir not in self.col.project_dirs:
                self.col.project_dirs.insert(0, selected_dir)  # Add to top of recent list
                # Limit number of recent directories to, say, 10
                self.col.project_dirs = self.col.project_dirs[:10]
                self.project_path_combobox['values'] = self.col.project_dirs  # Update dropdown values

    def _on_new_directory(self):
        new_dir = filedialog.askdirectory(title="Select New Project Directory")
        if new_dir:
            self.project_path_var.set(new_dir)
            self._on_directory_select(None)  # Manually trigger update of recent directories
            # Here you would typically initialize a new project structure or clear exclude data

    def _on_open_directory(self):
        dir = self.project_path_var.get()
        if dir:
            os.startfile(dir) # open dir in explorer

    # --- Other Callback Functions ---
    def _on_help_click(self):
        #messagebox.showinfo("Help", "This is a placeholder for help information.")
        os.startfile(os.path.join(self.col.mypfnets_dir,"pypfhelp.pdf"))

    def _on_add_proximity_data(self):
        filepath = filedialog.askopenfilenames(
            title="Select Proximity Data File",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.project_path_var.get() if self.project_path_var else "")
        if filepath:
            for f in filepath:
                try:
                    prx = Proximity(filepath=f)
                    if prx.name in self.col.proximities:
                        messagebox.showwarning("Duplicate Proximity",
                                               f"Proximity '{prx.name}' already exists. Replacing it.")
                    self.col.proximities[prx.name] = prx
                    self._update_proximity_listbox()  # Update the listbox
                    self.col.selected_prxs = []
                except Exception as e:
                    messagebox.showerror("Error", f"Could not load proximity data: {e}")

    def _on_proximity_select(self, event):
        selected_indices = self.proximity_listbox.curselection()
        self.col.selected_prxs = [self.proximity_listbox.get(i) for i in selected_indices]

    def _on_get_proximity_info(self):
        self.col.get_proximity_info()
        prxlist = self.proximity_listbox.get(0, tk.END)
        infolist = []
        for prx_name in prxlist:
            prx = self.proximities[prx_name]
            info = prx.get_info() # info is a dictionary
            infolist.append(info)
        # create dataframe with no index
        df = pd.DataFrame(infolist)
        f = os.path.join(self.project_path_var.get(),"proximity_info.csv")
        df.to_csv(f)
        os.startfile(f)

    def _on_data_correlations(self):
        self.col.get_proximity_correlations()

    def _on_average_proximities(self):
        self.col.average_proximities()
        self._update_proximity_listbox()
        self.col.selected_prxs = []

    def _on_delete_proximity(self):
        self.col.delete_proximity()
        self._update_proximity_listbox()

    def _on_network_type_change(self):
        selected_type = self.network_type_var.get()
        if selected_type == "Pathfinder":
            self.q_entry.config(state="normal")
            self.r_entry.config(state="normal")
        else:
            self.q_entry.config(state="disabled")
            self.r_entry.config(state="disabled")

    def _on_derive_network(self):
        if self.col.selected_prxs:
            prxlist = self.col.selected_prxs
        else:
            prxlist = list(self.col.proximities.keys())

        q_val_str = self.q_entry.get().strip().lower()
        r_val_str = self.r_entry.get().strip().lower()
        q = int(q_val_str) if q_val_str != 'inf' else float('inf')
        r = int(r_val_str) if r_val_str != 'inf' else float('inf')

        for prx_name in prxlist:
            prx = self.col.proximities[prx_name]
            match self.network_type_var.get():
                case "Pathfinder":
                    net = PFnet(proximity=prx, q=q, r=r)
                case "Threshold":
                    net = PFnet(proximity=prx, type="th")
                case "Nearest Neighbor":
                    net = PFnet(proximity=prx, type="nn")
            name = net.name
            counter = 1
            while net.name in self.col.pfnets:
                net.name = f"{name}_{counter}"
                counter += 1

            self.col.pfnets[net.name] = net
            self._update_pfnet_listbox()
            self.pfnet_listbox.selection_clear(0, tk.END)
            self.col.selected_prxs = []

    def _on_pfnet_select(self, event):
        selected_indices = self.pfnet_listbox.curselection()
        self.col.selected_nets = [self.pfnet_listbox.get(i) for i in selected_indices]

    def _on_get_network_info(self):
        self.col.get_pfnet_info()

    def _on_merge_networks(self):
        self.col.merge_networks()
        self._update_pfnet_listbox()
        self.col.selected_nets = []

    def _on_network_link_list(self):
        self.col.network_link_list()

    def _on_network_similarity(self):
        self.col.network_similarity()

    def _on_display_network(self):
        if self.col.selected_nets:
            pf_name = self.col.selected_nets[0]
            pf = self.col.pfnets[pf_name]
            try:
                # Call the shownet function, passing self (the main Tkinter window) as the parent
                shownet(self, pf, method=self.layout_method.get(), width=800, height=600)
            except Exception as e:
                messagebox.showerror("Display Error", f"Could not display network: {e}")
        else:
            messagebox.showwarning("No Network Selected",
                                   "Please select a network from the list to display.")

    def _on_delete_network(self):
        self.col.delete_pfnet()
        self._update_pfnet_listbox()

if __name__ == "__main__":
    app = PyPathfinder()
    app.mainloop()