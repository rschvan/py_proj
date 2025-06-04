import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
import numpy as np

from pypf.proximity import Proximity
from pypf.pfnet import PFnet
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

        # Internal data storage (dictionaries to store objects by name)
        self.proximities: dict[str, Proximity] = {}
        self.pfnets: dict[str, PFnet] = {}

        # Changed to store a list of selected names for multi-select
        self.current_proximity_names: list[str] = []
        self.current_pfnet_names: list[str] = []

        # List to store recent project directories for the dropdown
        userdir = mypfnets_dir()
        self.mypfnets_dir = userdir
        self.recent_project_dirs = []
        self.recent_project_dirs.append(userdir)

        self._create_widgets()
        # Initialize the project path combobox
        self.project_path_var.set(userdir)

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
            values=self.recent_project_dirs,  # Set initial dropdown values
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
        get_network_info_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        merge_networks_button = tk.Button(network_info_frame, text="Merge Networks", command=self._on_merge_networks)
        merge_networks_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        network_link_list_button = tk.Button(network_info_frame, text="Network Link List",
                                             command=self._on_network_link_list)
        network_link_list_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        network_similarity_button = tk.Button(network_info_frame, text="Network Similarity",
                                              command=self._on_network_similarity)
        network_similarity_button.grid(row=1, column=0, columnspan=1, padx=5, pady=5, sticky="ew")

        # Display Network Button
        display_network_button = tk.Button(network_info_frame, text="Display Network", command=self._on_display_network)
        display_network_button.grid(row=0, column=0, columnspan=1, padx=5, pady=10, sticky="ew")

        # Delete Network Button
        delete_network_button = tk.Button(network_info_frame, text="Delete Network", command=self._on_delete_network)
        delete_network_button.grid(row=3, column=1, columnspan=1, padx=5, pady=5, sticky="ew")

    # --- Utility methods for updating listboxes ---
    def _update_proximity_listbox(self):
        self.proximity_listbox.delete(0, tk.END)  # Clear current items
        for name in self.proximities.keys():
            self.proximity_listbox.insert(tk.END, name)

    def _update_pfnet_listbox(self):
        self.pfnet_listbox.delete(0, tk.END)  # Clear current items
        for name in self.pfnets.keys():
            self.pfnet_listbox.insert(tk.END, name)

    # --- Project Directory Callbacks ---
    def _on_directory_select(self, event):
        """Called when a directory is selected from the combobox dropdown or typed in."""
        selected_dir = self.project_path_var.get()
        if selected_dir:
            # Normalize path (e.g., remove trailing slashes) for consistent storage
            selected_dir = os.path.normpath(selected_dir)
            if os.path.isdir(selected_dir) and selected_dir not in self.recent_project_dirs:
                self.recent_project_dirs.insert(0, selected_dir)  # Add to top of recent list
                # Limit number of recent directories to, say, 10
                self.recent_project_dirs = self.recent_project_dirs[:10]
                self.project_path_combobox['values'] = self.recent_project_dirs  # Update dropdown values

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
        os.startfile(os.path.join(self.mypfnets_dir,"pypfhelp.pdf"))

    def _on_add_proximity_data(self):
        filepath = filedialog.askopenfilenames(
            title="Select Proximity Data File",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.project_path_var.get() if self.project_path_var else "")
        if filepath:
            for f in filepath:
                try:
                    prx = Proximity(filepath=f)
                    if prx.name in self.proximities:
                        messagebox.showwarning("Duplicate Proximity",
                                               f"Proximity '{prx.name}' already exists. Replacing it.")
                    self.proximities[prx.name] = prx
                    self._update_proximity_listbox()  # Update the listbox

                    # Select the newly added item (only one item selected)
                    index = list(sorted(self.proximities.keys())).index(prx.name)
                    self.proximity_listbox.selection_clear(0, tk.END)  # Clear any previous multi-selection
                    # self.proximity_listbox.selection_set(index)
                    # self.proximity_listbox.see(index)
                    self._on_proximity_select(None)  # Manually trigger selection update
                except Exception as e:
                    messagebox.showerror("Error", f"Could not load proximity data: {e}")

    def _on_proximity_select(self, event):
        # Get all currently selected items
        selected_indices = self.proximity_listbox.curselection()
        self.current_proximity_names = [self.proximity_listbox.get(i) for i in selected_indices]

    def _on_get_proximity_info(self):
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
        from pypf.utility import discorr
        # get proximity names
        prxnames = self.proximity_listbox.get(0, tk.END)
        n = len(prxnames)
        cors = np.zeros((n,n))
        # set diagonal to 1
        np.fill_diagonal(cors, 1)
        if n > 1:
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    dis1 = self.proximities[prxnames[i]].dismat
                    dis2 = self.proximities[prxnames[j]].dismat
                    cors[i,j] = discorr(dis1,dis2)
            df = pd.DataFrame(cors, index=prxnames, columns=prxnames)
            f = os.path.join(self.project_path_var.get(), "proximity_corrs.csv")
            df.to_csv(f)
            os.startfile(f)
        else:
            messagebox.showwarning("Too Few Proximities", "Must have at least two.")

    def _on_average_proximities(self):
        from pypf.utility import average_proximity
        if len(self.current_proximity_names) < 2:
            messagebox.showwarning("Not Enough Proximities", "Please select at least two proximities to average.")
            return
        #ans = tk.messagebox.askquestion("Method?","Select Means or Medians",default="yes")
        yes_no = messagebox.askyesno("Average Proximities", "Yes for means? No for medians")
        if yes_no:
            choice = "mean"
        else:
            choice = "median"
        proxes = [self.proximities[p] for p in self.current_proximity_names]
        ave_prx = average_proximity(proxes, method=choice)
        if ave_prx is None:
            messagebox.showwarning("Average Failed", "Proximities not compatible.")
            return
        self.proximities[ave_prx.name] = ave_prx
        self._update_proximity_listbox()  # Update the listbox
        self._on_proximity_select(None)

    def _on_delete_proximity(self):
        if self.current_proximity_names:
            if messagebox.askyesno("Confirm Delete",
                                   f"Delete selected proximities: {', '.join(self.current_proximity_names)}?"):
                for name in self.current_proximity_names:
                    if name in self.proximities:
                        del self.proximities[name]
                self._update_proximity_listbox()  # Update the listbox
                self.current_proximity_names = []  # Clear selection
                #messagebox.showinfo("Deleted", "Selected proximities deleted.")
        else:
            messagebox.showwarning("No Proximity Selected", "No proximity selected to delete.")

    def _on_network_type_change(self):
        selected_type = self.network_type_var.get()
        if selected_type == "Pathfinder":
            self.q_entry.config(state="normal")
            self.r_entry.config(state="normal")
        else:
            self.q_entry.config(state="disabled")
            self.r_entry.config(state="disabled")

    def _on_derive_network(self):
        if self.current_proximity_names:
            prxlist = self.current_proximity_names
        else:
            prxlist = self.proximity_listbox.get(0, tk.END)

        q_val_str = self.q_entry.get().strip().lower()
        r_val_str = self.r_entry.get().strip().lower()
        q = int(q_val_str) if q_val_str != 'inf' else float('inf')
        r = int(r_val_str) if r_val_str != 'inf' else float('inf')

        for prx_name in prxlist:
            prx = self.proximities[prx_name]
            match self.network_type_var.get():
                case "Pathfinder":
                    net = PFnet(proximity=prx, q=q, r=r)
                case "Threshold":
                    net = PFnet(proximity=prx, type="th")
                case "Nearest Neighbor":
                    net = PFnet(proximity=prx, type="nn")
            name = net.name
            counter = 1
            while net.name in self.pfnets:
                net.name = f"{name}_{counter}"
                counter += 1

            self.pfnets[net.name] = net
            self._update_pfnet_listbox()
            self.pfnet_listbox.selection_clear(0, tk.END)
            self._on_pfnet_select(None)

    def _on_pfnet_select(self, event):
        # Get all currently selected items
        selected_indices = self.pfnet_listbox.curselection()
        self.current_pfnet_names = [self.pfnet_listbox.get(i) for i in selected_indices]

    def _on_get_network_info(self):
        nets = self.pfnet_listbox.get(0, tk.END)
        n = len(nets)
        if nets:
            df = self.pfnets[nets[0]].get_info()
        else:
            return
        for i in range(1,n):
            net = self.pfnets[nets[i]]
            df = pd.concat([df, net.get_info()], axis=0)
            f = os.path.join(self.project_path_var.get(), "network_info.csv")
        df.to_csv(f)
        os.startfile(f)

    def _on_merge_networks(self):
        from pypf.utility import merge_networks
        netnames = self.current_pfnet_names
        nets = []
        if len(netnames) > 1:
            for name in netnames:
                nets.append(self.pfnets[name])
            mrg = merge_networks(nets)
            self.pfnets[mrg.name] = mrg
            self._update_pfnet_listbox()
            # self.on_pfnet_select(None)
        else:
            messagebox.showwarning("Not Enough Networks", "Please select at least two networks to merge.")

    def _on_network_link_list(self):

        if self.current_pfnet_names:
            # For now, show link list for the first selected network
            pf_name = self.current_pfnet_names[0]
            pf = self.pfnets[pf_name]
            link_list = list(pf.graph.edges())
            df = pd.DataFrame(link_list, columns=["from", "to"])
            f = os.path.join(self.project_path_var.get(), "link_list.csv")
            df.to_csv(f)
            os.startfile(f)
        else:
            messagebox.showwarning("No Network Selected", "Please select one or more networks first.")

    def _on_network_similarity(self):
        from pypf.utility import netsim
        netlist = self.pfnet_listbox.get(0, tk.END)
        n = len(netlist)
        if n > 1:
            simmat = np.zeros((n,n))
            for i in range(n):
                inet = self.pfnets[netlist[i]]
                for j in range(n):
                    jnet = self.pfnets[netlist[j]]
                    if i == j:
                        simmat[i,j] = 1
                    elif inet.nnodes != jnet.nnodes:
                        simmat[i,j] = np.nan
                    else:
                        sim = netsim(inet.adjmat, jnet.adjmat)
                        simmat[i,j] = sim["similarity"]
        else:
            messagebox.showwarning("Not Enough Networks", "Must have at least two networks for similarity.")
        df = pd.DataFrame(simmat, index=netlist, columns=netlist)
        #messagebox.showinfo("Network Similarity", df.to_string())
        # write df to a file
        f = os.path.join(self.project_path_var.get(), "net_sim.csv")
        pd.DataFrame(simmat, index=netlist, columns=netlist).to_csv(f)
        # open the file
        os.startfile(f)

    def _on_display_network(self):
        if self.current_pfnet_names:
            pf_name = self.current_pfnet_names[0]
            pf = self.pfnets[pf_name]

            try:
                # Call the shownet function, passing self (the main Tkinter window) as the parent
                shownet(self, pf, method="gravity", width=800, height=600)

            except Exception as e:
                messagebox.showerror("Display Error", f"Could not display network: {e}")
        else:
            messagebox.showwarning("No Network Selected",
                                   "Please select a network from the list to display.")

    def _on_delete_network(self):
        if self.current_pfnet_names:
            if messagebox.askyesno("Confirm Delete",
                                   f"Delete selected networks: {', '.join(self.current_pfnet_names)}?"):
                for name in self.current_pfnet_names:
                    if name in self.pfnets:
                        del self.pfnets[name]
                self._update_pfnet_listbox()  # Update the listbox
                self.current_pfnet_names = []  # Clear selection
        else:
            messagebox.showwarning("No Network Selected", "No network selected to delete.")

if __name__ == "__main__":
    app = PyPathfinder()
    app.mainloop()