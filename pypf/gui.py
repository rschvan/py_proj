import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

# Assuming these are available from your pypf package
# You would need to ensure these files (proximity.py, pfnet.py, networkData.py, networkDisplay.py, utility.py)
# are in a 'pypf' directory in the same location as this script, or accessible via your Python path.
from pypf.proximity import Proximity
from pypf.pfnet import PFnet
from pypf.networkData import networkData
from pypf.networkDisplay import NetworkDisplay
from pypf.utility import get_parent_dir  # For default path - now optional initial value


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
        self.recent_project_dirs = []
        # Optionally add the initial parent directory if it's not empty, or leave blank
        initial_parent_dir = get_parent_dir()
        if initial_parent_dir and os.path.isdir(initial_parent_dir):  # Check if it's a valid directory
            self.recent_project_dirs.append(initial_parent_dir)

        self._create_widgets()
        # Initialize the project path combobox with a blank default
        self.project_path_var.set("")  # Start with a blank default

    def _create_widgets(self):
        # --- Help Button (Top Right) ---
        help_button = tk.Button(self, text="Help", command=self._on_help_click)
        help_button.grid(row=0, column=2, padx=10, pady=5, sticky="ne")

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
            state="editable"  # Allows typing and selecting from dropdown
        )
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
        add_proximity_button.grid(row=1, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

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
        get_proximity_info_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        data_correlations_button = tk.Button(info_correlations_frame, text="Data Correlations",
                                             command=self._on_data_correlations)
        data_correlations_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Average Proximities Button
        avg_proximities_button = tk.Button(left_panel, text="Average Proximities", command=self._on_average_proximities)
        avg_proximities_button.grid(row=4, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        # Delete Proximity Button
        delete_proximity_button = tk.Button(left_panel, text="Delete Proximity", command=self._on_delete_proximity)
        delete_proximity_button.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

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
        derive_network_button.grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

        # Force Undirected Checkbox
        self.force_undirected_var = tk.BooleanVar()
        force_undirected_check = ttk.Checkbutton(right_panel, text="Force Undirected",
                                                 variable=self.force_undirected_var,
                                                 command=self._on_force_undirected_toggle)
        force_undirected_check.grid(row=1, column=2, padx=5, pady=10, sticky="w")

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
        network_info_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")  # Placed below listbox
        network_info_frame.grid_columnconfigure(0, weight=1)
        network_info_frame.grid_columnconfigure(1, weight=1)
        network_info_frame.grid_columnconfigure(2, weight=1)

        get_network_info_button = tk.Button(network_info_frame, text="Get Network Info",
                                            command=self._on_get_network_info)
        get_network_info_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        net_properties_button = tk.Button(network_info_frame, text="Net Properties", command=self._on_net_properties)
        net_properties_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        merge_networks_button = tk.Button(network_info_frame, text="Merge Networks", command=self._on_merge_networks)
        merge_networks_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        network_link_list_button = tk.Button(network_info_frame, text="Network Link List",
                                             command=self._on_network_link_list)
        network_link_list_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        network_similarity_button = tk.Button(network_info_frame, text="Network Similarity",
                                              command=self._on_network_similarity)
        network_similarity_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Display Network Button
        display_network_button = tk.Button(right_panel, text="Display Network", command=self._on_display_network)
        display_network_button.grid(row=4, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        # Delete Network Button
        delete_network_button = tk.Button(right_panel, text="Delete Network", command=self._on_delete_network)
        delete_network_button.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

    # --- Utility methods for updating listboxes ---
    def _update_proximity_listbox(self):
        self.proximity_listbox.delete(0, tk.END)  # Clear current items
        for name in sorted(self.proximities.keys()):
            self.proximity_listbox.insert(tk.END, name)

    def _update_pfnet_listbox(self):
        self.pfnet_listbox.delete(0, tk.END)  # Clear current items
        for name in sorted(self.pfnets.keys()):
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
        print(f"Project directory set to: {selected_dir}")

    def _on_new_directory(self):
        print("New Directory button clicked!")
        new_dir = filedialog.askdirectory(title="Select New Project Directory")
        if new_dir:
            self.project_path_var.set(new_dir)
            self._on_directory_select(None)  # Manually trigger update of recent directories
            print(f"New project directory: {new_dir}")
            # Here you would typically initialize a new project structure or clear existing data

    def _on_open_directory(self):
        print("Open Directory button clicked!")
        open_dir = filedialog.askdirectory(title="Open Existing Project Directory")
        if open_dir:
            self.project_path_var.set(open_dir)
            self._on_directory_select(None)  # Manually trigger update of recent directories
            print(f"Opened project directory: {open_dir}")
            # Here you would typically load existing project data and populate listboxes

    # --- Other Callback Functions ---
    def _on_help_click(self):
        print("Help button clicked!")
        messagebox.showinfo("Help", "This is a placeholder for help information.")

    def _on_add_proximity_data(self):
        print("Add Proximity Data button clicked!")
        filepath = filedialog.askopenfilename(
            title="Select Proximity Data File",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            try:
                prx = Proximity(filepath=filepath)
                if prx.name in self.proximities:
                    messagebox.showwarning("Duplicate Proximity",
                                           f"Proximity '{prx.name}' already exists. Replacing it.")
                self.proximities[prx.name] = prx
                self._update_proximity_listbox()  # Update the listbox

                # Select the newly added item (only one item selected)
                index = list(sorted(self.proximities.keys())).index(prx.name)
                self.proximity_listbox.selection_clear(0, tk.END)  # Clear any previous multi-selection
                self.proximity_listbox.selection_set(index)
                self.proximity_listbox.see(index)
                self._on_proximity_select(None)  # Manually trigger selection update

                print(f"Loaded proximity: {prx.name} from {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load proximity data: {e}")
                print(f"Error loading proximity: {e}")

    def _on_proximity_select(self, event):
        # Get all currently selected items
        selected_indices = self.proximity_listbox.curselection()
        self.current_proximity_names = [self.proximity_listbox.get(i) for i in selected_indices]
        print(f"Proximities selected: {self.current_proximity_names}")

    def _on_get_proximity_info(self):
        print("Get Proximity Info button clicked!")
        if self.current_proximity_names:
            # For now, show info for the first selected proximity if multiple are selected
            # This logic can be expanded to show info for all, or ask user to pick one
            prx_name = self.current_proximity_names[0]
            prx = self.proximities[prx_name]
            info = f"Proximity: {prx.name}\n"
            info += f"  Terms: {prx.nterms}\n"
            info += f"  Mean Distance: {prx.mean:.2f}\n"
            info += f"  Std Dev: {prx.sd:.2f}\n"
            messagebox.showinfo("Proximity Info", info)
        else:
            messagebox.showwarning("No Proximity Selected", "Please select one or more proximities from the list.")

    def _on_data_correlations(self):
        print("Data Correlations button clicked!")
        if self.current_proximity_names:
            messagebox.showinfo("Data Correlations",
                                f"This would show correlations for {', '.join(self.current_proximity_names)}.")
        else:
            messagebox.showwarning("No Proximity Selected", "Please select one or more proximities first.")

    def _on_average_proximities(self):
        print("Average Proximities button clicked!")
        if len(self.current_proximity_names) > 1:
            messagebox.showinfo("Average Proximities",
                                f"This would average selected proximities: {', '.join(self.current_proximity_names)}.")
        else:
            messagebox.showwarning("Not Enough Proximities", "Please select at least two proximities to average.")

    def _on_delete_proximity(self):
        print("Delete Proximity button clicked!")
        if self.current_proximity_names:
            if messagebox.askyesno("Confirm Delete",
                                   f"Delete selected proximities: {', '.join(self.current_proximity_names)}?"):
                for name in self.current_proximity_names:
                    if name in self.proximities:
                        del self.proximities[name]
                        print(f"Deleted proximity: {name}")
                self._update_proximity_listbox()  # Update the listbox
                self.current_proximity_names = []  # Clear selection
                messagebox.showinfo("Deleted", "Selected proximities deleted.")
        else:
            messagebox.showwarning("No Proximity Selected", "No proximity selected to delete.")

    def _on_network_type_change(self):
        selected_type = self.network_type_var.get()
        print(f"Network type changed to: {selected_type}")
        if selected_type == "Pathfinder":
            self.q_entry.config(state="normal")
            self.r_entry.config(state="normal")
        else:
            self.q_entry.config(state="disabled")
            self.r_entry.config(state="disabled")

    def _on_derive_network(self):
        print("Derive Network button clicked!")
        if not self.current_proximity_names:
            messagebox.showwarning("Missing Proximity",
                                   "Please select at least one proximity to derive a network from.")
            return

        # For now, derive network from the first selected proximity
        prx_name = self.current_proximity_names[0]
        prx = self.proximities[prx_name]

        try:
            q_val_str = self.q_entry.get().strip().lower()
            r_val_str = self.r_entry.get().strip().lower()

            q = int(q_val_str) if q_val_str not in ('inf', 'n-1') else (
                prx.nterms - 1 if q_val_str == 'n-1' else float('inf'))
            r = int(r_val_str) if r_val_str != 'inf' else float('inf')

            force_undirected = self.force_undirected_var.get()

            pf = PFnet(proximity=prx, q=q, r=r)
            if force_undirected:
                pf.graph = pf.graph.to_undirected()
                pf.nlinks = pf.graph.number_of_edges()
                pf.name = f"{pf.name}_undirected"

            original_pf_name = pf.name
            counter = 1
            while pf.name in self.pfnets:
                pf.name = f"{original_pf_name}_{counter}"
                counter += 1

            self.pfnets[pf.name] = pf
            self._update_pfnet_listbox()

            # Select the newly derived item (only one item selected)
            index = list(sorted(self.pfnets.keys())).index(pf.name)
            self.pfnet_listbox.selection_clear(0, tk.END)  # Clear any previous multi-selection
            self.pfnet_listbox.selection_set(index)
            self.pfnet_listbox.see(index)
            self._on_pfnet_select(None)  # Manually trigger selection update

            print(f"Derived network: {pf.name}")

        except ValueError as e:
            messagebox.showerror("Input Error",
                                 f"Invalid input for q or r: {e}\nPlease enter an integer or 'inf'. For q, 'n-1' is also allowed.")
            print(f"Input error: {e}")
        except Exception as e:
            messagebox.showerror("Error Deriving Network", f"Error: {e}")
            print(f"Error deriving network: {e}")

    def _on_pfnet_select(self, event):
        # Get all currently selected items
        selected_indices = self.pfnet_listbox.curselection()
        self.current_pfnet_names = [self.pfnet_listbox.get(i) for i in selected_indices]
        print(f"Networks selected: {self.current_pfnet_names}")

    def _on_force_undirected_toggle(self):
        print(f"Force Undirected checkbox toggled to: {self.force_undirected_var.get()}")

    def _on_get_network_info(self):
        print("Get Network Info button clicked!")
        if self.current_pfnet_names:
            # For now, show info for the first selected network
            pf_name = self.current_pfnet_names[0]
            pf = self.pfnets[pf_name]
            info = f"Network: {pf.name}\n"
            info += f"  Nodes: {pf.nnodes}\n"
            info += f"  Links: {pf.nlinks}\n"
            info += f"  q: {pf.q}\n"
            info += f"  r: {pf.r}\n"
            messagebox.showinfo("Network Info", info)
        else:
            messagebox.showwarning("No Network Selected", "Please select one or more networks from the list.")

    def _on_net_properties(self):
        print("Net Properties button clicked!")
        if self.current_pfnet_names:
            messagebox.showinfo("Net Properties",
                                f"This would display properties for {', '.join(self.current_pfnet_names)}.")
        else:
            messagebox.showwarning("No Network Selected", "Please select one or more networks first.")

    def _on_merge_networks(self):
        print("Merge Networks button clicked!")
        if len(self.current_pfnet_names) > 1:
            messagebox.showinfo("Merge Networks",
                                f"This would merge selected networks: {', '.join(self.current_pfnet_names)}.")
        else:
            messagebox.showwarning("Not Enough Networks", "Please select at least two networks to merge.")

    def _on_network_link_list(self):
        print("Network Link List button clicked!")
        if self.current_pfnet_names:
            # For now, show link list for the first selected network
            pf_name = self.current_pfnet_names[0]
            pf = self.pfnets[pf_name]
            link_list = "\n".join([f"{u} - {v}" for u, v in pf.graph.edges()])
            messagebox.showinfo("Network Link List",
                                f"Links in {pf.name}:\n{link_list[:500]}...")  # Show first 500 chars
        else:
            messagebox.showwarning("No Network Selected", "Please select one or more networks first.")

    def _on_network_similarity(self):
        print("Network Similarity button clicked!")
        if len(self.current_pfnet_names) > 1:
            messagebox.showinfo("Network Similarity",
                                f"This would compare similarities between networks: {', '.join(self.current_pfnet_names)}.")
        else:
            messagebox.showwarning("Not Enough Networks", "Please select at least two networks to compare similarity.")

    def _on_display_network(self):
        print("Display Network button clicked!")
        if self.current_pfnet_names:
            # For now, display only the first selected network.
            # Multi-network display would require more complex visualization logic.
            pf_name = self.current_pfnet_names[0]
            pf = self.pfnets[pf_name]

            display_window = tk.Toplevel(self)
            display_window.title(f"Network: {pf.name}")
            display_window.geometry("800x600")

            display_window.grid_rowconfigure(0, weight=1)
            display_window.grid_columnconfigure(0, weight=1)

            try:
                # Ensure the window is rendered to get accurate dimensions
                display_window.update_idletasks()
                canvas_width = display_window.winfo_width()
                canvas_height = display_window.winfo_height()

                # Fallback in case window dimensions are too small initially
                if canvas_width < 10 or canvas_height < 10:
                    canvas_width = 800
                    canvas_height = 600

                node_data, edge_data = networkData(pf, method="gravity",
                                                   canvas_width=canvas_width,
                                                   canvas_height=canvas_height)

                network_canvas = NetworkDisplay(
                    display_window,
                    nodes=node_data,
                    edges=edge_data,
                    width=canvas_width,
                    height=canvas_height,
                    bg="white"
                )
                network_canvas.grid(row=0, column=0, sticky="nsew")
                network_canvas.draw_network()

                # Bind the resize event for the new window
                network_canvas.bind("<Configure>",
                                    lambda event, canvas=network_canvas, pf_obj=pf: self._on_network_display_resize(
                                        event, canvas, pf_obj))

            except Exception as e:
                messagebox.showerror("Display Error", f"Could not display network: {e}")
                print(f"Error displaying network: {e}")
        else:
            messagebox.showwarning("No Network Selected",
                                   "Please select one or more networks from the list to display.")

    def _on_network_display_resize(self, event, canvas: NetworkDisplay, pf_obj: PFnet):
        """Callback for resizing the network display canvas."""
        print(f"Resizing network canvas to {event.width}x{event.height}")
        try:
            # Recalculate layout based on new canvas size
            node_data, edge_data = networkData(pf_obj, method="force",
                                               canvas_width=event.width,
                                               canvas_height=event.height)
            canvas.nodes = node_data
            canvas.edges = edge_data
            canvas.canvas_height = event.height  # Update internal canvas height for correct y-axis inversion
            canvas.draw_network()  # Redraw the network
        except Exception as e:
            print(f"Error during network redraw on resize: {e}")

    def _on_delete_network(self):
        print("Delete Network button clicked!")
        if self.current_pfnet_names:
            if messagebox.askyesno("Confirm Delete",
                                   f"Delete selected networks: {', '.join(self.current_pfnet_names)}?"):
                for name in self.current_pfnet_names:
                    if name in self.pfnets:
                        del self.pfnets[name]
                        print(f"Deleted network: {name}")
                self._update_pfnet_listbox()  # Update the listbox
                self.current_pfnet_names = []  # Clear selection
                messagebox.showinfo("Deleted", "Selected networks deleted.")
        else:
            messagebox.showwarning("No Network Selected", "No network selected to delete.")


if __name__ == "__main__":
    app = PyPathfinder()
    app.mainloop()