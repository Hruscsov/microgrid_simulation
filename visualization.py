import matplotlib as mpl
import numpy as np
import pandas as pd
import pandapower.plotting as plot
import networkx as nx
import matplotlib.pyplot as plt


def visualize_network(net, k=None, iterations=50, figsize=(10, 8)):
    """
    Egyszerű vizualizáció pandapower hálózathoz koordináták nélkül.
    - ha nincs bus_geodata, készít egy spring layoutot networkx-szel
    - majd simple_plot-tal kirajzolja
    Paraméterek:
        net: pandapower net
        figsize: matplotlib méret
        k: spring layout "távolság" paramétere (ha None, automatikus)
        iterations: spring layout iterációszám
    """
    # 1) Ha már van geodata, akkor csak rajzolunk
    if hasattr(net, "bus_geodata") and len(net.bus_geodata) == len(net.bus):
        # van minden buszhoz koordináta
        plot.simple_plot(net, plot_loads=True, plot_sgens=True)
        plt.gcf().set_size_inches(figsize)
        plt.show()
        return

    # 2) Ha nincs geodata, építünk egy networkx gráfot a pandapowerből
    G = nx.Graph()

    # buszok
    for bus_idx in net.bus.index:
        G.add_node(bus_idx)

    # vonalak
    for _, line in net.line.iterrows():
        G.add_edge(int(line.from_bus), int(line.to_bus))

    # trafók
    for _, trafo in net.trafo.iterrows():
        G.add_edge(int(trafo.hv_bus), int(trafo.lv_bus))

    # bus-bus switchek
    for _, sw in net.switch.iterrows():
        if sw.et == "b":
            G.add_edge(int(sw.bus), int(sw.element))

    # 3) spring layout számolása
    # (ha nagyon nagy a háló, lehet kamadakawai vagy shell is)
    pos = nx.spring_layout(G, k=k, iterations=iterations, seed=42)

    # 4) ezt betöltjük a pandapower bus_geodata táblájába
    # (pandapower elvárja: oszlopok: x, y; index=bus index)
    import pandas as pd
    bus_geo = []
    for bus_idx in net.bus.index:
        x, y = pos[bus_idx]
        bus_geo.append({"bus": bus_idx, "x": x, "y": y})
    bus_geo_df = pd.DataFrame(bus_geo).set_index("bus")

    net.bus_geodata = bus_geo_df

    # 5) és most már lehet egyszerűen rajzolni
    ax = plot.simple_plot(net, plot_loads=True, plot_sgens=True, respect_switches=True)
    ax.set_title("Berkenye hálózat – sablonos elrendezés")
    plt.show()


def plot_loadflow_results(net, net_name, show=True, dpi=300):
    # --- 1) buszfeszültség színezés ---
    vm = net.res_bus.vm_pu.values  # p.u.
    vmin, vmax = 0.9, 0.95
    vm_clipped = np.clip(vm, vmin, vmax)
    vm_norm = (vm_clipped - vmin) / (vmax - vmin + 1e-9)
    bus_colors = [plt.cm.viridis(val) for val in vm_norm]

    bus_collection = plot.create_bus_collection(
        net,
        buses=net.bus.index.tolist(),
        size=20,
        color=bus_colors
    )

    # --- 2) vonalterhelés színezés ---
    line_collections = []
    line_colors = None
    if len(net.line):
        loading = net.res_line.loading_percent.values  # %
        loading_clipped = np.clip(loading, 0, 100)
        loading_norm = loading_clipped / 100.0
        line_colors = [plt.cm.inferno(val) for val in loading_norm]

        lc = plot.create_line_collection(
            net,
            lines=net.line.index.tolist(),
            use_bus_geodata=True,
            color=line_colors,
            linewidths=2
        )
        line_collections.append(lc)

    # --- 3) kirajzolás ---
    ax = plot.draw_collections(
        [bus_collection] + line_collections,
        figsize=(10, 8)
    )
    ax.set_title("Berkenye – busz feszültség (szín) és vonalterhelés (szín)")

    # --- 4) BUSZ FELIRATOK ---
    # csak azokhoz tudunk írni, amiknek van geodata-ja; használjuk a create_bus
    # során megadott geodata-t (net.bus_geodata vagy net.bus['geodata'] / 'geo' / 'x','y').
    import json
    import pandas as pd
    # build a mapping bus_idx -> (x, y) from the geodata that was passed to create_bus
    pos_map = {}

    # 1) try net.bus_geodata (usual pandapower location)
    if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
        for bus_idx in net.bus_geodata.index:
            try:
                pos_map[bus_idx] = (net.bus_geodata.at[bus_idx, "x"], net.bus_geodata.at[bus_idx, "y"])
            except Exception:
                continue

    # 2) try a 'geodata' or 'geo' column on net.bus (possible if create_bus stored it there)
    if not pos_map and "geodata" in net.bus.columns:
        for bus_idx, row in net.bus.iterrows():
            gd = row.get("geodata")
            if pd.isna(gd):
                continue
            if isinstance(gd, (list, tuple)) and len(gd) >= 2:
                pos_map[bus_idx] = (gd[0], gd[1])
                continue
            if isinstance(gd, dict) and "x" in gd and "y" in gd:
                pos_map[bus_idx] = (gd["x"], gd["y"])
                continue
            if isinstance(gd, str):
                try:
                    parsed = json.loads(gd)
                    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                        pos_map[bus_idx] = (parsed[0], parsed[1])
                        continue
                    if isinstance(parsed, dict) and "x" in parsed and "y" in parsed:
                        pos_map[bus_idx] = (parsed["x"], parsed["y"])
                        continue
                except Exception:
                    pass

    # 3) try separate 'x' and 'y' columns on net.bus
    if not pos_map and "x" in net.bus.columns and "y" in net.bus.columns:
        for bus_idx, row in net.bus.iterrows():
            xval = row.get("x")
            yval = row.get("y")
            if pd.notna(xval) and pd.notna(yval):
                pos_map[bus_idx] = (xval, yval)

    # If still empty, we skip label placement (user requested to remove graph fallback)
    if not pos_map:
        print("No bus geodata found in net.bus_geodata or net.bus; skipping bus labels.")
    else:
        # place labels using pos_map; skip buses without coordinates
        for bus_idx in net.bus.index:
            if bus_idx not in pos_map:
                continue
            x, y = pos_map[bus_idx]
            vm_pu = net.res_bus.at[bus_idx, "vm_pu"]
            bus_name = net.bus.at[bus_idx, "name"] if "name" in net.bus.columns else ""
            if bus_name:
                label = f"{bus_idx}----------------->"
            else:
                label = f"{bus_idx} / {vm_pu:.3f} pu"
            # small offset so the text does not overlap the marker
            ax.text(
                x,
                y + 0.05,
                label,
                fontsize=6,
                ha="center",
                va="bottom",
            )

        # --- 4b) PV (sgen) and Storage markers ---
        # Plot static generators (PV) and storages at their buses if geodata available
        pv_xs, pv_ys = [], []
        if hasattr(net, "sgen") and len(net.sgen):
            for _, srow in net.sgen.iterrows():
                b = int(srow.get("bus"))
                if b in pos_map:
                    xx, yy = pos_map[b]
                    pv_xs.append(xx)
                    pv_ys.append(yy)
        if pv_xs:
            ax.scatter(pv_xs, pv_ys, c="lime", marker="^", s=120, edgecolors="k", zorder=6, label="PV (sgen)")

        st_xs, st_ys = [], []
        if hasattr(net, "storage") and len(net.storage):
            for _, srow in net.storage.iterrows():
                b = int(srow.get("bus"))
                if b in pos_map:
                    xx, yy = pos_map[b]
                    st_xs.append(xx)
                    st_ys.append(yy)
        if st_xs:
            ax.scatter(st_xs, st_ys, c="orange", marker="s", s=120, edgecolors="k", zorder=6, label="Storage")

        # LOADokra ugyanez
        # Kiplott load markereket is
        # Feliratot hogyan kell a load mellé tenni az ábrán? Csak a loadok id-jére vagyunk kíváncsiak.

        # Add a small legend for the additional markers
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", fontsize=8)

    # --- 5) SZÍNSKÁLÁK (colorbar) ---

    # busz feszültség skála
    sm_bus = mpl.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm_bus.set_array([])
    # place colorbars in their own axes to the right of the main axes so they
    # don't overlap. compute positions from the main axes bbox.
    fig = ax.get_figure()
    # let Matplotlib lay out the main axes first (so get_position is stable)
    fig.tight_layout()
    # make sure renderer has laid out the axes
    fig.canvas.draw()
    pos = ax.get_position()  # Bbox in figure coordinates

    # parameters for colorbar axes
    cb_width = 0.025
    cb_gap = -0.2  # gap between plot and first colorbar
    cb_gap_between = 0.08  # gap between two colorbars

    # first colorbar (bus voltage)
    cax_bus = fig.add_axes([pos.x1 + cb_gap, pos.y0, cb_width, pos.height])
    cbar_bus = fig.colorbar(sm_bus, cax=cax_bus)
    cbar_bus.set_label("Bus voltage [p.u.]", fontsize=9)
    cbar_bus.ax.tick_params(labelsize=9)

    # vonalterhelés skála (ha van line)
    if len(net.line):
        sm_line = mpl.cm.ScalarMappable(
            cmap=plt.cm.inferno,
            norm=mpl.colors.Normalize(vmin=0, vmax=100)
        )
        sm_line.set_array([])
        # second colorbar placed to the right of the first one
        cax_line = fig.add_axes([pos.x1 + cb_gap + cb_width + cb_gap_between, pos.y0, cb_width, pos.height])
        cbar_line = fig.colorbar(sm_line, cax=cax_line)
        cbar_line.set_label("Line loading [%]", fontsize=9)
        cbar_line.ax.tick_params(labelsize=9)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=(0, 0, 0.85, 1))  # leave space on the right for colorbars

    plt.savefig(f"loadflow_results_{net_name}.png", dpi=dpi)
    if show:
        plt.show()
    plt.close()


def create_subplot_grid(nets_dict, scenario_names, timestamp, timestep_idx, dpi=120):
    """
    Create a grid of loadflow results for all scenarios at a given timestep.
    Includes all visual elements: voltage/loading colors, bus labels, PV/storage markers, and colorbars.

    Parameters:
        nets_dict: dict mapping scenario name -> pandapower network
        scenario_names: list of scenario names (keys in nets_dict)
        timestamp: string timestamp to display (e.g., "14:00")
        timestep_idx: integer index of the timestep (for filenames)
        dpi: dots per inch for saved image (100=fast, 150=balanced, 300=high quality)

    Returns:
        path to the saved grid image, or None if failed
    """
    import os

    n_scenarios = len(scenario_names)
    n_cols = min(3, n_scenarios)  # max 3 columns
    n_rows = (n_scenarios + n_cols - 1) // n_cols  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))

    # Flatten axes for easier indexing
    if n_scenarios == 1:
        axes_flat = [axes]
    elif n_rows == 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = axes.flatten()

    # Add overall title with timestamp
    fig.suptitle(f"Network State at {timestamp}", fontsize=18, fontweight='bold', y=0.98)

    for idx, scen_name in enumerate(scenario_names):
        ax = axes_flat[idx]
        net = nets_dict[scen_name]

        # --- Prepare voltage and loading data ---
        vm = net.res_bus.vm_pu.values  # p.u.
        vmin, vmax = 0.9, 0.95
        vm_clipped = np.clip(vm, vmin, vmax)
        vm_norm = (vm_clipped - vmin) / (vmax - vmin + 1e-9)
        bus_colors = [plt.cm.viridis(val) for val in vm_norm]

        bus_collection = plot.create_bus_collection(
            net,
            buses=net.bus.index.tolist(),
            size=18,
            color=bus_colors
        )

        line_collections = []
        if len(net.line):
            loading = net.res_line.loading_percent.values  # %
            loading_clipped = np.clip(loading, 0, 100)
            loading_norm = loading_clipped / 100.0
            line_colors = [plt.cm.inferno(val) for val in loading_norm]

            lc = plot.create_line_collection(
                net,
                lines=net.line.index.tolist(),
                use_bus_geodata=True,
                color=line_colors,
                linewidths=2
            )
            line_collections.append(lc)

        # --- Draw on subplot ---
        plot.draw_collections(
            [bus_collection] + line_collections,
            ax=ax
        )
        ax.set_title(f"Scenario: {scen_name}", fontsize=13, fontweight='bold')

        # --- Add bus labels with voltage values ---
        pos_map = {}

        # Try net.bus_geodata (usual pandapower location)
        if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
            for bus_idx in net.bus_geodata.index:
                try:
                    pos_map[bus_idx] = (net.bus_geodata.at[bus_idx, "x"], net.bus_geodata.at[bus_idx, "y"])
                except Exception:
                    continue

        # Try separate 'x' and 'y' columns on net.bus
        if not pos_map and "x" in net.bus.columns and "y" in net.bus.columns:
            for bus_idx, row in net.bus.iterrows():
                xval = row.get("x")
                yval = row.get("y")
                if pd.notna(xval) and pd.notna(yval):
                    pos_map[bus_idx] = (xval, yval)

        # Place bus labels
        if pos_map:
            for bus_idx in net.bus.index:
                if bus_idx not in pos_map:
                    continue
                x, y = pos_map[bus_idx]
                vm_pu = net.res_bus.at[bus_idx, "vm_pu"]
                label = f"{bus_idx}\n{vm_pu:.3f}pu"
                ax.text(
                    x, y + 0.08,
                    label,
                    fontsize=5,
                    ha="center",
                    va="bottom",
                )

            # --- Add PV (sgen) markers ---
            pv_xs, pv_ys = [], []
            if hasattr(net, "sgen") and len(net.sgen):
                for _, srow in net.sgen.iterrows():
                    b = int(srow.get("bus"))
                    if b in pos_map:
                        xx, yy = pos_map[b]
                        pv_xs.append(xx)
                        pv_ys.append(yy)
            if pv_xs:
                ax.scatter(pv_xs, pv_ys, c="lime", marker="^", s=100, edgecolors="k", zorder=6, label="PV (sgen)", linewidths=0.5)

            # --- Add Storage markers ---
            st_xs, st_ys = [], []
            if hasattr(net, "storage") and len(net.storage):
                for _, srow in net.storage.iterrows():
                    b = int(srow.get("bus"))
                    if b in pos_map:
                        xx, yy = pos_map[b]
                        st_xs.append(xx)
                        st_ys.append(yy)
            if st_xs:
                ax.scatter(st_xs, st_ys, c="orange", marker="s", s=100, edgecolors="k", zorder=6, label="Storage", linewidths=0.5)

            # Add legend if there are markers
            handles, labels_list = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

        ax.axis('off')  # Hide axis labels

    # Hide any unused subplots
    for idx in range(n_scenarios, len(axes_flat)):
        axes_flat[idx].axis('off')

    # --- Add colorbars below the grid ---
    # Create colorbar axes at the bottom
    cbar_ax_voltage = fig.add_axes([0.15, 0.05, 0.3, 0.02])
    cbar_ax_loading = fig.add_axes([0.55, 0.05, 0.3, 0.02])

    # Voltage colorbar
    sm_bus = mpl.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm_bus.set_array([])
    cbar_bus = fig.colorbar(sm_bus, cax=cbar_ax_voltage, orientation='horizontal')
    cbar_bus.set_label("Bus voltage [p.u.]", fontsize=10)
    cbar_bus.ax.tick_params(labelsize=8)

    # Loading colorbar
    sm_line = mpl.cm.ScalarMappable(
        cmap=plt.cm.inferno,
        norm=mpl.colors.Normalize(vmin=0, vmax=100)
    )
    sm_line.set_array([])
    cbar_line = fig.colorbar(sm_line, cax=cbar_ax_loading, orientation='horizontal')
    cbar_line.set_label("Line loading [%]", fontsize=10)
    cbar_line.ax.tick_params(labelsize=8)

    # Save the grid figure
    output_path = f"grid_frames/grid_frame_{timestep_idx:04d}.png"
    os.makedirs("grid_frames", exist_ok=True)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])  # Leave space for colorbars

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path

