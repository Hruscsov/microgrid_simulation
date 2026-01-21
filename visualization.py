import matplotlib as mpl
import numpy as np
import pandapower.plotting as plot
import networkx as nx
import matplotlib.pyplot as plt
from numpy.ma.core import left_shift


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


def plot_loadflow_results(net):
    # --- 1) buszfeszültség színezés ---
    vm = net.res_bus.vm_pu.values  # p.u.
    vmin, vmax = 0.95, 1.05
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
    # csak azokhoz tudunk írni, amiknek van geodata-ja
    if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
        for bus_idx in net.bus.index:
            if bus_idx in net.bus_geodata.index:
                x = net.bus_geodata.at[bus_idx, "x"]
                y = net.bus_geodata.at[bus_idx, "y"]
                vm_pu = net.res_bus.at[bus_idx, "vm_pu"]
                # a busz neve alapból a net.bus.name-ben van
                bus_name = net.bus.at[bus_idx, "name"]
                ax.text(
                    x,
                    y + 0.08,  # kicsit fölé
                    f"{bus_name} / {vm_pu:.3f} pu",
                    fontsize=6,
                    ha="center",
                    va="bottom",
                )

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
    plt.tight_layout(rect=(0, 0, 0.85, 1))  # leave space on the right for colorbars
    plt.savefig("loadflow_results.png", dpi=300)
    plt.show()
    plt.close()
