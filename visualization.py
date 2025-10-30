import pandapower.plotting as plot
import networkx as nx
import matplotlib.pyplot as plt

def visualize_network(net, k=None, iterations=50):
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
