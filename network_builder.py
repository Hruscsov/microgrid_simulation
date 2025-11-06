import json
from os.path import exists

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
import pandapower.plotting as plot
import pandas as pd


def ensure_bus_geodata_from_column(net):
    """
    Ha a buszoknál a net.bus['geodata'] oszlopban van JSON,
    akkor ezt átalakítjuk a szokásos net.bus_geodata DataFrame-fé.
    """
    rows = []
    if "geo" not in net.bus.columns:
        print("no geodata found")
        return  # nincs mit csinálni

    for bus_idx, row in net.bus.iterrows():
        gd = row.get("geo")
        if pd.isna(gd):
            continue
        # ha string json
        if isinstance(gd, str):
            try:
                gd = json.loads(gd)
            except json.JSONDecodeError:
                continue
        # elvárjuk, hogy legyen benne x,y
        if isinstance(gd, dict) and "x" in gd and "y" in gd:
            rows.append({"bus": bus_idx, "x": gd["x"], "y": gd["y"]})

    # TODO: kidebuggolni, hogy miért nincs  buszoknak geocoordinátája
    if rows:
        net.bus_geodata = pd.DataFrame(rows).set_index("bus")


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
    cbar_bus = plt.colorbar(sm_bus, ax=ax, fraction=0.03, pad=0.01)
    cbar_bus.set_label("Bus voltage [p.u.]", fontsize=8)

    # vonalterhelés skála (ha van line)
    if len(net.line):
        sm_line = mpl.cm.ScalarMappable(
            cmap=plt.cm.inferno,
            norm=mpl.colors.Normalize(vmin=0, vmax=100)
        )
        sm_line.set_array([])
        cbar_line = plt.colorbar(sm_line, ax=ax, fraction=0.03, pad=0.06)
        cbar_line.set_label("Line loading [%]", fontsize=8)

    plt.tight_layout()
    plt.show()


def build_network(read_from_file=False):
    network_dump = "network_dump/network.p"
    if read_from_file and exists(network_dump):
        print("Reading network from file")
        return pp.from_pickle(network_dump)

    # --- 1. Adatok beolvasása ---
    file_path = r"C:\Users\gytom\szakgyak\Berkenye_modell.xlsx"

    # Topology a hálózat logikájához
    topo = pd.read_excel(file_path, sheet_name="Topology")

    # Graphic_data a koordinátákhoz
    gdata = pd.read_excel(file_path, sheet_name="Graphic_data")
    gdata = gdata.set_index("NEPID")

    gdata_links = pd.read_excel(file_path, sheet_name="Graphic_links")
    gdata_links = gdata_links.set_index("NEPID")

    # --- 2. Üres hálózat ---
    net = pp.create_empty_network(sn_mva=0.4)

    # --- 3. Minden érintett node összegyűjtése ---
    node_cols = ["NODE1", "NODE2"]
    all_nodes = set()

    # for col in node_cols:
    #     if col in topo.columns:
    #         all_nodes.update(topo[col].dropna().unique().tolist())

    # csomópont-jellegű sorok (BUSBAR-NODE) NEPID-jei
    bus_like_types = {"BUSBAR-NODE"}
    bus_rows = topo[topo["TYPE"].isin(bus_like_types)]
    all_nodes.update(bus_rows["NEPID"].dropna().unique().tolist())

    line_like_types = {"LINE", "CIRC_BREAKER"}
    line_rows = topo[topo["TYPE"].isin(line_like_types)]
    line_start_map = {int(row["NEPID"]): int(row["NODE1"]) for lid, row in line_rows.iterrows()}

    # stringesítsük
    all_nodes = {str(int(n)) for n in all_nodes if pd.notna(n)}

    # --- 4. Buszok létrehozása ---
    bus_map = {}
    medium_voltage_buses = bus_rows.loc[bus_rows.NAME.str.contains("KÖF")].NEPID.astype(int).values
    for nid in sorted(all_nodes):
        nid = int(nid)
        geodata = tuple()
        if nid in gdata.index:
            geodata = tuple(gdata.loc[int(nid), ["XPOS", "YPOS"]])
        elif nid in gdata_links.index:
            geodata = tuple(gdata_links.loc[int(nid), ["XCOORD", "YCOORD"]].iloc[0])

        voltage = 20 if nid in medium_voltage_buses else 0.4
        bus_idx = pp.create_bus(net, vn_kv=voltage, name=nid, geodata=geodata)
        bus_map[nid] = bus_idx

    def get_bus(node_val):
        if pd.isna(node_val):
            return None
        key = int(node_val)
        if key in bus_map:
            return bus_map[key]
        elif key in line_start_map:
            return bus_map.get(line_start_map[key])
        return None

    # --- 5. Hálózati elemek létrehozása a Topology alapján ---
    for _, row in topo.iterrows():
        etype = str(row["TYPE"]).strip().upper()
        n1 = get_bus(row["NODE1"]) if "NODE1" in topo.columns else None
        n2 = get_bus(row["NODE2"]) if "NODE2" in topo.columns else None
        name = str(row.get("NAME", row.get("NEPID", "")))

        if etype == "BUSBAR-NODE":
            continue

        elif etype == "FEEDER":
            if n1 is not None:
                pp.create_ext_grid(net, bus=n1, vm_pu=1.0, name=f"FEEDER_{name}")

        elif etype == "LINE":
            if (n1 is not None) and (n2 is not None):
                pp.create_line_from_parameters(
                    net,
                    from_bus=n1,
                    to_bus=n2,
                    length_km=0.1,
                    r_ohm_per_km=0.642,
                    x_ohm_per_km=0.083,
                    c_nf_per_km=210,
                    max_i_ka=0.1,
                    name=f"LINE_{name}_{row['NODE1']}_{row['NODE2']}"
                )

        elif etype == "CIRC_BREAKER":
            if (n1 is not None) and (n2 is not None):
                pp.create_switch(
                    net,
                    bus=n1,
                    element=n2,
                    et="b",
                    closed=True,
                    type="CB",
                    name=f"SW_{name}_{row['NODE1']}_{row['NODE2']}"
                )

        elif etype == "TRANSFORMER":
            if (n1 is not None) and (n2 is not None):
                pp.create_transformer_from_parameters(
                    net,
                    hv_bus=n1,
                    lv_bus=n2,
                    sn_mva=0.4,
                    vn_hv_kv=20.0,
                    vn_lv_kv=0.4,
                    vk_percent=4,
                    vkr_percent=1,
                    pfe_kw=1,
                    i0_percent=0.1,
                    name=f"TR_{name}_{row['NODE1']}_{row['NODE2']}"
                )

        elif etype == "LOAD":
            if n1 is None:
                print(name, row["NODE1"])
            else:
                pp.create_load(net, bus=n1, p_mw=0.01, q_mvar=0.005, name=f"LOAD_{name}")

        elif etype == "LINE-LOAD":
            if n1 is None:
                print(name, row["NODE1"])
            else:
                pp.create_load(net, bus=n1, p_mw=0.005, q_mvar=0.002, name=f"LL_{name}")

        else:
            print("Nem kezelt típus")
            pass

    print(net)
    ensure_bus_geodata_from_column(net)
    pp.to_pickle(net, network_dump)
    return net


if __name__ == "__main__":
    print("Build network")
    net = build_network(read_from_file=False)

    # --- 8. Load flow (power flow)  ---
    # pp.diagnostic(net)

    print("Run loadflow")
    pp.runpp(net, algorithm='nr')

    print("Plot loadflow")
    plot_loadflow_results(net)
