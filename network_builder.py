import json
from os.path import exists

import pandapower as pp
import pandas as pd

from visualization import plot_loadflow_results

def create_load_or_pv(net, bus, p_mw, q_mvar, name):
    """
    Létrehoz egy terhelést vagy napelemet a megadott buszon.
    A név alapján dönt, hogy melyiket hozza létre.
    """
    name = str(name).upper()
    p_mw = abs(p_mw)
    q_mvar = abs(q_mvar)
    if "HMKE" in name:
        pp.create_sgen(net, bus=bus, p_mw=-p_mw, q_mvar=-q_mvar, name=name)
    else:
        pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=name)

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


def build_network(read_from_file=False):
    network_dump = "network_dump/network.p"
    if read_from_file and exists(network_dump):
        print("Reading network from file")
        return pp.from_pickle(network_dump)

    # --- 1. Adatok beolvasása ---
    file_path = r"Berkenye_modell.xlsx"

    # Topology a hálózat logikájához
    topo = pd.read_excel(file_path, sheet_name="Topology")

    # Graphic_data a koordinátákhoz
    gdata = pd.read_excel(file_path, sheet_name="Graphic_data")
    gdata = gdata.set_index("NEPID")

    gdata_links = pd.read_excel(file_path, sheet_name="Graphic_links")
    gdata_links = gdata_links.set_index("NEPID")

    loads = pd.read_excel(file_path, sheet_name="Load")
    loads = loads.set_index("NEPID")

    lineloads = pd.read_excel(file_path, sheet_name="Lineload")
    lineloads = lineloads.set_index("NEPID")

    lines = pd.read_excel(file_path, sheet_name="Line")
    lines = lines.set_index("NEPID")

    trafo = pd.read_excel(file_path, sheet_name="Trafo")
    trafo = trafo.set_index("NEPID")

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

        # TODO: vezetékek paraméterezése a táblázat alapján
        elif etype == "LINE":
            if (n1 is not None) and (n2 is not None):
                params = lines.loc[int(row["NEPID"])]
                pp.create_line_from_parameters(
                    net,
                    from_bus=n1,
                    to_bus=n2,
                    length_km=params["LENGTH"],
                    r_ohm_per_km=params["R1"],
                    x_ohm_per_km=params["X1"],
                    c_nf_per_km=params["C1"]*1000,  # csak ha C1 µF/km! (ha már nF/km, akkor ne szorozz)
                    max_i_ka=params["IRMAX"]/1000,
                    type="cs" if params["CABLE"] == 1 else "ol",
                    parallel=int(params["NUMPARALLEL"]) if not pd.isna(params["NUMPARALLEL"]) else 1,
                    # opcionális, ha a pandapower verziód támogatja:
                    r0_ohm_per_km=params["R0"],
                    x0_ohm_per_km=params["X0"],
                    c0_nf_per_km=params["C0"] * 1000,  # ugyanaz az egységlogika, mint C1-nél
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
                params = trafo.loc[int(row["NEPID"])]
                pp.create_transformer_from_parameters(
                    net,
                    hv_bus=n1,
                    lv_bus=n2,
                    sn_mva=params["SR"],
                    vn_hv_kv=params["UR1"],
                    vn_lv_kv=params["UR2"],
                    vk_percent=params["UKR"],
                    vkr_percent=params["URR"],
                    pfe_kw=params["PFE"],
                    i0_percent=params["I0"] if not pd.isna(params["I0"]) else 0.0,
                    name=f"TR_{params['NAME']}_{row['NODE1']}_{row['NODE2']}"
                )

        elif etype == "LOAD":
            if n1 is None:
                print(name, row["NODE1"])
            else:
                params = loads.loc[int(row["NEPID"])]
                create_load_or_pv(net, n1, p_mw=params['P']/1000, q_mvar=params['Q']/1000, name=name)
        elif etype == "LINE-LOAD":
            if n1 is None:
                print(name, row["NODE1"])
            else:
                params = lineloads.loc[int(row["NEPID"])]
                create_load_or_pv(net, n1, p_mw=params['P']/1000, q_mvar=params['Q']/1000, name=name)

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
