import json
from os.path import exists

import pandapower as pp
import pandas as pd
from visualization import plot_loadflow_results
import pandapower.topology as ppt
import networkx as nx

def _consolidate_bus_geodata(net):
    """Ensure `net.bus_geodata` exists and contains x,y for buses when possible.
    This function looks for geodata in several places that `create_bus` or the
    builder might have stored it:
      - net.bus_geodata (preferred)
      - net.bus['geodata'] (may be dict, list/tuple or JSON string)
      - net.bus['x'] and net.bus['y'] columns
    If enough positions are found, net.bus_geodata is created/overwritten with
    a DataFrame indexed by bus and columns x,y.
    """
    pos = {}

    # 1) existing net.bus_geodata
    if hasattr(net, "bus_geodata") and getattr(net, "bus_geodata") is not None and not net.bus_geodata.empty:
        try:
            for bus_idx in net.bus_geodata.index:
                x = net.bus_geodata.at[bus_idx, "x"]
                y = net.bus_geodata.at[bus_idx, "y"]
                if pd.notna(x) and pd.notna(y):
                    pos[bus_idx] = (float(x), float(y))
        except Exception:
            # if the structure is unexpected, ignore and continue with other sources
            pos = {}

    # 2) geodata column on net.bus
    if not pos and "geodata" in net.bus.columns:
        for bus_idx, row in net.bus.iterrows():
            gd = row.get("geodata")
            if pd.isna(gd):
                continue
            # list or tuple
            if isinstance(gd, (list, tuple)) and len(gd) >= 2:
                try:
                    pos[bus_idx] = (float(gd[0]), float(gd[1]))
                    continue
                except Exception:
                    pass
            # dict with x,y
            if isinstance(gd, dict) and "x" in gd and "y" in gd:
                try:
                    pos[bus_idx] = (float(gd["x"]), float(gd["y"]))
                    continue
                except Exception:
                    pass
            # json string
            if isinstance(gd, str):
                try:
                    parsed = json.loads(gd)
                    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                        pos[bus_idx] = (float(parsed[0]), float(parsed[1]))
                        continue
                    if isinstance(parsed, dict) and "x" in parsed and "y" in parsed:
                        pos[bus_idx] = (float(parsed["x"]), float(parsed["y"]))
                        continue
                except Exception:
                    pass

    # 3) separate x and y columns on net.bus
    if not pos and "x" in net.bus.columns and "y" in net.bus.columns:
        for bus_idx, row in net.bus.iterrows():
            xval = row.get("x")
            yval = row.get("y")
            if pd.notna(xval) and pd.notna(yval):
                try:
                    pos[bus_idx] = (float(xval), float(yval))
                except Exception:
                    pass

    # write net.bus_geodata if we found anything
    if pos:
        df = pd.DataFrame([{"bus": int(b), "x": xy[0], "y": xy[1]} for b, xy in pos.items()]).set_index("bus")
        net.bus_geodata = df
        return True
    return False


def create_load_or_pv(net, bus, p_mw, q_mvar, name):
    """
    Létrehoz egy terhelést vagy napelemet a megadott buszon.
    A név alapján dönt, hogy melyiket hozza létre.
    """
    name = str(name).upper()
    p_mw = abs(p_mw)
    q_mvar = abs(q_mvar)
    if "HMKE" in name:
        idx = pp.create_sgen(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=name)
        net.sgen.loc[idx, ["p_mw0", "p_qmvar0"]] = p_mw, q_mvar
    else:
        idx = pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=name)
        net.load.loc[idx, ["p_mw0", "p_qmvar0"]] = p_mw, q_mvar

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


def _merge_switch_connected_buses(net):
    """Merge bus pairs that are connected only via a single bus-bus switch.
    For each bus-bus switch (et=='b'), if there are no other lines/trasformers/switches
    between the two buses, reassign all elements from one bus to the other and
    remove the redundant bus and the switch.
    This function mutates `net` in-place.
    """
    # helper to detect other connections besides a particular switch
    def has_other_connection(b1, b2, sw_idx):
        # lines
        if hasattr(net, "line") and len(net.line):
            cond = ((net.line['from_bus'] == b1) & (net.line['to_bus'] == b2)) | ((net.line['from_bus'] == b2) & (net.line['to_bus'] == b1))
            if cond.any():
                return True
        # trafos
        if hasattr(net, "trafo") and len(net.trafo):
            cond = ((net.trafo['hv_bus'] == b1) & (net.trafo['lv_bus'] == b2)) | ((net.trafo['hv_bus'] == b2) & (net.trafo['lv_bus'] == b1))
            if cond.any():
                return True
        # other switches connecting same pair (exclude the current)
        if hasattr(net, "switch") and len(net.switch):
            sws = net.switch
            cond = (sws['et'] == 'b') & (((sws['bus'] == b1) & (sws['element'] == b2)) | ((sws['bus'] == b2) & (sws['element'] == b1)))
            if cond.any():
                # if more than one such switch or the only one is not the current, treat as other connection
                matches = sws[cond]
                if len(matches) > 1:
                    return True
                if sw_idx not in matches.index:
                    return True
        return False

    def reassign_bus(from_bus, to_bus):
        # update common element tables with a 'bus' column
        tables_with_bus = ['load', 'sgen', 'storage', 'ext_grid', 'gen', 'shunt', 'ward', 'xward']
        for tab in tables_with_bus:
            if hasattr(net, tab) and len(getattr(net, tab)):
                df = getattr(net, tab)
                if 'bus' in df.columns:
                    df.loc[df['bus'] == from_bus, 'bus'] = to_bus
        # update lines
        if hasattr(net, 'line') and len(net.line):
            net.line.loc[net.line['from_bus'] == from_bus, 'from_bus'] = to_bus
            net.line.loc[net.line['to_bus'] == from_bus, 'to_bus'] = to_bus
        # update trafos
        if hasattr(net, 'trafo') and len(net.trafo):
            net.trafo.loc[net.trafo['hv_bus'] == from_bus, 'hv_bus'] = to_bus
            net.trafo.loc[net.trafo['lv_bus'] == from_bus, 'lv_bus'] = to_bus
        # update switches
        if hasattr(net, 'switch') and len(net.switch):
            sw = net.switch
            # bus column
            if 'bus' in sw.columns:
                sw.loc[sw['bus'] == from_bus, 'bus'] = to_bus
            # element when et=='b'
            mask = (sw['et'] == 'b') & (sw['element'] == from_bus)
            if mask.any():
                sw.loc[mask, 'element'] = to_bus
        # update bus_geodata
        if hasattr(net, 'bus_geodata') and getattr(net, 'bus_geodata') is not None and not net.bus_geodata.empty:
            if from_bus in net.bus_geodata.index:
                if to_bus not in net.bus_geodata.index:
                    # move geodata
                    net.bus_geodata.loc[to_bus] = net.bus_geodata.loc[from_bus]
                # drop the from_bus row
                try:
                    net.bus_geodata = net.bus_geodata.drop(index=from_bus)
                except Exception:
                    pass
        # finally drop the bus from net.bus
        if from_bus in net.bus.index:
            try:
                net.bus = net.bus.drop(index=from_bus)
            except Exception:
                pass

    # loop until no more merges can be done
    merged_any = True
    while merged_any:
        merged_any = False
        if not (hasattr(net, 'switch') and len(net.switch)):
            break
        # iterate over a snapshot of switches
        for sw_idx, sw in list(net.switch.iterrows()):
            try:
                if sw.et != 'b':
                    continue
            except Exception:
                continue
            b1 = int(sw['bus'])
            b2 = int(sw['element'])
            if has_other_connection(b1, b2, sw_idx):
                continue
            # choose keep bus: prefer ext_grid bus, otherwise smaller idx
            keep = None
            if hasattr(net, 'ext_grid') and len(net.ext_grid) and (net.ext_grid['bus'] == b1).any():
                keep = b1
            elif hasattr(net, 'ext_grid') and len(net.ext_grid) and (net.ext_grid['bus'] == b2).any():
                keep = b2
            else:
                keep = min(b1, b2)
            remove = b2 if keep == b1 else b1

            # reassign everything from 'remove' to 'keep'
            reassign_bus(remove, keep)
            # remove the switch that connected them
            try:
                net.switch = net.switch.drop(index=sw_idx)
            except Exception:
                pass
            merged_any = True
            # restart scanning after a merge
            break


def build_network(read_from_file=False, filename=None):
    network_dump = filename or "network_dump/network.p"
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
    bus_geo_rows = []
    for nid in sorted(all_nodes):
        nid = int(nid)
        # geodata should be None or a (x,y) tuple of floats; avoid empty tuple
        geodata = None
        if nid in gdata.index:
            xpos, ypos = gdata.loc[int(nid), ["XPOS", "YPOS"]]
            geodata = (float(xpos), float(ypos))
        elif nid in gdata_links.index:
            xcoord, ycoord = gdata_links.loc[int(nid), ["XCOORD", "YCOORD"]].iloc[0]
            geodata = (float(xcoord), float(ycoord))

        voltage = 20 if nid in medium_voltage_buses else 0.4
        # name should be a string (pandapower hints expect str), keep original nid as label
        bus_idx = pp.create_bus(net, vn_kv=voltage, name=str(nid), geodata=geodata)
        # store geodata for later consolidation (use pandapower bus index)
        if geodata is not None:
            bus_geo_rows.append({"bus": int(bus_idx), "x": geodata[0], "y": geodata[1]})
        bus_map[nid] = bus_idx

    # if we collected geodata while creating buses, set net.bus_geodata now
    if bus_geo_rows:
        net.bus_geodata = pd.DataFrame(bus_geo_rows).set_index("bus")

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
    # try to consolidate any geodata from different sources into net.bus_geodata
    try:
        _consolidate_bus_geodata(net)
    except Exception:
        pass
    # additional conversion from 'geo' column if present
    try:
        ensure_bus_geodata_from_column(net)
    except Exception:
        pass
    # merge bus pairs that are only connected by a single switch
    try:
        _merge_switch_connected_buses(net)
    except Exception:
        pass
    # Akkumulátor
    pp.create_storage(net, 0, p_mw=0.0, max_e_mwh=0.05,
                      soc_percent=50, min_e_mwh=0.01, max_p_mw=0.03, min_p_mw=-0.03)
    return net


if __name__ == "__main__":
    print("Build network")
    net = build_network(read_from_file=False)

    # --- SZÉTVÁLASZTÁS ---
    mg = ppt.create_nxgraph(net, respect_switches=True)
    components = list(nx.connected_components(mg))
    components = sorted(components, key=len, reverse=True)

    net_A = pp.select_subnet(net, components[0])
    net_B = pp.select_subnet(net, components[1])

    # --- 8. Load flow (power flow)  ---
    # pp.diagnostic(net)

    pp.runpp(net_A)
    plot_loadflow_results(net_A, "A")
    pp.to_pickle(net_A, "network_dump/network_A.p")

    pp.runpp(net_B)
    plot_loadflow_results(net_B, "B")
    pp.to_pickle(net_B, "network_dump/network_B.p")

