import pandas as pd
import numpy as np
import pandapower as pp
import networkx as nx
import matplotlib.pyplot as plt

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

for col in node_cols:
    if col in topo.columns:
        all_nodes.update(topo[col].dropna().unique().tolist())

# csomópont-jellegű sorok (BUSBAR-NODE) NEPID-jei
bus_like_types = {"BUSBAR-NODE"}
bus_rows = topo[topo["TYPE"].isin(bus_like_types)]
all_nodes.update(bus_rows["NEPID"].dropna().unique().tolist())

# stringesítsük
all_nodes = {str(int(n)) for n in all_nodes if pd.notna(n)}


# --- 4. Buszok létrehozása ---
bus_map = {}
for nid in sorted(all_nodes):
    nid = int(nid)
    geodata = tuple()
    if nid in gdata.index:
        geodata=tuple(gdata.loc[int(nid), ["XPOS", "YPOS"]])
    elif nid in gdata_links.index:
        geodata = tuple(gdata_links.loc[int(nid), ["XCOORD", "YCOORD"]].iloc[0])
    bus_idx = pp.create_bus(net, vn_kv=0.4, name=nid, geodata=geodata)
    bus_map[nid] = bus_idx

def get_bus(node_val):
    if pd.isna(node_val):
        return None
    key = int(node_val)
    return bus_map.get(key)


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
        if n1 is not None:
            pp.create_load(net, bus=n1, p_mw=0.01, q_mvar=0.005, name=f"LOAD_{name}")

    elif etype == "LINE-LOAD":
        if n1 is not None:
            pp.create_load(net, bus=n1, p_mw=0.005, q_mvar=0.002, name=f"LL_{name}")

    else:
        print("Nem kezelt típus")
        pass

print(net)
# --- 8. Kirajzolás ---
import pandapower.plotting as plot

ax = plot.simple_plot(net, plot_loads=True, plot_sgens=True, respect_switches=True)
ax.set_title("Berkenye hálózat – Graphic_data + spring layout")
plt.show()

# --- 9. (opcionális) mentés ---
pp.to_json(net, "berkenye_network_with_geodata.json")
print("✅ Hálózat elkészült és koordinátákkal bővült.")
