import numpy as np
import pandas as pd
import pandapower as pp

# --- 1. Adatok beolvasása ---
file_path = "C:\\Users\\gytom\\szakgyak\\Berkenye_modell.xlsx"
df = pd.read_excel(file_path, sheet_name="Graphic_links")

# --- 2. Előkészítés ---
df = df.drop_duplicates(subset=["FROM_ELEM", "TO_ELEM"]).reset_index(drop=True)
links = df[["FROM_ELEM", "TO_ELEM", "LAYER", "TYPE"]].copy()

# --- 3. Hálózat létrehozása ---
net = pp.create_empty_network(sn_mva=0.4)

# --- 4. Buszok létrehozása ---
elements = pd.unique(links[["FROM_ELEM", "TO_ELEM"]].values.ravel())
bus_map = {elem: pp.create_bus(net, vn_kv=0.4, name=str(elem)) for elem in elements}

# --- 5. Kapcsolatok és elemek felvétele ---
for _, row in links.iterrows():
    if np.isnan(row["FROM_ELEM"]):
        continue
    f = bus_map[row["FROM_ELEM"]]
    t = bus_map[row["TO_ELEM"]]
    layer = str(row["LAYER"]).lower()

    # Vezeték
    print("Creating ")
    if "vezeték" in layer:
        pp.create_line_from_parameters(net, from_bus=f, to_bus=t,
                                       length_km=0.1,
                                       r_ohm_per_km=0.642,
                                       x_ohm_per_km=0.083,
                                       c_nf_per_km=210,
                                       max_i_ka=0.1,
                                       name=f"Line_{row['FROM_ELEM']}_{row['TO_ELEM']}")
    # Szakaszoló biztosító
    elif "szakaszbiztosító" in layer:
        pp.create_switch(net, bus=f, element=t, et="b", name=f"Switch_{row['FROM_ELEM']}_{row['TO_ELEM']}", type="DS")

    # Transzformátor vagy forrás
    elif "transzformátor" in layer or "forrás" in layer:
        pp.create_transformer_from_parameters(net,
                                              hv_bus=f,
                                              lv_bus=t,
                                              sn_mva=0.4,
                                              vn_hv_kv=20,
                                              vn_lv_kv=0.4,
                                              vk_percent=4,
                                              vkr_percent=1,
                                              pfe_kw=1,
                                              i0_percent=0.1,
                                              name=f"Trafo_{row['FROM_ELEM']}_{row['TO_ELEM']}")

    # Fogyasztó
    elif "fogy" in layer:
        pp.create_load(net, bus=t, p_mw=0.01, q_mvar=0.005,
                       name=f"Load_{row['TO_ELEM']}")

geodata = df[["FROM_ELEM", "XCOORD", "YCOORD"]].drop_duplicates(subset=["FROM_ELEM"])

net.bus_geodata = pd.DataFrame(columns=["x", "y"])
for _, row in geodata.iterrows():
    bus_name = str(row["FROM_ELEM"])
    if bus_name in net.bus.name.values:
        bus_idx = net.bus.index[net.bus.name == bus_name][0]
        net.bus_geodata.loc[bus_idx, ["x", "y"]] = [row["XCOORD"], row["YCOORD"]]

# --- 6. Eredmény kiírása ---
print(net)
pp.to_json(net, "berkenye_network.json")
pp.plotting.simple_plot(net)

print("✅ Hálózat létrehozva és elmentve: berkenye_network.json")
