import pandas as pd
import pandapower as pp

# 1. Fájlbeolvasás
load_profile = pd.read_csv("load_profile.csv", index_col=0, parse_dates=True)
pv_profile = pd.read_csv("pv_profile.csv", index_col=0, parse_dates=True)

# 2. Hálózatmodell
net = pp.create_empty_network()

# Buszok
bus_slack = pp.create_bus(net, vn_kv=0.4, name="Slack")
bus1 = pp.create_bus(net, vn_kv=0.4, name="Bus 1")
bus2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")
bus3 = pp.create_bus(net, vn_kv=0.4, name="Bus 3")

# Slack és vezetékek
pp.create_ext_grid(net, bus_slack, vm_pu=1.0)
pp.create_line_from_parameters(net, bus_slack, bus1, 0.1, 0.1, 0.1, 10, 1)
pp.create_line_from_parameters(net, bus1, bus2, 0.1, 0.1, 0.1, 10, 1)
pp.create_line_from_parameters(net, bus1, bus3, 0.1, 0.1, 0.1, 10, 1)

# Load-ok
for i, bus in enumerate([bus2, bus3, bus3]):
    pp.create_load(net, bus, p_mw=0.0, q_mvar=0.0, name=f"Load {i+1}")

# PV-k
for i, bus in enumerate([bus2, bus3, bus3]):
    pp.create_sgen(net, bus, p_mw=0.0, q_mvar=0.0, name=f"PV {i+1}")

# Akkumulátor
storage_idx = pp.create_storage(net, bus1, p_mw=0.0, max_e_mwh=0.05,
                                soc_percent=50, min_e_mwh=0.01, max_p_mw=0.03, min_p_mw=-0.03)

# 3. Szimuláció
soc = 0.025  # Kezdeti SoC
capacity = 0.05
results = []
dt = 0.25
max_p = 0.03

for t in range(len(load_profile)):
    # Beállítjuk az időlépés adatait
    for i in range(3):
        net.load.at[i, 'p_mw'] = load_profile.iloc[t, i]
        net.sgen.at[i, 'p_mw'] = -pv_profile.iloc[t, i]  # negatív a termelés

    net_demand = net.load.p_mw.sum() + net.sgen.p_mw.sum()

    # Egyszerű akku vezérlés
    if net_demand > 0:
        discharge = min(net_demand, soc / dt, max_p)
        soc -= discharge * dt
        net.storage.at[storage_idx, 'p_mw'] = discharge
    else:
        charge = min(-net_demand, (capacity - soc) / dt, max_p)
        soc += charge * dt
        net.storage.at[storage_idx, 'p_mw'] = -charge

    pp.runpp(net)

    results.append({
        "time": load_profile.index[t],
        "soc": soc,
        "load_total": net.load.p_mw.sum(),
        "pv_total": -net.sgen.p_mw.sum(),
        "battery_p_mw": net.storage.at[storage_idx, 'p_mw'],
        "grid_p_mw": net.res_ext_grid.p_mw.iloc[0]
    })

df = pd.DataFrame(results).set_index("time")
df.to_csv("simulation_results.csv")
