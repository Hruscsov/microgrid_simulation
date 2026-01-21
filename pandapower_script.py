import pandas as pd
import pandapower as pp
from matplotlib import pyplot as plt

from network_builder import build_network

# 1. Fájlbeolvasás
# A Fogyasztasi_adatok.csv fájlból jöjjenek a profil adatok
load_profile = pd.read_csv("load_profile.csv", index_col=0, parse_dates=True)
# A PV adatok pedig a Puspokszilagy_meteo_adatsor_2023_with_PV_forecast.csv fájlból
pv_profile = pd.read_csv("pv_profile.csv", index_col=0, parse_dates=True)

# 2. Hálózatmodell
net = build_network(read_from_file=True)

# 3. Szimuláció
capacity = net.storage.loc[0, "max_e_mwh"]  # Akkumulátor kapacitás MWh
soc = net.storage.loc[0, "soc_percent"] / 100 * capacity # Kezdeti SoC
results = []
dt = 0.25
max_p = net.storage.loc[0, "max_p_mw"]

for t in range(len(load_profile)):

    # Beállítjuk az időlépés adatait
    for i in net.load.index:
        net.load.at[i, 'p_mw'] = load_profile.iloc[t, i%len(pv_profile.columns)]/1000*4

    for i in net.sgen.index:
        net.sgen.at[i, 'p_mw'] = pv_profile.loc[t, "PV_Output_kW"]/1000*4*net.sgen.at[i, 'p_mw0']

    # Megnézni, hogy hogyan alakul a net demand, ha azt a PV_forecast_kW_<> alapján számoljuk ki
    # Szükség van rá, hogy napelemnként összeadjuk a PV előrejelzést minden napelemre a t. időlépésben
    net_demand = net.load.p_mw.sum() - net.sgen.p_mw.sum()

    print(f"{t}: {net_demand:.4f} MW")
    # Egyszerű akku vezérlés
    if net_demand > 0:
        discharge = min(net_demand, soc / dt, max_p)
        soc -= discharge * dt
        net.storage.at[0, 'p_mw'] = discharge
    else:
        charge = min(-net_demand, (capacity - soc) / dt, max_p)
        soc += charge * dt
        net.storage.at[0, 'p_mw'] = -charge

    pp.runpp(net)

    results.append({
        "time": load_profile.index[t],
        "soc": soc,
        "load_total": net.load.p_mw.sum(),
        "pv_total": -net.sgen.p_mw.sum(),
        "battery_p_mw": net.storage.at[0, 'p_mw'],
        "grid_p_mw": net.res_ext_grid.p_mw.iloc[0]
    })

df = pd.DataFrame(results).set_index("time")
df.plot(subplots=True, figsize=(10, 8))
plt.tight_layout()
plt.savefig("simulation_results.png")
df.to_csv("simulation_results.csv")
