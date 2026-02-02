import pandas as pd
import pandapower as pp
from matplotlib import pyplot as plt

from network_builder import build_network

#load_trafo1 = [11, 116, 111, 122, 13, 17, 77, 10, 55, 51, 37, 39, 26, 41, 49, 43, 47, 57, 59, 60, 61, 19, 79, 81, 15, 25, 83, 170, 0, 1, 175, 27, 174, 23, 45, 31, 29, 172, 173, 33, 35]
#load_trafo2 = [3, 4, 5]

# 1. Fájlbeolvasás
# A Fogyasztasi_adatok.csv fájlból jöjjenek a profil adatok
load_profile = pd.read_csv("load_profile.csv", index_col=0, parse_dates=True)
# A PV adatok pedig a Puspokszilagy_meteo_adatsor_2023_with_PV_forecast.csv fájlból
pv_profile = pd.read_csv("Puspokszilagy_meteo_adatsor_2023_with_PV_forecast.csv", index_col=0, parse_dates=True)

# 2. Hálózatmodell
net = build_network(read_from_file=True, filename="network_dump/network_B.p")

# 3. Szimuláció
dt = 0.25
max_p = net.storage.loc[0, "max_p_mw"]

error_types = ["original", "0.05", "0.1", "0.2"]

for sig in error_types:
    print("--- Simulating with PV forecast type:", sig, "---")
    results = []
    capacity = net.storage.loc[0, "max_e_mwh"]  # Akkumulátor kapacitás MWh
    soc = net.storage.loc[0, "soc_percent"] / 100 * capacity # Kezdeti SoC
    for t, T in enumerate(load_profile.index):

        # Beállítjuk az időlépés adatait
        for i in net.load.index:
            # Feltetelvizsgalat, hogy i benne van-e a load_trafo1 listában
            net.load.at[i, 'p_mw'] = load_profile.iloc[t, i%len(load_profile.columns)]/1000*4

        fictive_production = 0
        for i in net.sgen.index:
            net.sgen.at[i, 'p_mw'] = pv_profile.loc[T, f"PV_forecast_kW_original"]*net.sgen.at[i, 'p_mw0'] * 4 * 10
            fictive_production += pv_profile.loc[T, f"PV_forecast_kW_{sig}"]*net.sgen.at[i, 'p_mw0'] * 4 * 10
        # Megnézni, hogy hogyan alakul a net demand, ha azt a PV_forecast_kW_<> alapján számoljuk ki
        # Szükség van rá, hogy napelemnként összeadjuk a PV előrejelzést minden napelemre a t. időlépésben

        net_demand = net.load.p_mw.sum() - fictive_production
        production_error = fictive_production - net.sgen.p_mw.sum()

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
            "grid_p_mw": net.res_ext_grid.p_mw.iloc[0],
            "production_error": production_error,
            "fictive_pv_production": fictive_production,
            "net_demand": net.load.p_mw.sum().sum() - net.sgen.p_mw.sum().sum()
        })

    df = pd.DataFrame(results).set_index("time")
    df.plot(subplots=True, figsize=(10, 8))
    plt.tight_layout()
    plt.savefig(f"simulation_results_{sig}.png")
    df.to_csv(f"simulation_results_{sig}.csv")
