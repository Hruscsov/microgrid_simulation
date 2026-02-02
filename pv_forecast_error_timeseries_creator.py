#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("Puspokszilagy_meteo_adatsor_2023.csv", parse_dates=True, index_col="Time", header=0)

# Szimuláljuk, hogy hogy néz ki egy PV előrejelzés
# Tudjuk, hogy minden időlépésen van hiba
# Végigitrálva a df-nek a sorain, a PV_output_kW oszlop értékéhez hozzáadunk egy hibát

for i, row in df.iterrows():
    true_value = row["PV_Output_kW"]

    # Hibát generálunk: normális eloszlás, 0 középérték, 10% szórás
    for s in (0.05, 0.1, 0.2):
        error = true_value * np.random.laplace(scale=s)
        df.loc[i, f"PV_forecast_kW_{s}"] = true_value + error

df.to_csv("Puspokszilagy_meteo_adatsor_2023_with_PV_forecast.csv")
df.loc["2023-05-16"].plot(y=["PV_Output_kW", "PV_forecast_kW_0.05", "PV_forecast_kW_0.1", "PV_forecast_kW_0.2"],
                          title="PV output and forecasts on 2023-05-16")
plt.savefig("pv_forecast_error_timeseries.png")
plt.show()
