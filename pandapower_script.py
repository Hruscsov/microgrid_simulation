from os.path import exists

import pandas as pd
import numpy as np
import pandapower as pp

from matplotlib import pyplot as plt

from network_builder import build_network
from visualization import create_subplot_grid
import copy
import os
import time
from PIL import Image, ImageDraw, ImageFont

# ==========================
# KONFIGURÁCIÓ
# ==========================
DT = 0.25  # óra (0.25 = 15 perc)

# --- Futási módok ---
RUN_MODE = "continuous"  # "continuous" vagy "year_daily_window"
SCENARIO_MODE = "all"   # "all" vagy "original_only"

# continuous módban ez számít (első N időlépést futtatja)
TIMESTEPS = None

# year_daily_window módban:
WINDOW_START = "12:00"
WINDOW_HOURS = 1
RESET_SOC_EACH_DAY = True
YEAR_SAVE_EVERY_STEP_IN_WINDOW = False

# Vizualizáció / frame
FRAME_SAMPLE_RATE = 96
PLOT_DPI = 80
NETWORKS = ["A", "B"]

# Fájlok
LOAD_FILE = "measurements_clean.csv"
PV_FILE = "Puspokszilagy_meteo_adatsor_2023_with_PV_forecast.csv"

# ========= ÚJ: kapcsolók (nem nyúlunk a logikához, csak ki-be) =========
SAVE_FRAMES = False     # ha False: nincs create_subplot_grid, nincs képfájl
MAKE_GIF = False        # ha False: nincs GIF (akkor se, ha vannak frame-ek)
SAVE_PLOTS = True       # ha False: nem ment matplotlib idősor ábrát

# Progress frissítés (lépésekben)
PROGRESS_EVERY = 4      # pl. 4 = óránként (15 perces lépésnél), 96 = naponta, 500 = ritkán

# Battery paraméterek
BATTERY_SIZE_MWH = 0.2  # MWh-ban
BATTERY_POWER_MW = 0.025
# ==========================
# SEGÉDFÜGGVÉNYEK
# ==========================

def add_pvs(net, num_pvs=10, p_mw=0.005):
    pv_buses = net.sgen.bus if len(net.sgen) > 0 else []
    trafo_buses = net.trafo.hv_bus.tolist() + net.trafo.lv_bus.tolist()
    ext_grid_buses = net.ext_grid.bus.tolist()
    forbidden = set(ext_grid_buses + trafo_buses + list(pv_buses))
    candidate_buses = [b for b in net.bus.index if b not in forbidden][:num_pvs]

    for i, bus in enumerate(candidate_buses):
        idx = pp.create_sgen(net, bus=bus, p_mw=p_mw, name=f"PV_{i}")
        net.sgen.at[idx, "p_mw0"] = p_mw


def get_storage_index(net, network_type):
    if not hasattr(net, "storage") or net.storage.empty:
        return None

    target = f"BAT_{network_type}"
    if "name" in net.storage.columns:
        hits = net.storage.index[net.storage["name"].astype(str).str.upper() == target]
        if len(hits):
            return int(hits[0])

    return int(net.storage.index[0])


def stamp_image(img, text):
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bar_h = 34
    draw.rectangle([0, 0, img.width, bar_h], fill=(0, 0, 0, 180))

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    draw.text((10, 7), text, fill=(255, 255, 255, 255), font=font)
    out = Image.alpha_composite(img, overlay).convert("RGB")
    return out


def format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def progress_bar(prefix: str, i: int, n: int, start_ts: float, extra: str = "", width: int = 24):
    now = time.time()
    done = i + 1
    frac = done / n if n else 1.0
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)

    elapsed = now - start_ts
    rate = elapsed / done if done > 0 else 0
    eta = rate * (n - done) if done > 0 else 0

    line = (
        f"\r{prefix} [{bar}] {frac*100:6.2f}%  "
        f"{done}/{n}  elapsed {format_hms(elapsed)}  ETA {format_hms(eta)}"
    )
    if extra:
        line += f"  | {extra}"

    print(line, end="", flush=True)

    if done == n:
        print("", flush=True)


def make_timestamped_gif(grid_frames, network_type, out_path, duration_ms=500):
    if not grid_frames:
        print("  (nincs frame, GIF kihagyva)")
        return

    grid_frames = sorted(grid_frames, key=lambda x: x[1])
    images = []

    for path, current_time in grid_frames:
        if not os.path.exists(path):
            continue

        time_str = current_time.strftime("%Y-%m-%d %H:%M")
        img = Image.open(path)
        img = stamp_image(img, f"Network {network_type} - {time_str}")
        img.thumbnail((800, 600), Image.Resampling.LANCZOS)
        images.append(img)

    if not images:
        print("  (nem találtam képfájlokat, GIF kihagyva)")
        return

    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True
    )

    print(f"  ✅ GIF mentve: {out_path} ({len(images)} frame)")


def parse_hhmm(s: str):
    h, m = s.split(":")
    return int(h), int(m)


def build_sim_positions(load_index: pd.DatetimeIndex):
    """
    - continuous: 0..TIMESTEPS-1 (ha TIMESTEPS=None -> teljes fájl)
    - year_daily_window: minden nap WINDOW_START..(+WINDOW_HOURS) ablak (vagy csak WINDOW_START)
    """
    if RUN_MODE == "continuous":
        if TIMESTEPS is None:
            return list(range(len(load_index)))
        else:
            return list(range(min(TIMESTEPS, len(load_index))))

    if RUN_MODE == "year_daily_window":
        h0, m0 = parse_hhmm(WINDOW_START)
        step_minutes = int(DT * 60)
        steps_in_window = int((WINDOW_HOURS * 60) / step_minutes)

        days = pd.Series(load_index).dt.normalize().unique()
        pos_map = {ts: i for i, ts in enumerate(load_index)}

        pos_list = []
        for d in days:
            start = pd.Timestamp(d) + pd.Timedelta(hours=h0, minutes=m0)

            if YEAR_SAVE_EVERY_STEP_IN_WINDOW:
                for k in range(steps_in_window):
                    ts = start + pd.Timedelta(minutes=step_minutes * k)
                    if ts in pos_map:
                        pos_list.append(pos_map[ts])
            else:
                ts = start
                if ts in pos_map:
                    pos_list.append(pos_map[ts])

        return pos_list

    raise ValueError(f"Ismeretlen RUN_MODE: {RUN_MODE}")


# ==========================
# SZIMULÁCIÓ
# ==========================

def modify_storage(base_net, storage_idx, max_p_mw, max_e_mwh):
    base_net.storage.at[storage_idx, "max_p_mw"] = max_p_mw
    base_net.storage.at[storage_idx, "min_p_mw"] = -max_p_mw
    base_net.storage.at[storage_idx, "max_e_mwh"] = max_e_mwh
    base_net.storage.at[storage_idx, "min_e_mwh"] = max_e_mwh*0.05


def run_simulation(NETWORK_TYPE):
    print("=" * 60)
    print(f"SZIMULÁCIÓ - Network_{NETWORK_TYPE} | RUN_MODE={RUN_MODE}")
    print("=" * 60)

    # LOAD
    load_df = pd.read_csv(LOAD_FILE, index_col=0, parse_dates=True)
    load_data = load_df.values

    # PV
    if not exists(PV_FILE):
        raise ValueError("PV fájl nem található: " + PV_FILE)
    pv_profile_df = pd.read_csv(
        PV_FILE,
        index_col=0,
        parse_dates=True
    )
    pv_data = pv_profile_df.values
    pv_cols = {col: i for i, col in enumerate(pv_profile_df.columns)}

    # szimulációs sorpozíciók
    sim_pos = build_sim_positions(load_df.index)
    total_steps = len(sim_pos)
    print(f"Sim lépések: {total_steps} / összes sor: {len(load_df)}")

    if total_steps == 0:
        print("⚠ Nincs egyetlen kiválasztott timestep sem (ellenőrizd a dátum/indexeket).")
        return

    # Hálózat
    base_net = build_network(read_from_file=True, filename=f"network_dump/net_{NETWORK_TYPE}.p")
    add_pvs(base_net, num_pvs=5, p_mw=0.008)

    # Storage
    storage_idx = get_storage_index(base_net, NETWORK_TYPE)
    modify_storage(base_net, storage_idx, max_p_mw=BATTERY_POWER_MW, max_e_mwh=BATTERY_SIZE_MWH)
    HAS_STORAGE = storage_idx is not None

    if HAS_STORAGE:
        max_p = float(base_net.storage.loc[storage_idx, "max_p_mw"])
        max_e = float(base_net.storage.loc[storage_idx, "max_e_mwh"])
        base_net.storage.loc[storage_idx, "soc_percent"] = 50.0
        initial_soc = 0.5 * max_e
        print(f"Storage: idx={storage_idx}, max_e={max_e:.2f} MWh")
    else:
        max_p = 0.0
        max_e = 0.0
        initial_soc = 0.0
        print("Nincs storage ebben a hálóban.")

    # Szenáriók
    all_error_types = ["original", "0.05", "0.1", "0.2", "no_control"]
    if SCENARIO_MODE == "original_only":
        error_types = ["original"]
    else:
        error_types = all_error_types

    nets = {scen: copy.deepcopy(base_net) for scen in error_types}
    socs = {scen: initial_soc for scen in error_types}

    # Pre-extract load indices and values for fast access
    load_indices = list(base_net.load.index)
    load_count = len(load_indices)

    # Pre-extract sgen indices and their p_mw0 values for fast access
    sgen_indices = list(base_net.sgen.index)
    sgen_p_mw0 = [float(base_net.sgen.at[idx, "p_mw0"]) for idx in sgen_indices]

    # Mappák (csak ha tényleg mentünk képet)
    os.makedirs(f"results_{NETWORK_TYPE}", exist_ok=True)
    if SAVE_FRAMES:
        for scen in error_types:
            os.makedirs(f"frames_{NETWORK_TYPE}_{scen}", exist_ok=True)
        os.makedirs(f"grid_frames_{NETWORK_TYPE}", exist_ok=True)

    results = {scen: [] for scen in error_types}
    grid_frames = []

    start_ts = time.time()

    for step, t in enumerate(sim_pos):
        current_time = load_df.index[t]
        time_str = current_time.strftime("%Y-%m-%d %H:%M")

        # Éves metszetnél SOC reset
        if RUN_MODE == "year_daily_window" and RESET_SOC_EACH_DAY:
            if current_time.strftime("%H:%M") == WINDOW_START:
                for scen in socs:
                    socs[scen] = initial_soc

        row = load_data[t, :]

        base_pv = float(pv_data[t, pv_cols["PV_forecast_kW_original"]])

        for scen in error_types:
            net = nets[scen]
            soc = socs[scen]

            # LOAD (ITT MARAD A *4, ahogy nálad működik!)
            for i, load_idx in enumerate(load_indices):
                net.load.at[load_idx, "p_mw"] = (row[i] / 1000.0 * 4) if i < load_count else 0.0

            # PV
            fictive = 0.0
            for i, sgen_idx in enumerate(sgen_indices):
                pv_val = base_pv * sgen_p_mw0[i]
                net.sgen.at[sgen_idx, "p_mw"] = pv_val

                if scen != "no_control":
                    error_pv = float(pv_data[t, pv_cols[f"PV_forecast_kW_{scen}"]])
                    fictive += error_pv * sgen_p_mw0[i]

            # BATTERY
            battery_p = 0.0
            if HAS_STORAGE and scen != "no_control":
                net_demand = float(net.load.p_mw.sum()) - fictive

                if net_demand > 0:
                    discharge = min(net_demand, soc / DT, max_p)
                    soc -= discharge * DT
                    battery_p = discharge
                else:
                    charge = min(-net_demand, (max_e - soc) / DT, max_p)
                    soc += charge * DT
                    battery_p = -charge

                net.storage.at[storage_idx, "p_mw"] = battery_p

            socs[scen] = soc

            # Power flow (NEM BÁNTJUK!)
            try:
                pp.runpp(net, numba=True)
                grid_p = float(net.res_ext_grid.p_mw.iloc[0])
            except Exception:
                print(f"Couldn't run powerflow for {time_str}")
                grid_p = 0.0

            results[scen].append({
                "time": time_str,
                "load_mw": float(net.load.p_mw.sum()),
                "pv_mw": float(net.sgen.p_mw.sum()),
                "battery_mw": float(battery_p),
                "grid_mw": float(grid_p),
                "soc_mwh": float(soc),
            })

        # ====== FRAME MENTÉS (csak ha SAVE_FRAMES=True) ======
        if SAVE_FRAMES:
            if RUN_MODE == "continuous":
                save_frame_now = (t % FRAME_SAMPLE_RATE == 0)
            else:
                save_frame_now = True

            if save_frame_now:
                try:
                    nets_at_t = {scen: nets[scen] for scen in error_types}
                    saved_path = create_subplot_grid(
                        nets_at_t,
                        error_types,
                        f"{NETWORK_TYPE} {time_str}",
                        t,
                        dpi=PLOT_DPI
                    )

                    if saved_path and os.path.exists(saved_path):
                        safe_name = current_time.strftime("%Y-%m-%d_%H-%M")
                        new_name = f"grid_frames_{NETWORK_TYPE}/{safe_name}.png"
                        os.replace(saved_path, new_name)
                        grid_frames.append((new_name, current_time))

                except Exception as e:
                    print(f"\nGrid hiba: {e}")

        # progress (konfig alapján)
        do_print = (step % PROGRESS_EVERY == 0) or (step == total_steps - 1)
        if do_print:
            extra = f"time={time_str}"
            if SAVE_FRAMES:
                extra += f" frames={len(grid_frames)}"
            progress_bar(f"[{NETWORK_TYPE}]", step, total_steps, start_ts, extra=extra, width=28)

    # CSV + battery metrics + plotok
    battery_metrics = pd.DataFrame(columns=error_types,
                                   index=["throughput", "fec", "round_trip_efficiency", "standby_share",
                                          "grid_exchange", "self_consumption", "self_sufficiency"])

    for scen in error_types:
        df = pd.DataFrame(results[scen])
        df.to_csv(f"results_{NETWORK_TYPE}/results_{scen}_bess_{BATTERY_SIZE_MWH}.csv", index=False)

        battery_metrics.loc["throughput", scen] = df["battery_mw"].abs().sum() * DT
        battery_metrics.loc["fec", scen] = (
            (battery_metrics.loc["throughput", scen] / nets[scen].storage.at[storage_idx, "max_e_mwh"])
            if HAS_STORAGE else 0.0
        )
        battery_metrics.loc["round_trip_efficiency", scen] = (
            ((df[df["battery_mw"] < 0]["battery_mw"].abs().sum() * DT) /
             (df[df["battery_mw"] > 0]["battery_mw"].sum() * DT))
            if HAS_STORAGE and (df["battery_mw"] > 0).sum() > 0 else 0.0
        )
        battery_metrics.loc["standby_share", scen] = (
            (df[df["battery_mw"] == 0].shape[0] / df.shape[0])
            if HAS_STORAGE else 1.0
        )
        battery_metrics.loc["grid_exchange", scen] = df["grid_mw"].abs().sum() * DT

        pv_sum = df["pv_mw"].sum()
        battery_metrics.loc["self_consumption", scen] = (
            np.minimum(df["pv_mw"], df["load_mw"] + df["battery_mw"].clip(upper=0).abs()).sum() / pv_sum
            if pv_sum > 0 else 0.0
        )

        load_sum = df["load_mw"].sum()
        battery_metrics.loc["self_sufficiency", scen] = (
            np.minimum(df["pv_mw"]+ df["battery_mw"].clip(lower=0), df["load_mw"]).sum() / load_sum
            if load_sum > 0 else 0.0
        )

        if SAVE_PLOTS:
            df.plot(subplots=True, figsize=(10, 8))
            plt.tight_layout()
            plt.savefig(f"results_{NETWORK_TYPE}/simulation_results_{scen}_bess_{BATTERY_SIZE_MWH}.png")
            plt.close("all")

    battery_metrics.to_csv(f"results_{NETWORK_TYPE}/battery_metrics_bess_{BATTERY_SIZE_MWH}.csv")

    # GIF (csak ha kell és van frame)
    if SAVE_FRAMES and MAKE_GIF:
        make_timestamped_gif(
            grid_frames,
            NETWORK_TYPE,
            f"results_{NETWORK_TYPE}/animation_grid.gif"
        )

    print(f"\n✅ Network_{NETWORK_TYPE} kész!\n")


# ==========================
# FUTTATÁS
# ==========================
if __name__ == "__main__":
    for net_type in NETWORKS:
        run_simulation(net_type)

    print("=" * 60)
    print("MINDKÉT HÁLÓZAT LEFUTOTT ✔")
    print("=" * 60)
