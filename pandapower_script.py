import pandas as pd
import pandapower as pp

from network_builder import build_network
from visualization import plot_loadflow_results, create_subplot_grid
import os
import copy

def add_extra_pvs(net, num_pvs, peak_power_mw=0.01):
    # Add extra PVs to the network based on the profile
    pv_buses = net.sgen.bus
    trafo_buses = net.trafo.hv_bus + net.trafo.lv_bus
    ext_grid_buses = net.ext_grid.bus
    candidate_buses = net.bus.index.difference(ext_grid_buses + trafo_buses + pv_buses)  # Buses that are not already connected to ext_grid or trafo
    for i in range(num_pvs):
        bus = candidate_buses[i % len(candidate_buses)]
        pp.create_sgen(net, bus=bus, p_mw=peak_power_mw, name=f"PV_{i}")
        net.sgen.at[net.sgen.index[-1], "p_mw0"] = peak_power_mw  # Store original power for scaling


# ========== OPTIMIZATION SETTINGS ==========
# MEDIUM VIDEO (36 seconds) - Balanced
FRAME_SAMPLE_RATE = 4
FRAME_DURATION = 1.5
fps_multiplier = 3

SKIP_INDIVIDUAL_GIFS = False  # Skip creating individual GIFs per scenario (only create grid GIF)
PLOT_DPI = 100  # Lower DPI = faster plotting (100 = good balance, 300 = high quality but slow)
CREATE_MP4 = True  # Create MP4 video instead of GIF (better playback control, consistent speed)
# ==========================================

# 1. Fájlbeolvasás
# A Fogyasztasi_adatok.csv fájlból jöjjenek a profil adatok
load_profile = pd.read_csv("load_profile.csv", index_col=0, parse_dates=True)
# A PV adatok pedig a Puspokszilagy_meteo_adatsor_2023_with_PV_forecast.csv fájlból
pv_profile = pd.read_csv("Puspokszilagy_meteo_adatsor_2023_with_PV_forecast.csv", index_col=0, parse_dates=True)

# 2. Hálózatmodell
# NOTE: Network_A has issues loading. Using Network_B instead.
# If you need Network_A, the issue is in the network_builder.build_network() function
base_net = build_network(read_from_file=True, filename="network_dump/net_B.p")
add_extra_pvs(base_net, num_pvs=10, peak_power_mw=0.01)  # Add 10 extra PVs with 10 kW peak power

pp.runpp(base_net)
plot_loadflow_results(base_net, "base", show=True)

# Storage index for this network (network_B uses 0, network_A uses 1)
STORAGE_INDEX = 0  # Change to 1 if using network_A

# 3. Szimuláció
dt = 0.25

# Try to get storage info, but handle errors gracefully
try:
    max_p = base_net.storage.loc[STORAGE_INDEX, "max_p_mw"]
    HAS_STORAGE = True
    print(f"✓ Storage found at index {STORAGE_INDEX}")
except Exception as e:
    print(f"⚠ Storage error: {e}")
    print(f"⚠ Disabling battery control for this network")
    max_p = 0
    HAS_STORAGE = False

error_types = ["original", "0.05", "0.1", "0.2", "no_control"]

# Storage for all-scenario results for grid plotting
all_scenario_results = {scen: [] for scen in error_types}
all_scenario_frames = {scen: [] for scen in error_types}
all_scenario_nets = {scen: copy.deepcopy(base_net) for scen in error_types}

# Initialize SOC (only if storage exists)
all_scenario_soc = {}
for scen in error_types:
    if HAS_STORAGE:
        try:
            all_scenario_soc[scen] = all_scenario_nets[scen].storage.loc[STORAGE_INDEX, "soc_percent"] / 100 * all_scenario_nets[scen].storage.loc[STORAGE_INDEX, "max_e_mwh"]
        except Exception:
            all_scenario_soc[scen] = 0
    else:
        all_scenario_soc[scen] = 0

grid_frames = []
os.makedirs("grid_frames", exist_ok=True)

# Run all scenarios in parallel per timestep
print(f"Starting simulation (frame sampling: every {FRAME_SAMPLE_RATE} steps)...")
for t, T in enumerate(load_profile.index):
    if t % max(1, len(load_profile) // 10) == 0:
        print(f"Progress: {t}/{len(load_profile)} timesteps")

    # Update all scenarios simultaneously
    for scen in error_types:
        net = all_scenario_nets[scen]
        soc = all_scenario_soc[scen]
        capacity = net.storage.loc[STORAGE_INDEX, "max_e_mwh"] if HAS_STORAGE else 0

        # Load update
        for i in net.load.index:
            net.load.at[i, 'p_mw'] = load_profile.iloc[t, i % len(load_profile.columns)] / 1000 * 4

        # PV update
        fictive_production = 0
        for i in net.sgen.index:
            net.sgen.at[i, 'p_mw'] = pv_profile.loc[T, f"PV_forecast_kW_original"] * net.sgen.at[i, 'p_mw0'] * 4 * 20
            if scen != "no_control":
                fictive_production += pv_profile.loc[T, f"PV_forecast_kW_{scen}"] * net.sgen.at[i, 'p_mw0'] * 4 * 20

        net_demand = net.load.p_mw.sum() - fictive_production
        production_error = fictive_production - net.sgen.p_mw.sum()

        # Battery control (only if storage exists)
        if HAS_STORAGE and scen != "no_control":
            try:
                capacity = net.storage.loc[STORAGE_INDEX, "max_e_mwh"]
                if net_demand > 0:
                    discharge = min(net_demand, soc / dt, max_p)
                    soc -= discharge * dt
                    net.storage.at[STORAGE_INDEX, 'p_mw'] = discharge
                else:
                    charge = min(-net_demand, (capacity - soc) / dt, max_p)
                    soc += charge * dt
                    net.storage.at[STORAGE_INDEX, 'p_mw'] = -charge
            except Exception:
                pass  # Storage operations failed, skip battery control

        all_scenario_soc[scen] = soc

        # Run loadflow
        pp.runpp(net)

        # Store results
        try:
            battery_p = net.storage.at[STORAGE_INDEX, 'p_mw'] if HAS_STORAGE else 0
        except Exception:
            battery_p = 0

        result = {
            "time": load_profile.index[t],
            "soc": soc,
            "load_total": net.load.p_mw.sum(),
            "pv_total": -net.sgen.p_mw.sum(),
            "battery_p_mw": battery_p,
            "grid_p_mw": net.res_ext_grid.p_mw.iloc[0],
            "production_error": production_error,
            "fictive_pv_production": fictive_production,
            "net_demand": net_demand
        }
        all_scenario_results[scen].append(result)

        # Save individual frames (sampled)
        if t % FRAME_SAMPLE_RATE == 0:
            frame_name = f"loadflow_results_{scen}_{t}.png"
            plot_loadflow_results(net, f"{scen}_{t}", show=False, dpi=PLOT_DPI)
            frames_dir = f"frames_{scen}"
            os.makedirs(frames_dir, exist_ok=True)
            if os.path.exists(frame_name):
                dst = os.path.join(frames_dir, frame_name)
                os.replace(frame_name, dst)
                all_scenario_frames[scen].append(dst)

    # Create grid frame (sampled)
    if t % FRAME_SAMPLE_RATE == 0:
        timestamp = load_profile.index[t].strftime("%H:00")
        nets_at_t = {scen: all_scenario_nets[scen] for scen in error_types}
        grid_path = create_subplot_grid(nets_at_t, error_types, timestamp, t, dpi=PLOT_DPI)
        if grid_path:
            grid_frames.append(grid_path)


print("\n--- Simulation complete ---")
print(f"Created {len(grid_frames)} grid frames (sampled at rate {FRAME_SAMPLE_RATE})")

# Save per-scenario CSVs and plots (fast - no plotting delay)
print("\n--- Saving per-scenario data ---")
for scen in error_types:
    df = pd.DataFrame(all_scenario_results[scen]).set_index("time")
    df.to_csv(f"simulation_results_{scen}.csv")
    print(f"Saved simulation_results_{scen}.csv")

# Create per-scenario GIFs only if requested (can skip to save time)
if not SKIP_INDIVIDUAL_GIFS and all_scenario_frames:
    print("\n--- Creating per-scenario GIFs ---")
    for scen in error_types:
        frames = all_scenario_frames[scen]
        if frames:
            try:
                import imageio.v2 as imageio
                images = [imageio.imread(p) for p in frames]
                imageio.mimsave(f"simulation_animation_{scen}.gif", images, duration=FRAME_DURATION)
                print(f"Saved animation simulation_animation_{scen}.gif with {len(frames)} frames.")
            except Exception:
                try:
                    from PIL import Image
                    imgs = [Image.open(p).convert('RGBA') for p in frames]
                    if imgs:
                        imgs[0].save(f"simulation_animation_{scen}.gif", save_all=True, append_images=imgs[1:], duration=int(FRAME_DURATION * 1000), loop=0)
                        print(f"Saved animation simulation_animation_{scen}.gif with {len(frames)} frames (Pillow fallback).")
                except Exception as e:
                    print(f"Failed to create animation for {scen}:", e)

# Create grid GIF animation
if grid_frames:
    print(f"\n--- Creating combined grid animation with {len(grid_frames)} frames ---")
    try:
        import imageio.v2 as imageio
        images = [imageio.imread(p) for p in grid_frames]
        imageio.mimsave("simulation_animation_grid.gif", images, duration=FRAME_DURATION)
        print("✓ Saved animation simulation_animation_grid.gif")
    except Exception:
        try:
            from PIL import Image
            imgs = [Image.open(p).convert('RGBA') for p in grid_frames]
            if imgs:
                imgs[0].save("simulation_animation_grid.gif", save_all=True, append_images=imgs[1:], duration=int(FRAME_DURATION * 1000), loop=0)
                print("✓ Saved animation simulation_animation_grid.gif (Pillow fallback)")
        except Exception as e:
            print("Failed to create grid animation:", e)
else:
    print("No grid frames created!")

print("\n========== SIMULATION SUMMARY ==========")
print(f"Total timesteps: {len(load_profile)}")
print(f"Sampled frames (1/{FRAME_SAMPLE_RATE}): {len(grid_frames)}")
print(f"Output GIF frames: {len(grid_frames)}")
print(f"Frame sampling rate: {FRAME_SAMPLE_RATE}")
print(f"Plot DPI: {PLOT_DPI}")
print(f"Individual GIFs: {'Yes' if not SKIP_INDIVIDUAL_GIFS else 'No'}")
print("========================================")

# Create MP4 video (better playback control than GIF)
if CREATE_MP4 and grid_frames:
    print(f"\n--- Creating MP4 video (more reliable playback) ---")
    try:
        import imageio.v2 as imageio
        # Calculate FPS: multiply by fps_multiplier for faster playback
        fps = fps_multiplier / FRAME_DURATION
        images = [imageio.imread(p) for p in grid_frames]
        imageio.mimsave("simulation_animation_grid.mp4", images, fps=fps)
        print(f"✓ Saved video simulation_animation_grid.mp4")
        print(f"  FPS: {fps:.2f} (playback speed: {fps_multiplier}x)")
        print(f"  Total duration: {len(grid_frames) * FRAME_DURATION / fps_multiplier:.1f} seconds")
    except Exception as e:
        print(f"Could not create MP4 (ffmpeg may not be installed): {e}")
        print("  Using GIF instead...")



