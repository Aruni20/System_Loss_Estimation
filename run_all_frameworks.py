"""
run_all_frameworks.py
=====================
Main execution script for the PV System Loss Estimation study.

I'm estimating system-level losses for solar PV installations across India
using 6 independent frameworks — ranging from detailed physics-based module
models (SAPM, CEC) to data-driven latent representations (Encoder-Decoder).

The key contribution here is that I'm using REAL meteorological observations
(hourly GHI, DNI, DHI, temperature, humidity, wind) fetched from the
Open-Meteo Historical Weather API, which provides satellite-derived data
consistent with IMD ground-truth for Indian locations.

Previous versions of this analysis used synthetic sinusoidal weather
generators — this was adequate for validating the framework logic, but
produced unrealistic spatial uniformity in the loss estimates. Switching
to real 2023 weather data fixed that and gave physically meaningful
regional variation (e.g., higher losses in hot/coastal regions).

Outputs (one CSV per framework + a merged master file):
  - sapm_system_loss_india.csv       (L1: Sandia Array Performance Model)
  - cec_system_loss_india.csv        (L2: CEC Single-Diode Model)
  - sam_system_loss_india.csv        (L3: SAM Engineering Model)
  - pvwatts_system_loss_india.csv    (L4: PVWatts Static Baseline)
  - latent_system_loss_india.csv     (L5: Encoder-Decoder Consensus)
  - L6_spatiotemporal_physics_loss_india.csv  (L6: Physics-Guided Model)
  - results_master_imd.csv           (all 6 merged for comparison)

Author: Aruki
"""

# -- Import ordering matters on Windows --
# torch loads certain DLLs that conflict with scipy/h5py if loaded later.
# Learned this the hard way — keeping torch first avoids the WinError 1114.
import torch
import torch.nn as nn
import pvlib
import pandas as pd
import numpy as np
import requests
import time
import warnings
import os
from pvlib.pvsystem import retrieve_sam
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from scipy.stats import wilcoxon

# Suppress pvlib's RuntimeWarnings (division by zero at night) and
# pandas FutureWarnings about deprecated .fillna(method=...) syntax
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Make sure we're working from the script's own directory,
# so relative CSV paths resolve correctly regardless of where we run from
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

print("=" * 65)
print("  SYSTEM LOSS ESTIMATION — IMD REAL-DATA EDITION")
print("=" * 65)


# ============================================================
# STEP 1: Load the India Cities dataset
# ============================================================
# The CSV has ~188 cities with lat/lng/state/population.
# I'm keeping only the top 3 cities per state (by population)
# to keep the API calls manageable — fetching hourly weather
# for 72 cities already takes ~30 minutes with rate limiting.

print("\n[1/8] Loading India cities...")
cities_df = pd.read_csv("India Cities LatLng.csv")
india_df = cities_df[cities_df["country"] == "India"]

india_df = (
    india_df
    .sort_values("population", ascending=False)
    .groupby("admin_name")
    .head(3)
    .reset_index(drop=True)
)

# Build a nested dict: state -> city -> (lat, lon)
# This structure makes it easy to iterate state-by-state later
INDIA_STATES = {}
for _, row in india_df.iterrows():
    state = row["admin_name"]
    city = row["city"]
    lat = row["lat"]
    lon = row["lng"]
    if pd.isna(state):
        continue
    INDIA_STATES.setdefault(state, {})
    INDIA_STATES[state][city] = (lat, lon)

total_cities = sum(len(c) for c in INDIA_STATES.values())
print(f"   Loaded {len(INDIA_STATES)} states, {total_cities} cities")


# ============================================================
# STEP 2: Fetch REAL hourly weather from Open-Meteo
# ============================================================
# I chose Open-Meteo over other options (Solcast, PVGIS, NSRDB)
# because it's free, requires no API key, and provides the full
# set of variables I need at hourly resolution:
#   - shortwave_radiation  = GHI (W/m²)
#   - direct_normal_irradiance = DNI (W/m²)
#   - diffuse_radiation = DHI (W/m²)
#   - temperature_2m = ambient air temp (°C)
#   - relative_humidity_2m = RH (%)
#   - wind_speed_10m = wind speed at 10m (m/s)
#
# The data comes from ERA5 reanalysis + satellite models, which
# gives IMD-consistent observations for India. I'm pulling 2023
# as the reference year since it's the most recent complete year.

def fetch_imd_weather(lat, lon, year=2023):
    """
    Pulls one full year of hourly weather for a given location.
    Retries up to 3 times with exponential backoff in case of
    network hiccups (the API is occasionally slow for India).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": (
            "shortwave_radiation,"
            "direct_normal_irradiance,"
            "diffuse_radiation,"
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m"
        ),
        "timezone": "Asia/Kolkata"
    }

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)  # 1s, 2s backoff
            else:
                raise RuntimeError(
                    f"Failed to fetch weather for ({lat}, {lon}): {e}"
                )

    hourly = data["hourly"]
    index = pd.to_datetime(hourly["time"])

    # Rename the API field names to shorter, pvlib-compatible names
    df = pd.DataFrame({
        "ghi": hourly["shortwave_radiation"],
        "dni": hourly["direct_normal_irradiance"],
        "dhi": hourly["diffuse_radiation"],
        "temp_air": hourly["temperature_2m"],
        "wind_speed": hourly["wind_speed_10m"],
        "relative_humidity": hourly["relative_humidity_2m"]
    }, index=index)

    # Forward-fill any rare gaps (usually just 1-2 missing hours
    # at year boundaries), then zero-fill anything still missing
    df = df.ffill().fillna(0)
    return df


print("\n[2/8] Fetching REAL weather data from Open-Meteo API...")
print("      (Satellite-derived IMD-grade data for year 2023)\n")

weather_data = {}
count = 0

for state, cities in INDIA_STATES.items():
    weather_data[state] = {}
    for city, (lat, lon) in cities.items():
        count += 1
        print(f"  [{count:3d}/{total_cities}] {city}, {state} "
              f"({lat:.2f}N, {lon:.2f}E)...", end=" ", flush=True)
        try:
            w = fetch_imd_weather(lat, lon, year=2023)
            weather_data[state][city] = w
            print(f"OK  (GHI avg={w['ghi'].mean():.0f} W/m², "
                  f"T avg={w['temp_air'].mean():.1f}°C)")
        except Exception as e:
            print(f"FAIL: {e}")

        # Open-Meteo asks for ~0.5s between requests for free-tier
        time.sleep(0.4)

print(f"\n   Fetched weather for {count} cities\n")


# ============================================================
# FRAMEWORK L1: SAPM (Sandia Array Performance Model)
# ============================================================
# The SAPM uses empirical polynomial coefficients (fitted to lab
# measurements) to predict module power output as a function of
# irradiance and cell temperature. It's one of the most widely
# used models in pvlib — I'm using the Canadian Solar CS5P-220M
# module because it has well-validated Sandia coefficients.
#
# The "loss" is computed as:  L = 1 - (E_actual / E_ideal)
# where E_ideal is what you'd get at STC (25°C, 1000 W/m²)
# scaled by the actual plane-of-array irradiance.

print("[3/8] Framework L1: SAPM (Sandia Array Performance Model)...")

sandia_modules = retrieve_sam("SandiaMod")
module_sapm = sandia_modules["Canadian_Solar_CS5P_220M___2009_"]


def sapm_city_energy(weather, lat, lon):
    """Actual energy output: SAPM model with real irradiance + cell temp."""
    location = pvlib.location.Location(lat, lon)
    solpos = location.get_solarposition(weather.index)

    # Tilt = latitude (standard rule of thumb for annual optimization)
    # Azimuth = 180° (south-facing, appropriate for Northern hemisphere)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=abs(lat), surface_azimuth=180,
        dni=weather["dni"], ghi=weather["ghi"], dhi=weather["dhi"],
        solar_zenith=solpos["zenith"], solar_azimuth=solpos["azimuth"]
    )

    # Cell temperature using the Sandia thermal model
    # "open_rack_glass_glass" is the standard residential mounting config
    temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    Tcell = pvlib.temperature.sapm_cell(
        poa["poa_global"], weather["temp_air"],
        weather["wind_speed"], **temp_params
    )

    sapm_out = pvlib.pvsystem.sapm(poa["poa_global"], Tcell, module_sapm)

    # Clip negatives — SAPM can produce small negative values at very
    # low irradiance (below ~20 W/m²), which are physically meaningless
    return sapm_out["p_mp"].clip(lower=0).sum()


def sapm_ideal_energy(weather, lat, lon):
    """Reference energy: module at STC scaled by actual irradiance."""
    location = pvlib.location.Location(lat, lon)
    solpos = location.get_solarposition(weather.index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=abs(lat), surface_azimuth=180,
        dni=weather["dni"], ghi=weather["ghi"], dhi=weather["dhi"],
        solar_zenith=solpos["zenith"], solar_azimuth=solpos["azimuth"]
    )

    # P_STC = Impo * Vmpo (maximum power point at standard conditions)
    p_stc = module_sapm["Impo"] * module_sapm["Vmpo"]
    return (p_stc * (poa["poa_global"].clip(lower=0) / 1000.0)).sum()


# Run L1 for every city
results_sapm = []
for state, cities in INDIA_STATES.items():
    for city, (lat, lon) in cities.items():
        if city not in weather_data.get(state, {}):
            continue
        w = weather_data[state][city]
        E_act = sapm_city_energy(w, lat, lon)
        E_id = sapm_ideal_energy(w, lat, lon)
        loss = (1 - E_act / E_id) * 100 if E_id > 0 else 0
        results_sapm.append({
            "State": state, "City": city,
            "Lat": lat, "Lng": lon,
            "SAPM_System_Loss_%": round(loss, 2)
        })

df_sapm = pd.DataFrame(results_sapm)
df_sapm.to_csv("sapm_system_loss_india.csv", index=False)
print(f"   Loss range: {df_sapm['SAPM_System_Loss_%'].min():.2f}% – "
      f"{df_sapm['SAPM_System_Loss_%'].max():.2f}%\n")


# ============================================================
# FRAMEWORK L2: CEC (California Energy Commission) Model
# ============================================================
# The CEC model uses the single-diode equivalent circuit, which
# is more physically grounded than SAPM — it solves for the I-V
# curve parameters (IL, I0, Rs, Rsh, nNsVth) at each timestep
# given the actual irradiance and cell temperature.
#
# I picked the Aavid Solar ASMS-235M because it's a representative
# Indian-market module with validated CEC parameters in the SAM db.

print("[4/8] Framework L2: CEC (California Energy Commission)...")

cec_modules = retrieve_sam("CECMod")
module_cec = cec_modules["Aavid_Solar_ASMS_235M"]


def cec_city_energy(weather, lat, lon):
    """Single-diode model: solves the full I-V curve at each hour."""
    location = pvlib.location.Location(lat, lon)
    solpos = location.get_solarposition(weather.index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=abs(lat), surface_azimuth=180,
        dni=weather["dni"], ghi=weather["ghi"], dhi=weather["dhi"],
        solar_zenith=solpos["zenith"], solar_azimuth=solpos["azimuth"]
    )

    temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    Tcell = pvlib.temperature.sapm_cell(
        poa["poa_global"], weather["temp_air"],
        weather["wind_speed"], **temp_params
    )

    # Calculate the five single-diode parameters
    IL, I0, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_cec(
        effective_irradiance=poa["poa_global"], temp_cell=Tcell,
        alpha_sc=module_cec["alpha_sc"], a_ref=module_cec["a_ref"],
        I_L_ref=module_cec["I_L_ref"], I_o_ref=module_cec["I_o_ref"],
        R_sh_ref=module_cec["R_sh_ref"], R_s=module_cec["R_s"],
        Adjust=module_cec["Adjust"]
    )

    # Solve for maximum power point
    sd = pvlib.pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth)
    return sd["p_mp"].clip(lower=0).sum()


def cec_ideal_energy(weather, lat, lon):
    """STC-scaled reference, same approach as SAPM ideal."""
    location = pvlib.location.Location(lat, lon)
    solpos = location.get_solarposition(weather.index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=abs(lat), surface_azimuth=180,
        dni=weather["dni"], ghi=weather["ghi"], dhi=weather["dhi"],
        solar_zenith=solpos["zenith"], solar_azimuth=solpos["azimuth"]
    )
    p_stc = module_cec["V_mp_ref"] * module_cec["I_mp_ref"]
    return (p_stc * (poa["poa_global"].clip(lower=0) / 1000.0)).sum()


# Run L2 for every city
results_cec = []
for state, cities in INDIA_STATES.items():
    for city, (lat, lon) in cities.items():
        if city not in weather_data.get(state, {}):
            continue
        w = weather_data[state][city]
        E_act = cec_city_energy(w, lat, lon)
        E_id = cec_ideal_energy(w, lat, lon)
        loss = (1 - E_act / E_id) * 100 if E_id > 0 else 0
        results_cec.append({
            "State": state, "City": city,
            "CEC_System_Loss_%": round(loss, 2)
        })

df_cec = pd.DataFrame(results_cec)
df_cec.to_csv("cec_system_loss_india.csv", index=False)
print(f"   Loss range: {df_cec['CEC_System_Loss_%'].min():.2f}% – "
      f"{df_cec['CEC_System_Loss_%'].max():.2f}%\n")


# ============================================================
# FRAMEWORK L3: SAM-style Engineering Model
# ============================================================
# Unlike L1/L2 which simulate the module physics directly, this
# framework uses a component-wise approach: each loss mechanism
# (temperature, soiling, humidity, wiring, mismatch, etc.) is
# estimated separately and then summed.
#
# The key improvement over the original is that temperature, soiling,
# and humidity losses are now derived from REAL weather data rather
# than latitude-based heuristics. Specifically:
#   - Temperature loss: uses actual mean ambient temp + NOCT offset
#   - Soiling loss: uses daily GHI patterns as a proxy for dry/wet periods
#   - Humidity loss: uses actual RH data to estimate corrosion risk
#
# The fixed losses (mismatch, wiring, degradation, availability) are
# kept at standard PVWatts values since they're site-independent.

print("[5/8] Framework L3: SAM-style Engineering Model (IMD-driven)...")

# These are standard PVWatts defaults (site-independent)
LOSS_MISMATCH = 2.0       # Module mismatch in array
LOSS_WIRING = 2.0         # DC + AC wiring losses
LOSS_DEGRADATION = 0.5    # First-year degradation (LID)
LOSS_AVAILABILITY = 3.0   # Grid/inverter downtime


def compute_imd_sam_loss(weather, lat):
    """
    Compute SAM-style losses using actual weather measurements.
    Returns total loss (%) and individual components.
    """
    # -- Temperature derating --
    # NOCT approximation: cell temp ≈ ambient + 25°C under typical conditions
    # Then apply standard temp coefficient of ~0.4%/°C above STC (25°C)
    T_cell_avg = weather["temp_air"].mean() + 25
    L_temp = min(max(0, 0.4 * (T_cell_avg - 25)), 10)

    # -- Soiling loss (dry-period proxy) --
    # I'm using a simple but effective proxy: days where total GHI exceeds
    # 5000 Wh/m² are likely dry/sunny days where dust accumulates.
    # Days with lower GHI suggest cloud/rain → natural cleaning.
    # The fraction of "dry days" maps linearly to soiling loss (2-6%).
    daily_ghi = weather["ghi"].resample("D").sum()
    dry_fraction = (daily_ghi > 5000).mean()
    L_soil = 2.0 + 4.0 * dry_fraction

    # -- Humidity / corrosion loss --
    # Fraction of hours with RH > 70% gives corrosion risk exposure.
    # Coastal India (Tamil Nadu, Kerala) sees 60-80% of hours above this
    # threshold, while Rajasthan sits around 20-30%.
    high_rh = (weather["relative_humidity"] > 70).mean()
    L_hum = 0.5 + 2.0 * high_rh

    total = (L_temp + L_soil + L_hum +
             LOSS_MISMATCH + LOSS_WIRING +
             LOSS_DEGRADATION + LOSS_AVAILABILITY)

    # Clamp to realistic bounds: 12% is roughly the theoretical minimum
    # (you always have wiring + mismatch), 25% is PVWatts upper bound
    return np.clip(total, 12, 25), L_temp, L_soil, L_hum


# Run L3 for every city
results_sam = []
for state, cities in INDIA_STATES.items():
    for city, (lat, lon) in cities.items():
        if city not in weather_data.get(state, {}):
            continue
        w = weather_data[state][city]
        total, lt, ls, lh = compute_imd_sam_loss(w, lat)
        results_sam.append({
            "State": state, "City": city,
            "SAM_System_Loss_%": round(total, 2),
            "Temp_Loss_%": round(lt, 2),
            "Soil_Loss_%": round(ls, 2),
            "Humid_Loss_%": round(lh, 2)
        })

df_sam = pd.DataFrame(results_sam)
df_sam.to_csv("sam_system_loss_india.csv", index=False)
print(f"   Loss range: {df_sam['SAM_System_Loss_%'].min():.2f}% – "
      f"{df_sam['SAM_System_Loss_%'].max():.2f}%\n")


# ============================================================
# FRAMEWORK L4: PVWatts Static Baseline
# ============================================================
# This is intentionally the simplest framework — it assigns the
# standard NREL PVWatts default of 14.1% to every location.
# The whole point is to serve as a comparison baseline: if our
# spatially-varying models don't differ meaningfully from a flat
# 14.1%, then they add no information.

print("[6/8] Framework L4: PVWatts Static Loss (14.1% baseline)...")

PVWATTS_TOTAL_LOSS = 14.1

df_pvwatts = india_df[["admin_name", "city"]].copy()
df_pvwatts = df_pvwatts.rename(columns={"admin_name": "State", "city": "City"})
df_pvwatts["PVWatts_System_Loss_%"] = PVWATTS_TOTAL_LOSS
df_pvwatts.to_csv("pvwatts_system_loss_india.csv", index=False)
print(f"   Static loss: {PVWATTS_TOTAL_LOSS}% (all cities)\n")


# ============================================================
# FRAMEWORK L5: Latent Loss via Encoder-Decoder
# ============================================================
# This is where it gets interesting. I have 4 independent loss
# estimates per city (L1-L4), and I want to find a single "consensus"
# loss that best summarizes all four. The idea is:
#
#   Encoder: maps the 4 framework losses → 1 latent value
#   Decoder: maps the latent back → reconstructed 4 losses
#
# By training to minimize reconstruction error (unsupervised),
# the encoder learns to extract the underlying common signal
# across all frameworks — essentially a nonlinear weighted average
# that adapts to the data.
#
# Important: I normalize the inputs using min-max scaling (not /100)
# because the frameworks have very different ranges (CEC: 3-11%,
# SAM: 17-25%). Dividing by 100 would make them all look tiny and
# similar, killing the gradient signal. Then I rescale the output
# to match the range of per-city framework averages.

print("[7/8] Framework L5: Latent Loss (Encoder-Decoder Alignment)...")

# Load the computed per-framework results
sapm_l = pd.read_csv("sapm_system_loss_india.csv")
cec_l = pd.read_csv("cec_system_loss_india.csv")
sam_l = pd.read_csv("sam_system_loss_india.csv")
pvw_l = pd.read_csv("pvwatts_system_loss_india.csv")

sapm_l = sapm_l.rename(columns={"SAPM_System_Loss_%": "SAPM"})
cec_l = cec_l.rename(columns={"CEC_System_Loss_%": "CEC"})
sam_l = sam_l.rename(columns={"SAM_System_Loss_%": "SAM"})
pvw_l = pvw_l.rename(columns={"PVWatts_System_Loss_%": "PVWatts"})

# Merge all four into one table (inner join ensures only cities
# that succeeded in all frameworks are included)
df_merged = (
    sapm_l[["State", "City", "Lat", "Lng", "SAPM"]]
    .merge(cec_l[["State", "City", "CEC"]], on=["State", "City"])
    .merge(sam_l[["State", "City", "SAM"]], on=["State", "City"])
    .merge(pvw_l[["State", "City", "PVWatts"]], on=["State", "City"])
)

loss_cols = ["SAPM", "CEC", "SAM", "PVWatts"]
X_raw = df_merged[loss_cols].values  # shape: (n_cities, 4), in %

# Per-feature min-max normalization to [0, 1]
X_min = X_raw.min(axis=0, keepdims=True)
X_max = X_raw.max(axis=0, keepdims=True)
X_norm = (X_raw - X_min) / (X_max - X_min + 1e-8)
X = torch.tensor(X_norm, dtype=torch.float32)

# I'll rescale the encoder output (which is Sigmoid → [0,1]) to match
# the range of per-city mean losses across all 4 frameworks.
# This gives the latent variable a physically interpretable scale.
latent_target_min = X_raw.mean(axis=1).min()
latent_target_max = X_raw.mean(axis=1).max()


class Encoder(nn.Module):
    """Compresses 4 framework losses into a single latent scalar."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()  # output bounded to [0, 1]
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Reconstructs 4 framework losses from the latent scalar."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, z):
        return self.net(z)


encoder = Encoder()
decoder = Decoder()
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
)
loss_fn = nn.MSELoss()

# 5000 epochs is enough for convergence on ~70 samples with 4 features.
# I tried 3000 initially but the reconstruction error was still noisy.
EPOCHS = 5000
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    z = encoder(X)
    X_hat = decoder(z)
    loss = loss_fn(X_hat, X)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"   Epoch {epoch}, Loss = {loss.item():.6f}")

print(f"   Final Loss = {loss.item():.6f}")

# Extract the learned latent and rescale to physical units (%)
with torch.no_grad():
    latent_raw = encoder(X).numpy().squeeze()  # in [0, 1]

latent = latent_target_min + latent_raw * (latent_target_max - latent_target_min)

df_merged["Latent_System_Loss_%"] = np.round(latent, 2)
df_merged.to_csv("latent_system_loss_india.csv", index=False)
print(f"   Latent loss range: {latent.min():.2f}% – {latent.max():.2f}%\n")


# ============================================================
# FRAMEWORK L6: Spatio-Temporal Physics-Guided Model
# ============================================================
# This is my most physics-grounded framework. Instead of using
# black-box module models (L1/L2) or simple heuristics (L3),
# I model three distinct loss mechanisms with actual dynamics:
#
# 1. SOILING: Dynamic dust accumulation on dry days, with rain
#    events acting as a cleaning mechanism. The accumulation rate
#    scales with GHI (proxy for dry sunny conditions), and I use
#    a rain proxy (days with GHI < 2000 Wh/m²) for cleaning.
#
# 2. TEMPERATURE DERATING: Cell temperature is estimated from
#    ambient temp + irradiance-driven heating (NOCT formula),
#    then a 0.4%/°C penalty is applied above 25°C. I only count
#    daytime hours (GHI > 50 W/m²) since nighttime temps don't
#    affect generation.
#
# 3. HUMIDITY/CORROSION: Prolonged high humidity (RH > 70%)
#    degrades encapsulant and contacts over time. I model this
#    as an exponential saturation with a rate constant that's
#    doubled for coastal states (salt spray amplification).
#
# The calibration values (deposition rate, cleaning efficiency,
# beta for humidity) were tuned against SCADA data from DAIICT
# to produce ~14-15% total loss for Gujarat (inland), which is
# consistent with published field measurements.

print("[8/8] Framework L6: Physics-Guided Loss (IMD weather)...")

# Fixed system losses: wiring, inverter, mismatch, availability
# I'm using 10% here, which is standard for utility-scale plants
L_SYS = 0.10

COASTAL_STATES = {
    "Gujarāt", "Mahārāshtra", "Tamil Nādu", "Kerala",
    "Andhra Pradesh", "Odisha", "West Bengal", "Goa", "Karnātaka"
}


def physics_loss_imd(weather, lat, coastal):
    """
    Compute environment-driven PV losses from real weather data.
    Returns total environmental loss + individual components.
    """
    # Aggregate hourly data to daily for soiling dynamics
    daily = weather.resample("D").agg({
        "ghi": "sum", "temp_air": "mean",
        "relative_humidity": "mean", "wind_speed": "mean"
    })
    T = len(daily)
    ghi_d = daily["ghi"].values
    rh_d = daily["relative_humidity"].values

    # -- Dynamic soiling model --
    # Rain proxy: overcast/rainy days have daily GHI < 2000 Wh/m²
    rain_proxy = ghi_d < 2000
    D = np.zeros(T)  # dust accumulation index
    for t in range(1, T):
        # Deposition: base rate 0.005/day, slightly higher on sunny days
        dep = 0.005 * (1 + 0.0002 * ghi_d[t])
        # Cleaning: rain washes off ~40% of accumulated dust
        clean = 0.4 if rain_proxy[t] else 0.0
        D[t] = max(0, min(D[t - 1] + dep - clean, 1.0))

    # Soiling loss follows Beer-Lambert-type attenuation
    L_soil = np.mean(1 - np.exp(-0.3 * D))

    # -- Temperature derating --
    # NOCT formula: T_cell = T_amb + (NOCT - 20) / 800 * G
    ghi_h = weather["ghi"].values
    temp_h = weather["temp_air"].values
    T_cell = temp_h + (45 - 20) / 800 * ghi_h

    # Only daytime hours contribute (nighttime is irrelevant)
    daytime = ghi_h > 50
    if daytime.sum() > 0:
        L_temp = np.mean(0.004 * np.maximum(T_cell[daytime] - 25, 0))
    else:
        L_temp = 0.0

    # -- Humidity-driven degradation --
    # Fraction of days with mean RH > 70%
    f_humid = np.mean(rh_d > 70)
    # Coastal locations get double the degradation rate (salt spray)
    beta = 0.008 * (2 if coastal else 1)
    L_humid = 1 - np.exp(-beta * f_humid)

    L_env = L_soil + L_temp + L_humid
    return L_env, L_soil, L_temp, L_humid


# Run L6 for every city
results_L6 = []
for state, cities in INDIA_STATES.items():
    for city, (lat, lon) in cities.items():
        if city not in weather_data.get(state, {}):
            continue
        w = weather_data[state][city]
        coastal = state in COASTAL_STATES
        L_env, ls, lt, lh = physics_loss_imd(w, lat, coastal)
        L_total = L_SYS + L_env
        results_L6.append({
            "State": state, "City": city,
            "Physics_System_Loss_%": round(L_total * 100, 2),
            "Soiling_Loss_%": round(ls * 100, 2),
            "Temp_Derate_%": round(lt * 100, 2),
            "Humidity_Loss_%": round(lh * 100, 2)
        })

df_L6 = pd.DataFrame(results_L6)
df_L6.to_csv("L6_spatiotemporal_physics_loss_india.csv", index=False)
print(f"   Loss range: {df_L6['Physics_System_Loss_%'].min():.2f}% – "
      f"{df_L6['Physics_System_Loss_%'].max():.2f}%\n")


# ============================================================
# HYPOTHESIS TEST: L6 (Physics) vs L5 (Latent)
# ============================================================
# Wilcoxon signed-rank test checks whether the paired differences
# between L6 and L5 are symmetrically distributed around zero.
# It's non-parametric, so it doesn't assume normality — important
# because loss distributions across cities are usually skewed.

print("=" * 65)
print("  HYPOTHESIS TEST: L6 (Physics) vs L5 (Latent)")
print("=" * 65)

df_L5t = pd.read_csv("latent_system_loss_india.csv")
df_L6t = pd.read_csv("L6_spatiotemporal_physics_loss_india.csv")
df_L5t = df_L5t.rename(columns={"Latent_System_Loss_%": "L5"})
df_L6t = df_L6t.rename(columns={"Physics_System_Loss_%": "L6"})

df_test = (
    df_L5t[["State", "City", "L5"]]
    .merge(df_L6t[["State", "City", "L6"]], on=["State", "City"])
)

stat, pval = wilcoxon(df_test["L6"], df_test["L5"])
mean_d = np.mean(df_test["L6"] - df_test["L5"])
med_d = np.median(df_test["L6"] - df_test["L5"])

print(f"  Paired samples: {len(df_test)}")
print(f"  Wilcoxon stat:  {stat}")
print(f"  p-value:        {pval:.2e}")
print(f"  Mean diff:      {mean_d:.3f}%")
print(f"  Median diff:    {med_d:.3f}%")

if pval < 0.05:
    print("  Result: REJECT H0 — L6 is statistically different from L5")
else:
    print("  Result: FAIL to reject H0 — No significant difference")


# ============================================================
# MASTER RESULTS TABLE
# ============================================================
# Merge all 6 frameworks into a single CSV for easy comparison.
# This is the file I use for generating figures and tables in
# the manuscript.

print("\n" + "=" * 65)
print("  MASTER RESULTS TABLE")
print("=" * 65)

master = (
    df_sapm[["State", "City", "Lat", "Lng", "SAPM_System_Loss_%"]]
    .merge(df_cec[["State", "City", "CEC_System_Loss_%"]], on=["State", "City"])
    .merge(df_sam[["State", "City", "SAM_System_Loss_%"]], on=["State", "City"])
    .merge(df_pvwatts[["State", "City", "PVWatts_System_Loss_%"]], on=["State", "City"])
)

lat_df = pd.read_csv("latent_system_loss_india.csv")
phy_df = pd.read_csv("L6_spatiotemporal_physics_loss_india.csv")

master = master.merge(
    lat_df[["State", "City", "Latent_System_Loss_%"]],
    on=["State", "City"], how="left"
)
master = master.merge(
    phy_df[["State", "City", "Physics_System_Loss_%"]],
    on=["State", "City"], how="left"
)

master.to_csv("results_master_imd.csv", index=False)

all_loss_cols = [
    "SAPM_System_Loss_%", "CEC_System_Loss_%",
    "SAM_System_Loss_%", "PVWatts_System_Loss_%",
    "Latent_System_Loss_%", "Physics_System_Loss_%"
]

summary = master[all_loss_cols].describe().T[
    ["mean", "std", "min", "25%", "50%", "75%", "max"]
].round(2)

print(f"\n  Cities: {len(master)} | States: {master['State'].nunique()}\n")
print(summary.to_string())

print(f"\n  All results saved to: results_master_imd.csv")
print("\n" + "=" * 65)
print("  DONE")
print("=" * 65)
