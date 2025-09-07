"""
Deribit ATM IV → 1‑day annualized IV (robust) — Flask microservice

Overview
--------
This service pulls option chains from Deribit for a set of assets (BTC, ETH, PAXG, SOL), filters
for options around-the-money and within user-configurable time windows, extracts (or infers)
ATM implied volatilities (annualized, as quoted by Deribit), and converts those to a *1‑day
annualized IV* using a robust total-variance interpolation/extrapolation procedure.

Key Concepts
------------
• Annualized IV (σ_ann): the volatility quoted by venues. For a maturity T (in DAYS here),
  the *total variance* is w(T) = σ_ann(T)^2 * T. At exactly T=1 day, σ_1d_ann = sqrt(w(1)).

• Why total variance:
  Variance aggregates (approximately) linearly in time. That makes w(T) the most stable
  dimension to interpolate/extrapolate across nearby maturities. Working in DAYS is fine
  so long as we are consistent; at T=1 day, sqrt(w(1)) is numerically the 1‑day annualized IV.

• Robust 1‑day inference:
  - If we have a “bracket” with one expiry ≤1 day and another ≥1 day, we *linearly interpolate*
    w(T) to T=1 day.
  - If both expiries are on the same side of 1 day (both <1d or both >1d), we build THREE
    candidates and take their MEDIAN (robust to outliers/shape assumptions):
      A) Flat‑σ:    use the IV of the nearest expiry (assumes local flat σ_ann).
      B) Linear σ²: linearly extrapolate r(T)=σ_ann(T)^2 to T=1 day, then σ_1d=√r(1).
      C) Power‑law in w: fit w(T)=c*T^β on the two nearest maturities, clip β∈[0.5,1.5],
         then σ_1d = sqrt(c) because w(1)=c.
  - If only one expiry is available, we use Flat‑σ.

Outputs
-------
For each asset, the cache stores:
  • price                         : current Deribit index price
  • raw_1d_vol_per_day            : daily volatility (σ_1d_ann / sqrt(365))
  • smoothed_1d_vol_per_day       : EMA + soft-roll version of the above (daily units)
  • timestamp                     : ISO time (UTC)

Smoothing
---------
We maintain a small on-disk state per asset to:
  • Soft-roll across changes in the (two) expiries used (avoid sudden jumps).
  • Apply an exponential moving average (EMA) in the 1‑day annualized IV domain.
"""

import requests
import datetime
from flask import Flask, jsonify, request
from threading import Thread
import time
import os
import json
from math import log, sqrt
from typing import Dict, List, Tuple, Optional

app = Flask(__name__)

BASE_URL = "https://www.deribit.com/api/v2/"

# Per-asset filter knobs and per-asset smoothing state file.
ASSETS: Dict[str, Dict[str, float | str]] = {
    "BTC": {
        "strike_pct_band": 0.10,   # keep strikes within ±10% of spot
        "min_life_days": 0.5,      # option must have lived at least 0.5 day since listing
        "min_expiry_days": 0.3,    # minimum time-to-expiry we accept
        "max_expiry_days": 7.0,    # maximum time-to-expiry we accept
        "state_path": "./BTC_iv_smoothing_state.json"
    },
    "ETH": {
        "strike_pct_band": 0.10,
        "min_life_days": 0.5,
        "min_expiry_days": 0.3,
        "max_expiry_days": 7.0,
        "state_path": "./ETH_iv_smoothing_state.json"
    },
    "PAXG": {
        "strike_pct_band": 0.10,
        "min_life_days": 0.5,
        "min_expiry_days": 0.3,
        "max_expiry_days": 7.0,
        "state_path": "./PAXG_iv_smoothing_state.json"
    },
    "SOL": {
        "strike_pct_band": 0.10,
        "min_life_days": 0.5,
        "min_expiry_days": 0.3,
        "max_expiry_days": 7.0,
        "state_path": "./SOL_iv_smoothing_state.json"
    }
}

CACHE: Dict[str, Dict[str, float | str]] = {}
STATE_PATH: Optional[str] = None  # not used directly; each asset has its own path

# When a user queries XAU, we return the cached PAXG data.
QUERY_ALIASES = {"XAU": "PAXG"}


def cache_key(asset: str) -> str:
    """
    Return the canonical cache key for an asset.

    We keep asset symbols uppercase throughout the cache to avoid duplicate keys like "btc"/"BTC".
    """
    return asset.upper()


def get_asset_price(asset: str = "BTC") -> float:
    """
    Fetch the Deribit index price for a given asset (USD index).

    Returns:
        float: index_price as provided by Deribit.
    """
    resp = requests.get(BASE_URL + f"public/get_index_price?index_name={asset.lower()}_usd")
    data = resp.json()
    return float(data["result"]["index_price"])


def get_price(asset: str) -> float:
    """Thin wrapper for get_asset_price()."""
    return get_asset_price(asset)


def _expiry_keyset(expiry_list: List[int]) -> str:
    """Stable string key for a set/list of expiries (ms). Used to detect pair changes."""
    return ",".join(str(int(x)) for x in sorted(expiry_list))


def _load_state(path: str = STATE_PATH) -> dict:
    """
    Load on-disk smoothing state. If file is missing/corrupt, return {}.
    State fields used:
      - last_pair               : str key of the last pair of expiries used
      - last_pair_change_ts     : float seconds (UNIX time)
      - roll_anchor             : last value to blend from when a pair changes
      - last_time               : last update ts (for EMA)
      - last_output             : last (pre-EMA) blended output
      - last_ema                : last EMA value
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict, path: str = STATE_PATH) -> None:
    """Persist smoothing state to disk. Swallow errors silently."""
    try:
        with open(path, "w") as f:
            json.dump(state, f)
    except Exception:
        pass


def _try_extract_iv(ticker: dict) -> Optional[float]:
    """
    Extract annualized IV (decimal) from a ticker payload.

    Preference order:
      1) mark_iv (if present and > 0)
      2) mid of bid_iv and ask_iv (if both present and > 0)
    Returns:
      float in *decimal* (e.g., 0.45 for 45%), or None if not available.
    """
    r = ticker.get("result", {})
    mark_iv = r.get("mark_iv")
    if mark_iv and mark_iv > 0:
        return float(mark_iv) / 100.0

    bid_iv = r.get("bid_iv") or 0.0
    ask_iv = r.get("ask_iv") or 0.0
    if bid_iv > 0 and ask_iv > 0:
        return (float(bid_iv) + float(ask_iv)) / 2.0 / 100.0

    return None


def fetch_options_with_iv(
    asset: str = "BTC",
    price: float = 0.0,
    strike_pct_band: float = 0.10,
    min_life_days: float = 0.5,
    min_expiry_days: float = 0.3,
    max_expiry_days: float = 7.0
) -> List[dict]:
    """
    Pull current (non-expired) option instruments for ALL currencies from Deribit, then:
      • Filter to the target `asset`.
      • Keep only options whose time-to-expiry is within [min_expiry_days, max_expiry_days].
      • Ensure the instrument has lived at least `min_life_days` since listing (filters “newborns”).
      • Keep strikes within ±`strike_pct_band` of the spot price.
      • For each passing instrument, fetch its ticker and attach a decimal annualized IV ('iv') and
        an 'option_type' ('call'/'put') derived from the instrument name suffix.

    Returns:
        list of dicts, each enriched with:
          - "iv": annualized IV in decimal (e.g., 0.30 for 30%)
          - "option_type": "call" or "put"
          - plus some optional metadata (_best_bid, _best_ask, _open_interest)
    """
    # List all option instruments; Deribit supports 'currency=any' here
    resp = requests.get(BASE_URL + "public/get_instruments?currency=any&kind=option&expired=false")
    resp.raise_for_status()
    instruments_data = resp.json()

    now = datetime.datetime.now(datetime.timezone.utc)
    lo, hi = (1.0 - strike_pct_band) * price, (1.0 + strike_pct_band) * price

    out: List[dict] = []

    for instr in instruments_data.get("result", []):
        if instr.get("base_currency") != asset.upper():
            continue

        expiry = datetime.datetime.fromtimestamp(instr["expiration_timestamp"] / 1000.0, datetime.timezone.utc)
        creation = datetime.datetime.fromtimestamp(instr["creation_timestamp"] / 1000.0, datetime.timezone.utc)

        # Filter by time since listing and time to expiry (in DAYS)
        t_to_exp = (expiry - now).total_seconds() / 86400.0
        t_life = (now - creation).total_seconds() / 86400.0
        if not (min_expiry_days <= t_to_exp <= max_expiry_days):
            continue
        if t_life < min_life_days:
            continue

        # Filter by strike proximity to spot
        K = float(instr["strike"]) if instr.get("strike") is not None else None
        if K is None or not (lo <= K <= hi):
            continue

        # Pull ticker to get IVs
        tkr = requests.get(BASE_URL + f"public/ticker?instrument_name={instr['instrument_name']}").json()
        iv = _try_extract_iv(tkr)
        if iv is None or iv <= 0:
            continue

        # Deribit names end with 'C' or 'P'
        opt_type = "call" if instr["instrument_name"].endswith("C") else "put"

        out.append({
            **instr,
            "iv": iv,                       # annualized (decimal)
            "option_type": opt_type,
            "_best_bid": tkr.get("result", {}).get("best_bid_price"),
            "_best_ask": tkr.get("result", {}).get("best_ask_price"),
            "_open_interest": tkr.get("result", {}).get("open_interest"),
        })

    return out


def compute_atm_iv_per_expiry(opts_for_expiry: List[dict], spot_price: float) -> Optional[float]:
    """
    Compute a single ATM (k=ln(K/spot)=0) IV per expiry via *local* interpolation in k=ln(K/spot).

    Steps:
      • Group quotes by strike K.
      • For each strike, average call/put IV if both exist; else use what's available.
      • Convert strike to moneyness k=ln(K/spot), and build points (k, iv).
      • Find the two points around k=0 and linearly interpolate in k. If only one side exists,
        use the nearest point.

    Returns:
        iv_atm (annualized, decimal) for this expiry, or None if insufficient data.
    """
    by_strike: Dict[float, Dict[str, Optional[float]]] = {}
    for o in opts_for_expiry:
        K = float(o["strike"])
        iv = float(o["iv"])  # annualized (decimal)
        d = by_strike.setdefault(K, {"call": None, "put": None})
        if o.get("option_type") == "call":
            d["call"] = iv
        else:
            d["put"] = iv

    pts: List[Tuple[float, float, float]] = []  # (k, iv, K)
    for K, cp in by_strike.items():
        vals = [v for v in (cp["call"], cp["put"]) if v is not None]
        if not vals:
            continue
        ivK = sum(vals) / len(vals)
        k = log(K / spot_price)
        pts.append((k, ivK, K))

    if not pts:
        return None

    pts.sort(key=lambda x: x[0])

    lower = None
    upper = None
    for k, ivK, K in pts:
        if k <= 0:
            lower = (k, ivK, K)
        if k >= 0 and upper is None:
            upper = (k, ivK, K)
            break

    if lower and upper and lower[2] != upper[2]:
        # Linear in k between lower(k0,iv0) and upper(k1,iv1) at k=0
        k0, iv0, _ = lower
        k1, iv1, _ = upper
        if abs(k1 - k0) < 1e-12:
            return (iv0 + iv1) / 2.0
        w = (0.0 - k0) / (k1 - k0)
        return iv0 * (1.0 - w) + iv1 * w

    # Fallback: only one side → nearest point
    nearest = min(pts, key=lambda x: abs(x[0]))
    return nearest[1]


def total_variance_interpolate_to_1d(expiry_to_iv: Dict[int, float], now_ts: float) -> Tuple[Optional[float], List[int]]:
    """
    Convert per-expiry ATM IVs to a *1‑day annualized IV* via robust total-variance methods.

    Units & idea:
      - T is measured in DAYS (T = time_to_expiry_days).
      - Total variance: w(T) = [iv_ann(T)]^2 * T_days.  At T=1 day, iv_1d_ann = sqrt(w(1)).

    Method:
      1) If we have a true bracket around 1 day (one expiry ≤1d and another ≥1d), *linearly interpolate*
         w(T) to T=1d.
      2) Otherwise (both expiries on the same side of 1 day), build THREE candidates and take their MEDIAN:
         • Flat‑σ:     iv_1d = iv of the expiry closest to 1 day.
         • Linear σ²:  define r(T) = iv(T)^2. Using the two nearest expiries, linearly extrapolate r(T) to T=1d,
                       then iv_1d = sqrt(max(r(1), 0)).
         • Power‑law w: fit w(T) = c * T^β on the two nearest expiries, clip β to [0.5, 1.5].
                        Then w(1) = c ⇒ iv_1d = sqrt(max(c, 0)).
      3) If only one expiry is available, just use Flat‑σ.

    Args:
        expiry_to_iv: mapping {expiry_ms: iv_annualized_decimal}
        now_ts      : UNIX time in seconds

    Returns:
        (iv_1d_annualized_decimal | None, used_expiries_list_ms)
    """
    # Build positive-time items: (T_days, iv_ann, exp_ms)
    items: List[Tuple[float, float, int]] = []
    for exp_ms, iv in expiry_to_iv.items():
        T_days = (exp_ms / 1000.0 - now_ts) / 86400.0
        if T_days > 0:
            items.append((float(T_days), float(iv), int(exp_ms)))

    if not items:
        return None, []

    # Sort by time to expiry
    items.sort(key=lambda x: x[0])

    # --- Single point: Flat‑σ (nearest is the only one) ---
    if len(items) == 1:
        T, iv, exp = items[0]
        return iv, [exp]

    # Helper: linear in total variance between two points to T=1d
    def _iv1_from_two_totalvar(p, q) -> Tuple[float, List[int]]:
        (Tb, ivb, expb), (Ta, iva, expa) = sorted([p, q], key=lambda x: x[0])
        wb = (ivb ** 2) * Tb
        wa = (iva ** 2) * Ta
        if abs(Ta - Tb) < 1e-12:
            closer = p if abs(p[0] - 1.0) <= abs(q[0] - 1.0) else q
            return closer[1], [closer[2]]
        t = (1.0 - Tb) / (Ta - Tb)
        w1 = wb + (wa - wb) * t
        iv1 = sqrt(max(w1, 0.0))
        return iv1, [expb, expa]

    # Try to find a bracket around 1 day
    below = None
    above = None
    for T, iv, exp in items:
        if T <= 1.0:
            below = (T, iv, exp)
        if T >= 1.0 and above is None:
            above = (T, iv, exp)

    # Case 1: true bracket (≤1 and ≥1, distinct expiries) → interpolate total variance
    if below and above and below[2] != above[2]:
        return _iv1_from_two_totalvar(below, above)

    # Case 2: same-side (both <1 or both >1). Use robust median of three candidates.
    # Pick the two expiries closest to 1 day
    by_dist = sorted(items, key=lambda x: abs(x[0] - 1.0))
    p = by_dist[0]
    q = by_dist[1]

    # Candidate A: Flat‑σ (nearest maturity IV)
    iv_flat = p[1]

    # Candidate B: Linear in σ² (variance rate)
    # r(T) = iv(T)^2, interpolate/extrapolate r at T=1
    if abs(q[0] - p[0]) < 1e-12:
        iv_lin_sig2 = iv_flat
    else:
        r_b = p[1] ** 2
        r_a = q[1] ** 2
        t = (1.0 - p[0]) / (q[0] - p[0])
        r1 = r_b + (r_a - r_b) * t
        iv_lin_sig2 = sqrt(max(r1, 0.0))

    # Candidate C: Power-law in total variance: w(T) = c * T^β (β clipped)
    wb = (p[1] ** 2) * p[0]
    wa = (q[1] ** 2) * q[0]
    if wb <= 0 or wa <= 0 or abs(q[0] - p[0]) < 1e-12:
        iv_powerlaw = iv_flat
        used = list({p[2], q[2]})
    else:
        try:
            beta_raw = log(wa / wb) / log(q[0] / p[0])
            beta = min(1.5, max(0.5, beta_raw))  # clip β to keep behavior reasonable
            c = wb / (p[0] ** beta)              # w(1) = c ⇒ σ_1d = sqrt(c)
            iv_powerlaw = sqrt(max(c, 0.0))
        except Exception:
            iv_powerlaw = iv_flat
        used = list({p[2], q[2]})

    # Robust aggregation: median of three candidates
    candidates = sorted([iv_flat, iv_lin_sig2, iv_powerlaw])
    iv1 = candidates[1]  # median
    return iv1, used


def soft_roll_and_ema(
    new_iv_1d: float,
    used_pair: List[int],
    now_dt: datetime.datetime,
    ema_half_life_sec: int = 3600,
    roll_window_sec: int = 1800,
    path: str = STATE_PATH
) -> float:
    """
    Smooth the 1‑day annualized IV stream with:
      1) Soft-roll across changes of the expiry pair used for interpolation:
         for `roll_window_sec` after a pair change, blend linearly between the previous output
         (roll_anchor) and the new value; afterwards use the new value fully.
      2) EMA in the 1‑day annualized IV domain with half-life `ema_half_life_sec`.

    Args:
        new_iv_1d         : latest raw 1‑day annualized IV (decimal)
        used_pair         : expiries (ms) used to compute new_iv_1d
        now_dt            : current UTC datetime
        ema_half_life_sec : EMA half-life in seconds
        roll_window_sec   : soft-roll window length in seconds
        path              : path to state JSON (per-asset)

    Returns:
        float: smoothed 1‑day annualized IV (decimal)
    """
    state = _load_state(path=path)

    now_ts = now_dt.timestamp()
    pair_key = _expiry_keyset(used_pair)

    last_pair = state.get("last_pair")
    last_pair_change_ts = state.get("last_pair_change_ts", now_ts)
    last_output = state.get("last_output")

    # Detect change in the pair of expiries and initialize a new roll
    if pair_key != last_pair:
        state["last_pair"] = pair_key
        state["last_pair_change_ts"] = now_ts
        state["roll_anchor"] = last_output if last_output is not None else new_iv_1d

    roll_anchor = state.get("roll_anchor", new_iv_1d)
    t0 = state.get("last_pair_change_ts", now_ts)
    dt_roll = max(0.0, now_ts - t0)

    # Soft-roll blend
    if dt_roll >= roll_window_sec:
        blended = new_iv_1d
    else:
        alpha = max(0.0, 1.0 - dt_roll / roll_window_sec)  # alpha decreases to 0 over the window
        blended = alpha * roll_anchor + (1.0 - alpha) * new_iv_1d

    # EMA step
    last_ema = state.get("last_ema")
    last_time = state.get("last_time", now_ts)
    dt = max(1.0, now_ts - last_time)  # avoid zero-division and large jumps if clocks stutter
    lam = 1.0 - pow(2.0, -dt / max(1.0, ema_half_life_sec))  # convert half-life to step coefficient
    ema = blended if last_ema is None else (lam * blended + (1.0 - lam) * last_ema)

    # Persist state
    state["last_time"] = now_ts
    state["last_output"] = blended
    state["last_ema"] = ema
    _save_state(state, path=path)

    return ema


def compute_continuous_1d_iv(options: List[dict], spot_price: float, path: str) -> Dict[str, object]:
    """
    Pipeline to compute a *continuous* 1‑day annualized IV for an asset at the current time.

    Steps:
      1) Group option quotes by expiry.
      2) For each expiry, compute an ATM IV (annualized) via local interpolation in k=ln(K/spot).
      3) Interpolate/extrapolate *total variance* to exactly T=1 day to get σ_1d_ann.
      4) Apply soft-roll + EMA smoothing in σ_1d_ann domain.

    Returns:
        {
          "raw_1d_iv"        : 1‑day annualized IV (decimal) BEFORE smoothing,
          "smoothed_1d_iv"   : 1‑day annualized IV (decimal) AFTER smoothing,
          "used_pair"        : [expiry_ms, ...] maturities used to compute the raw value,
          "per_expiry_atm_iv": {expiry_ms: iv_ann_decimal},
        }
    """
    # 1) Group by expiry (ms)
    expiry_groups: Dict[int, List[dict]] = {}
    for o in options:
        expiry_groups.setdefault(o["expiration_timestamp"], []).append(o)

    # 2) ATM IV per expiry (annualized)
    per_expiry_atm: Dict[int, float] = {}
    for exp_ms, opts in expiry_groups.items():
        iv_atm = compute_atm_iv_per_expiry(opts, spot_price)
        if iv_atm is not None and iv_atm > 0:
            per_expiry_atm[exp_ms] = iv_atm

    # 3) Interpolate/extrapolate to exactly 1 day in total variance space
    now_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
    iv1d, used_pair = total_variance_interpolate_to_1d(per_expiry_atm, now_ts)
    if iv1d is None:
        return {"raw_1d_iv": None, "smoothed_1d_iv": None, "used_pair": [], "per_expiry_atm_iv": per_expiry_atm}

    # 4) Smooth (EMA + soft-roll)
    smoothed = soft_roll_and_ema(iv1d, used_pair, datetime.datetime.now(datetime.timezone.utc), path=path)

    return {
        "raw_1d_iv": iv1d,                 # annualized implied σ at T=1 day
        "smoothed_1d_iv": smoothed,        # EMA + soft-roll in σ_1d_ann space
        "used_pair": used_pair,            # expiries used for the interpolation/extrapolation
        "per_expiry_atm_iv": per_expiry_atm,
    }


def update_cache() -> None:
    """
    Background loop:
      • For each asset, fetch spot, gather options + IVs, compute raw & smoothed 1‑day annualized IV,
        convert both to *per‑day* σ by dividing by sqrt(365), and store in CACHE with a timestamp.

    Notes:
      • We skip assets for which the calculation fails (e.g., not enough valid expiries).
      • In production you may want request timeouts/retries; kept simple here for clarity.
    """
    while True:
        for asset, settings in ASSETS.items():
            print(asset, settings)  # basic telemetry

            strike_pct_band = float(settings["strike_pct_band"])
            min_life_days   = float(settings["min_life_days"])
            min_expiry_days = float(settings["min_expiry_days"])
            max_expiry_days = float(settings["max_expiry_days"])
            state_path      = str(settings["state_path"])

            try:
                price = get_price(asset)

                options = fetch_options_with_iv(
                    asset=asset,
                    price=price,
                    strike_pct_band=strike_pct_band,
                    min_life_days=min_life_days,
                    min_expiry_days=min_expiry_days,
                    max_expiry_days=max_expiry_days
                )
                if not options:
                    continue

                res = compute_continuous_1d_iv(options, price, path=state_path)
                raw_1d_iv_annualized = res["raw_1d_iv"]
                smoothed_1d_iv_annualized = res["smoothed_1d_iv"]

                # Guard against None or zero results
                if not raw_1d_iv_annualized or not smoothed_1d_iv_annualized:
                    continue
                if raw_1d_iv_annualized * smoothed_1d_iv_annualized == 0:
                    continue

                # Convert annualized 1‑day σ to per‑day σ by dividing by sqrt(365).
                # (If you prefer 252 trading days, replace 365 with 252 consistently.)
                daily_raw = float(raw_1d_iv_annualized) / sqrt(365.0)
                daily_smooth = float(smoothed_1d_iv_annualized) / sqrt(365.0)

                key = cache_key(asset)
                CACHE[key] = {
                    "price": float(price),
                    "raw_1d_vol_per_day": round(daily_raw, 6),
                    "smoothed_1d_vol_per_day": round(daily_smooth, 6),
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().rsplit(sep="+")[0],
                }

            except Exception as e:
                print(f"Error updating {asset}: {e}")

        time.sleep(30)  # refresh cadence


@app.route("/")
def index():
    """Simple landing endpoint with usage instructions."""
    return jsonify({
        "message": "Use /volatility?asset=BTC, ETH, SOL or asset=XAU (PAXG) to get volatility data"
    })


@app.route("/volatility")
def volatility():
    """
    Query endpoint:
      - asset: one of BTC, ETH, SOL, PAXG (or XAU which aliases to PAXG)
    Returns the latest cached snapshot for the asset, or an error if not available yet.
    """
    asset = request.args.get("asset", "BTC").upper()
    asset = QUERY_ALIASES.get(asset, asset)
    if asset not in CACHE:
        return jsonify({"error": "Asset not found or not yet cached"}), 400
    return jsonify(CACHE[asset])


# Start the background updater on process start
Thread(target=update_cache, daemon=True).start()

if __name__ == "__main__":
    # Bind to all interfaces, default port 5500 (override with PORT env var)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5532)))
