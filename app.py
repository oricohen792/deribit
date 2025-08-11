import requests
import datetime
import numpy as np
from flask import Flask, jsonify, request
from threading import Thread
import time
import os
import json
from math import log, sqrt

app = Flask(__name__)

BASE_URL = "https://www.deribit.com/api/v2/"
ASSETS = {
    "BTC": {
        "strike_pct_band": 0.10,
        "min_life_days": 0.5,
        "min_expiry_days": 0.3,
        "max_expiry_days": 7.0,
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
CACHE = {}
STATE_PATH = None

# map query aliases to cache keys
# When a user queries XAU, return the cached PAXG data
QUERY_ALIASES = {"XAU": "PAXG"}


def cache_key(asset: str) -> str:
    """Return the key used for caching a given asset."""
    # Keep the original asset name in the cache map
    return asset.upper()


def get_asset_price(asset="BTC"):
    response = requests.get(BASE_URL + f"public/get_index_price?index_name={asset.lower()}_usd")
    data = response.json()
    return data['result']['index_price']


def get_price(asset):
    return get_asset_price(asset)


def _expiry_keyset(expiry_list) -> str:
    return ",".join(str(int(x)) for x in sorted(expiry_list))


def _load_state(path: str = STATE_PATH) -> dict:
    try:
        print(path)
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict, path: str = STATE_PATH):
    try:
        print(path)
        with open(path, "w") as f:
            json.dump(state, f)
    except Exception:
        pass


def _try_extract_iv(ticker: dict) -> float | None:
    """Prefer mark_iv; if missing/zero, try mid of bid/ask IVs. Returns decimal IV or None."""
    r = ticker.get("result", {})
    mark_iv = r.get("mark_iv")
    if mark_iv and mark_iv > 0:
        return mark_iv / 100.0
    bid_iv = r.get("bid_iv") or 0.0
    ask_iv = r.get("ask_iv") or 0.0
    if bid_iv > 0 and ask_iv > 0:
        return ((bid_iv + ask_iv) / 2.0) / 100.0
    return None


def fetch_options_with_iv(asset: str = "BTC",
                          price: float= 0,
                          strike_pct_band: float = 0.10,
                          min_life_days: float = 0.5,
                          min_expiry_days: float = 0.3,
                          max_expiry_days: float = 7.0) -> list[dict]:
    """
    Pull non-expired options for any currency, filter to target asset, time window, and strike band.
    For each instrument attach an 'iv' field (decimal). Returns a flat list of option dicts.
    """
    # NOTE: 'currency=any' is supported by Deribit to list all option instruments
    resp = requests.get(BASE_URL + "public/get_instruments?currency=any&kind=option&expired=false")
    resp.raise_for_status()
    instruments_data = resp.json()

    now = datetime.datetime.now(datetime.timezone.utc)
    lo, hi = (1.0 - strike_pct_band) * price, (1.0 + strike_pct_band) * price

    out: list[dict] = []

    for instr in instruments_data.get("result", []):
        if instr.get("base_currency") != asset.upper():
            continue

        expiry = datetime.datetime.fromtimestamp(instr["expiration_timestamp"] / 1000.0, datetime.timezone.utc)
        creation = datetime.datetime.fromtimestamp(instr["creation_timestamp"] / 1000.0, datetime.timezone.utc)

        t_to_exp = (expiry - now).total_seconds() / 86400.0
        t_life = (now - creation).total_seconds() / 86400.0

        if not (min_expiry_days <= t_to_exp <= max_expiry_days):
            continue
        if t_life < min_life_days:
            continue

        K = float(instr["strike"]) if instr.get("strike") is not None else None
        if K is None or not (lo <= K <= hi):
            continue

        # Pull ticker to get IVs
        tkr = requests.get(BASE_URL + f"public/ticker?instrument_name={instr['instrument_name']}").json()
        iv = _try_extract_iv(tkr)
        if iv is None or iv <= 0:
            continue

        # Deribit naming ends with C/P for calls/puts
        opt_type = 'call' if instr['instrument_name'].endswith('C') else 'put'

        out.append({
            **instr,
            "iv": iv,
            "option_type": opt_type,
            # Helpful extras for later weighting/filters
            "_best_bid": tkr.get("result", {}).get("best_bid_price"),
            "_best_ask": tkr.get("result", {}).get("best_ask_price"),
            "_open_interest": tkr.get("result", {}).get("open_interest"),
        })

    return out


def compute_atm_iv_per_expiry(opts_for_expiry: list[dict], spot_price: float) -> float | None:
    """Compute ATM (k=0) IV per expiry via local interpolation in k=ln(K/spot)."""
    by_strike: dict[float, dict] = {}
    for o in opts_for_expiry:
        K = float(o["strike"])
        iv = float(o["iv"])  # decimal
        d = by_strike.setdefault(K, {"call": None, "put": None})
        if o.get("option_type") == "call":
            d["call"] = iv
        else:
            d["put"] = iv

    pts = []  # (k, iv, K)
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
        k0, iv0, _ = lower
        k1, iv1, _ = upper
        if abs(k1 - k0) < 1e-12:
            return (iv0 + iv1) / 2.0
        w = (0 - k0) / (k1 - k0)
        return iv0 * (1 - w) + iv1 * w

    # Fallback: only one side
    nearest = min(pts, key=lambda x: abs(x[0]))
    return nearest[1]


# -----------------------------
# Interpolate total variance to exactly 1d
# -----------------------------

def total_variance_interpolate_to_1d(expiry_to_iv: dict[int, float], now_ts: float):
    """Return (iv_1d, used_expiries_list)."""
    items = []  # (T_days, iv, exp_ms)
    for exp_ms, iv in expiry_to_iv.items():
        T_days = (exp_ms / 1000.0 - now_ts) / 86400.0
        if T_days > 0:
            items.append((T_days, iv, exp_ms))
    if not items:
        return None, []

    items.sort(key=lambda x: x[0])

    below = None
    above = None
    for T, iv, exp in items:
        if T <= 1.0:
            below = (T, iv, exp)
        if T >= 1.0 and above is None:
            above = (T, iv, exp)
            break

    if below and above and below[2] != above[2]:
        Tb, ivb, expb = below
        Ta, iva, expa = above
        wb = (ivb ** 2) * Tb
        wa = (iva ** 2) * Ta
        w1 = wb + (wa - wb) * ((1.0 - Tb) / (Ta - Tb))
        iv1 = sqrt(max(w1, 0.0))  # T=1 day
        return iv1, [expb, expa]

    # Fallback: single side → scale to 1d
    T, iv, exp = min(items, key=lambda x: abs(x[0] - 1.0))
    iv1 = iv * sqrt(1.0 / T)
    return iv1, [exp]


def soft_roll_and_ema(new_iv_1d: float,
                      used_pair: list[int],
                      now_dt: datetime.datetime,
                      ema_half_life_sec: int = 3600,
                      roll_window_sec: int = 1800,
                      path: str = STATE_PATH) -> float:
    state = _load_state(path=path)

    now_ts = now_dt.timestamp()
    pair_key = _expiry_keyset(used_pair)

    last_pair = state.get("last_pair")
    last_pair_change_ts = state.get("last_pair_change_ts", now_ts)
    last_output = state.get("last_output")

    if pair_key != last_pair:
        state["last_pair"] = pair_key
        state["last_pair_change_ts"] = now_ts
        state["roll_anchor"] = last_output if last_output is not None else new_iv_1d

    roll_anchor = state.get("roll_anchor", new_iv_1d)
    t0 = state.get("last_pair_change_ts", now_ts)
    dt_roll = max(0.0, now_ts - t0)
    if dt_roll >= roll_window_sec:
        blended = new_iv_1d
    else:
        alpha = max(0.0, 1.0 - dt_roll / roll_window_sec)
        blended = alpha * roll_anchor + (1 - alpha) * new_iv_1d

    last_ema = state.get("last_ema")
    last_time = state.get("last_time", now_ts)
    dt = max(1.0, now_ts - last_time)
    lam = 1.0 - pow(2.0, -dt / max(1.0, ema_half_life_sec))
    ema = blended if last_ema is None else (lam * blended + (1 - lam) * last_ema)

    state["last_time"] = now_ts
    state["last_output"] = blended
    state["last_ema"] = ema
    _save_state(state, path=path)

    return ema


def compute_continuous_1d_iv(options: list[dict], spot_price: float, path: str):
    # Group by expiry and compute ATM IV per expiry
    expiry_groups: dict[int, list[dict]] = {}
    for o in options:
        expiry_groups.setdefault(o["expiration_timestamp"], []).append(o)

    per_expiry_atm: dict[int, float] = {}
    for exp_ms, opts in expiry_groups.items():
        iv_atm = compute_atm_iv_per_expiry(opts, spot_price)
        if iv_atm is not None and iv_atm > 0:
            per_expiry_atm[exp_ms] = iv_atm

    now_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
    iv1d, used_pair = total_variance_interpolate_to_1d(per_expiry_atm, now_ts)
    if iv1d is None:
        return {"raw_1d_iv": None, "smoothed_1d_iv": None, "used_pair": [], "per_expiry_atm_iv": per_expiry_atm}

    smoothed = soft_roll_and_ema(iv1d, used_pair, datetime.datetime.now(datetime.timezone.utc), path=path)

    return {
        "raw_1d_iv": iv1d,  # annualized implied sigma at T=1d
        "smoothed_1d_iv": smoothed,  # EMA + soft-roll
        "used_pair": used_pair,  # expiries used for the interpolation
        "per_expiry_atm_iv": per_expiry_atm,
    }


def calculate_volatility(daily_vols, expiry_times):
    distances = [abs(t - 1) for t in expiry_times]
    inv_distances = [1 / d if d != 0 else 0 for d in distances]
    #weighted_avg = np.average(daily_vols, weights=inv_distances)
    weighted_avg = np.min(daily_vols)
    simple_avg = np.mean(daily_vols)
    return simple_avg / 365 ** 0.5, weighted_avg / 365 ** 0.5


def update_cache():
    while True:
        for asset,settings in ASSETS.items():
            print(asset,settings)
            strike_pct_band = settings["strike_pct_band"]
            min_life_days   = settings["min_life_days"]
            min_expiry_days = settings["min_expiry_days"]
            max_expiry_days = settings["max_expiry_days"]
            state_path      = settings["state_path"]
            try:
                price = get_price(asset)
                options = fetch_options_with_iv(asset,
                                                price,
                                                strike_pct_band,
                                                min_life_days,
                                                min_expiry_days,
                                                max_expiry_days
                                                )
                if not options:
                    continue
                res = compute_continuous_1d_iv(options, price, path=state_path)
                raw_1d_iv_annualized = res["raw_1d_iv"]
                smoothed_1d_iv_annualized = res["smoothed_1d_iv"]
                if raw_1d_iv_annualized * smoothed_1d_iv_annualized == 0 : # couldn't calc
                    continue
                else:
                    key = cache_key(asset)
                    CACHE[key] = {
                        "price": price,
                        "weighted_avg_vol": round(raw_1d_iv_annualized, 6),
                        "smoothed_avg_vol": round(smoothed_1d_iv_annualized, 6),
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().rsplit(sep="+")[0]
                    }
            except Exception as e:
                print(f"Error updating {asset}: {e}")
        time.sleep(30)


@app.route("/")
def index():
    return jsonify({
        "message": "Use /volatility?asset=BTC, ETH, SOL or asset=XAU (PAXG) to get volatility data"
    })


@app.route("/volatility")
def volatility():
    asset = request.args.get("asset", "BTC").upper()
    asset = QUERY_ALIASES.get(asset, asset)
    if asset not in CACHE:
        return jsonify({"error": "Asset not found or not yet cached"}), 400
    return jsonify(CACHE[asset])


Thread(target=update_cache, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5500)))
