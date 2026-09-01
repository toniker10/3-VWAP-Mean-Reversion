import pandas as pd
import numpy as np
import os, glob
import kagglehub

# pip install kagglehub -q   (run once, on Colab put it in its own cell)

dataset_path = kagglehub.dataset_download(
    "salaheddineelkhirani/5-year-data-for-s-and-p-500-and-nasdaq-100"
)
print("Dataset path:", dataset_path)

all_csvs = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
print(f"Found {len(all_csvs)} CSV files.")
for f in all_csvs:
    print(" -", os.path.basename(f))

# strategy params, same defaults as the Pine script, shared by both instruments
BAND          = 0.00616
ADX_LEN       = 14
ADX_THRESHOLD = 20.93
VOL_MULT      = 1.90
VOL_MULT_FAST = 1.10
SL_PCT        = 0.377 / 100
TP_PCT        = 0.377 / 100
ALLOW_LONG    = False
ALLOW_SHORT   = True

COMMISSION    = 0.0001
QTY_PCT       = 1.0


def rolling_vwap(src, vol, length):
    num = (src * vol).rolling(length).mean() * length
    den = vol.rolling(length).mean() * length
    return np.where(den != 0, num / den, src)


def dmi(high, low, close, length):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/length, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/length, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return plus_di, minus_di, adx


def load_and_prepare(csv_path):
    _raw_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    _lower = {c.lower(): c for c in _raw_cols}

    def _find(*candidates):
        for c in candidates:
            if c in _lower:
                return _lower[c]
        return None

    col_map = {
        "timestamp": _find("timestamp", "datetime", "date", "time"),
        "open":      _find("open"),
        "high":      _find("high"),
        "low":       _find("low"),
        "close":     _find("close", "adj close", "adjclose"),
        "volume":    _find("volume", "vol"),
    }
    if any(v is None for v in col_map.values()):
        raise ValueError(f"Could not detect all columns. Available: {_raw_cols}")

    df = pd.read_csv(csv_path)
    df = df.rename(columns={v: k for k, v in col_map.items()})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hl2"] = (df["high"] + df["low"]) / 2

    df["day"] = df["timestamp"].dt.date
    df["pv"] = df["hl2"] * df["volume"]
    grp = df.groupby("day")
    df["sess_pv_cum"] = grp["pv"].cumsum()
    df["sess_v_cum"] = grp["volume"].cumsum()
    df["vwap_session"] = np.where(df["sess_v_cum"] != 0,
                                   df["sess_pv_cum"] / df["sess_v_cum"], df["hl2"])

    df["vwap12"] = rolling_vwap(df["hl2"], df["volume"], 12)
    df["vwap78"] = rolling_vwap(df["hl2"], df["volume"], 78)
    df["di_plus"], df["di_minus"], df["adx"] = dmi(df["high"], df["low"], df["close"], ADX_LEN)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ma10"] = df["volume"].rolling(10).mean()

    df["far_above"] = (
        (df["close"] > df["vwap_session"] * (1 + BAND)) &
        (df["close"] > df["vwap12"] * (1 + BAND)) &
        (df["close"] > df["vwap78"] * (1 + BAND))
    )
    df["far_below"] = (
        (df["close"] < df["vwap_session"] * (1 - BAND)) &
        (df["close"] < df["vwap12"] * (1 - BAND)) &
        (df["close"] < df["vwap78"] * (1 - BAND))
    )
    df["is_ranging"] = df["adx"] < ADX_THRESHOLD
    df["has_volume"] = (df["volume"] > df["vol_ma20"] * VOL_MULT) & (df["volume"] > df["vol_ma10"] * VOL_MULT_FAST)
    df["long_signal"] = df["far_below"] & df["is_ranging"] & df["has_volume"] & ALLOW_LONG
    df["short_signal"] = df["far_above"] & df["is_ranging"] & df["has_volume"] & ALLOW_SHORT
    return df


def run_backtest(df):
    cum_return = 1.0
    position = 0
    entry_price = 0.0
    entry_i = 0
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        if position == 0:
            if row["long_signal"]:
                position, entry_price, entry_i = 1, row["close"], i
            elif row["short_signal"]:
                position, entry_price, entry_i = -1, row["close"], i

        elif position == 1:
            sl = entry_price * (1 - SL_PCT)
            tp = entry_price * (1 + TP_PCT)
            exit_price = None
            if row["low"] <= sl:
                exit_price = sl
            elif row["high"] >= tp:
                exit_price = tp
            if exit_price is not None:
                ret = (exit_price - entry_price) / entry_price - COMMISSION * 2
                cum_return *= (1 + ret * QTY_PCT)
                trades.append({"entry_time": df.iloc[entry_i]["timestamp"], "exit_time": row["timestamp"],
                                "side": "long", "entry": entry_price, "exit": exit_price,
                                "ret_pct": ret * 100})
                position = 0

        elif position == -1:
            sl = entry_price * (1 + SL_PCT)
            tp = entry_price * (1 - TP_PCT)
            exit_price = None
            if row["high"] >= sl:
                exit_price = sl
            elif row["low"] <= tp:
                exit_price = tp
            if exit_price is not None:
                ret = (entry_price - exit_price) / entry_price - COMMISSION * 2
                cum_return *= (1 + ret * QTY_PCT)
                trades.append({"entry_time": df.iloc[entry_i]["timestamp"], "exit_time": row["timestamp"],
                                "side": "short", "entry": entry_price, "exit": exit_price,
                                "ret_pct": ret * 100})
                position = 0

    return pd.DataFrame(trades), cum_return


INSTRUMENTS = {"ES": "S&P 500", "NQ": "NASDAQ 100"}
summary_rows = []

for hint, label in INSTRUMENTS.items():
    matches = [f for f in all_csvs if hint.lower() in os.path.basename(f).lower()]
    if not matches:
        print(f"\nNo file found for {hint}, skipping.")
        continue
    csv_path = matches[0]

    print(f"\n{hint} ({label}) - {os.path.basename(csv_path)}")

    df = load_and_prepare(csv_path)

    n = len(df)
    print(f"Total bars:  {n}")
    print(f"far_above:   {df['far_above'].sum():6d}  ({df['far_above'].mean()*100:.2f}%)")
    print(f"far_below:   {df['far_below'].sum():6d}  ({df['far_below'].mean()*100:.2f}%)")
    print(f"is_ranging:  {df['is_ranging'].sum():6d}  ({df['is_ranging'].mean()*100:.2f}%)")
    print(f"has_volume:  {df['has_volume'].sum():6d}  ({df['has_volume'].mean()*100:.2f}%)")
    print(f"long_signal:  {df['long_signal'].sum():6d}")
    print(f"short_signal: {df['short_signal'].sum():6d}")

    trades_df, cum_return = run_backtest(df)

    if hint == "ES":
        df_es, trades_es = df.copy(), trades_df.copy()
    elif hint == "NQ":
        df_nq, trades_nq = df.copy(), trades_df.copy()

    print("\n--- RESULTS ---")
    if len(trades_df) == 0:
        print("No trades were made.")
        summary_rows.append({"instrument": hint, "trades": 0, "win_rate": None, "total_return_pct": None})
    else:
        win_rate = (trades_df["ret_pct"] > 0).mean() * 100
        total_ret_pct = (cum_return - 1) * 100
        print(f"Trades:       {len(trades_df)}")
        print(f"Win rate:     {win_rate:.1f}%")
        print(f"Total return: {total_ret_pct:.2f}%")
        print(f"Avg trade:    {trades_df['ret_pct'].mean():.3f}%")
        print(f"Best trade:   {trades_df['ret_pct'].max():.3f}%")
        print(f"Worst trade:  {trades_df['ret_pct'].min():.3f}%")
        summary_rows.append({"instrument": hint, "trades": len(trades_df),
                              "win_rate": round(win_rate, 1), "total_return_pct": round(total_ret_pct, 2)})

print("\nES vs NQ comparison")
print(pd.DataFrame(summary_rows).to_string(index=False))

# quick check that ES/NQ signal counts and trade lists actually differ
print("\nSanity check")
print(f"ES signals: {df_es['short_signal'].sum()} -> trades: {len(trades_es)}")
print(f"NQ signals: {df_nq['short_signal'].sum()} -> trades: {len(trades_nq)}")
print("\nES entry dates:")
print(trades_es["entry_time"].tolist())
print("\nNQ entry dates:")
print(trades_nq["entry_time"].tolist())
