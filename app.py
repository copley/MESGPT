#!/usr/bin/env python3
"""
Full Example of a Terminal-Based Python Application
Integrating:
- IBAPI for MES Futures (technical indicators),
- Alpha Vantage for SPY Fundamentals (proxy for S&P 500),
- OpenAI for AI-Generated Trade Recommendations.

NOTE: 
1) Install packages with:
   pip install ibapi openai requests python-dotenv pandas numpy

2) You must have a valid IB TWS or IB Gateway session running 
   with API enabled on port 7497 (or update the port below).

3) Create a .env file (or use environment variables) with:
   OPENAI_API_KEY=<your_openai_key>
   ALPHAVANTAGE_API_KEY=<your_alpha_vantage_key>
"""

import os
import time
import requests
import openai
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.utils import iswrapper
import threading

###############################################################################
# LOAD ENVIRONMENT VARIABLES
###############################################################################
load_dotenv()  # Loads .env file if present
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Error: OPENAI_API_KEY environment variable is not set.")

if not ALPHAVANTAGE_API_KEY:
    raise ValueError("Error: ALPHAVANTAGE_API_KEY environment variable is not set.")

# Set the API key for OpenAI
openai.api_key = OPENAI_API_KEY

###############################################################################
# 1. Helper Functions for Technical Indicators
###############################################################################

def compute_moving_average(series, window):
    """Simple moving average over a specified window."""
    return series.rolling(window=window).mean()

def compute_price_change(series, window):
    """Difference between current close and close 'window' days ago."""
    return series - series.shift(window)

def compute_percent_change(series, window):
    """Percent change from 'window' days ago."""
    old_price = series.shift(window)
    return (series - old_price) / (old_price) * 100

def compute_average_volume(volume_series, window):
    """Average volume over 'window' days."""
    return volume_series.rolling(window=window).mean()

def compute_stochastics(df, period=14, k_smooth=3, d_smooth=3):
    """
    Compute Raw Stochastic, %K, %D for given period.
    Raw = (Close - LowestLow(period)) / (HighestHigh(period) - LowestLow(period)) * 100
    %K = SMA of Raw (k_smooth)
    %D = SMA of %K (d_smooth)
    """
    low_min = df['low'].rolling(period).min()
    high_max = df['high'].rolling(period).max()
    raw_stoch = (df['close'] - low_min) / (high_max - low_min) * 100
    
    # %K is an EMA or SMA of raw_stoch (many variations exist, we'll do SMA)
    k = raw_stoch.rolling(k_smooth).mean()
    # %D is an SMA of %K
    d = k.rolling(d_smooth).mean()
    
    return raw_stoch, k, d

def compute_atr(df, period=14):
    """
    Average True Range.
    True Range = max of:
      (high - low),
      abs(high - prev_close),
      abs(low - prev_close)
    ATR = SMA of True Range over 'period'.
    """
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['true_range'].rolling(period).mean()
    return df['atr']

def compute_rsi(series, period=14):
    """
    Classic 14-day RSI.
    """
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Reindex to original length
    return rsi.reindex(series.index)

def compute_percent_r(df, period=14):
    """
    Williams %R = (HighestHigh - Close) / (HighestHigh - LowestLow) * -100
    Usually period=14
    """
    high_max = df['high'].rolling(period).max()
    low_min = df['low'].rolling(period).min()
    return (high_max - df['close']) / (high_max - low_min) * -100

def compute_historic_volatility(series, period=20):
    """
    Historical Volatility, annualized (approx).
    1) daily returns = ln(today_close / yesterday_close)
    2) stdev of daily returns over 'period'
    3) multiply stdev by sqrt(260) to annualize (some use 252 or 365)
    """
    log_ret = np.log(series / series.shift(1))
    stdev = log_ret.rolling(window=period).std()
    hv = stdev * np.sqrt(260) * 100  # in %, approx
    return hv

def compute_macd(series, fast=12, slow=26, signal=9):
    """
    MACD = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal)
    MACD Oscillator = MACD - Signal line
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram  # "MACD Oscillator"

###############################################################################
# 2. Trader's Cheat Sheet & Pivot Points, S/R
###############################################################################

def compute_pivots_s_r(high, low, close):
    """
    Standard pivot point calculations for the next session.
    Returns PP, R1, R2, R3, S1, S2, S3
    """
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (r1 - s1)
    s2 = pp - (r1 - s1)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    return pp, r1, r2, r3, s1, s2, s3

def compute_std_devs(series_close, stdev_multiplier=[1,2,3]):
    """
    Returns a dict with {1: stdev1, 2: stdev2, 3: stdev3}.
    Uses last 5 closes to match Barchart’s example or modify as needed.
    """
    last_5 = series_close.iloc[-5:]
    avg = last_5.mean()
    diffs = last_5 - avg
    squared = diffs**2
    sum_sq = squared.sum()
    var = sum_sq / (len(last_5) - 1)
    base_stdev = np.sqrt(var)

    out = {}
    for m in stdev_multiplier:
        out[m] = base_stdev * m
    return out, avg

def compute_fib_levels(recent_high, recent_low):
    """
    Compute 38.2%, 50%, 61.8% retracements from a given high/low.
    """
    diff = recent_high - recent_low
    fib_382 = recent_high - 0.382 * diff
    fib_50  = (recent_high + recent_low) / 2
    fib_618 = recent_high - 0.618 * diff
    return fib_382, fib_50, fib_618

###############################################################################
# 3. Barchart Opinion Calculation
###############################################################################

def mock_trend_seeker(df):
    """
    Placeholder for the proprietary Trend Seeker® logic.
    """
    short_ema = df['close'].ewm(span=10).mean().iloc[-1]
    long_ema  = df['close'].ewm(span=50).mean().iloc[-1]
    last_price = df['close'].iloc[-1]

    # Naive approach: If short EMA > long EMA and price > short EMA => "Buy"; else "Sell" or "Hold"
    if short_ema > long_ema and last_price > short_ema:
        return 1  # "Buy"
    elif short_ema < long_ema and last_price < short_ema:
        return -1  # "Sell"
    return 0  # "Hold"

def indicator_signal(value, threshold_up=0, threshold_down=0):
    """
    Converts a numeric value into buy/sell/hold signals for demonstration.
    """
    if value > threshold_up:
        return 1
    elif value < threshold_down:
        return -1
    else:
        return 0

def barchart_opinion_logic(df):
    """
    Summarize 13 (or more) indicators into short-, medium-, and long-term signals.
    Then produce an overall Barchart-style opinion (e.g., "64% Buy").
    """
    last = df.iloc[-1]

    # 1) Short Term (≈20 days)
    short_signals = []
    price20ma = last['close'] - last['MA_20']
    short_signals.append(indicator_signal(price20ma))

    short_slope_7 = df['close'].diff().tail(7).mean()
    short_signals.append(indicator_signal(short_slope_7))

    # placeholder for Bollinger or other short-term
    short_signals.append(0)

    # 20-50 day MA crossover
    if last['MA_20'] > last['MA_50']:
        short_signals.append(1)
    else:
        short_signals.append(-1)

    short_signals.append(0)  # another placeholder

    # 2) Medium Term (≈50 days)
    medium_signals = []
    price50ma = last['close'] - last['MA_50']
    medium_signals.append(indicator_signal(price50ma))

    if last['MA_20'] > last['MA_100']:
        medium_signals.append(1)
    else:
        medium_signals.append(-1)

    medium_signals.append(0)  # placeholder
    medium_signals.append(0)  # placeholder

    # 3) Long Term (≈100-200 days)
    long_signals = []
    price100ma = last['close'] - last['MA_100']
    long_signals.append(indicator_signal(price100ma))

    if last['MA_50'] > last['MA_100']:
        long_signals.append(1)
    else:
        long_signals.append(-1)

    price200ma = last['close'] - last['MA_200']
    long_signals.append(indicator_signal(price200ma))

    # Trend Seeker®
    trend_seeker_signal = mock_trend_seeker(df)

    # Combine all
    combined = short_signals + medium_signals + long_signals + [trend_seeker_signal]
    overall = sum(combined) / len(combined)

    # Then factor for Barchart style
    raw_percent = overall * 100
    factored_percent = abs(raw_percent) * 1.04
    possible_vals = [8,16,24,32,40,48,56,64,72,80,88,96,100]

    def nearest_opinion_pct(x):
        if x >= 95:
            return 100
        return min(possible_vals, key=lambda v: abs(v - x))

    final_opinion_pct = nearest_opinion_pct(factored_percent)
    sign = np.sign(overall)
    if sign > 0:
        final_opinion_str = f"{final_opinion_pct}% Buy"
    elif sign < 0:
        final_opinion_str = f"{final_opinion_pct}% Sell"
    else:
        final_opinion_str = "Hold"

    return {
        'short_avg': sum(short_signals)/len(short_signals),
        'medium_avg': sum(medium_signals)/len(medium_signals),
        'long_avg': sum(long_signals)/len(long_signals),
        'trend_seeker_signal': trend_seeker_signal,
        'overall_numeric': overall,
        'overall_opinion': final_opinion_str
    }

###############################################################################
# ALPHA VANTAGE FUNCTIONS
###############################################################################

def fetch_alpha_vantage_fundamentals():
    """
    Fetch SPY fundamentals (OVERVIEW endpoint) as a proxy for S&P 500,
    returning relevant data in a dict.
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": "SPY",
        "apikey": ALPHAVANTAGE_API_KEY
    }
    resp = requests.get(base_url, params=params)
    data = resp.json()

    # Extract what we need; handle missing fields gracefully
    # fallback to "N/A" if not present
    fundamentals = {
        "pe_ratio": data.get("PERatio", "N/A"),
        "forward_pe": data.get("ForwardPE", "N/A"),
        "earnings_growth": data.get("QuarterlyEarningsGrowthYOY", "N/A"),
        "dividend_yield": data.get("DividendYield", "N/A"),
        "market_cap": data.get("MarketCapitalization", "N/A")
    }
    # Optionally fetch more fields as needed
    return fundamentals

###############################################################################
# IBAPI APPLICATION
###############################################################################

class IBApp(EWrapper, EClient):
    def __init__(self, ipaddress, portid, clientid):
        EClient.__init__(self, self)
        self.ipaddress = ipaddress
        self.portid = portid
        self.clientid = clientid

        # Data storage
        self.historical_data = []
        self.request_completed = False

    def connect_and_run(self):
        self.connect(self.ipaddress, self.portid, self.clientid)
        thread = threading.Thread(target=self.run)
        thread.start()

    @iswrapper
    def nextValidId(self, orderId: int):
        """
        Called when the API receives the next valid order ID.
        We can safely request data here.
        """
        print(f"[IBApp] nextValidId called with orderId={orderId}")
        self.request_historical_data()

    def request_historical_data(self):
        print("[IBApp] Requesting historical data for MES (1 year, daily bars)...")
        contract = self.create_mes_contract()
        # ~1 year of daily data
        self.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime="",
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

    def create_mes_contract(self):
        """
        Adjust localSymbol or lastTradeDateOrContractMonth as needed.
        Example: 'MESM5' for June 2025.
        """
        contract = Contract()
        contract.symbol = "MES"
        contract.secType = "FUT"
        contract.exchange = "GLOBEX"
        contract.currency = "USD"
        # For demonstration, using 'MESM5'. 
        # In real usage, pick the correct front-month or rolling logic.
        contract.localSymbol = "MESM5"
        return contract

    @iswrapper
    def historicalData(self, reqId, bar):
        """
        Called for each bar of historical data.
        """
        self.historical_data.append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    @iswrapper
    def historicalDataEnd(self, reqId, start, end):
        """
        Called once no more historical bars are coming.
        """
        print("[IBApp] Historical data download complete.")
        self.request_completed = True
        # Process data
        self.process_data()
        # Disconnect from IB
        self.disconnect()

    def process_data(self):
        """
        Transform the raw data into a DataFrame, compute analytics, 
        print the results, etc.
        """
        df = pd.DataFrame(self.historical_data)
        # Sort by date (just in case)
        df.sort_values(by="date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Compute Indicators
        df["MA_5"]   = compute_moving_average(df['close'], 5)
        df["MA_20"]  = compute_moving_average(df['close'], 20)
        df["MA_50"]  = compute_moving_average(df['close'], 50)
        df["MA_100"] = compute_moving_average(df['close'], 100)
        df["MA_200"] = compute_moving_average(df['close'], 200)

        df["pc_5"]   = compute_price_change(df['close'], 5)
        df["pc_20"]  = compute_price_change(df['close'], 20)
        df["pc_50"]  = compute_price_change(df['close'], 50)
        df["pc_100"] = compute_price_change(df['close'], 100)
        df["pc_200"] = compute_price_change(df['close'], 200)

        df["pct_5"]   = compute_percent_change(df['close'], 5)
        df["pct_20"]  = compute_percent_change(df['close'], 20)
        df["pct_50"]  = compute_percent_change(df['close'], 50)
        df["pct_100"] = compute_percent_change(df['close'], 100)
        df["pct_200"] = compute_percent_change(df['close'], 200)

        df["vol_5"]   = compute_average_volume(df['volume'], 5)
        df["vol_20"]  = compute_average_volume(df['volume'], 20)
        df["vol_50"]  = compute_average_volume(df['volume'], 50)
        df["vol_100"] = compute_average_volume(df['volume'], 100)
        df["vol_200"] = compute_average_volume(df['volume'], 200)

        df["raw_9"], df["k_9"], df["d_9"]     = compute_stochastics(df, 9)
        df["raw_14"], df["k_14"], df["d_14"]  = compute_stochastics(df, 14)
        df["raw_20"], df["k_20"], df["d_20"]  = compute_stochastics(df, 20)

        df["atr_14"] = compute_atr(df, 14)

        df["rsi_9"]  = compute_rsi(df['close'], 9)
        df["rsi_14"] = compute_rsi(df['close'], 14)
        df["rsi_20"] = compute_rsi(df['close'], 20)

        df["pr_9"]   = compute_percent_r(df, 9)
        df["pr_14"]  = compute_percent_r(df, 14)
        df["pr_20"]  = compute_percent_r(df, 20)

        df["hv_20"]  = compute_historic_volatility(df['close'], 20)

        df["macd"]   = compute_macd(df['close'])

        # Grab last row
        last = df.iloc[-1]
        last_date = last["date"]

        # Print a Technical Analysis summary 
        print("\n============================================================")
        print(f"Technical Analysis for MES - Last Date: {last_date}")
        print("============================================================\n")

        def safe_round(val, decimals=2):
            return round(val, decimals) if pd.notnull(val) else None

        print("Period | Moving Average | Price Change | Percent Change | Avg Volume")
        print("-------+----------------+--------------+----------------+-----------")

        for (label, ma_col, pc_col, pct_col, vol_col) in [
            ("5-Day",   'MA_5',   'pc_5',   'pct_5',   'vol_5'),
            ("20-Day",  'MA_20',  'pc_20',  'pct_20',  'vol_20'),
            ("50-Day",  'MA_50',  'pc_50',  'pct_50',  'vol_50'),
            ("100-Day", 'MA_100', 'pc_100', 'pct_100', 'vol_100'),
            ("200-Day", 'MA_200', 'pc_200', 'pct_200', 'vol_200'),
        ]:
            print(
                f"{label:7} "
                f"{safe_round(last[ma_col],2):>10} "
                f"{safe_round(last[pc_col],2):>12} "
                f"{safe_round(last[pct_col],2):>10}% "
                f"{int(safe_round(last[vol_col],0) or 0):>12}"
            )

        print("\nPeriod  | Raw Stochastic | Stoch %K  | Stoch %D  | ATR")
        print("--------+----------------+----------+-----------+-------")
        for (label, rcol, kcol, dcol, atr_col) in [
            ("9-Day",  'raw_9', 'k_9', 'd_9', 'atr_14'),
            ("14-Day", 'raw_14','k_14','d_14','atr_14'),
            ("20-Day", 'raw_20','k_20','d_20','atr_14'),
        ]:
            print(
                f"{label:7} "
                f"{safe_round(last[rcol],2):>14}% "
                f"{safe_round(last[kcol],2):>8}% "
                f"{safe_round(last[dcol],2):>8}% "
                f"{safe_round(last[atr_col],2):>8}"
            )

        print("\nPeriod  | Relative Strength | Percent R | Historic Vol | MACD Osc")
        print("--------+-------------------+-----------+--------------+---------")
        for (label, rsi_col, pr_col, hv_col) in [
            ("9-Day",  'rsi_9',  'pr_9',  'hv_20'),
            ("14-Day", 'rsi_14', 'pr_14', 'hv_20'),
            ("20-Day", 'rsi_20', 'pr_20', 'hv_20'),
        ]:
            print(
                f"{label:7} "
                f"{safe_round(last[rsi_col],2):>17}% "
                f"{safe_round(last[pr_col],2):>10}% "
                f"{safe_round(last[hv_col],2):>12}% "
                f"{safe_round(last['macd'],2):>9}"
            )

        # Trader's Cheat Sheet
        (pivot_pp, pivot_r1, pivot_r2, pivot_r3,
         pivot_s1, pivot_s2, pivot_s3) = compute_pivots_s_r(last['high'], last['low'], last['close'])

        stdev_dict, avg_5day = compute_std_devs(df['close'], [1,2,3])
        p1_res = avg_5day + stdev_dict[1]
        p1_sup = avg_5day - stdev_dict[1]
        p2_res = avg_5day + stdev_dict[2]
        p2_sup = avg_5day - stdev_dict[2]
        p3_res = avg_5day + stdev_dict[3]
        p3_sup = avg_5day - stdev_dict[3]

        print("\nTrader's Cheat Sheet")
        print("-------------------")
        print(f"Pivot Point (PP): {safe_round(pivot_pp,2)}")
        print(f"R1: {safe_round(pivot_r1,2)}   R2: {safe_round(pivot_r2,2)}   R3: {safe_round(pivot_r3,2)}")
        print(f"S1: {safe_round(pivot_s1,2)}   S2: {safe_round(pivot_s2,2)}   S3: {safe_round(pivot_s3,2)}\n")
        print(f"5-Day Avg Price (for SD calc): {safe_round(avg_5day,2)}")
        print(f"Price 1 SD Resistance: {safe_round(p1_res,2)}   Support: {safe_round(p1_sup,2)}")
        print(f"Price 2 SD Resistance: {safe_round(p2_res,2)}   Support: {safe_round(p2_sup,2)}")
        print(f"Price 3 SD Resistance: {safe_round(p3_res,2)}   Support: {safe_round(p3_sup,2)}\n")

        # Barchart-Style Opinion
        opinion_results = barchart_opinion_logic(df)
        print("Barchart Opinion")
        print("------------------------------------------------------------")
        print(f"Short-Term Avg Signal : {opinion_results['short_avg']}")
        print(f"Medium-Term Avg Signal: {opinion_results['medium_avg']}")
        print(f"Long-Term Avg Signal  : {opinion_results['long_avg']}")
        print(f"Trend Seeker (mock)   : {opinion_results['trend_seeker_signal']}")
        print(f"Overall Numeric Avg   : {opinion_results['overall_numeric']}")
        print(f"Final Opinion         : {opinion_results['overall_opinion']}")
        print("============================================================\n")

        # For AI prompt usage, we'll store some final data in a dictionary
        self.final_technical_data = {
            "last_close": last["close"],
            "ma_20": safe_round(last["MA_20"],2),
            "ma_50": safe_round(last["MA_50"],2),
            "rsi_14": safe_round(last["rsi_14"],2),
            "macd": safe_round(last["macd"],2),
            "atr_14": safe_round(last["atr_14"],2),
            "s1": safe_round(pivot_s1,2),
            "s2": safe_round(pivot_s2,2),
            "r1": safe_round(pivot_r1,2),
            "r2": safe_round(pivot_r2,2)
        }

###############################################################################
# OPENAI PROMPT + CALL
###############################################################################

def build_ai_prompt(fundamentals, technicals):
    """
    Construct a GPT prompt that merges fundamental data (SPY) + MES technicals.
    Requests trade recommendation with probability and brief rationale.
    """
    prompt = f"""
    You are an advanced trading assistant analyzing the MES futures contract.

    ### Fundamental Data (via Alpha Vantage):
    - SPY P/E Ratio: {fundamentals['pe_ratio']}
    - SPY Forward P/E: {fundamentals['forward_pe']}
    - Earnings Growth (YoY): {fundamentals['earnings_growth']}
    - Dividend Yield: {fundamentals['dividend_yield']}

    ### Technical Data (via IBAPI for MES):
    - Current MES Price: {technicals['last_close']}
    - 20-day MA: {technicals['ma_20']}, 50-day MA: {technicals['ma_50']}
    - RSI (14-day): {technicals['rsi_14']}
    - MACD: {technicals['macd']}
    - ATR (14): {technicals['atr_14']}
    - Support Levels: {technicals['s1']}, {technicals['s2']}
    - Resistance Levels: {technicals['r1']}, {technicals['r2']}

    ### TASK
    1. Provide a trade recommendation (Buy/Sell/Hold).
    2. Estimate the probability (0–100%) that MES will move in the recommended direction over the next 2 weeks.
    3. Give a brief fundamental justification (based on SPY's metrics).
    4. Give a brief technical justification (RSI, MACD, S/R, etc.).
    5. Mention one macro risk that might invalidate this trade.

    Rules:
    - Probability must be a single integer from 40–80%.
    - Keep your answer under 100 words total.
    """
    return prompt

def get_ai_analysis(prompt):
    """
    Calls OpenAI ChatCompletion with the given prompt.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

###############################################################################
# MAIN SCRIPT
###############################################################################

def main():
    # 1) Connect to IB and fetch MES data
    app = IBApp("127.0.0.1", 7497, clientid=1)
    app.connect_and_run()

    # Wait for data retrieval
    timeout = 60
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        if app.request_completed:
            break
        time.sleep(1)
    if not app.request_completed:
        print("[Main] Timed out waiting for historical data.")
        return

    # 2) Fetch fundamentals from Alpha Vantage (SPY)
    fundamentals = fetch_alpha_vantage_fundamentals()

    # 3) Build an AI prompt using the final technical data from IB
    technicals = getattr(app, "final_technical_data", None)
    if not technicals:
        print("[Main] No final technical data found. Exiting.")
        return

    prompt = build_ai_prompt(fundamentals, technicals)

    # 4) Get AI's trade recommendation
    ai_decision = get_ai_analysis(prompt)

    # 5) Print AI's result in the terminal
    print("AI Analysis:\n", ai_decision)


if __name__ == "__main__":
    main()
