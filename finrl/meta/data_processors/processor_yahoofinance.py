"""Reference: https://github.com/AI4Finance-LLC/FinRL"""

from __future__ import annotations

import datetime
import logging
import time
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from finrl.meta.data_processors.base_processor import BaseDataProcessor

logger = logging.getLogger(__name__)

YAHOO_CHART_ENDPOINTS = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
    "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
)
YAHOO_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


### Added by aymeric75 for scrap_data function


class YahooFinanceProcessor(BaseDataProcessor):
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _download_chart_api(
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        interval: str,
        proxy: str | dict | None = None,
    ) -> pd.DataFrame:
        """Fetch Yahoo's crumb-free chart endpoint as a server fallback."""
        start_utc = pd.Timestamp(start)
        end_utc = pd.Timestamp(end)
        start_utc = (
            start_utc.tz_localize("UTC")
            if start_utc.tzinfo is None
            else start_utc.tz_convert("UTC")
        )
        end_utc = (
            end_utc.tz_localize("UTC")
            if end_utc.tzinfo is None
            else end_utc.tz_convert("UTC")
        )
        params: dict[str, Any] = {
            "period1": int(start_utc.timestamp()),
            "period2": int(end_utc.timestamp()),
            "interval": interval,
            "events": "div,splits,capitalGains",
            "includeAdjustedClose": "true",
        }
        proxies = proxy if isinstance(proxy, dict) else None
        if isinstance(proxy, str) and proxy:
            proxies = {"http": proxy, "https": proxy}
        failures: list[str] = []

        for endpoint in YAHOO_CHART_ENDPOINTS:
            url = endpoint.format(ticker=ticker)
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers={"User-Agent": YAHOO_USER_AGENT, "Accept": "application/json"},
                    proxies=proxies,
                    timeout=(5, 20),
                )
                response.raise_for_status()
                chart = response.json().get("chart", {})
                if chart.get("error"):
                    raise RuntimeError(str(chart["error"]))
                results = chart.get("result") or []
                if not results:
                    raise RuntimeError("respons chart tidak berisi result")
                result = results[0]
                timestamps = result.get("timestamp") or []
                quotes = (result.get("indicators", {}).get("quote") or [{}])[0]
                if not timestamps or not quotes:
                    raise RuntimeError("respons chart tidak berisi OHLCV")

                frame = pd.DataFrame(
                    {
                        "Open": quotes.get("open", []),
                        "High": quotes.get("high", []),
                        "Low": quotes.get("low", []),
                        "Close": quotes.get("close", []),
                        "Volume": quotes.get("volume", []),
                    },
                    index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
                )
                frame.index.name = "Datetime" if interval != "1d" else "Date"
                adjusted = result.get("indicators", {}).get("adjclose") or []
                adjusted_close = adjusted[0].get("adjclose", []) if adjusted else []
                if len(adjusted_close) == len(frame):
                    raw_close = pd.to_numeric(frame["Close"], errors="coerce")
                    factor = pd.Series(adjusted_close, index=frame.index).div(
                        raw_close.replace(0, np.nan)
                    )
                    for column in ("Open", "High", "Low", "Close"):
                        frame[column] = pd.to_numeric(frame[column], errors="coerce") * factor
                frame = frame.dropna(subset=["Close"])
                if not frame.empty:
                    return frame
                raise RuntimeError("semua bar chart kosong")
            except Exception as error:  # noqa: BLE001 - aggregate provider diagnostics
                failures.append(f"{url}: {type(error).__name__}: {error}")
                logger.warning("Yahoo chart fallback gagal untuk %s: %s", ticker, error)
                time.sleep(0.5)

        raise RuntimeError("; ".join(failures))

    """
    Param
    ----------
        start_date : str
            start date of the data
        end_date : str
            end date of the data
        ticker_list : list
            a list of stock tickers
    Example
    -------
    input:
    ticker_list = config_tickers.DOW_30_TICKER
    start_date = '2009-01-01'
    end_date = '2021-10-31'
    time_interval == "1D"

    output:
        date	    tic	    open	    high	    low	        close	    volume
    0	2009-01-02	AAPL	3.067143	3.251429	3.041429	2.767330	746015200.0
    1	2009-01-02	AMGN	58.590000	59.080002	57.750000	44.523766	6547900.0
    2	2009-01-02	AXP	    18.570000	19.520000	18.400000	15.477426	10955700.0
    3	2009-01-02	BA	    42.799999	45.560001	42.779999	33.941093	7010200.0
    ...
    """

    ######## ADDED BY aymeric75 ###################

    def date_to_unix(self, date_str) -> int:
        """Convert a date string in yyyy-mm-dd format to Unix timestamp."""
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())

    def fetch_stock_data(self, stock_name, period1, period2) -> pd.DataFrame:
        # Base URL
        url = f"https://finance.yahoo.com/quote/{stock_name}/history/?period1={period1}&period2={period2}&filter=history"

        # Selenium WebDriver Setup
        options = Options()
        options.add_argument("--headless")  # Headless for performance
        options.add_argument("--disable-gpu")  # Disable GPU for compatibility
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

        # Navigate to the URL
        driver.get(url)
        driver.maximize_window()
        time.sleep(5)  # Wait for redirection and page load

        # Handle potential popup
        try:
            RejectAll = driver.find_element(
                By.XPATH, '//button[@class="btn secondary reject-all"]'
            )
            action = ActionChains(driver)
            action.click(on_element=RejectAll)
            action.perform()
            time.sleep(5)

        except Exception as e:
            print("Popup not found or handled:", e)

        # Parse the page for the table
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table")
        if not table:
            raise Exception("No table found after handling redirection and popup.")

        # Extract headers
        headers = [th.text.strip() for th in table.find_all("th")]
        headers[4] = "Close"
        headers[5] = "Adj Close"
        headers = ["date", "open", "high", "low", "close", "adjcp", "volume"]
        # , 'tic', 'day'

        # Extract rows
        rows = []
        for tr in table.find_all("tr")[1:]:  # Skip header row
            cells = [td.text.strip() for td in tr.find_all("td")]
            if len(cells) == len(headers):  # Only add rows with correct column count
                rows.append(cells)

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Convert columns to appropriate data types
        def safe_convert(value, dtype):
            try:
                return dtype(value.replace(",", ""))
            except ValueError:
                return value

        df["open"] = df["open"].apply(lambda x: safe_convert(x, float))
        df["high"] = df["high"].apply(lambda x: safe_convert(x, float))
        df["low"] = df["low"].apply(lambda x: safe_convert(x, float))
        df["close"] = df["close"].apply(lambda x: safe_convert(x, float))
        df["adjcp"] = df["adjcp"].apply(lambda x: safe_convert(x, float))
        df["volume"] = df["volume"].apply(lambda x: safe_convert(x, int))

        # Add 'tic' column
        df["tic"] = stock_name

        # Add 'day' column
        start_date = datetime.datetime.fromtimestamp(period1)
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = (df["date"] - start_date).dt.days
        df = df[df["day"] >= 0]  # Exclude rows with days before the start date

        # Reverse the DataFrame rows
        df = df.iloc[::-1].reset_index(drop=True)

        return df

    def scrap_data(self, stock_names, start_date, end_date) -> pd.DataFrame:
        """Fetch and combine stock data for multiple stock names."""
        period1 = self.date_to_unix(start_date)
        period2 = self.date_to_unix(end_date)

        all_dataframes = []
        total_stocks = len(stock_names)

        for i, stock_name in enumerate(stock_names):
            try:
                print(
                    f"Processing {stock_name} ({i + 1}/{total_stocks})... {(i + 1) / total_stocks * 100:.2f}% complete."
                )
                df = self.fetch_stock_data(stock_name, period1, period2)
                all_dataframes.append(df)
            except Exception as e:
                print(f"Error fetching data for {stock_name}: {e}")

        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df = combined_df.sort_values(by=["day", "tic"]).reset_index(drop=True)

        return combined_df

    ######## END ADDED BY aymeric75 ###################

    def convert_interval(self, time_interval: str) -> str:
        # Convert FinRL 'standardised' time periods to Yahoo format: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        yahoo_intervals = [
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ]
        if time_interval in yahoo_intervals:
            return time_interval
        if time_interval in [
            "1Min",
            "2Min",
            "5Min",
            "15Min",
            "30Min",
            "60Min",
            "90Min",
        ]:
            time_interval = time_interval.replace("Min", "m")
        elif time_interval in ["1H", "1D", "5D", "1h", "1d", "5d"]:
            time_interval = time_interval.lower()
        elif time_interval == "1W":
            time_interval = "1wk"
        elif time_interval in ["1M", "3M"]:
            time_interval = time_interval.replace("M", "mo")
        else:
            raise ValueError("wrong time_interval")

        return time_interval

    def download_data(
        self,
        ticker_list: list[str],
        start_date: str,
        end_date: str,
        time_interval: str,
        proxy: str | dict = None,
    ) -> pd.DataFrame:
        time_interval = self.convert_interval(time_interval)

        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        # Download and save the data in a pandas DataFrame
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        is_intraday = self.time_interval not in {"1d", "5d", "1wk", "1mo", "3mo"}
        chunk_size = timedelta(days=7 if self.time_interval == "1m" else 59)
        frames = []
        failures: dict[str, list[str]] = {}
        for tic in ticker_list:
            current_tic_start_date = start_date
            while current_tic_start_date <= end_date:
                request_end = (
                    min(current_tic_start_date + chunk_size, end_date + timedelta(days=1))
                    if is_intraday
                    else end_date + timedelta(days=1)
                )
                try:
                    temp_df = yf.download(
                        tic,
                        start=current_tic_start_date,
                        end=request_end,
                        interval=self.time_interval,
                        proxy=proxy,
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                        timeout=20,
                    )
                except Exception as error:
                    failures.setdefault(tic, []).append(
                        f"yfinance: {type(error).__name__}: {error}"
                    )
                    temp_df = pd.DataFrame()
                if temp_df.empty:
                    try:
                        temp_df = self._download_chart_api(
                            tic,
                            current_tic_start_date,
                            request_end,
                            self.time_interval,
                            proxy,
                        )
                    except Exception as error:
                        failures.setdefault(tic, []).append(
                            f"chart-api: {type(error).__name__}: {error}"
                        )
                        if not is_intraday:
                            break
                        current_tic_start_date = request_end
                        continue
                if temp_df.columns.nlevels != 1:
                    temp_df.columns = temp_df.columns.droplevel(1)

                temp_df["tic"] = tic
                frames.append(temp_df)
                if not is_intraday:
                    break
                current_tic_start_date = request_end

        if not frames:
            summary = " | ".join(
                f"{ticker}: {'; '.join(messages[-2:])}"
                for ticker, messages in failures.items()
            )
            raise ValueError(
                "Yahoo Finance tidak mengembalikan data. Periksa akses keluar VPS, "
                "rate limit Yahoo, ticker, dan rentang interval."
                + (f" Detail provider: {summary}" if summary else "")
            )
        if failures:
            logger.warning(
                "Sebagian request Yahoo gagal tetapi data parsial tersedia: %s",
                " | ".join(
                    f"{ticker}: {'; '.join(messages[-2:])}"
                    for ticker, messages in failures.items()
                ),
            )
        data_df = pd.concat(frames)

        data_df = data_df.reset_index().drop(columns=["Adj Close"], errors="ignore")
        # Map column names explicitly by name instead of positional indexing
        rename_map = {
            "Date": "timestamp",
            "Datetime": "timestamp",
            "date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        data_df = data_df.rename(columns=rename_map)
        data_df.columns = [str(c).lower() for c in data_df.columns]
        if "date" in data_df.columns and "timestamp" not in data_df.columns:
            data_df = data_df.rename(columns={"date": "timestamp"})
        data_df = data_df[["timestamp", "open", "high", "low", "close", "volume", "tic"]]

        return data_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        tic_list = np.unique(df.tic.values)
        NY = "America/New_York"

        # produce full timestamp index
        if self.time_interval.lower() in {"1d", "5d", "1wk", "1mo"}:
            # Use the union of actual exchange dates. This supports IDX and
            # other non-US markets without imposing a NYSE calendar.
            times = sorted(pd.to_datetime(df["timestamp"]).dropna().unique())
        elif self.time_interval in {"1m", "1Min"} and not any(str(t).endswith(".JK") for t in tic_list):
            trading_days = self.get_trading_days(start=self.start, end=self.end)
            times = []
            for day in trading_days:
                current_time = pd.Timestamp(day + " 09:30:00").tz_localize(NY)
                for i in range(390):  # 390 minutes in trading day
                    times.append(current_time)
                    current_time += pd.Timedelta(minutes=1)
        else:
            # For intraday or non-US markets, use the observed timestamps from the data
            times = sorted(pd.to_datetime(df["timestamp"]).dropna().unique())

        # create a new dataframe with full timestamp series
        new_df = pd.DataFrame()
        for tic in tic_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_timestamp = tic_df.iloc[i]["timestamp"]
                if self.time_interval == "1d":
                    tmp_timestamp = pd.Timestamp(tmp_timestamp).to_datetime64()
                elif tmp_timestamp.tzinfo is None:
                    tmp_timestamp = tmp_timestamp.tz_localize(NY)
                else:
                    tmp_timestamp = tmp_timestamp.tz_convert(NY)
                tmp_df.loc[tmp_timestamp] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

            tmp_df = self.fill_nan_with_previous(tmp_df)

            # merge single ticker data to new DataFrame
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        # reset index and rename columns
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        return new_df

    def add_vix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add VIX data from Yahoo Finance."""
        vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
        cleaned_vix = self.clean_data(vix_df)
        vix = cleaned_vix[["timestamp", "close"]]
        vix = vix.rename(columns={"close": "VIXY"})

        df = data.copy()
        df = df.merge(vix, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def fetch_latest_data(
        self,
        ticker_list: list[str],
        time_interval: str,
        tech_indicator_list: list[str],
        limit: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fetch latest data for real-time trading."""
        time_interval = self.convert_interval(time_interval)

        end_datetime = datetime.datetime.now()
        start_datetime = end_datetime - datetime.timedelta(minutes=limit + 1)

        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = yf.download(
                tic, start_datetime, end_datetime, interval=time_interval
            )
            barset["tic"] = tic
            data_df = pd.concat([data_df, barset])

        data_df = data_df.reset_index().drop(columns=["Adj Close"])
        data_df.columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tic",
        ]

        start_time = data_df.timestamp.min()
        end_time = data_df.timestamp.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["timestamp"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

            tmp_df = self.fill_nan_with_previous(tmp_df)
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        start_datetime = end_datetime - datetime.timedelta(minutes=1)
        turb_df = yf.download("VIXY", start_datetime, limit=1)
        latest_turb = turb_df["Close"].values
        return latest_price, latest_tech, latest_turb
