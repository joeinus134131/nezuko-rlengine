from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from finrl.meta.data_processors.base_processor import BaseDataProcessor

logger = logging.getLogger(__name__)


class AlpacaProcessor(BaseDataProcessor):
    def __init__(self, API_KEY=None, API_SECRET=None, API_BASE_URL=None, client=None):
        super().__init__()
        if client is None:
            try:
                self.client = StockHistoricalDataClient(API_KEY, API_SECRET)
            except BaseException:
                raise ValueError("Wrong Account Info!")
        else:
            self.client = client

    def _fetch_data_for_ticker(self, ticker, start_date, end_date, time_interval):
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=start_date,
            end=end_date,
        )
        bars = self.client.get_stock_bars(request_params).df

        return bars

    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        """
        Downloads data using Alpaca's tradeapi.REST method.

        Parameters:
        - ticker_list : list of strings, each string is a ticker
        - start_date : string in the format 'YYYY-MM-DD'
        - end_date : string in the format 'YYYY-MM-DD'
        - time_interval: string representing the interval ('1D', '1Min', etc.)

        Returns:
        - pd.DataFrame with the requested data
        """
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        NY = "America/New_York"
        start_date = pd.Timestamp(start_date + " 09:30:00", tz=NY)
        end_date = pd.Timestamp(end_date + " 15:59:00", tz=NY)
        data_list = []
        # Use ThreadPoolExecutor to fetch data for multiple tickers concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    self._fetch_data_for_ticker,
                    ticker,
                    start_date,
                    end_date,
                    time_interval,
                )
                for ticker in ticker_list
            ]
        for future in futures:

            bars = future.result()
            # fix start
            # Reorganize the dataframes to be in original alpaca_trade_api structure
            # Rename the existing 'symbol' column if it exists
            if not bars.empty:

                # Now reset the index
                bars.reset_index(inplace=True)

                # Set 'timestamp' as the new index
                if "level_1" in bars.columns:
                    bars.rename(columns={"level_1": "timestamp"}, inplace=True)
                if "level_0" in bars.columns:
                    bars.rename(columns={"level_0": "symbol"}, inplace=True)

                bars.set_index("timestamp", inplace=True)

                # Reorder and rename columns as needed
                bars = bars[
                    [
                        "close",
                        "high",
                        "low",
                        "trade_count",
                        "open",
                        "volume",
                        "vwap",
                        "symbol",
                    ]
                ]

                data_list.append(bars)
            else:
                print("empty")

        # Combine the data
        data_df = pd.concat(data_list, axis=0)

        # Convert the timezone
        data_df = data_df.tz_convert(NY)

        # If time_interval is less than a day, filter out the times outside of NYSE trading hours
        if pd.Timedelta(time_interval) < pd.Timedelta(days=1):
            data_df = data_df.between_time("09:30", "15:59")

        # Reset the index and rename the columns for consistency
        data_df = data_df.reset_index().rename(
            columns={"index": "timestamp", "symbol": "tic"}
        )

        # Sort the data by both timestamp and tic for consistent ordering
        data_df = data_df.sort_values(by=["tic", "timestamp"])

        # Reset the index and drop the old index column
        data_df = data_df.reset_index(drop=True)

        return data_df

    @staticmethod
    def clean_individual_ticker(args):
        tic, df, times = args
        tmp_df = pd.DataFrame(index=times)
        tic_df = df[df.tic == tic].set_index("timestamp")

        # Step 1: Merging dataframes to avoid loop
        tmp_df = tmp_df.merge(
            tic_df[["open", "high", "low", "close", "volume"]],
            left_index=True,
            right_index=True,
            how="left",
        )

        # Step 2: Handling NaN values efficiently
        if pd.isna(tmp_df.iloc[0]["close"]):
            first_valid_index = tmp_df["close"].first_valid_index()
            if first_valid_index is not None:
                first_valid_price = tmp_df.loc[first_valid_index, "close"]
                print(
                    f"The price of the first row for ticker {tic} is NaN. It will be filled with the first valid price."
                )
                tmp_df.iloc[0] = [first_valid_price] * 4 + [0.0]  # Set volume to zero
            else:
                print(
                    f"Missing data for ticker: {tic}. The prices are all NaN. Fill with 0."
                )
                tmp_df.iloc[0] = [0.0] * 5

        for i in range(1, tmp_df.shape[0]):
            if pd.isna(tmp_df.iloc[i]["close"]):
                previous_close = tmp_df.iloc[i - 1]["close"]
                tmp_df.iloc[i] = [previous_close] * 4 + [0.0]

        # Setting the volume for the market opening timestamp to zero - Not needed
        # tmp_df.loc[tmp_df.index.time == pd.Timestamp("09:30:00").time(), 'volume'] = 0.0

        # Step 3: Data type conversion
        tmp_df = tmp_df.astype(float)

        tmp_df["tic"] = tic

        return tmp_df

    def clean_data(self, df):
        logger.info("Data cleaning started")
        tic_list = np.unique(df.tic.values)
        n_tickers = len(tic_list)

        logger.info("Aligning start and end dates")
        grouped = df.groupby("timestamp")
        filter_mask = grouped.transform("count")["tic"] >= n_tickers
        df = df[filter_mask]

        trading_days = self.get_trading_days(start=self.start, end=self.end)

        # produce full timestamp index
        logger.info("Producing full timestamp index")
        times = []
        for day in trading_days:
            NY = "America/New_York"
            current_time = pd.Timestamp(day + " 09:30:00").tz_localize(NY)
            for i in range(390):
                times.append(current_time)
                current_time += pd.Timedelta(minutes=1)

        logger.info("Processing tickers")

        future_results = []
        for tic in tic_list:
            result = self.clean_individual_ticker((tic, df.copy(), times))
            future_results.append(result)

        logger.info("Concatenating and renaming")
        new_df = pd.concat(future_results)
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        logger.info("Data cleaning finished")
        return new_df

    def add_vix(self, data):
        """Add VIX data using multithreading."""
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.download_and_clean_data)
            cleaned_vix = future.result()

        vix = cleaned_vix[["timestamp", "close"]]
        merge_column = "date" if "date" in data.columns else "timestamp"
        vix = vix.rename(columns={"timestamp": merge_column, "close": "VIXY"})

        data = data.copy()
        data = data.merge(vix, on=merge_column)
        data = data.sort_values([merge_column, "tic"]).reset_index(drop=True)
        return data

    def download_and_clean_data(self):
        """Download and clean VIX data."""
        vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
        return self.clean_data(vix_df)

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            request_params = StockBarsRequest(
                symbol_or_symbols=[tic], timeframe=TimeFrame.Minute, limit=limit
            )

            barset = self.client.get_stock_bars(request_params).df
            # Reorganize the dataframes to be in original alpaca_trade_api structure
            # Rename the existing 'symbol' column if it exists
            if "symbol" in barset.columns:
                barset.rename(columns={"symbol": "symbol_old"}, inplace=True)

            # Now reset the index
            barset.reset_index(inplace=True)

            # Set 'timestamp' as the new index
            if "level_0" in barset.columns:
                barset.rename(columns={"level_0": "symbol"}, inplace=True)
            if "level_1" in barset.columns:
                barset.rename(columns={"level_1": "timestamp"}, inplace=True)
            barset.set_index("timestamp", inplace=True)

            # Reorder and rename columns as needed
            barset = barset[
                [
                    "close",
                    "high",
                    "low",
                    "trade_count",
                    "open",
                    "volume",
                    "vwap",
                    "symbol",
                ]
            ]

            barset["tic"] = tic
            barset = barset.reset_index()
            data_df = pd.concat([data_df, barset])

        data_df = data_df.reset_index(drop=True)
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

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        previous_close = 0.0
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
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
        request_params = StockBarsRequest(
            symbol_or_symbols="VIXY", timeframe=TimeFrame.Minute, limit=1
        )
        turb_df = self.client.get_stock_bars(request_params).df
        latest_turb = turb_df["close"].values
        return latest_price, latest_tech, latest_turb
