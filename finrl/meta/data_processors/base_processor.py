"""Base data processor with common methods shared between data source processors."""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd
import pandas_market_calendars as tc
from stockstats import StockDataFrame as Sdf

logger = logging.getLogger(__name__)


class BaseDataProcessor(ABC):
    """Abstract base class for data processors.

    Provides common functionality for downloading, cleaning, and processing
    financial data from different sources.

    Attributes:
        start: Start date for data.
        end: End date for data.
        time_interval: Time interval for data ('1D', '1Min', etc.).
    """

    def __init__(self):
        self.start: str = ""
        self.end: str = ""
        self.time_interval: str = ""

    @abstractmethod
    def download_data(
        self, ticker_list: list[str], start_date: str, end_date: str, time_interval: str
    ) -> pd.DataFrame:
        """Download data from the data source.

        Args:
            ticker_list: List of ticker symbols.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            time_interval: Time interval ('1D', '1Min', etc.).

        Returns:
            DataFrame with downloaded data.
        """
        pass

    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw data.

        Args:
            df: Raw DataFrame to clean.

        Returns:
            Cleaned DataFrame.
        """
        pass

    def add_technical_indicator(
        self,
        df: pd.DataFrame,
        tech_indicator_list: list[str],
    ) -> pd.DataFrame:
        """Add technical indicators using stockstats.

        Args:
            df: DataFrame with OHLCV data.
            tech_indicator_list: List of indicator names to add.

        Returns:
            DataFrame with technical indicators added.
        """
        logger.info("Adding technical indicators")
        df = df.copy()
        df = df.sort_values(by=["tic", "timestamp"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["timestamp"] = df[df.tic == unique_ticker[i]][
                        "timestamp"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], ignore_index=True
                    )
                except Exception as e:
                    logger.warning(f"Error calculating {indicator} for {unique_ticker[i]}: {e}")
            df = df.merge(
                indicator_df[["tic", "timestamp", indicator]],
                on=["tic", "timestamp"],
                how="left",
            )
        df = df.sort_values(by=["timestamp", "tic"])
        logger.info("Finished adding technical indicators")
        return df

    def calculate_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        """Calculate turbulence index based on price covariance.

        Args:
            data: DataFrame with OHLCV data.
            time_period: Lookback period for covariance calculation.

        Returns:
            DataFrame with turbulence index.
        """
        logger.info("Calculating turbulence index")
        df = data.copy()
        df_price_pivot = df.pivot(index="timestamp", columns="tic", values="close")
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.timestamp.unique()
        # A short smoke-test period may contain fewer observations than the
        # covariance lookback. In that case risk cannot yet be estimated, so
        # emit a correctly-sized zero warm-up series instead of creating a
        # length mismatch.
        start = min(time_period, len(unique_date))
        turbulence_index = [0] * start
        count = 0

        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"timestamp": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        """Add turbulence index to the DataFrame.

        Args:
            data: DataFrame with OHLCV data.
            time_period: Lookback period for turbulence calculation.

        Returns:
            DataFrame with turbulence column added.
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: list[str], if_vix: bool
    ) -> list[np.ndarray]:
        """Convert DataFrame to numpy arrays for RL training.

        Args:
            df: DataFrame with OHLCV and indicator data.
            tech_indicator_list: List of technical indicator column names.
            if_vix: Whether VIX data is included.

        Returns:
            Tuple of (price_array, tech_array, turbulence_array).
        """
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True

        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIXY"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )

        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start: str, end: str) -> list[str]:
        """Get list of trading days between start and end dates.

        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).

        Returns:
            List of trading day strings (YYYY-MM-DD).
        """
        nyse = tc.get_calendar("NYSE")
        df = nyse.date_range_htf("1D", pd.Timestamp(start), pd.Timestamp(end))
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])
        return trading_days

    def fill_nan_with_previous(self, tmp_df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values in price data with previous valid values.

        Args:
            tmp_df: DataFrame with potential NaN values.

        Returns:
            DataFrame with NaN values filled.
        """
        # Fill NaN on first row with first valid close
        if str(tmp_df.iloc[0]["close"]) == "nan":
            logger.info("NaN data on start date, filling with first valid data")
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

        # If still NaN, fill with 0
        if str(tmp_df.iloc[0]["close"]) == "nan":
            logger.warning("All prices are NaN. Filling with 0.")
            tmp_df.iloc[0] = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Fill remaining NaN with previous close
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

        return tmp_df
