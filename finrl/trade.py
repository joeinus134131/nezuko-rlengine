from __future__ import annotations

from finrl.test import test


def trade(
    start_date: str,
    end_date: str,
    ticker_list: list[str],
    data_source: str,
    time_interval: str,
    technical_indicator_list: list[str],
    drl_lib: str,
    env: type,
    model_name: str,
    API_KEY: str,
    API_SECRET: str,
    API_BASE_URL: str,
    trade_mode: str = "backtesting",
    if_vix: bool = True,
    **kwargs,
) -> None:
    """Execute trading using a trained DRL model.

    Args:
        start_date: Start date for trading data (YYYY-MM-DD).
        end_date: End date for trading data (YYYY-MM-DD).
        ticker_list: List of stock ticker symbols.
        data_source: Data source identifier ('yahoofinance', 'alpaca', 'wrds').
        time_interval: Time interval for data ('1D', '1Min', etc.).
        technical_indicator_list: List of technical indicator names.
        drl_lib: DRL library used for training ('elegantrl', 'rllib', 'stable_baselines3').
        env: Environment class to use.
        model_name: Name of the model ('a2c', 'ppo', 'ddpg', 'td3', 'sac').
        API_KEY: Alpaca API key (required for paper trading).
        API_SECRET: Alpaca API secret (required for paper trading).
        API_BASE_URL: Alpaca API base URL (required for paper trading).
        trade_mode: Trading mode ('backtesting' or 'paper_trading').
        if_vix: Whether to include VIX data.
        **kwargs: Additional arguments for data processing and trading.

    Raises:
        ValueError: If trade_mode is not 'backtesting' or 'paper_trading'.
    """
    if trade_mode == "backtesting":
        # use test function for backtesting mode
        test(
            start_date,
            end_date,
            ticker_list,
            data_source,
            time_interval,
            technical_indicator_list,
            drl_lib,
            env,
            model_name,
            if_vix=if_vix,
            **kwargs,
        )

    elif trade_mode == "paper_trading":
        # Alpaca is optional for users who only run training or backtests.
        from finrl.integrations.alpaca import validate_alpaca_paper_connection
        from finrl.meta.env_stock_trading.env_stock_papertrading import (
            AlpacaPaperTrading,
        )

        # read parameters
        net_dim = kwargs.get("net_dimension", 2**7)  # dimension of NNs
        cwd = kwargs.get("cwd", "./" + str(model_name))  # current working directory
        state_dim = kwargs.get("state_dim")  # dimension of state/observations space
        action_dim = kwargs.get("action_dim")  # dimension of action space
        if state_dim is None or action_dim is None:
            raise ValueError(
                "state_dim and action_dim must be provided for paper trading."
            )

        # Validate endpoint and credentials with a read-only request before the
        # model starts its long-running loop or touches existing paper orders.
        API_BASE_URL, _ = validate_alpaca_paper_connection(
            API_KEY, API_SECRET, API_BASE_URL
        )

        # initialize paper trading env
        paper_trading = AlpacaPaperTrading(
            ticker_list,
            time_interval,
            drl_lib,
            model_name,
            cwd,
            net_dim,
            state_dim,
            action_dim,
            API_KEY,
            API_SECRET,
            API_BASE_URL,
            technical_indicator_list,
            turbulence_thresh=30,
            max_stock=1e2,
            latency=None,
        )

        # AlpacaPaperTrading.run()  # run paper trading
        paper_trading.run()
        # bug fix run is a instance function not static

    else:
        raise ValueError(
            "Invalid mode input! Please input either 'backtesting' or 'paper_trading'."
        )
