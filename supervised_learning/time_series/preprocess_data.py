#!/usr/bin/env python3
"""Preprocess raw BTC exchange data into hourly features."""
import sys

import numpy as np
import pandas as pd


RAW_FEATURES = [
    'open', 'high', 'low', 'close',
    'volume_btc', 'volume_currency', 'weighted_price'
]


def _normalize_columns(frame):
    """Normalize column names to lowercase snake case."""
    frame = frame.copy()
    frame.columns = [
        column.strip().lower().replace('(', '').replace(')', '').replace(
            ' ', '_').replace('-', '_').replace('/', '_')
        for column in frame.columns
    ]
    return frame


def _load_exchange_data(path):
    """Load and clean one exchange dataset."""
    frame = pd.read_csv(path)
    frame = _normalize_columns(frame)

    if 'timestamp' not in frame.columns:
        raise ValueError('missing timestamp column in {}'.format(path))

    frame['timestamp'] = pd.to_datetime(frame['timestamp'], unit='s')
    frame = frame.set_index('timestamp').sort_index()
    frame = frame.loc[:, [column for column in RAW_FEATURES if column in frame.columns]]
    frame = frame.apply(pd.to_numeric, errors='coerce')
    frame = frame.dropna()
    return frame


def _aggregate_hourly(frame):
    """Aggregate a minute-level exchange frame to hourly candles."""
    aggregations = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume_btc': 'sum',
        'volume_currency': 'sum',
        'weighted_price': 'mean'
    }
    available = {
        column: method for column, method in aggregations.items()
        if column in frame.columns
    }
    hourly = frame.resample('1h').agg(available)
    hourly = hourly.dropna()
    return hourly


def preprocess_data(coinbase_path='coinbase.csv',
                    bitstamp_path='bitstamp.csv',
                    output_path='btc_hourly.csv'):
    """Preprocess raw BTC data and save an hourly CSV."""
    coinbase = _aggregate_hourly(_load_exchange_data(coinbase_path))
    bitstamp = _aggregate_hourly(_load_exchange_data(bitstamp_path))

    combined = coinbase.join(bitstamp, how='inner', lsuffix='_coinbase',
                             rsuffix='_bitstamp')

    hourly = pd.DataFrame(index=combined.index)
    hourly['open'] = combined[['open_coinbase', 'open_bitstamp']].mean(axis=1)
    hourly['high'] = combined[['high_coinbase', 'high_bitstamp']].mean(axis=1)
    hourly['low'] = combined[['low_coinbase', 'low_bitstamp']].mean(axis=1)
    hourly['close'] = combined[['close_coinbase', 'close_bitstamp']].mean(axis=1)
    hourly['volume_btc'] = combined[['volume_btc_coinbase',
                                     'volume_btc_bitstamp']].sum(axis=1)
    hourly['volume_currency'] = combined[['volume_currency_coinbase',
                                          'volume_currency_bitstamp']].sum(axis=1)
    hourly['weighted_price'] = combined[['weighted_price_coinbase',
                                         'weighted_price_bitstamp']].mean(axis=1)

    hourly = hourly.reset_index().rename(columns={'index': 'timestamp'})
    hourly.to_csv(output_path, index=False)
    return output_path, hourly.shape[0]


def main(argv=None):
    """Run the preprocessing pipeline from the command line."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (0, 2, 3):
        raise SystemExit(
            'Usage: ./preprocess_data.py [coinbase.csv bitstamp.csv [output.csv]]'
        )

    if len(args) == 0:
        output_path, count = preprocess_data()
    elif len(args) == 2:
        output_path, count = preprocess_data(args[0], args[1])
    else:
        output_path, count = preprocess_data(args[0], args[1], args[2])

    print('Saved {} hourly rows to {}'.format(count, output_path))


if __name__ == '__main__':
    main()
