from .utils import load_dataset as _load_dataset


# Load FOREX datasets
FOREX_EURUSD_RENKO = _load_dataset('renko_full_data_81', 'datetime')

# Load Stocks datasets
STOCKS_GOOGL = _load_dataset('STOCKS_GOOGL', 'Date')
