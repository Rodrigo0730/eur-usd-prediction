import yfinance as yf

data = yf.download('EURUSD=X', start='2015-01-01', end='2023-01-01', interval='1d')

data.to_csv('eur_usd_data.csv', index=True)
