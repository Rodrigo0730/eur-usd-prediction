import yfinance as yf

data = yf.download('EURUSD=X', start='2004-01-01', end='2024-10-10', interval='1d')

data.to_csv('eur_usd_data.csv', index=True)
