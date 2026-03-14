# Colab

This project now includes an offline entrypoint for Google Colab:

`main_colab.py`

It does not use MetaTrader5. Instead, it reads uploaded CSV files for:

- `M1`
- `M5`
- `M15`
- `H1`

You can export those CSV files directly from your Windows MT5 machine with:

```powershell
python .\export_mt5_offline.py --symbol XAUUSD --years-back 1 --output-dir offline_exports
python .\export_mt5_offline.py --symbol BTCUSDm --output-symbol BTCUSD --years-back 1 --output-dir offline_exports
```

## Expected CSV names

Put your files into one folder and use names like:

- `XAUUSD_M1.csv`
- `XAUUSD_M5.csv`
- `XAUUSD_M15.csv`
- `XAUUSD_H1.csv`

or:

- `BTCUSD_M1.csv`
- `BTCUSD_M5.csv`
- `BTCUSD_M15.csv`
- `BTCUSD_H1.csv`

The loader also accepts simpler names like `M1.csv`, `M5.csv`, `M15.csv`, `H1.csv` if there is only one symbol in the folder.

## Required columns

Each CSV should contain at least:

- `time`
- `open`
- `high`
- `low`
- `close`

Optional columns:

- `tick_volume`
- `spread`

If `spread` is missing, the offline backtest will assume `0`.

## Colab setup

```python
!git clone <your-repo-url>
%cd /content/Experts
!pip install -r requirements-colab.txt
```

Upload your CSV files into a folder such as `/content/data`.

## Train only

```python
!python main_colab.py --symbol XAUUSD --profile xau_active --mode train --years-back 1 --data-dir /content/data
```

## Full pipeline

```python
!python main_colab.py --symbol XAUUSD --profile xau_active --mode pipeline --years-back 1 --data-dir /content/data
```

For BTC active:

```python
!python main_colab.py --symbol BTCUSD --profile btc_active --mode pipeline --years-back 1 --data-dir /content/data
```

## Backtest only using saved artifacts

```python
!python main_colab.py --symbol XAUUSD --profile xau_active --mode backtest --years-back 1 --data-dir /content/data
```

## Optional symbol spec JSON

If you want more accurate point/tick/volume settings for backtest sizing, create a JSON file like:

```json
{
  "digits": 2,
  "point": 0.01,
  "contract_size": 100.0,
  "trade_tick_size": 0.01,
  "trade_tick_value": 1.0,
  "volume_min": 0.01,
  "volume_max": 100.0,
  "volume_step": 0.01
}
```

Then pass it with:

```python
!python main_colab.py --symbol XAUUSD --profile xau_active --mode pipeline --years-back 1 --data-dir /content/data --symbol-spec /content/xauusd_spec.json
```

Sample spec files are already included in this repo:

- `symbol_specs/xauusd.json`
- `symbol_specs/btcusd.json`
