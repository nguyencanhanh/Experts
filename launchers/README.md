# Launchers

These PowerShell launchers set the correct `TRADE_BOT_*` environment variables for each asset/profile before running `main.py`.

Examples:

```powershell
cd C:\work\canhanh\trade_bot\Experts

.\launchers\run_xau_base.ps1 -Mode train -YearsBack 2
.\launchers\run_xau_active.ps1 -Mode pipeline -YearsBack 2
.\launchers\run_btc_base.ps1 -Mode train -YearsBack 2
.\launchers\run_btc_active.ps1 -Mode pipeline -YearsBack 2
```

If your broker uses a different BTC symbol name, pass it explicitly:

```powershell
.\launchers\run_btc_base.ps1 -Symbol BTCUSDm -Mode train -YearsBack 2
.\launchers\run_btc_active.ps1 -Symbol BTCUSDm -Mode pipeline -YearsBack 2
```
