param(
    [ValidateSet("pipeline", "train", "backtest", "live", "paper")]
    [string]$Mode = "train",

    [int]$YearsBack = 1,

    [string]$Symbol = "BTCUSD"
)

$launcher = Join-Path $PSScriptRoot "Invoke-TradeBotPreset.ps1"
& $launcher -Symbol $Symbol -Profile "btc_base" -Mode $Mode -YearsBack $YearsBack
