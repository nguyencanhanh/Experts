param(
    [Parameter(Mandatory = $true)]
    [string]$Symbol,

    [Parameter(Mandatory = $true)]
    [string]$Profile,

    [ValidateSet("pipeline", "train", "backtest", "live", "paper")]
    [string]$Mode = "train",

    [int]$YearsBack = 2,

    [string]$ArtifactVersion = "v7"
)

$ErrorActionPreference = "Stop"

function Set-Or-RestoreEnv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [AllowNull()]
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        Remove-Item "Env:$Name" -ErrorAction SilentlyContinue
    }
    else {
        Set-Item "Env:$Name" $Value
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")

$previousEnv = @{
    TRADE_BOT_SYMBOL = $env:TRADE_BOT_SYMBOL
    TRADE_BOT_PROFILE = $env:TRADE_BOT_PROFILE
    TRADE_BOT_YEARS_BACK = $env:TRADE_BOT_YEARS_BACK
    TRADE_BOT_ARTIFACT_VERSION = $env:TRADE_BOT_ARTIFACT_VERSION
}

Push-Location $projectRoot
try {
    $env:TRADE_BOT_SYMBOL = $Symbol.Trim().ToUpper()
    $env:TRADE_BOT_PROFILE = $Profile.Trim().ToLower()
    $env:TRADE_BOT_YEARS_BACK = [string][Math]::Max($YearsBack, 1)
    $env:TRADE_BOT_ARTIFACT_VERSION = $ArtifactVersion.Trim()

    Write-Host "Running main.py | symbol=$($env:TRADE_BOT_SYMBOL) profile=$($env:TRADE_BOT_PROFILE) years_back=$($env:TRADE_BOT_YEARS_BACK) mode=$Mode"
    & python .\main.py --mode $Mode
}
finally {
    Pop-Location
    foreach ($entry in $previousEnv.GetEnumerator()) {
        Set-Or-RestoreEnv -Name $entry.Key -Value $entry.Value
    }
}
