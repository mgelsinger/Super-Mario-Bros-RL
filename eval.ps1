# eval.ps1 - record Mario PPO rollouts to MP4 (Windows/conda, PowerShell-safe)
# Usage (PowerShell switches, not GNU-style):
#   .\eval.ps1                           # record default 30s clip
#   .\eval.ps1 -Seconds 30 -FPS 60       # control clip duration
#   .\eval.ps1 -Episodes 3               # kept for compatibility (unused for length)
#   .\eval.ps1 -Model runs\mario_ppo\ppo_mario_1000000_steps.zip
#   .\eval.ps1 -OutDir runs\mario_eval\myclip -OpenFolder
#   .\eval.ps1 -ForceDummy               # if subprocess vec envs complain on Windows
#   .\eval.ps1 -Quiet                    # suppress warnings/verbose prints
# Note: Do not pass "--record-only" to this script; it is added automatically.

param(
  [string]$EnvName   = 'smbrl',
  [string]$LogRoot   = 'runs\mario_ppo',
  [string]$Model     = '',                  # optional explicit checkpoint (.zip)
  [string]$OutDir    = '',                  # optional output folder
  [int]   $Episodes  = 1,                   # how many episodes to record
  [int]   $Seconds   = 30,                  # desired clip length in seconds
  [int]   $FPS       = 60,                  # assumed steps per second
  [switch]$Quiet,                            # suppress python warnings/verbose prints
  [switch]$ForceDummy,
  [switch]$OpenFolder
)

$ErrorActionPreference = 'Stop'

# Guard: if a GNU-style arg (e.g. --record-only) is given first, PS treats it
# as a positional argument and we'd try to `conda activate` it. Ignore it.
if ($EnvName -like '--*') {
  Write-Warning "Ignoring unexpected argument '$EnvName'. Use PowerShell switches like -Episodes, -Model, -OutDir."
  $EnvName = 'smbrl'
}

# Activate conda env
try {
  conda activate $EnvName | Out-Null
} catch {
  Write-Error "Couldn't activate conda env '$EnvName'. Make sure 'conda init powershell' has been run, then restart PowerShell."
  exit 1
}

# CUDA sanity (PowerShell here-string piped to python)
$pyInfo = @'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
'@
if (-not $Quiet) { $pyInfo | python - }  # runs the inline Python

# Ensure video deps are present (idempotent)
python -m pip install --quiet imageio imageio-ffmpeg | Out-Null

# Resolve checkpoint to use
$ckptPath = $null
if ($Model -and (Test-Path $Model)) {
  $ckptPath = (Resolve-Path $Model).Path
} else {
  $latest = Get-ChildItem -Path $LogRoot -Recurse -Filter 'ppo_mario_*_steps.zip' -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latest) {
    $final = Get-ChildItem -Path $LogRoot -Recurse -Filter 'ppo_mario_final.zip' -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($final) { $ckptPath = $final.FullName }
  } else {
    $ckptPath = $latest.FullName
  }
}
if (-not $ckptPath) {
  Write-Error "No checkpoint found in '$LogRoot' and no -Model provided."
  exit 1
}

# Output dir
if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $OutDir = Join-Path 'runs\mario_eval' $stamp
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Using checkpoint: $ckptPath"
Write-Host "Saving videos to: $OutDir"

# Compute desired video length in steps
$VideoLen = [int]($Seconds * $FPS)
if ($VideoLen -le 0) { $VideoLen = 1800 }

# Build python args (record-only evaluation)
$pyArgs = @(
  "train_ppo_sb3.py",
  "--record-only",
  "--resume", $ckptPath,
  "--eval-episodes", $Episodes,
  "--record-dir", $OutDir,
  "--video-length", $VideoLen
)
if ($ForceDummy) { $pyArgs += "--force-dummy" }

# Run recorder (optionally quiet)
if ($Quiet) {
  $env:PYTHONWARNINGS = 'ignore'
  python -W ignore @pyArgs
} else {
  python @pyArgs
}

if ($OpenFolder) {
  Start-Process $OutDir
}
