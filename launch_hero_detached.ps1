<#
.SYNOPSIS
    Launch the 7 high-resolution HERO runs DETACHED + in PARALLEL (survives a Claude
    restart / window close), each a single `python _hero_run.py <config>` child.

.DESCRIPTION
    The hero configs (sweeps/hero/NN_*.json) are known-good cells (v8-confirmed best-observer
    < LCDM) at 100k particles / 3000 steps (converged). Barnes-Hut is single-threaded, so the
    7 children run truly in parallel on the 8 cores (~6 h wall). Each writes
    results/hero/<tag>.npz (snapshots for images) + <tag>_result.json (chi2 numbers) and a
    per-run log in results/logs/hero_<tag>.out/.err.

    Like launch_sweep_detached.ps1 this ORPHANS the work via Start-Process. The config list is
    a STATIC literal array (no disk glob, no untrusted input); each child gets a STATIC
    argument vector @('_hero_run.py', <literal path>).

.PARAMETER Stop
    Kill any running hero children (matched by `_hero_run.py` in the CommandLine) and exit.
#>
[CmdletBinding()]
param([switch]$Stop)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$LogDir = Join-Path $RepoRoot "results\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "results\hero") | Out-Null

function Get-HeroProcs {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match '_hero_run\.py'
    }
}

if ($Stop) {
    $procs = @(Get-HeroProcs)
    foreach ($p in $procs) { Write-Host ("stopping PID {0}" -f $p.ProcessId); try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {} }
    Write-Host "stopped $($procs.Count) hero process(es)."
    return
}

$existing = @(Get-HeroProcs)
if ($existing.Count -gt 0) {
    Write-Host "Hero runs already in progress ($($existing.Count)). Use -Stop first, or wait."
    $existing | ForEach-Object { Write-Host ("   PID {0}" -f $_.ProcessId) }
    return
}

# STATIC literal config list (repo-relative). Nothing here comes from user input.
$Configs = @(
    "sweeps/hero/01_M300_S20_sig6.json",
    "sweeps/hero/02_M400_S22_sig5.json",
    "sweeps/hero/03_M200_S20_sig7.json",
    "sweeps/hero/04_M300_S20_sig5.json",
    "sweeps/hero/05_M400_S20_sig5.json",
    "sweeps/hero/06_M300_S22_sig7.json",
    "sweeps/hero/07_M200_S22_sig6.json"
)

$pids = @()
foreach ($cfg in $Configs) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($cfg)
    $out = Join-Path 'results\logs' "hero_$name.out"
    $err = Join-Path 'results\logs' "hero_$name.err"
    # STATIC argument vector: _hero_run.py + the (literal) config path.
    $p = Start-Process -FilePath 'python' -ArgumentList @('_hero_run.py', $cfg) `
        -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $pids += $p.Id
    Start-Sleep -Milliseconds 400   # stagger numba/IC startup a touch
}

Write-Host "=================================================================="
Write-Host " 7 HERO runs launched DETACHED (100k particles / 3000 steps each)."
Write-Host "   python PIDs : $($pids -join ', ')"
Write-Host "   per-run log : results/logs/hero_<tag>.out / .err"
Write-Host "   outputs     : results/hero/<tag>.npz + <tag>_result.json"
Write-Host "   ~6 h wall (Barnes-Hut single-threaded, 7 on 8 cores)."
Write-Host " Survives a Claude restart / window close. Stop with:"
Write-Host "   powershell -ExecutionPolicy Bypass -File .\launch_hero_detached.ps1 -Stop"
Write-Host "=================================================================="
