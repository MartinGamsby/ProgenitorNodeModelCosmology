<#
.SYNOPSIS
    Launch the core_v3 sweep family DETACHED and RESUMABLE so a multi-day run
    survives a Claude restart / terminal close.

.DESCRIPTION
    The core_v3 family (sweeps/core_v3/NN_*.json) is the redesigned headline
    comparison that SUPERSEDES comparison_v2: FEWER but HIGHER-quality sims
    (2000 particles / 546 steps, M down to 1, S co-fit [3..35] ternary, GRF + a
    uniform_sphere control on cube26). The 12 core arms isolate
    cube-vs-virialized-vs-softening cleanly at MATCHED close-range treatments
    {none / bounded+substep / Plummer 1 Gpc}; arm 13 is the B3a geometry-seed sweep
    (5 virialized realizations). Each arm is a separate single-driver sweep that
    writes results/ws1_sweep_<tag>.csv, and sweep.py CHECKPOINTS every finished cell
    to that CSV and SKIPS done cells on a re-run -- so the whole family is resumable:
    re-launch and it picks up where it stopped (per arm, per cell).

    CORE-ONLY by DEFAULT. The 12 core arms + the seed arm are the HEADLINE. The
    secondary SATELLITE shape studies (sweeps/satellite_*/) are run ONLY when you
    pass -IncludeSatellites:
      - satellite_startsize/ (6 arms): start_size_scale in {0.5..2.0} (PF10 lever)
      - satellite_convergence/ (3 arms): N in {1000,2000,4000} on ONE cube26 cell
      - satellite_extent/ (3 arms): vir_extent {1.0,1.5,2.0} coupled to node count
        (150/506/1200 nodes, PF14)
    The satellites use a lower particle_count (1000) where N is NOT the variable, so
    they are a SHAPE study, not a converged chi2 band -- see each _manifest.json.

    This script ORPHANS the work from the launching shell via Start-Process, so a
    Claude update / window close cannot kill it (a prior run_in_background sweep was
    lost to a restart; the detached one survived). The detached worker runs the arms
    in sequence; one arm failing does not abort the rest.

    Run order == the NN prefix on the arm filenames: the three GRF geometries
    (cube26 control, virialized Option A lattice, Option B gradient) x three
    treatments (01-09), then the cube26 uniform_sphere control x three treatments
    (10-12), then the seed sweep (13). With -IncludeSatellites the start_size,
    convergence, and extent satellite arms run AFTER the core+seed.

    RUNTIME -- run `python _calibrate_runtime.py` FIRST to print the projected hours
    per arm + grand total on THIS machine (NO blind launch). Measured s/sim at
    2000p/546 (cache bypassed): cube26 none/plummer ~23 s, bounded+substep ~88 s;
    virialized-A (150 nodes) none/plummer ~30 s, bounded+substep ~243 s (the substep
    dominates); virialized-B (gradient) bounded ~65 s (fewer substep refinements).
    A ternary co-fit cell runs ~6-7 sims. Projected on this machine: CORE ~9.5 h +
    SEED ~7 h = ~16.5 h (~0.7 day); +satellites ~+11 h => GRAND TOTAL ~27 h (~1.1 day).
    (See results/runtime_projection.csv -- re-run the calibrator for current numbers.)

.PARAMETER NoResume
    Pass -NoResume to force every cell to recompute (adds sweep.py's --no-resume).
    Default: resume (skip cells already in each arm's CSV).

.PARAMETER IncludeSatellites
    Pass -IncludeSatellites to ALSO run the secondary satellite shape studies
    (start_size / convergence / extent) after the core + seed arms. Default:
    CORE-ONLY (the 12 core arms + the seed arm).

.SECURITY
    The argument vector handed to python is STATIC: a fixed @("sweep.py","--config",
    <literal path>) array per arm, built only from the hardcoded arm lists in this
    file -- NO string-built command line, NO interpolation of untrusted input, NO
    network. The arm set is selected by SWITCHING between two hardcoded static arrays
    ($CoreArms and $CoreArms+$SatelliteArms), NOT by building paths at runtime, so the
    -IncludeSatellites flag cannot widen the set beyond these literals. The only
    optional flag (--no-resume) is a fixed literal gated by the boolean switch above.
    Keep it this way: do not concatenate a command string and do not glob the disk.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1
    # launches the CORE (12 arms + seed) detached; prints the worker PID + resume cmd.

    powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1 -IncludeSatellites
    # core + seed THEN the start_size / convergence / extent satellites.

    powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1 -NoResume
    # same as default (core-only), but recomputes every cell from scratch.
#>
[CmdletBinding()]
param(
    [switch]$NoResume,
    [switch]$IncludeSatellites
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Repo root = this script's directory (fixed, no user input).
$RepoRoot = $PSScriptRoot
$ResultsDir = Join-Path $RepoRoot "results"
$LogDir = Join-Path $ResultsDir "logs"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Sweep arms, in run order. STATIC literal lists (repo-relative). The detached
# worker iterates exactly these; nothing here comes from user input.
#
# $CoreArms = the HEADLINE (12 core arms + the seed arm). This is the default set.
$CoreArms = @(
    "sweeps/core_v3/01_cube26_grf_soft0.json",
    "sweeps/core_v3/02_cube26_grf_bounded.json",
    "sweeps/core_v3/03_cube26_grf_plummer1.json",
    "sweeps/core_v3/04_virA_grf_soft0.json",
    "sweeps/core_v3/05_virA_grf_bounded.json",
    "sweeps/core_v3/06_virA_grf_plummer1.json",
    "sweeps/core_v3/07_virB_grf_soft0.json",
    "sweeps/core_v3/08_virB_grf_bounded.json",
    "sweeps/core_v3/09_virB_grf_plummer1.json",
    "sweeps/core_v3/10_cube26_uniform_soft0.json",
    "sweeps/core_v3/11_cube26_uniform_bounded.json",
    "sweeps/core_v3/12_cube26_uniform_plummer1.json",
    "sweeps/core_v3/13_virA_grf_bounded_seedsweep.json"
)

# $SatelliteArms = the SECONDARY shape studies, appended ONLY with -IncludeSatellites.
# STATIC literal list (no disk glob, no runtime path building -> the flag cannot widen
# the set beyond these literals).
$SatelliteArms = @(
    "sweeps/satellite_startsize/01_startsize_0p5.json",
    "sweeps/satellite_startsize/02_startsize_0p8.json",
    "sweeps/satellite_startsize/03_startsize_1p0.json",
    "sweeps/satellite_startsize/04_startsize_1p2.json",
    "sweeps/satellite_startsize/05_startsize_1p5.json",
    "sweeps/satellite_startsize/06_startsize_2p0.json",
    "sweeps/satellite_convergence/01_conv_1000p.json",
    "sweeps/satellite_convergence/02_conv_2000p.json",
    "sweeps/satellite_convergence/03_conv_4000p.json",
    "sweeps/satellite_extent/01_extent_1p0.json",
    "sweeps/satellite_extent/02_extent_1p5.json",
    "sweeps/satellite_extent/03_extent_2p0.json"
)

# Select the run set by SWITCHING between the two static arrays (core-only default).
if ($IncludeSatellites) {
    $Arms = $CoreArms + $SatelliteArms
} else {
    $Arms = $CoreArms
}

# The flag appended to EVERY arm call. Fixed literals only.
$ExtraArgs = @()
if ($NoResume) { $ExtraArgs = @("--no-resume") }

# ---------------------------------------------------------------------------
# The detached worker: a single -Command script block that loops the arms,
# calling python sweep.py per arm with a STATIC argument array. It logs to
# results/logs/ and keeps going if one arm fails.
# Pass the arm list + flag in via a here-string-free, array-literal $args block
# that we render as PowerShell source (still static: only our own literals).
# ---------------------------------------------------------------------------

# Render the arm array + extra-args as PowerShell literals for the child. These
# come ONLY from the static $Arms / $ExtraArgs above (no untrusted input), so the
# rendered source is safe; we single-quote each path to avoid any interpolation.
$ArmsLiteral = ($Arms | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ", "
$ExtraLiteral = ($ExtraArgs | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ", "

$WorkerScript = @"
Set-StrictMode -Version Latest
`$ErrorActionPreference = 'Continue'
`$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath '$($RepoRoot.Replace("'", "''"))'
`$arms = @($ArmsLiteral)
`$extra = @($ExtraLiteral)
`$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
`$summary = Join-Path 'results/logs' "core_v3_run_`$stamp.log"
"core_v3 detached run started `$(Get-Date -Format o)" | Out-File -FilePath `$summary -Encoding utf8
foreach (`$cfg in `$arms) {
    `$name = [System.IO.Path]::GetFileNameWithoutExtension(`$cfg)
    `$out = Join-Path 'results/logs' "`$name.out"
    `$err = Join-Path 'results/logs' "`$name.err"
    "[`$(Get-Date -Format o)] START `$cfg" | Tee-Object -FilePath `$summary -Append
    # STATIC argument vector: sweep.py + --config + the (literal) config path + flags.
    `$pyArgs = @('sweep.py', '--config', `$cfg) + `$extra
    & python `$pyArgs 1> `$out 2> `$err
    "[`$(Get-Date -Format o)] DONE  `$cfg (exit `$LASTEXITCODE)" | Tee-Object -FilePath `$summary -Append
}
"core_v3 detached run finished `$(Get-Date -Format o)" | Tee-Object -FilePath `$summary -Append
"@

# Persist the worker to a file so Start-Process launches it cleanly (and it is
# auditable). Written under results/logs (gitignored).
$WorkerPath = Join-Path $LogDir "core_v3_worker.ps1"
$WorkerScript | Out-File -FilePath $WorkerPath -Encoding utf8

# STATIC argument vector for the detached PowerShell host: fixed flags + the
# worker script path (our own file, not user input).
$psArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-WindowStyle", "Hidden",
    "-File", $WorkerPath
)

$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList $psArgs `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -PassThru

if ($IncludeSatellites) { $Mode = "CORE + SEED + SATELLITES" } else { $Mode = "CORE + SEED (core-only)" }

Write-Host ""
Write-Host "=================================================================="
Write-Host " core_v3 sweep launched DETACHED."
Write-Host "   mode       : $Mode"
Write-Host "   worker PID : $($proc.Id)"
Write-Host "   worker     : $WorkerPath"
Write-Host "   arms       : $($Arms.Count) (run in NN order)"
Write-Host "   per-arm log: results/logs/<arm>.out / .err"
Write-Host "   run log    : results/logs/core_v3_run_<stamp>.log"
Write-Host "   results    : results/ws1_sweep_*.csv (one CSV per arm)"
Write-Host ""
Write-Host " It survives a Claude restart / window close (orphaned process)."
Write-Host ""
Write-Host " PROJECT RUNTIME FIRST (no blind launch):"
Write-Host "   python _calibrate_runtime.py   # -> results/runtime_projection.csv"
Write-Host ""
Write-Host " RESUME (just relaunch -- sweep.py skips done cells per arm):"
Write-Host "   powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1"
Write-Host "   (add -IncludeSatellites to also run the satellite shape studies)"
Write-Host ""
Write-Host " STOP the worker:"
Write-Host "   Stop-Process -Id $($proc.Id)"
Write-Host ""
Write-Host " CHECK progress:"
Write-Host "   Get-Content results/logs/core_v3_run_*.log -Tail 20"
Write-Host "=================================================================="
