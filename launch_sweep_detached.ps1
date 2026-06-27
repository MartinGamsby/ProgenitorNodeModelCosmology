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

    This script ORPHANS the work from the launching shell via Start-Process, so a
    Claude update / window close cannot kill it (a prior run_in_background sweep was
    lost to a restart; the detached one survived). The detached worker runs the arms
    in sequence; one arm failing does not abort the rest.

    Run order == the NN prefix on the arm filenames: the three GRF geometries
    (cube26 control, virialized Option A lattice, Option B gradient) x three
    treatments (01-09), then the cube26 uniform_sphere control x three treatments
    (10-12), then the seed sweep (13).

    RUNTIME (R5 calibration): at 2000p/546 with the bounded+substep treatment, a
    single sim costs ~99 s (cube26) / ~239 s (virialized, vir_n_nodes=150) on this
    machine; a ternary co-fit cell runs several such sims. Budget a MULTI-DAY run.

.PARAMETER NoResume
    Pass -NoResume to force every cell to recompute (adds sweep.py's --no-resume).
    Default: resume (skip cells already in each arm's CSV).

.SECURITY
    The argument vector handed to python is STATIC: a fixed @("sweep.py","--config",
    <literal path>) array per arm, built only from the hardcoded arm list in this
    file -- NO string-built command line, NO interpolation of untrusted input, NO
    network. The only optional flag (--no-resume) is a fixed literal gated by the
    boolean switch above. Keep it this way: do not concatenate a command string.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1
    # launches detached; prints the worker PID and the resume command.

    powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1 -NoResume
    # same, but recomputes every cell from scratch.
#>
[CmdletBinding()]
param(
    [switch]$NoResume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Repo root = this script's directory (fixed, no user input).
$RepoRoot = $PSScriptRoot
$ResultsDir = Join-Path $RepoRoot "results"
$LogDir = Join-Path $ResultsDir "logs"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Sweep arms, in run order. STATIC literal list (repo-relative). The detached
# worker iterates exactly these; nothing here comes from user input.
$Arms = @(
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

Write-Host ""
Write-Host "=================================================================="
Write-Host " core_v3 sweep launched DETACHED."
Write-Host "   worker PID : $($proc.Id)"
Write-Host "   worker     : $WorkerPath"
Write-Host "   arms       : $($Arms.Count) (run in NN order)"
Write-Host "   per-arm log: results/logs/<arm>.out / .err"
Write-Host "   run log    : results/logs/core_v3_run_<stamp>.log"
Write-Host "   results    : results/ws1_sweep_core3_*.csv (one CSV per arm)"
Write-Host ""
Write-Host " It survives a Claude restart / window close (orphaned process)."
Write-Host ""
Write-Host " RESUME (just relaunch -- sweep.py skips done cells per arm):"
Write-Host "   powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1"
Write-Host ""
Write-Host " STOP the worker:"
Write-Host "   Stop-Process -Id $($proc.Id)"
Write-Host ""
Write-Host " CHECK progress:"
Write-Host "   Get-Content results/logs/core_v3_run_*.log -Tail 20"
Write-Host "=================================================================="
