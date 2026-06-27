<#
.SYNOPSIS
    Launch the comparison_v2 sweep family DETACHED and RESUMABLE so a multi-day run
    survives a Claude restart / terminal close.

.DESCRIPTION
    The comparison_v2 family (sweeps/comparison_v2/NN_*.json) isolates ONE variable
    per arm against the cube26/no-softening control (item 9 attribution). Each arm is
    a separate single-driver sweep that writes results/ws1_sweep_<tag>.csv, and
    sweep.py CHECKPOINTS every finished cell to that CSV and SKIPS done cells on a
    re-run -- so the whole family is resumable: re-launch and it picks up where it
    stopped (per arm, per cell).

    This script ORPHANS the work from the launching shell via Start-Process, so a
    Claude update / window close cannot kill it (a prior run_in_background sweep was
    lost to a restart; the detached one survived). The detached worker runs the arms
    in sequence; one arm failing does not abort the rest.

    Run order == the NN prefix on the arm filenames (01 control first, then the
    softening / Option-A / Option-B / bounded / extent / co-fit / start-size /
    convergence-ladder arms).

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
    "sweeps/comparison_v2/01_cube26_baseline.json",
    "sweeps/comparison_v2/02_cube26_soft1.json",
    "sweeps/comparison_v2/03_cube26_bounded.json",
    "sweeps/comparison_v2/04_virA_soft0.json",
    "sweeps/comparison_v2/05_virA_soft1.json",
    "sweeps/comparison_v2/06_virB_soft0.json",
    "sweeps/comparison_v2/07_virB_soft1.json",
    "sweeps/comparison_v2/08_virA_bounded.json",
    "sweeps/comparison_v2/09_virA_extentcouple.json",
    "sweeps/comparison_v2/10_cube26_ternary.json",
    "sweeps/comparison_v2/11_virA_ternary.json",
    "sweeps/comparison_v2/12_virA_ss05.json",
    "sweeps/comparison_v2/13_virA_ss08.json",
    "sweeps/comparison_v2/14_virA_ss12.json",
    "sweeps/comparison_v2/15_virA_ss15.json",
    "sweeps/comparison_v2/16_virA_ss20.json",
    "sweeps/comparison_v2/17_ladder_1000p_273.json",
    "sweeps/comparison_v2/18_ladder_2000p_546.json",
    "sweeps/comparison_v2/19_ladder_4000p_1092.json"
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
`$summary = Join-Path 'results/logs' "comparison_v2_run_`$stamp.log"
"comparison_v2 detached run started `$(Get-Date -Format o)" | Out-File -FilePath `$summary -Encoding utf8
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
"comparison_v2 detached run finished `$(Get-Date -Format o)" | Tee-Object -FilePath `$summary -Append
"@

# Persist the worker to a file so Start-Process launches it cleanly (and it is
# auditable). Written under results/logs (gitignored).
$WorkerPath = Join-Path $LogDir "comparison_v2_worker.ps1"
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
Write-Host " comparison_v2 sweep launched DETACHED."
Write-Host "   worker PID : $($proc.Id)"
Write-Host "   worker     : $WorkerPath"
Write-Host "   arms       : $($Arms.Count) (run in NN order)"
Write-Host "   per-arm log: results/logs/<arm>.out / .err"
Write-Host "   run log    : results/logs/comparison_v2_run_<stamp>.log"
Write-Host "   results    : results/ws1_sweep_cmp2_*.csv (one CSV per arm)"
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
Write-Host "   Get-Content results/logs/comparison_v2_run_*.log -Tail 20"
Write-Host "=================================================================="
