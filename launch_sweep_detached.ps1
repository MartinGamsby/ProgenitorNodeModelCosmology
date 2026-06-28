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
    re-launch and it picks up where it stopped (per arm, per COMPLETED cell).

    RESUME GRANULARITY (important): the checkpoint is one fully-COMPLETED cell. A
    co-fit cell runs ~6-7 sims internally and a single 2000p cell takes minutes
    (cube26 ~3 min; virialized bounded ~30 min). If you kill MID-cell, that cell has
    no checkpoint yet and re-runs from scratch -- that is expected, not a resume bug.
    On a relaunch the run log prints a "[resume-info] N cell(s) already ... skipping"
    line per arm so you can SEE resume working.

    SINGLE INSTANCE: only ONE worker may run at a time. A bare relaunch while a worker
    is alive REFUSES (it does not stack a second racing worker -- the old failure mode
    where Stop-Process on the shell orphaned the python child and re-launching piled up
    concurrent sweeps corrupting the shared cache). Use -Stop to end the current run
    cleanly, or -Force to stop-then-relaunch.

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
    A ternary co-fit cell runs ~6-7 sims. Projected SEQUENTIALLY (72-cell core after the M grid
    was trimmed to {1,5,10,50,100,500}): CORE ~8 h + SEED ~7 h = ~15 h; +satellites ~+11 h => ~26 h.
    With -Parallel N this divides by ~N (minus merge/oversubscription overhead). ALWAYS re-run
    `python _calibrate_runtime.py` for current numbers (-> results/runtime_projection.csv).

.PARAMETER NoResume
    Pass -NoResume to force every cell to recompute (adds sweep.py's --no-resume).
    Default: resume (skip cells already in each arm's CSV).

.PARAMETER IncludeSatellites
    Pass -IncludeSatellites to ALSO run the secondary satellite shape studies
    (start_size / convergence / extent) after the core + seed arms. Default:
    CORE-ONLY (the 12 core arms + the seed arm).

.PARAMETER Stop
    Pass -Stop to cleanly END the running sweep: it kills the detached worker shell
    AND its (orphaned-on-Stop-Process) python child, then clears stale data/*.lock.
    This is the CORRECT way to stop -- do NOT `Stop-Process -Id <shell PID>`, which
    leaves the python child running. Exits without launching.

.PARAMETER Force
    Pass -Force to stop any already-running worker and relaunch fresh in one step
    (equivalent to -Stop followed by a normal launch).

.PARAMETER Parallel
    How many arms to run AT ONCE (default 1 = sequential). With -Parallel 2 or 3 the
    detached worker keeps that many `python sweep.py` children busy, pulling the next
    arm from the list whenever one finishes. Parallel runs set HMEA_CACHE_CONCURRENT=1
    so the shared metrics cache uses its concurrency-safe merge mode (read-merge-write
    under a short lock + atomic rename) instead of the single-writer lifetime lock.
    A single worker (default) keeps the faster exclusive cache. Keep this modest
    (2-3): each sim is already CPU-heavy, so oversubscribing cores wastes time.

.SECURITY
    The argument vector handed to python is STATIC: a fixed @("sweep.py","--config",
    <literal path>) array per arm, built only from the hardcoded arm lists in this
    file -- NO string-built command line, NO interpolation of untrusted input, NO
    network. The arm set is selected by SWITCHING between two hardcoded static arrays
    ($CoreArms and $CoreArms+$SatelliteArms), NOT by building paths at runtime, so the
    -IncludeSatellites flag cannot widen the set beyond these literals. The only
    optional flag (--no-resume) is a fixed literal gated by the boolean switch above.
    The worker body IS rendered as a here-string, but ONLY from this file's own
    literals (the single-quoted static arm paths); no external/untrusted value is
    interpolated. The -Stop / -Force process matching is read-only CIM CommandLine
    matching against fixed patterns. Keep it this way: do not concatenate a command
    string from untrusted input and do not glob the disk for the arm set.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1
    # launches the CORE (12 arms + seed) detached; prints the worker + python PIDs.

    powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1 -IncludeSatellites
    # core + seed THEN the start_size / convergence / extent satellites.

    powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1 -Stop
    # cleanly stop the running sweep (shell + python child) and clear stale locks.

    powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1 -Force
    # stop whatever is running and relaunch the core fresh.
#>
[CmdletBinding()]
param(
    [switch]$NoResume,
    [switch]$IncludeSatellites,
    [switch]$Stop,
    [switch]$Force,
    [ValidateRange(1, 8)]
    [int]$Parallel = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Repo root = this script's directory (fixed, no user input).
$RepoRoot = $PSScriptRoot
$ResultsDir = Join-Path $RepoRoot "results"
$LogDir = Join-Path $ResultsDir "logs"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ---------------------------------------------------------------------------
# Single-instance management. The detached worker is a powershell shell running
# core_v3_worker.ps1 that spawns `python sweep.py --config sweeps/...` children.
# Stop-Process on the shell alone ORPHANS the python child (the bug that let
# relaunches stack concurrent racing sweeps). We identify BOTH by CommandLine and
# stop the whole set, then clear the stale per-process cache locks a force-kill
# leaves behind. Matching is read-only CIM against fixed patterns (no untrusted input).
# ---------------------------------------------------------------------------
function Get-SweepWorkers {
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and (
            ($_.Name -eq 'python.exe'     -and $_.CommandLine -match 'sweep\.py\s+--config\s+sweeps/') -or
            ($_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'core_v3_worker\.ps1')
        )
    }
}

function Stop-SweepWorkers {
    $workers = @(Get-SweepWorkers)
    foreach ($w in $workers) {
        Write-Host ("   stopping {0} PID {1}" -f $w.Name, $w.ProcessId)
        try { Stop-Process -Id $w.ProcessId -Force -ErrorAction Stop } catch { Write-Host "     (already gone)" }
    }
    if ($workers.Count -gt 0) { Start-Sleep -Milliseconds 800 }
    # Clear stale per-process cache locks left by a force-kill (gitignored).
    Get-ChildItem (Join-Path $RepoRoot 'data') -Filter '*.lock' -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue }
    return $workers.Count
}

# -Stop: end the current run cleanly and exit (no launch).
if ($Stop) {
    Write-Host "Stopping any running core_v3 sweep worker(s)..."
    $n = Stop-SweepWorkers
    if ($n -eq 0) { Write-Host "  none were running." }
    Write-Host "Stopped $n process(es); cleared stale data/*.lock."
    return
}

# Single-instance guard: refuse to stack a second worker (unless -Force).
$existing = @(Get-SweepWorkers)
if ($existing.Count -gt 0) {
    if ($Force) {
        Write-Host "[-Force] stopping $($existing.Count) existing worker process(es) before relaunch..."
        Stop-SweepWorkers | Out-Null
    } else {
        Write-Host ""
        Write-Host "A core_v3 sweep is ALREADY running -- refusing to launch a second (racing) worker:"
        $existing | ForEach-Object { Write-Host ("   PID {0}  {1}" -f $_.ProcessId, $_.Name) }
        Write-Host ""
        Write-Host " Resume is automatic; that worker is still going. Check progress:"
        Write-Host "   Get-Content results/logs/core_v3_run_*.log -Tail 20"
        Write-Host " Stop it cleanly (shell + python child + locks):"
        Write-Host "   powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1 -Stop"
        Write-Host " Or stop-and-relaunch in one step:"
        Write-Host "   powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1 -Force"
        Write-Host ""
        return
    }
}

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

# $ExploreArms = the EXPLORATION (parameter-finding) sweeps that sweep the node
# mass-function distribution-size axis (vir_mass_spreads). Appended with the
# satellites under -IncludeSatellites. STATIC literal list (no disk glob).
$ExploreArms = @(
    "sweeps/explore_vir_spread.json",
    "sweeps/explore_vir_spread_hi.json"
)

# Select the run set by SWITCHING between the static arrays (core-only default).
if ($IncludeSatellites) {
    $Arms = $CoreArms + $SatelliteArms + $ExploreArms
} else {
    $Arms = $CoreArms
}

# The flag appended to EVERY arm call. Fixed literals only.
$ExtraArgs = @()
if ($NoResume) { $ExtraArgs = @("--no-resume") }

# ---------------------------------------------------------------------------
# The detached worker: a single -File script that loops the arms, calling
# python sweep.py per arm with a STATIC argument array. It logs to results/logs/,
# echoes the per-arm resume line so resume is VISIBLE, and keeps going if one arm
# fails. The arm list + flag are rendered as PowerShell literals (only our own
# static literals -> safe; each path single-quoted to avoid interpolation).
# ---------------------------------------------------------------------------
$ArmsLiteral = ($Arms | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ", "
$ExtraLiteral = ($ExtraArgs | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ", "

# Common header for both worker bodies. In parallel mode (Parallel > 1) the shared
# metrics cache must use concurrency-safe merge mode -> set HMEA_CACHE_CONCURRENT=1.
# A single worker keeps the faster exclusive cache (env unset).
$ConcurrentLine = if ($Parallel -gt 1) { "`$env:HMEA_CACHE_CONCURRENT = '1'" } else { "" }
$WorkerHeader = @"
Set-StrictMode -Version Latest
`$ErrorActionPreference = 'Continue'
`$env:PYTHONIOENCODING = 'utf-8'
$ConcurrentLine
Set-Location -LiteralPath '$($RepoRoot.Replace("'", "''"))'
`$arms = @($ArmsLiteral)
`$extra = @($ExtraLiteral)
`$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
`$summary = Join-Path 'results/logs' "core_v3_run_`$stamp.log"
"@

if ($Parallel -gt 1) {
    # POOL worker: keep $Parallel `python sweep.py` children busy, pull the next arm
    # from the list whenever one finishes. Each child gets a STATIC argument vector.
    $WorkerScript = $WorkerHeader + @"

"core_v3 detached PARALLEL run started `$(Get-Date -Format o) (PID `$PID, parallel=$Parallel)" | Out-File -FilePath `$summary -Encoding utf8
`$maxParallel = $Parallel
`$idx = 0
`$running = @{}   # process Id -> @{ cfg=...; out=...; proc=... }
while (`$idx -lt `$arms.Count -or `$running.Count -gt 0) {
    while (`$running.Count -lt `$maxParallel -and `$idx -lt `$arms.Count) {
        `$cfg = `$arms[`$idx]; `$idx++
        `$name = [System.IO.Path]::GetFileNameWithoutExtension(`$cfg)
        `$out = Join-Path 'results/logs' "`$name.out"
        `$err = Join-Path 'results/logs' "`$name.err"
        "[`$(Get-Date -Format o)] START `$cfg" | Tee-Object -FilePath `$summary -Append
        # STATIC argument vector: sweep.py + --config + the (literal) config path + flags.
        `$pyArgs = @('sweep.py', '--config', `$cfg) + `$extra
        `$p = Start-Process -FilePath 'python' -ArgumentList `$pyArgs -NoNewWindow -PassThru ``
            -RedirectStandardOutput `$out -RedirectStandardError `$err
        `$running[`$p.Id] = @{ cfg = `$cfg; out = `$out; proc = `$p }
    }
    Start-Sleep -Seconds 3
    foreach (`$id in @(`$running.Keys)) {
        `$info = `$running[`$id]
        if (`$info.proc.HasExited) {
            `$code = `$info.proc.ExitCode
            `$r = Select-String -Path `$info.out -Pattern '[resume]' -SimpleMatch -ErrorAction SilentlyContinue | Select-Object -First 1
            if (`$r) { "[`$(Get-Date -Format o)] [resume-info] `$(`$r.Line.Trim())" | Tee-Object -FilePath `$summary -Append }
            "[`$(Get-Date -Format o)] DONE  `$(`$info.cfg) (exit `$code)" | Tee-Object -FilePath `$summary -Append
            `$running.Remove(`$id)
        }
    }
}
"core_v3 detached run finished `$(Get-Date -Format o)" | Tee-Object -FilePath `$summary -Append
"@
} else {
    # SEQUENTIAL worker (single process; exclusive cache).
    $WorkerScript = $WorkerHeader + @"

"core_v3 detached run started `$(Get-Date -Format o) (PID `$PID)" | Out-File -FilePath `$summary -Encoding utf8
foreach (`$cfg in `$arms) {
    `$name = [System.IO.Path]::GetFileNameWithoutExtension(`$cfg)
    `$out = Join-Path 'results/logs' "`$name.out"
    `$err = Join-Path 'results/logs' "`$name.err"
    "[`$(Get-Date -Format o)] START `$cfg" | Tee-Object -FilePath `$summary -Append
    # STATIC argument vector: sweep.py + --config + the (literal) config path + flags.
    `$pyArgs = @('sweep.py', '--config', `$cfg) + `$extra
    & python `$pyArgs 1> `$out 2> `$err
    `$code = `$LASTEXITCODE
    # Surface the resume line (sweep.py prints '[resume] N cell(s) already ... skipping')
    # so a relaunch VISIBLY shows what it skipped instead of looking like a fresh start.
    `$r = Select-String -Path `$out -Pattern '[resume]' -SimpleMatch -ErrorAction SilentlyContinue | Select-Object -First 1
    if (`$r) { "[`$(Get-Date -Format o)] [resume-info] `$(`$r.Line.Trim())" | Tee-Object -FilePath `$summary -Append }
    "[`$(Get-Date -Format o)] DONE  `$cfg (exit `$code)" | Tee-Object -FilePath `$summary -Append
}
"core_v3 detached run finished `$(Get-Date -Format o)" | Tee-Object -FilePath `$summary -Append
"@
}

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

# Try to discover the python child PID(s) (spawned by the worker shortly after start)
# so the user can see the actual compute process(es), not just the shell.
$pyPids = @()
for ($i = 0; $i -lt 12 -and $pyPids.Count -lt $Parallel; $i++) {
    Start-Sleep -Milliseconds 500
    $pyPids = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -and $_.CommandLine -match 'sweep\.py\s+--config\s+sweeps/'
    } | Select-Object -ExpandProperty ProcessId)
}

if ($IncludeSatellites) { $Mode = "CORE + SEED + SATELLITES + EXPLORE(vir_mass_spread)" } else { $Mode = "CORE + SEED (core-only)" }
if ($Parallel -gt 1) { $Mode = "$Mode  [parallel=$Parallel, shared cache concurrency-safe]" }

Write-Host ""
Write-Host "=================================================================="
Write-Host " core_v3 sweep launched DETACHED."
Write-Host "   mode        : $Mode"
Write-Host "   worker shell : PID $($proc.Id)  ($WorkerPath)"
if ($pyPids.Count -gt 0) {
    Write-Host "   python child : $($pyPids.Count) running -- PID(s) $($pyPids -join ', ')"
} else {
    Write-Host "   python child : (starting...) -- see results/logs/core_v3_run_*.log"
}
Write-Host "   arms        : $($Arms.Count) (run in NN order)"
Write-Host "   per-arm log : results/logs/<arm>.out / .err"
Write-Host "   run log     : results/logs/core_v3_run_<stamp>.log"
Write-Host "   results     : results/ws1_sweep_*.csv (one CSV per arm)"
Write-Host ""
Write-Host " It survives a Claude restart / window close (orphaned process)."
Write-Host " Single-instance: a bare relaunch while this is running is REFUSED"
Write-Host " (no stacked racing workers). Resume is per COMPLETED cell."
Write-Host ""
Write-Host " RESUME (after a clean -Stop, or any time): just relaunch --"
Write-Host "   powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1"
Write-Host "   (add -IncludeSatellites to also run the satellite shape studies)"
Write-Host "   the run log shows a [resume-info] line per arm for what it skipped."
Write-Host ""
Write-Host " STOP the worker (shell + python child + clears stale locks):"
Write-Host "   powershell -ExecutionPolicy Bypass -File .\launch_sweep_detached.ps1 -Stop"
Write-Host "   (do NOT 'Stop-Process -Id $($proc.Id)' alone -- it orphans the python child)"
Write-Host ""
Write-Host " CHECK progress:"
Write-Host "   Get-Content results/logs/core_v3_run_*.log -Tail 20"
Write-Host "=================================================================="
