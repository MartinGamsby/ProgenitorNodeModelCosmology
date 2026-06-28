"""
Observer-from-a-Particle Expansion / Distance Kernel  (PROTOTYPE — items 5 / D)

Pure-function module (no I/O, no plotting, no simulation imports) that answers
the user's hypothesis, framed as CORRECT INFERENCE rather than cherry-picking:

    "We are a RANDOM observer, not in the centre. Pantheon+ (the data) TELLS us
     WHERE we are. So compute mu(z) from the viewpoint of EACH particle, not the
     cloud centre, and ask: (i) DOES a Pantheon-matching observer EXIST in this
     universe, and (ii) WHAT FRACTION of observers see Pantheon-like expansion?"

Epistemology (the point the user insisted on): we occupy a random vantage and the
SNe data localise it, so SELECTING the best-matching observer is not a cheat — it
is the inference the data licenses. The honest question is therefore not "is the
best a fluke?" but TWO questions: does a viable observer EXIST (best chi2/dof), and
how GENERIC is that vantage (the FRACTION of observers at/below a reference fit).
A large fraction = a Pantheon-like view is a generic vantage; a small fraction =
our vantage is fine-tuned. We report the fraction so fine-tuning stays visible.

The default a(t) -> mu(z) pipeline (cosmo/sim_distance.py, consuming a_curve from
cosmo/factories.run_external_node_simulation) measures expansion as the RMS radius
of the inner observable cloud about its CENTER OF MASS. An observer sitting ON a
particle p, off-centre, sees a DIFFERENT and generally ANISOTROPIC expansion.
This module builds, for a chosen observer particle p, an inferred scale-factor
history a_p(t), then hands it to the SAME authoritative a(t) -> mu(z) -> chi2 path
the sweep uses (sim_to_distance_modulus + evaluate_precomputed). Nothing here
changes the default centre-based result; it is a pure ADD-ON analysis.

------------------------------------------------------------------------------
What does "expansion from particle p" MEAN? (the design decision — stated)
------------------------------------------------------------------------------
The handover flags this as needing care. There is no single right answer, so we
implement TWO physically defensible definitions and COMPARE them. Both reduce to
the existing centre-based a(t) in the appropriate limit (see invariants below).

  1. ``local_rms`` — LOCAL-NEIGHBOUR RMS GROWTH.
     a_p(t) = RMS distance of p's neighbour SET about p, normalized to 1 today.
     The neighbour SET is fixed ONCE (by k-nearest-neighbours of p in the TODAY
     snapshot, or "all" particles) and then TRACKED by identity across time, so
     a_p(t) measures how that fixed patch of universe grows around p — it is a
     Lagrangian local expansion. Robust, local, anisotropy-aware. This is the
     local analogue of the centre-based RMS measure.

       * Centre-observer limit: with the COM treated as the "particle" and the
         neighbour set = ALL particles, a_p(t) is IDENTICALLY the centre-based
         RMS a(t) (RMS about the mean). This is the reproduction invariant the
         unit tests pin.

  2. ``hubble_flow`` — RADIAL HUBBLE FLOW ABOUT p.
     Per snapshot, fit the local Hubble slope of p's neighbours about p,
         H_p(t) = sum(v_r * r) / sum(r^2)      [s^-1],   v_r = (v - v_p).rhat
     exactly as cosmo/anisotropy.hubble_dipole computes its GLOBAL slope (here
     centred on p and its peculiar velocity instead of the COM). Integrate H_p
     to a scale factor,
         a_p(t) = exp( integral_{t0}^{t} H_p(t') dt' ),
     then renormalize to 1 today. This is closest to how a REAL observer infers
     expansion (measure recession velocities, fit a Hubble law) and ties directly
     to the anisotropy diagnostic (PF2: anisotropy is the discriminating signal).

Scoring is IDENTICAL for both: a_p(t) -> sim_to_distance_modulus -> the in-range
Pantheon+ subset -> evaluate_precomputed -> chi2/dof. This is the one
authoritative chi2 (reconciled in Section 1); we do NOT invent a new one.

------------------------------------------------------------------------------
Units / inputs
------------------------------------------------------------------------------
positions : (n_snap, N, 3) float array, metres, in the SIM frame (NOT pre-centred;
            we subtract p / the COM internally).
velocities: (n_snap, N, 3) float array, m/s. Required only for ``hubble_flow``.
times_Gyr : (n_snap,) RELATIVE sim time, Gyr, starting at 0.0 (sim convention,
            matching ext_results['t_Gyr']).
The today snapshot is index ``today_index`` (default -1).

No randomness anywhere. Determinism: given the same arrays + observer index +
definition + k, the output is bit-reproducible.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
from scipy.integrate import cumulative_trapezoid

# Reuse the AUTHORITATIVE a(t)->mu(z) kernel and chi2 evaluator (Section 1 path).
from .sim_distance import sim_to_distance_modulus
from .hubble_diagram import evaluate_precomputed

# Sentinel meaning "use every particle as the neighbour set" (centre-like).
ALL_NEIGHBOURS = -1


def strided_observer_sample(n_total: int, sample: int):
    """The strided observer index sample used to score a cell.

    Shared by the sweep scorer (which picks the BEST observer) and the mu(z) figure
    (which must re-derive the SAME best observer to plot it), so both agree. Returns
    None (= score every particle) when sample is unset or >= n_total.
    """
    if sample and 0 < int(sample) < n_total:
        return np.unique(np.linspace(0, n_total - 1, int(sample)).astype(int))
    return None


# ---------------------------------------------------------------------------
# Snapshot-history extraction (the ONLY place that touches a live sim object)
# ---------------------------------------------------------------------------

def history_from_snapshots(snapshots: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a sim's saved snapshots into (positions, velocities, times_Gyr).

    Mirrors the snapshot contract in cosmo/integrator (each snapshot dict has
    'positions' (N,3) m, 'velocities' (N,3) m/s, 'time_s'). The sim ALREADY saves
    these at ``save_interval`` (see cosmo/simulation.run), so this is pure
    re-shaping — no new sim output contract, honouring the prototype invariant.

    Args:
        snapshots: list of snapshot dicts (sim.snapshots).

    Returns:
        positions  : (n_snap, N, 3) metres.
        velocities : (n_snap, N, 3) m/s.
        times_Gyr  : (n_snap,) RELATIVE Gyr, starting at 0.0 (t -= t[0]).
    """
    if len(snapshots) < 2:
        raise ValueError("need >= 2 snapshots to build a history.")
    pos = np.stack([np.asarray(s["positions"], dtype=float) for s in snapshots])
    vel = np.stack([np.asarray(s["velocities"], dtype=float) for s in snapshots])
    t_s = np.array([float(s["time_s"]) for s in snapshots], dtype=float)
    gyr_to_s = 1e9 * 365.25 * 24 * 3600          # matches simulation.py time_Gyr
    t_Gyr = (t_s - t_s[0]) / gyr_to_s
    return pos, vel, t_Gyr


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_history(
    positions: np.ndarray,
    velocities: Optional[np.ndarray],
    times_Gyr: np.ndarray,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(
            f"positions must be (n_snap, N, 3); got {pos.shape}"
        )
    t = np.asarray(times_Gyr, dtype=float)
    if t.ndim != 1 or t.shape[0] != pos.shape[0]:
        raise ValueError(
            f"times_Gyr must be (n_snap,) matching positions[0]; "
            f"got {t.shape} vs n_snap={pos.shape[0]}"
        )
    vel = None
    if velocities is not None:
        vel = np.asarray(velocities, dtype=float)
        if vel.shape != pos.shape:
            raise ValueError(
                f"velocities shape {vel.shape} != positions shape {pos.shape}"
            )
    return pos, vel, t


# ---------------------------------------------------------------------------
# Neighbour selection
# ---------------------------------------------------------------------------

def neighbour_indices(
    positions_today: np.ndarray,
    observer: int,
    k: int = ALL_NEIGHBOURS,
) -> np.ndarray:
    """Return the indices of the observer's neighbour set in the TODAY frame.

    The set EXCLUDES the observer itself (a_p measures the cloud around p, not p).
    Membership is fixed here and then tracked by identity across snapshots, so the
    expansion is Lagrangian (a fixed patch of universe), not a re-selected aperture.

    Args:
        positions_today: (N, 3) positions at the "today" snapshot, metres.
        observer:        index of the observer particle (0 <= observer < N), or
                         the special value -1 / ``"com"`` handled by callers that
                         want the COM observer (not a particle).
        k:               number of nearest neighbours to keep; ``ALL_NEIGHBOURS``
                         (-1) keeps every other particle (centre-like limit).

    Returns:
        1-D int array of neighbour indices (length min(k, N-1) or N-1).
    """
    pos = np.asarray(positions_today, dtype=float)
    N = pos.shape[0]
    others = np.array([i for i in range(N) if i != observer], dtype=int)
    if k == ALL_NEIGHBOURS or k >= others.size:
        return others
    d = np.linalg.norm(pos[others] - pos[observer], axis=1)
    nearest = others[np.argsort(d, kind="stable")[:k]]
    return np.sort(nearest)


# ---------------------------------------------------------------------------
# a_p(t) — the two observer definitions
# ---------------------------------------------------------------------------

def observer_a_curve_local_rms(
    positions: np.ndarray,
    times_Gyr: np.ndarray,
    observer: int,
    k: int = ALL_NEIGHBOURS,
    today_index: int = -1,
    *,
    about_com: bool = False,
) -> np.ndarray:
    """``local_rms`` definition: a_p(t) = RMS of p's (fixed) neighbour set about p.

    The neighbour set is chosen ONCE at ``today_index`` and tracked by identity.
    a_p is normalized so a_p[0] = 1 at the FIRST snapshot (the sim convention that
    sim_to_distance_modulus expects: a[0]=1 at t_start, renormalized to today
    downstream). Centre-relative RMS is taken about the observer particle's own
    position each snapshot (about_com=False) or about the neighbour-set COM
    (about_com=True). about_com=True with k=ALL reproduces the centre-based a(t).

    Args:
        positions:   (n_snap, N, 3) metres.
        times_Gyr:   (n_snap,) relative Gyr (unused here but kept for signature
                     symmetry with the hubble_flow variant / future use).
        observer:    observer particle index.
        k:           neighbour count (ALL_NEIGHBOURS = every other particle).
        today_index: snapshot used to FIX neighbour membership.
        about_com:   if True, RMS is about the neighbour-set centre of mass; if
                     False (default), about the observer particle p itself.

    Returns:
        (n_snap,) array a_p with a_p[0] == 1.0.
    """
    pos, _, _ = _validate_history(positions, None, times_Gyr)
    n_snap, N, _ = pos.shape

    nbr = neighbour_indices(pos[today_index], observer, k)
    if nbr.size < 2:
        raise ValueError(
            f"observer {observer} has < 2 neighbours (N={N}, k={k}); "
            "cannot define a local RMS."
        )

    rms = np.empty(n_snap, dtype=float)
    for s in range(n_snap):
        nbr_pos = pos[s][nbr]                       # (k, 3)
        if about_com:
            centre = nbr_pos.mean(axis=0)
        else:
            centre = pos[s][observer]
        r = np.linalg.norm(nbr_pos - centre, axis=1)
        rms[s] = np.sqrt(np.mean(r ** 2))

    if rms[0] <= 0.0:
        raise ValueError("Initial local RMS is non-positive; cannot normalize.")
    return rms / rms[0]


def center_a_curve(
    positions: np.ndarray,
    times_Gyr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reference centre-based a(t): RMS of the (masked) cloud about its COM.

    This is a stand-alone re-implementation of the measurement in
    cosmo/simulation._calculate_expansion_history (RMS about the mean of the inner
    observable subset, normalized to the first snapshot). It exists so the unit
    tests can assert the ``local_rms`` centre-observer reproduces it EXACTLY, and
    so the figure script can mark the centre baseline without re-running the sim.

    Args:
        positions: (n_snap, N, 3) metres (already the inner/observable subset, or
                   pass ``mask`` to subset here).
        times_Gyr: (n_snap,) relative Gyr (unused; signature symmetry).
        mask:      optional (N,) bool to restrict to the observable subset.

    Returns:
        (n_snap,) array a with a[0] == 1.0.
    """
    pos, _, _ = _validate_history(positions, None, times_Gyr)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        pos = pos[:, m, :]
    n_snap = pos.shape[0]
    rms = np.empty(n_snap, dtype=float)
    for s in range(n_snap):
        com = pos[s].mean(axis=0)
        r = np.linalg.norm(pos[s] - com, axis=1)
        rms[s] = np.sqrt(np.mean(r ** 2))
    if rms[0] <= 0.0:
        raise ValueError("Initial centre RMS is non-positive; cannot normalize.")
    return rms / rms[0]


def observer_a_curve_hubble_flow(
    positions: np.ndarray,
    velocities: np.ndarray,
    times_Gyr: np.ndarray,
    observer: int,
    k: int = ALL_NEIGHBOURS,
    today_index: int = -1,
) -> np.ndarray:
    """``hubble_flow`` definition: integrate the local Hubble slope about p.

    Per snapshot, with positions/velocities taken RELATIVE to the observer p (its
    own position and peculiar velocity subtracted), fit the local Hubble slope of
    p's fixed neighbour set,
        H_p(t) = sum(v_r * r) / sum(r^2),   v_r = (v - v_p) . rhat   [s^-1],
    exactly the global-slope formula in cosmo/anisotropy.hubble_dipole. Then
        a_p(t) = exp( integral_{t0}^{t} H_p dt' )   (a_p[0] = 1).

    The integral uses the RELATIVE sim time axis (Gyr -> s). This recovers a(t)
    from the velocity field the way an observer would, and is sensitive to the
    anisotropy a local frame feels (a particle on one face sees a biased flow).

    Args:
        positions:   (n_snap, N, 3) metres.
        velocities:  (n_snap, N, 3) m/s.
        times_Gyr:   (n_snap,) relative Gyr, starting at 0.0.
        observer:    observer particle index.
        k:           neighbour count (ALL_NEIGHBOURS = every other particle).
        today_index: snapshot used to FIX neighbour membership.

    Returns:
        (n_snap,) array a_p with a_p[0] == 1.0.
    """
    from .constants import CosmologicalConstants

    pos, vel, t = _validate_history(positions, velocities, times_Gyr)
    if vel is None:
        raise ValueError("hubble_flow requires velocities.")
    n_snap, N, _ = pos.shape

    nbr = neighbour_indices(pos[today_index], observer, k)
    if nbr.size < 2:
        raise ValueError(
            f"observer {observer} has < 2 neighbours (N={N}, k={k}); "
            "cannot fit a local Hubble slope."
        )

    H_p = np.empty(n_snap, dtype=float)            # s^-1
    gyr_to_s = CosmologicalConstants.Gyr_to_s
    for s in range(n_snap):
        rel_pos = pos[s][nbr] - pos[s][observer]   # (k, 3) metres
        rel_vel = vel[s][nbr] - vel[s][observer]   # (k, 3) m/s
        r = np.linalg.norm(rel_pos, axis=1)
        r_safe = np.where(r > 0, r, 1.0)
        r_hat = rel_pos / r_safe[:, None]
        v_r = np.sum(rel_vel * r_hat, axis=1)      # radial speed (k,)
        denom = float(np.dot(r, r))
        H_p[s] = float(np.dot(v_r, r) / denom) if denom > 0 else 0.0

    t_s = t * gyr_to_s                              # relative time in seconds
    # a_p(t) = exp(integral H_p dt); cumulative_trapezoid pads index 0 with 0.
    ln_a = cumulative_trapezoid(H_p, t_s, initial=0.0)
    a_p = np.exp(ln_a)
    # a_p[0] == exp(0) == 1.0 by construction.
    return a_p


# ---------------------------------------------------------------------------
# Scoring a single observer against Pantheon+ (authoritative chi2)
# ---------------------------------------------------------------------------

def score_observer(
    a_curve: np.ndarray,
    times_Gyr: np.ndarray,
    t_start_Gyr: float,
    pantheon_data: dict,
) -> dict:
    """Score one observer's a_p(t) against Pantheon+ via the AUTHORITATIVE path.

    Mirrors cosmo.parameter_sweep.compute_pantheon_metrics' core scoring (the same
    sim_to_distance_modulus + evaluate_precomputed used by the sweep), but WITHOUT
    the growth anchor: an off-centre observer's a_p legitimately has a different
    total growth than the centre, so anchoring to the centre's growth would
    wrongly reject every off-centre observer. We instead report ``growth_factor``
    so the caller can flag/inspect runaway observers honestly.

    Returns dict: chi2, chi2_dof, R2, n_sne_used, growth_factor, plus
    ``ok`` (False on any documented bad-input case; chi2_dof = +inf then).
    """
    a = np.asarray(a_curve, dtype=float)
    bad = {
        "chi2": float("inf"), "chi2_dof": float("inf"), "R2": -float("inf"),
        "n_sne_used": 0, "growth_factor": float("nan"), "ok": False,
    }
    if a.size < 2 or not np.all(np.isfinite(a)) or a[0] <= 0.0:
        return bad
    growth = float(a[-1] / a[0])

    try:
        dist = sim_to_distance_modulus(
            z_target=pantheon_data["z"],
            a=a,
            t_Gyr=times_Gyr,
            t_start_Gyr=t_start_Gyr,
        )
    except ValueError:
        bad["growth_factor"] = growth
        return bad

    in_range = dist["in_range"]
    z_in = pantheon_data["z"][in_range]
    mu_obs_in = pantheon_data["mu"][in_range]
    sigma_in = pantheon_data["sigma"][in_range]
    mu_model = dist["mu"]
    if len(z_in) < 2:
        bad["growth_factor"] = growth
        return bad

    try:
        ev = evaluate_precomputed(z_in, mu_obs_in, sigma_in, mu_model)
    except ValueError:
        bad["growth_factor"] = growth
        return bad

    chi2_dof = ev["chi2_dof"]
    if not np.isfinite(chi2_dof):
        bad["growth_factor"] = growth
        return bad

    return {
        "chi2": ev["chi2"],
        "chi2_dof": chi2_dof,
        "R2": ev["R2"],
        "n_sne_used": int(len(z_in)),
        "growth_factor": growth,
        "ok": True,
    }


# ---------------------------------------------------------------------------
# Fraction of VIABLE observers (the "how generic is our vantage" statistic)
# ---------------------------------------------------------------------------

def fraction_at_or_below(chi2_dof: np.ndarray, threshold: float) -> float:
    """Fraction of FINITE per-observer chi2/dof values at/below ``threshold``.

    The honest counterpart to "the best observer": given the full array of
    per-observer chi2/dof (np.inf marks observers that failed to score), report
    what FRACTION of the *scoreable* observers see a fit at least as good as a
    reference. Used with the LCDM reference (~0.436) and the EdS-null reference
    (~0.843) so the caller can say "a fraction f of random observers see
    Pantheon-like (sub-LCDM / sub-EdS) expansion". A large fraction = a
    Pantheon-matching vantage is GENERIC; a small fraction = our vantage is
    FINE-TUNED. This keeps the "take the best" inference honest.

    Pure function (no I/O, no plotting). Properties pinned by the unit tests:
      * monotone non-decreasing in ``threshold``;
      * 0.0 for any threshold below the minimum finite chi2;
      * 1.0 for any threshold at/above the maximum finite chi2;
      * exact count at a boundary (<= is inclusive);
      * NaN if there are NO finite observers (nothing to take a fraction of).

    Args:
        chi2_dof:  array of per-observer chi2/dof (np.inf for failed observers).
        threshold: reference chi2/dof to compare against (e.g. the LCDM or EdS
                   in-range chi2/dof for this run).

    Returns:
        float in [0, 1] = (# finite chi2_dof <= threshold) / (# finite chi2_dof),
        or float('nan') if no finite observers.
    """
    vals = np.asarray(chi2_dof, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float("nan")
    return float(np.count_nonzero(finite <= threshold) / finite.size)


# ---------------------------------------------------------------------------
# The prototype driver: score a sample of observers, report the distribution
# ---------------------------------------------------------------------------

def observer_chi2_distribution(
    positions: np.ndarray,
    velocities: np.ndarray,
    times_Gyr: np.ndarray,
    t_start_Gyr: float,
    pantheon_data: dict,
    *,
    definition: str = "local_rms",
    k: int = ALL_NEIGHBOURS,
    observers: Optional[Sequence[int]] = None,
    mask: Optional[np.ndarray] = None,
    today_index: int = -1,
    lcdm_ref: Optional[float] = None,
    eds_ref: Optional[float] = None,
) -> dict:
    """Score a SAMPLE of per-particle observers and summarise the distribution.

    This is the inference the user asked for: we are a RANDOM observer, Pantheon+
    localises us, so we compute the fit "from the point of view of all particles"
    and ask whether a Pantheon-matching observer EXISTS (best) and how GENERIC
    that vantage is (the FRACTION of observers at/below a reference fit).

    Args:
        positions:   (n_snap, N_total, 3) metres (full sim cloud).
        velocities:  (n_snap, N_total, 3) m/s.
        times_Gyr:   (n_snap,) relative Gyr.
        t_start_Gyr: absolute start time (Gyr) for the mu(z) kernel's today guard.
        pantheon_data: dict with 'z','mu','sigma' (cosmo.pantheon.load_pantheon).
        definition:  "local_rms" or "hubble_flow".
        k:            neighbour count per observer (ALL_NEIGHBOURS = whole cloud).
        observers:    explicit observer indices to score; default = every particle
                      in the observable subset.
        mask:         (N_total,) bool observable mask; observers and neighbours are
                      restricted to True entries (the inner cloud), matching the
                      centre-based a(t) which uses the observable subset only.
        today_index:  snapshot treated as "today" for neighbour membership.
        lcdm_ref:     optional LCDM in-range chi2/dof reference for this run; if
                      given, 'frac_below_lcdm' = fraction of observers at/below it.
        eds_ref:      optional EdS-null in-range chi2/dof reference; if given,
                      'frac_below_eds' = fraction of observers at/below it.

    Returns dict with keys:
        'definition', 'k', 'n_observers',
        'chi2_dof'        : (n_observers,) array (np.inf for failed observers),
        'growth_factor'   : (n_observers,) array,
        'observer_index'  : (n_observers,) array of the GLOBAL particle indices,
        'center_chi2_dof' : float — the centre-observer baseline (COM, all nbrs),
        'center_growth'   : float,
        'best_chi2_dof'   : float (min over finite observers; inf if none),
        'best_observer'   : int global index of the best observer (or -1),
        'median', 'p10', 'p90' : percentiles over the FINITE chi2_dof values,
        'n_finite'        : int count of observers with a finite chi2,
        'lcdm_ref', 'eds_ref'         : the reference values used (or None),
        'frac_below_lcdm', 'frac_below_eds' : fraction of FINITE observers at/below
                      each reference (the "how generic is our vantage" statistic;
                      np.nan if the corresponding *_ref was not supplied).
    """
    pos, vel, t = _validate_history(positions, velocities, times_Gyr)
    n_snap, N_total, _ = pos.shape

    if mask is None:
        sub = np.arange(N_total)
        pos_sub = pos
        vel_sub = vel
    else:
        m = np.asarray(mask, dtype=bool)
        sub = np.nonzero(m)[0]
        pos_sub = pos[:, m, :]
        vel_sub = vel[:, m, :]
    N = pos_sub.shape[1]

    if observers is None:
        local_observers = list(range(N))            # local indices into pos_sub
    else:
        # Map provided GLOBAL indices to local subset indices.
        global_to_local = {int(g): i for i, g in enumerate(sub)}
        local_observers = [global_to_local[int(o)] for o in observers
                           if int(o) in global_to_local]

    # ---- Centre baseline (COM "observer", whole subset) ----
    center_a = center_a_curve(pos_sub, t)
    center_score = score_observer(center_a, t, t_start_Gyr, pantheon_data)
    center_chi2 = center_score["chi2_dof"]
    center_growth = center_score["growth_factor"]

    # ---- Per-observer scoring ----
    chi2 = np.full(len(local_observers), np.inf, dtype=float)
    growth = np.full(len(local_observers), np.nan, dtype=float)
    for i, p_local in enumerate(local_observers):
        try:
            if definition == "local_rms":
                a_p = observer_a_curve_local_rms(
                    pos_sub, t, p_local, k=k, today_index=today_index)
            elif definition == "hubble_flow":
                a_p = observer_a_curve_hubble_flow(
                    pos_sub, vel_sub, t, p_local, k=k, today_index=today_index)
            else:
                raise ValueError(
                    f"unknown definition {definition!r}; "
                    "expected 'local_rms' or 'hubble_flow'."
                )
        except ValueError:
            continue
        sc = score_observer(a_p, t, t_start_Gyr, pantheon_data)
        chi2[i] = sc["chi2_dof"]
        growth[i] = sc["growth_factor"]

    finite = np.isfinite(chi2)
    n_finite = int(finite.sum())
    if n_finite > 0:
        finite_vals = chi2[finite]
        best_local = int(np.argmin(chi2))           # argmin over all (inf safe)
        best_chi2 = float(chi2[best_local])
        best_observer = int(sub[local_observers[best_local]])
        median = float(np.median(finite_vals))
        p10 = float(np.percentile(finite_vals, 10))
        p90 = float(np.percentile(finite_vals, 90))
    else:
        best_chi2 = float("inf")
        best_observer = -1
        median = p10 = p90 = float("nan")

    observer_index = np.array([int(sub[lo]) for lo in local_observers], dtype=int)

    frac_below_lcdm = (fraction_at_or_below(chi2, lcdm_ref)
                       if lcdm_ref is not None else float("nan"))
    frac_below_eds = (fraction_at_or_below(chi2, eds_ref)
                      if eds_ref is not None else float("nan"))

    return {
        "definition": definition,
        "k": k,
        "n_observers": len(local_observers),
        "chi2_dof": chi2,
        "growth_factor": growth,
        "observer_index": observer_index,
        "center_chi2_dof": float(center_chi2),
        "center_growth": float(center_growth),
        "best_chi2_dof": best_chi2,
        "best_observer": best_observer,
        "median": median,
        "p10": p10,
        "p90": p90,
        "n_finite": n_finite,
        "lcdm_ref": lcdm_ref,
        "eds_ref": eds_ref,
        "frac_below_lcdm": frac_below_lcdm,
        "frac_below_eds": frac_below_eds,
    }
