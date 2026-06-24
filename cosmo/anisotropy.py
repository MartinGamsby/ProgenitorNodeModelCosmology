"""
Directional Shear / Hubble-Dipole Anisotropy Diagnostic

Pure-function module for measuring directional expansion anisotropy in an
N-body particle cloud. Mirrors cosmo/sim_distance.py style: no I/O, no
plotting, no simulation imports. All functions operate on (N, 3) numpy arrays
of positions and velocities in SI units (metres, m/s).

Physics motivation
------------------
The HMEA lattice with EQUAL node masses produces a TRACELESS tidal tensor
(the isotropic contribution cancels). The section-3 per-node mass variation
breaks this symmetry: a seed with e.g. heavier nodes along the +x face
produces a stress tensor with a net shear component, stretching the expanding
cloud preferentially along x. This module MEASURES that anisotropy in the
evolved cloud without re-running the integrator.

Public API (all return plain dicts; dataclass-style keys document intent)
--------------------------------------------------------------------------
  shape_tensor(positions)
      Shape tensor S_ij = mean(x_i x_j); eigenvalues, principal axis, and a
      scalar shear index (lambda_max - lambda_min) / mean(lambda).

  axis_rms(positions)
      Per-axis RMS extent vector; max/min ratio as a simple anisotropy scalar.

  hubble_dipole(positions, velocities, axis=None)
      Radial Hubble flow fit; dipole (H_plus - H_minus) / H_mean and the
      best-fit dipole direction.

  expansion_anisotropy(positions_initial, positions_final)
      Per-axis growth factor rms_final / rms_initial; spread across axes.

  anisotropy_summary(positions, velocities,
                     positions_initial=None, positions_final=None)
      Convenience wrapper returning all four sub-dicts merged under named keys.

Edge cases
----------
  All functions validate input shapes and handle degenerate inputs (N < 2,
  all-zero positions, collinear points) by returning zero results with a
  `degenerate` flag set to True. No randomness anywhere.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_positions(positions: np.ndarray, name: str = "positions") -> np.ndarray:
    """Coerce and validate a (N, 3) position array."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim == 1 and pos.shape[0] == 3:
        pos = pos.reshape(1, 3)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(
            f"{name} must be shape (N, 3); got {positions.shape}"  # type: ignore[union-attr]
        )
    return pos


def _validate_positions_velocities(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos = _validate_positions(positions)
    vel = np.asarray(velocities, dtype=float)
    if vel.shape != pos.shape:
        raise ValueError(
            f"positions shape {pos.shape} != velocities shape {vel.shape}"
        )
    return pos, vel


def _subtract_com(
    positions: np.ndarray,
    velocities: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Return COM-centred positions (and COM-velocity-subtracted velocities)."""
    com_pos = positions.mean(axis=0)
    pos_c = positions - com_pos
    vel_c = None
    if velocities is not None:
        com_vel = velocities.mean(axis=0)
        vel_c = velocities - com_vel
    return pos_c, vel_c


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def shape_tensor(positions: np.ndarray) -> dict:
    """Compute the shape / inertia tensor of the particle cloud.

    S_ij = mean_k(x_ki * x_kj)  (symmetric, (3,3) matrix).

    Returns
    -------
    dict with keys:
      ``'S'``           : (3, 3) shape tensor.
      ``'eigenvalues'`` : 1-D array (3,) sorted ascending.
      ``'eigenvectors'``: (3, 3) column-matrix; eigenvectors[:, i] belongs to
                          eigenvalues[i].
      ``'principal_axis'``: unit vector (3,) = eigenvector of lambda_max
                            (the stretch direction).
      ``'shear_index'`` : (lambda_max - lambda_min) / mean(lambda); 0 = isotropic.
      ``'degenerate'``  : True if N < 2 or all positions are identical.
    """
    pos = _validate_positions(positions)
    N = pos.shape[0]

    if N < 2 or np.allclose(pos, pos[0]):
        return {
            "S": np.zeros((3, 3)),
            "eigenvalues": np.zeros(3),
            "eigenvectors": np.eye(3),
            "principal_axis": np.array([1.0, 0.0, 0.0]),
            "shear_index": 0.0,
            "degenerate": True,
        }

    # COM-centre
    pos_c, _ = _subtract_com(pos)

    S = (pos_c.T @ pos_c) / N   # (3, 3)

    eigenvalues, eigenvectors = np.linalg.eigh(S)   # ascending order
    lam_min, lam_mean, lam_max = eigenvalues[0], eigenvalues.mean(), eigenvalues[2]

    if lam_mean <= 0.0:
        shear_index = 0.0
    else:
        shear_index = float((lam_max - lam_min) / lam_mean)

    return {
        "S": S,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "principal_axis": eigenvectors[:, 2],   # eigenvector of lambda_max
        "shear_index": shear_index,
        "degenerate": False,
    }


def axis_rms(positions: np.ndarray) -> dict:
    """Compute per-axis RMS extent of the particle cloud.

    Returns
    -------
    dict with keys:
      ``'rms'``      : (3,) array — RMS extent along x, y, z.
      ``'max_min_ratio'``: rms.max() / rms.min(); 1 = isotropic.
      ``'degenerate'``: True if N < 2.
    """
    pos = _validate_positions(positions)
    N = pos.shape[0]

    if N < 2:
        return {"rms": np.zeros(3), "max_min_ratio": 1.0, "degenerate": True}

    pos_c, _ = _subtract_com(pos)
    rms = np.sqrt(np.mean(pos_c ** 2, axis=0))   # (3,)

    rms_min = rms.min()
    if rms_min <= 0.0:
        max_min_ratio = 1.0
    else:
        max_min_ratio = float(rms.max() / rms_min)

    return {"rms": rms, "max_min_ratio": max_min_ratio, "degenerate": False}


def hubble_dipole(
    positions: np.ndarray,
    velocities: np.ndarray,
    axis: Optional[np.ndarray] = None,
) -> dict:
    """Measure the Hubble-flow dipole along a given axis (default: shape-tensor
    principal axis).

    Algorithm
    ---------
    1. COM-centre positions and COM-velocity-subtract velocities.
    2. Compute radial speed v_r = v . r_hat for each particle.
    3. Fit a GLOBAL Hubble slope H_global = sum(v_r * r) / sum(r^2) [s^-1].
    4. Split particles by sign of x . axis_hat:
         H_plus  = Hubble slope in +axis hemisphere,
         H_minus = Hubble slope in -axis hemisphere.
    5. dipole = (H_plus - H_minus) / H_mean
       where H_mean = (H_plus + H_minus) / 2.

    The axis that MAXIMISES |dipole| is estimated by testing the three cardinal
    axes and the shape-tensor principal axis; the winner is reported as
    ``'best_axis'``.

    Returns
    -------
    dict with keys:
      ``'H_global'``  : global Hubble slope [s^-1].
      ``'H_plus'``    : Hubble slope in +axis hemisphere [s^-1].
      ``'H_minus'``   : Hubble slope in -axis hemisphere [s^-1].
      ``'H_mean'``    : (H_plus + H_minus) / 2 [s^-1].
      ``'dipole'``    : (H_plus - H_minus) / H_mean (dimensionless).
      ``'axis'``      : the axis used for hemisphere splitting (3,).
      ``'best_axis'`` : the cardinal/principal axis that maximises |dipole|.
      ``'best_dipole'``: |dipole| along best_axis.
      ``'degenerate'``: True if N < 4 or cloud is near-singular.
    """
    pos, vel = _validate_positions_velocities(positions, velocities)
    N = pos.shape[0]

    _DEGENERATE = {
        "H_global": 0.0,
        "H_plus": 0.0,
        "H_minus": 0.0,
        "H_mean": 0.0,
        "dipole": 0.0,
        "axis": np.array([1.0, 0.0, 0.0]),
        "best_axis": np.array([1.0, 0.0, 0.0]),
        "best_dipole": 0.0,
        "degenerate": True,
    }

    if N < 4:
        return _DEGENERATE

    pos_c, vel_c = _subtract_com(pos, vel)
    assert vel_c is not None

    r = np.linalg.norm(pos_c, axis=1)   # (N,)
    if np.all(r < 1e-30):
        return _DEGENERATE

    # Unit radial vectors (avoid /0 for the zero-position particle if any)
    r_safe = np.where(r > 0, r, 1.0)
    r_hat = pos_c / r_safe[:, None]

    # Radial speed v_r = v . r_hat
    v_r = np.sum(vel_c * r_hat, axis=1)   # (N,)

    # Global Hubble slope H = sum(v_r * r) / sum(r^2)
    denom_global = np.dot(r, r)
    H_global = float(np.dot(v_r, r) / denom_global) if denom_global > 0 else 0.0

    # Determine the probe axis (default: shape-tensor principal axis)
    if axis is not None:
        ax = np.asarray(axis, dtype=float)
        ax = ax / np.linalg.norm(ax)
    else:
        st = shape_tensor(pos_c)
        ax = st["principal_axis"]

    def _hemisphere_H(ax_hat: np.ndarray) -> tuple[float, float]:
        """Return (H_plus, H_minus) along ax_hat."""
        proj = pos_c @ ax_hat          # (N,) scalar projection
        mask_plus = proj >= 0
        mask_minus = ~mask_plus
        results = []
        for mask in (mask_plus, mask_minus):
            if mask.sum() < 2:
                results.append(H_global)  # fallback
                continue
            vr_h = v_r[mask]
            r_h = r[mask]
            d = np.dot(r_h, r_h)
            results.append(float(np.dot(vr_h, r_h) / d) if d > 0 else 0.0)
        return results[0], results[1]

    H_plus, H_minus = _hemisphere_H(ax)
    H_mean = (H_plus + H_minus) / 2.0
    dipole = float((H_plus - H_minus) / H_mean) if H_mean != 0.0 else 0.0

    # Find best cardinal / principal axis
    candidates = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        ax,
    ]
    best_axis = ax
    best_dipole = abs(dipole)
    for cand in candidates:
        hp, hm = _hemisphere_H(cand)
        hm_c = (hp + hm) / 2.0
        d_cand = abs((hp - hm) / hm_c) if hm_c != 0.0 else 0.0
        if d_cand > best_dipole:
            best_dipole = d_cand
            best_axis = cand

    return {
        "H_global": H_global,
        "H_plus": H_plus,
        "H_minus": H_minus,
        "H_mean": H_mean,
        "dipole": dipole,
        "axis": ax,
        "best_axis": best_axis,
        "best_dipole": best_dipole,
        "degenerate": False,
    }


def expansion_anisotropy(
    positions_initial: np.ndarray,
    positions_final: np.ndarray,
) -> dict:
    """Measure per-axis expansion factor between two snapshots.

    growth_factor_i = rms_final_i / rms_initial_i  for i in {x, y, z}.

    Returns
    -------
    dict with keys:
      ``'rms_initial'``  : (3,) RMS extents at the initial snapshot.
      ``'rms_final'``    : (3,) RMS extents at the final snapshot.
      ``'growth_factor'``: (3,) per-axis growth factor.
      ``'spread'``       : max/min of growth_factor (1 = isotropic expansion).
      ``'degenerate'``   : True if initial cloud is degenerate or any initial
                           RMS is zero.
    """
    pi = _validate_positions(positions_initial, "positions_initial")
    pf = _validate_positions(positions_final, "positions_final")

    ri = axis_rms(pi)
    rf = axis_rms(pf)

    if ri["degenerate"]:
        return {
            "rms_initial": ri["rms"],
            "rms_final": rf["rms"],
            "growth_factor": np.ones(3),
            "spread": 1.0,
            "degenerate": True,
        }

    rms_i = ri["rms"]
    rms_f = rf["rms"]

    # Avoid division by zero
    safe = np.where(rms_i > 0, rms_i, 1.0)
    growth = rms_f / safe
    growth = np.where(rms_i > 0, growth, 1.0)

    g_min = growth.min()
    spread = float(growth.max() / g_min) if g_min > 0 else 1.0

    return {
        "rms_initial": rms_i,
        "rms_final": rms_f,
        "growth_factor": growth,
        "spread": spread,
        "degenerate": False,
    }


def anisotropy_summary(
    positions: np.ndarray,
    velocities: np.ndarray,
    positions_initial: Optional[np.ndarray] = None,
    positions_final: Optional[np.ndarray] = None,
    axis: Optional[np.ndarray] = None,
) -> dict:
    """Convenience wrapper: run all four diagnostics and return as sub-dicts.

    Args:
        positions:          (N, 3) particle positions for shape/Hubble diagnostics.
        velocities:         (N, 3) particle velocities.
        positions_initial:  (N, 3) initial snapshot positions (for expansion_anisotropy).
                            If None, expansion_anisotropy is skipped.
        positions_final:    (N, 3) final snapshot positions (for expansion_anisotropy).
                            Defaults to ``positions`` if positions_initial is supplied.
        axis:               Hemisphere-split axis for hubble_dipole (default: shape principal axis).

    Returns
    -------
    dict with keys:
      ``'shape'``     : result of shape_tensor(positions).
      ``'axis_rms'``  : result of axis_rms(positions).
      ``'hubble_dipole'``: result of hubble_dipole(positions, velocities, axis).
      ``'expansion'`` : result of expansion_anisotropy(positions_initial, positions_final)
                        or None if positions_initial is not supplied.
    """
    pos, vel = _validate_positions_velocities(positions, velocities)

    result = {
        "shape": shape_tensor(pos),
        "axis_rms": axis_rms(pos),
        "hubble_dipole": hubble_dipole(pos, vel, axis=axis),
        "expansion": None,
    }

    if positions_initial is not None:
        pf = pos if positions_final is None else _validate_positions(positions_final, "positions_final")
        result["expansion"] = expansion_anisotropy(positions_initial, pf)

    return result
