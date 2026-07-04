"""The INFINITE virialized meta-structure: a periodic (torus = infinite tiling) self-gravitating
medium relaxed to VIRIAL EQUILIBRIUM (2K/|U| ~ 1).

Why this and not the lattice/glass/finite-cluster
--------------------------------------------------
A genuinely *virialized* system is held up by VELOCITY DISPERSION (2K/|U| ~ 1), not by every
member sitting at net-zero force. Its members MOVE (they orbit); each feels a net force at every
instant. The only STATIC structures with net-zero interior force are a crystal or a glass -- which
is why every "force-balance" construction collapses back toward a lattice. The honest "virialized
in an infinite theoretical universe" structure is therefore a SNAPSHOT of a self-gravitating medium
in virial equilibrium: homogeneous on average, disordered, locally clumpy (clumps + voids). Over
the ~8 Gyr simulation the nodes barely move (their dynamical time >> the sim), so freezing this
snapshot as static boundary conditions is a valid approximation.

The PERIODIC box is what makes it "infinite": every node is surrounded by images on all sides, so
the mean-density (background) force cancels by symmetry -- the Jeans cancellation a truly infinite
medium provides. A FINITE chunk cannot do this (its edge deficit leaves a ~r background pull). A
finite crystal only reads as "force-balanced" if a central node of mass ~ one full HMEA node is
added at the origin -- but that is NOT the physical central node: "us"/the progenitor has mass
~ centerM x observable = TINY next to an HMEA node, and cannot cancel the background. So the finite
force-balance was an artifact of over-massing the (real) central node, not evidence of virialization
-- virialization is 2K/|U| ~ 1, which this medium provides.

"Us" / the progenitor node
--------------------------
Node index 0 is OUR observable universe -- the progenitor node -- placed at the origin. Its mass is
``center_mass_frac`` x the mean node mass (default 1.0 = a typical node; physically ~1x observable
mass, i.e. tiny next to an HMEA node, so by default a light tracer). It PARTICIPATES in the medium
and in the virial ratio, so the virialization "takes us into account".

Note on the virial ratio: the periodic potential uses the minimum-image convention (each node feels
the nearest image of every other), a standard prototype approximation to the true (Ewald) periodic
sum -- adequate to demonstrate virial equilibrium of the medium.
"""
from __future__ import annotations

import numpy as np


def _mi_accel_pot(pos: np.ndarray, mass: np.ndarray, L: float, eps: float):
    """Minimum-image softened acceleration (G=1) and total potential energy for a periodic box."""
    d = pos[None, :, :] - pos[:, None, :]
    d -= L * np.round(d / L)                       # minimum image (periodic / "infinite")
    r2 = (d * d).sum(-1) + eps * eps
    inv_r = r2 ** -0.5
    inv_r3 = inv_r ** 3
    np.fill_diagonal(inv_r3, 0.0)
    accel = (mass[None, :, None] * d * inv_r3[:, :, None]).sum(1)
    iu = np.triu_indices(len(mass), 1)
    U = -(mass[:, None] * mass[None, :] * inv_r)[iu].sum()
    return accel, float(U)


def virial_ratio(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, L: float, eps: float) -> float:
    """2K/|U| in the periodic (infinite) sense. ==1 at virial equilibrium (the 'virialized' test)."""
    _, U = _mi_accel_pot(pos, mass, L, eps)
    K = 0.5 * (mass[:, None] * vel * vel).sum()
    return float(2.0 * K / abs(U)) if U != 0.0 else float("inf")


def density_std(pos: np.ndarray, L: float, ncell: int = 6) -> float:
    """std/mean of node counts on an ncell^3 grid -> 0 for a perfect crystal; grows with clumping.
    A homogeneous medium is O(1); a collapsed cluster is >> 1."""
    idx = np.floor((pos % L) / L * ncell).astype(int).clip(0, ncell - 1)
    cnt = np.bincount(idx[:, 0] * ncell * ncell + idx[:, 1] * ncell + idx[:, 2], minlength=ncell ** 3)
    return float(cnt.std() / cnt.mean())


def central_concentration(pos_centred: np.ndarray, box: float) -> float:
    """Fraction of nodes within 0.25*(box/2) of the origin. For a HOMOGENEOUS medium this ~ the
    uniform expectation (0.25^3 = 0.0156); a centrally-concentrated cluster gives a MUCH larger
    value. Distinguishes the homogeneous medium from the (wrong) collapsed halo."""
    r = np.linalg.norm(pos_centred, axis=1)
    return float(np.mean(r < 0.25 * (box / 2.0)))


def relax_virialized_medium(
    n_nodes: int = 216,
    sigma: float = 1.5,
    seed: int = 0,
    *,
    center_mass_frac: float = 1.0,
    box: float = 1.0,
    eps: float | None = None,
    dt: float = 0.004,
    n_steps: int = 2500,
    warm: float = 1.0,
    return_history: bool = False,
) -> dict:
    """Relax a periodic self-gravitating medium to virial equilibrium (2K/|U| ~ ``warm``).

    Node 0 is "us"/the progenitor: mass ``center_mass_frac`` x mean, translated to the origin in the
    returned (centred) coordinates. Pressure-supported (warm start, NO damping) so it stays
    HOMOGENEOUS instead of monolithically collapsing.

    Returns a dict with:
      pos            (N,3) node positions CENTRED on "us" (origin), in [-box/2, box/2)
      mass           (N,)  node masses (mean of the non-central nodes ~ 1; node 0 == center_mass_frac)
      vel            (N,3) node velocities (the dispersion that virializes the medium)
      virial_ratio   float 2K/|U| (==1 at equilibrium) -- the 'virialized in an infinite universe' metric
      density_std    float std/mean of cell counts (homogeneity; O(1) uniform, >>1 collapsed)
      concentration  float fraction within 0.25*(box/2) of us (uniform ~0.0156; collapsed >>)
      center_index   int   index of the "us" node (0)
      box, eps       the box size and softening used
      history        (optional) virial-ratio samples over the relaxation (bounded => stayed virial)
    """
    if int(n_nodes) < 8:
        raise ValueError(f"n_nodes must be >= 8 for a medium, got {n_nodes}")
    rng = np.random.default_rng(int(seed))
    N = int(n_nodes)
    if eps is None:
        # Softening = 2x the mean node spacing: COLLISIONLESS (each node feels the smooth field, not
        # close two-body encounters). Smaller softening lets close encounters inject energy and the
        # medium numerically HEATS -> evaporates (2K/|U| runs away); 2x spacing holds it at virial
        # (2K/|U| ~ 1.1, bounded) across N (verified N=400..1200).
        eps = 2.0 * box / N ** (1.0 / 3.0)

    w = np.exp(float(sigma) * rng.standard_normal(N))
    mass = w / w.mean()                                 # mean node mass == 1
    mass[0] = float(center_mass_frac)                   # node 0 = "us" (progenitor), scaled by centerM
    pos = rng.random((N, 3)) * box                      # homogeneous random start
    accel, U = _mi_accel_pot(pos, mass, box, eps)

    vel = rng.standard_normal((N, 3))
    vel -= np.average(vel, axis=0, weights=mass)        # zero net momentum
    K0 = 0.5 * (mass[:, None] * vel * vel).sum()
    vel *= np.sqrt(float(warm) * 0.5 * abs(U) / K0)     # set initial 2K/|U| == warm (pressure support)

    hist = []
    for s in range(int(n_steps)):
        if return_history and s % 50 == 0:
            hist.append(virial_ratio(pos, vel, mass, box, eps))
        vel += 0.5 * dt * accel
        pos = (pos + dt * vel) % box
        accel, U = _mi_accel_pot(pos, mass, box, eps)
        vel += 0.5 * dt * accel                          # NO damping -> stays hot -> homogeneous

    vr = virial_ratio(pos, vel, mass, box, eps)
    ds = density_std(pos, box)
    # Translate (periodically) so "us" sits at the origin, then express in [-box/2, box/2).
    pos_centred = ((pos - pos[0] + box / 2.0) % box) - box / 2.0
    conc = central_concentration(pos_centred, box)
    return {
        "pos": pos_centred, "mass": mass, "vel": vel,
        "virial_ratio": vr, "density_std": ds, "concentration": conc,
        "center_index": 0, "box": box, "eps": eps,
        "history": np.array(hist) if return_history else None,
    }
