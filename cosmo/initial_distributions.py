"""
Initial particle distribution generators.

Provides alternative position samplers for ParticleSystem:
  - uniform_sphere : rejection sampling (default, current behaviour, byte-identical to legacy)
  - grf            : Gaussian random field shaped by BBKS LCDM P(k) + Zel'dovich displacement

Both samplers return raw (N, 3) positions in metres BEFORE the shared
centre-of-mass / RMS-normalisation post-processing in particles.py.
"""

import numpy as np


# ---------------------------------------------------------------------------
# BBKS analytic transfer function (Bardeen, Bond, Kaiser, Szalay 1986)
# Adequate for a toy GRF initial condition; matches P(k) shape on large scales.
# ---------------------------------------------------------------------------

def _bbks_transfer(k_h: np.ndarray,
                   Omega_m: float = 0.3,
                   h: float = 0.70) -> np.ndarray:
    """BBKS transfer function T(k).

    Args:
        k_h : wavenumbers in h/Mpc  (any shape, must be > 0).
        Omega_m : matter density parameter.
        h : dimensionless Hubble parameter (H0 / (100 km/s/Mpc)).

    Returns:
        T(k) array matching the shape of k_h.
    """
    # Shape parameter Gamma (Sugiyama 1995 approximation, good enough for toy IC)
    Gamma = Omega_m * h * np.exp(-0.06 - np.sqrt(2 * h) * 0.06)
    q = k_h / Gamma  # dimensionless

    T = np.log(1.0 + 2.34 * q) / (2.34 * q)
    T /= (1.0 + 3.89 * q + (16.1 * q) ** 2
          + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** 0.25
    return T


def _power_spectrum(k_h: np.ndarray,
                    n_s: float = 0.965,
                    Omega_m: float = 0.3,
                    h: float = 0.70,
                    A_s: float = 1.0) -> np.ndarray:
    """LCDM-approximate power spectrum P(k) = A_s * k^n_s * T(k)^2.

    The overall amplitude A_s is arbitrary here because the positions are
    RMS-normalised to box_size/2 in the shared post-processing step, so
    only the SHAPE of P(k) matters for creating clustered structure.

    Args:
        k_h : wavenumbers in h/Mpc (> 0).

    Returns:
        P(k) array (same shape as k_h), zero where k_h == 0.
    """
    safe_k = np.where(k_h > 0, k_h, 1.0)  # avoid 0-division; masked below
    T = _bbks_transfer(safe_k, Omega_m=Omega_m, h=h)
    P = A_s * safe_k ** n_s * T ** 2
    P = np.where(k_h > 0, P, 0.0)
    return P


def _zeldovich_displacement_field(delta_k: np.ndarray, Ng: int) -> np.ndarray:
    """Compute the Zel'dovich displacement Psi(x) in each axis from delta_k.

    In k-space: Psi_i(k) = -i * k_i / k^2 * delta(k)  (Poisson-kernel)

    Args:
        delta_k : complex 3D array of shape (Ng, Ng, Ng), the density contrast
                  in Fourier space (output of np.fft.fftn on the density field).
        Ng      : grid size.

    Returns:
        psi : real array of shape (Ng, Ng, Ng, 3), the displacement field.
    """
    # k-coordinates (0..Ng/2, then -(Ng/2-1)..-1 convention from fftfreq)
    freq = np.fft.fftfreq(Ng) * Ng  # integer wavenumber indices
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing='ij')

    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = 1.0  # avoid division by zero; DC component -> 0 below

    psi_k = np.zeros((Ng, Ng, Ng, 3), dtype=complex)
    for axis, k_i in enumerate([kx, ky, kz]):
        psi_k[..., axis] = -1j * k_i / k2 * delta_k

    # Zero DC mode (no bulk translation)
    psi_k[0, 0, 0, :] = 0.0

    psi = np.fft.ifftn(psi_k, axes=(0, 1, 2)).real  # (Ng, Ng, Ng, 3)
    return psi


# ---------------------------------------------------------------------------
# Diagnostic helpers (read-only inspection; NOT on the sampler path)
# ---------------------------------------------------------------------------

def grf_density_field(box_size_m: float,
                      seed: int,
                      Ng: int = 64,
                      n_s: float = 0.965,
                      Omega_m: float = 0.3,
                      h: float = 0.70) -> np.ndarray:
    """Return the real-space density contrast delta(x) used by ``sample_grf``.

    This reproduces steps 1-3 of ``sample_grf`` (white noise -> P(k) shaping ->
    inverse FFT) so the density field can be inspected directly (mean ~ 0,
    variance, NaN check). It is a DIAGNOSTIC helper and is NOT called on the
    particle-sampling path, so it cannot change ``uniform_sphere`` or ``grf``
    output.

    Args:
        box_size_m : physical box size in metres (only sets the cell size,
                     unused by delta itself but kept for signature symmetry).
        seed       : integer seed (matches ``sample_grf``).
        Ng         : grid resolution.

    Returns:
        delta : real array of shape (Ng, Ng, Ng), the density contrast.
    """
    rng = np.random.default_rng(seed)
    white_noise = rng.standard_normal((Ng, Ng, Ng))
    noise_k = np.fft.fftn(white_noise)

    freq = np.fft.fftfreq(Ng) * Ng
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing='ij')
    k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    Pk = _power_spectrum(k_mag, n_s=n_s, Omega_m=Omega_m, h=h)
    delta_k = noise_k * np.sqrt(Pk)
    delta = np.fft.ifftn(delta_k).real
    return delta


def grf_field_stats(box_size_m: float,
                    seed: int,
                    Ng: int = 64,
                    n_s: float = 0.965,
                    Omega_m: float = 0.3,
                    h: float = 0.70) -> dict:
    """Statistics of the GRF density field + Zel'dovich displacement.

    Used by the WS5 GRF investigation to decide whether the GRF setup is healthy
    (H4) or broken. Returns a dict with:
      - delta_mean / delta_std : density-contrast moments (mean should be ~0).
      - delta_has_nan          : NaN/Inf present in the density field.
      - pk_low_mean / pk_high_mean / pk_decays : P(k) shape sanity (must decay
        from large to small scales, i.e. low-k power > high-k power).
      - disp_rms_over_cell     : RMS of the FINAL (rescaled) Zel'dovich
        displacement in units of the grid cell size. ``sample_grf`` rescales the
        displacement so this is ~0.5 by construction; >1 would mean particles are
        displaced more than a cell (over-perturbation -> clipping).
    """
    delta = grf_density_field(box_size_m, seed, Ng=Ng,
                              n_s=n_s, Omega_m=Omega_m, h=h)

    # P(k) shape sanity (independent of the realization).
    k_low = np.array([0.001, 0.01, 0.05])
    k_high = np.array([1.0, 5.0, 10.0])
    pk_low = float(np.mean(_power_spectrum(k_low, n_s=n_s, Omega_m=Omega_m, h=h)))
    pk_high = float(np.mean(_power_spectrum(k_high, n_s=n_s, Omega_m=Omega_m, h=h)))

    # Reconstruct the FINAL (rescaled) displacement exactly as sample_grf does,
    # so disp_rms_over_cell reflects what the sampler actually applies.
    rng = np.random.default_rng(seed)
    white_noise = rng.standard_normal((Ng, Ng, Ng))
    noise_k = np.fft.fftn(white_noise)
    freq = np.fft.fftfreq(Ng) * Ng
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing='ij')
    k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    delta_k = noise_k * np.sqrt(_power_spectrum(k_mag, n_s=n_s,
                                                Omega_m=Omega_m, h=h))
    psi = _zeldovich_displacement_field(delta_k, Ng)
    cell_size_m = box_size_m / Ng
    psi_rms_raw = np.sqrt(np.mean(psi ** 2))
    if psi_rms_raw > 0:
        psi = psi * (0.5 * cell_size_m / psi_rms_raw)
    disp_rms_over_cell = float(np.sqrt(np.mean(psi ** 2)) / cell_size_m)

    return {
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta)),
        "delta_has_nan": bool(np.any(~np.isfinite(delta))),
        "pk_low_mean": pk_low,
        "pk_high_mean": pk_high,
        "pk_decays": bool(pk_low > pk_high),
        "disp_rms_over_cell": disp_rms_over_cell,
    }


# ---------------------------------------------------------------------------
# Public samplers
# ---------------------------------------------------------------------------

def sample_uniform_sphere(n_particles: int,
                          sphere_radius_m: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Uniform sphere rejection sampler (legacy behaviour, kept here for clarity).

    This reproduces the ORIGINAL logic from particles.py:108-115 exactly —
    including the same loop order — so tests can verify byte-level compatibility
    by calling both paths with the same Generator.

    Args:
        n_particles   : N.
        sphere_radius_m: bounding sphere radius in metres.
        rng           : numpy Generator (use np.random.default_rng(seed)).

    Returns:
        positions : (N, 3) float64 array, raw (NOT centred/normalised).
    """
    positions = np.empty((n_particles, 3), dtype=np.float64)
    R = sphere_radius_m
    for i in range(n_particles):
        while True:
            pos = rng.uniform(-R, R, 3)
            if np.linalg.norm(pos) <= R:
                break
        positions[i] = pos
    return positions


def sample_grf(n_particles: int,
               box_size_m: float,
               seed: int,
               Ng: int = 64,
               n_s: float = 0.965,
               Omega_m: float = 0.3,
               h: float = 0.70,
               support: str = "sphere") -> np.ndarray:
    """Generate particle positions via a Gaussian random field + Zel'dovich displacement.

    Recipe (deterministic for a fixed seed):
    1. Draw a Gaussian white-noise field of shape (Ng^3) in real space.
    2. FFT to k-space, multiply by sqrt(P(k)) to impose BBKS LCDM spectrum.
    3. IFFT to get density contrast delta(x).
    4. Compute the Zel'dovich displacement field Psi(x) from delta(x).
    5. Place N particles on a regular Lagrangian grid and displace each by Psi at
       its grid cell.
    6. SUPPORT MASK (``support="sphere"``, default): keep only grid cells whose
       UNDISPLACED (Lagrangian) radius is within the same sphere radius
       ``(box/2)/sqrt(3/5)`` that ``uniform_sphere`` uses. This makes the GRF
       cloud a CLUSTERED SPHERE rather than a clustered CUBE — so GRF and
       uniform_sphere share the SAME bounding geometry and the ONLY remaining
       difference is clustering (the intended physics). ``support="box"`` keeps
       the legacy full-cube grid (a GRF-perturbed cube; kept for reproducibility
       / comparison only).
    7. Subsample / select N particles from the masked grid (random without
       replacement; if N exceeds the masked count we tile + jitter).
    8. Radial clip to the sphere radius (sphere support) — NOT a cube clip — so a
       wild Zel'dovich excursion cannot push a particle outside the spherical
       envelope. The shared RMS-normalisation in particles.py then rescales to the
       exact target, so only the SHAPE / clustering is baked in here.

    Args:
        n_particles : number of output particles N.
        box_size_m  : physical box size in metres (sets the Zel'dovich amplitude
                      so displacements are proportional to box/Ng).
        seed        : integer seed → deterministic, reproducible.
        Ng          : grid resolution (default 64; higher = more structure detail).
        n_s         : scalar spectral index (LCDM default 0.965).
        Omega_m     : matter fraction (LCDM default 0.3).
        h           : dimensionless Hubble (LCDM default 0.70).
        support     : "sphere" (default) confines the cloud to the uniform_sphere
                      radius (clustered sphere, comparable geometry); "box" keeps
                      the legacy GRF-perturbed cube.

    Returns:
        positions : (N, 3) float64 array in metres, raw (NOT centred/normalised).
                    Sphere support: |r| <= (box/2)/sqrt(3/5). Box support: within
                    [-box/2, box/2].
    """
    if support not in ("sphere", "box"):
        raise ValueError(
            f"Unknown GRF support {support!r}. Valid: 'sphere' (default), 'box'."
        )
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. White noise field in real space
    # ------------------------------------------------------------------
    white_noise = rng.standard_normal((Ng, Ng, Ng))

    # ------------------------------------------------------------------
    # 2. FFT + P(k) shaping
    # ------------------------------------------------------------------
    noise_k = np.fft.fftn(white_noise)

    # Physical k values: each grid mode corresponds to k_i = 2π * n_i / L
    # We use k in h/Mpc for BBKS but only the shape matters.
    freq = np.fft.fftfreq(Ng) * Ng  # integer wavenumber indices
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing='ij')
    k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    Pk = _power_spectrum(k_mag, n_s=n_s, Omega_m=Omega_m, h=h)
    sqrtPk = np.sqrt(Pk)  # zero at DC

    delta_k = noise_k * sqrtPk

    # ------------------------------------------------------------------
    # 3. Density field in real space
    # ------------------------------------------------------------------
    # (We only need delta_k for the displacement; real-space delta not used directly.)

    # ------------------------------------------------------------------
    # 4. Zel'dovich displacement field Psi(x): shape (Ng, Ng, Ng, 3)
    # ------------------------------------------------------------------
    psi = _zeldovich_displacement_field(delta_k, Ng)

    # Normalise displacement amplitude so it is proportional to the cell size.
    # Without this the raw Psi values are in arbitrary units.
    cell_size_m = box_size_m / Ng
    # RMS of the raw displacement over all voxels and axes
    psi_rms = np.sqrt(np.mean(psi ** 2))
    if psi_rms > 0:
        # Scale so typical displacement ~ 0.5 * cell_size  (mild clustering)
        psi = psi * (0.5 * cell_size_m / psi_rms)

    # ------------------------------------------------------------------
    # 5. Regular Lagrangian grid + displacement
    # ------------------------------------------------------------------
    # Grid points range from -box_size_m/2 to +box_size_m/2
    lin = np.linspace(-box_size_m / 2, box_size_m / 2, Ng, endpoint=False)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    g0 = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # Lagrangian (Ng^3, 3)
    # Eulerian positions after Zel'dovich displacement
    ex = (gx + psi[..., 0]).ravel()
    ey = (gy + psi[..., 1]).ravel()
    ez = (gz + psi[..., 2]).ravel()
    all_positions = np.stack([ex, ey, ez], axis=1)  # (Ng^3, 3)

    # ------------------------------------------------------------------
    # 6. SUPPORT MASK: confine to the uniform_sphere radius (default), so GRF
    #    and uniform_sphere share the SAME geometry and only clustering differs.
    #    The mask uses the UNDISPLACED Lagrangian radius (a fixed, seed-independent
    #    set of cells) so the cut is the cloud's bounding geometry, not a clip on
    #    the structure itself.
    # ------------------------------------------------------------------
    sphere_radius_m = (box_size_m / 2.0) / np.sqrt(3.0 / 5.0)
    if support == "sphere":
        inside = np.linalg.norm(g0, axis=1) <= sphere_radius_m
        all_positions = all_positions[inside]
    # support == "box": keep the full cube grid (legacy).

    # ------------------------------------------------------------------
    # 7. Subsample to exactly N particles
    # ------------------------------------------------------------------
    n_avail = all_positions.shape[0]
    if n_particles <= n_avail:
        # Random without-replacement subsample (deterministic)
        idx = rng.choice(n_avail, size=n_particles, replace=False)
        positions = all_positions[idx]
    else:
        # Tile the available cells and add jitter (edge case: N > available cells)
        repeats = int(np.ceil(n_particles / n_avail))
        tiled = np.tile(all_positions, (repeats, 1))[:n_particles]
        jitter = rng.uniform(-0.1 * cell_size_m, 0.1 * cell_size_m,
                             tiled.shape)
        positions = tiled + jitter

    # ------------------------------------------------------------------
    # 8. Clip excursions. Sphere support: radial clip to the sphere radius (so a
    #    rare large Zel'dovich displacement cannot push a particle outside the
    #    spherical envelope). Box support: legacy per-axis cube clip.
    # ------------------------------------------------------------------
    if support == "sphere":
        r = np.linalg.norm(positions, axis=1, keepdims=True)
        over = (r > sphere_radius_m).ravel()
        if np.any(over):
            positions[over] = positions[over] * (sphere_radius_m / r[over])
    else:
        limit = box_size_m / 2
        positions = np.clip(positions, -limit, limit)

    return positions.astype(np.float64)


def sample_grf_mass(n_particles: int,
                    box_size_m: float,
                    seed: int,
                    Ng: int = 64,
                    n_s: float = 0.965,
                    Omega_m: float = 0.3,
                    h: float = 0.70,
                    support: str = "sphere",
                    delta_rms: float = 0.5,
                    weight_floor: float = 0.05) -> tuple:
    """MASS-WEIGHTED GRF: carry the density contrast in per-particle MASSES on
    QUASI-UNIFORM positions (init_distribution="grfmass").

    The Zel'dovich-displaced ``sample_grf`` encodes delta(x) GEOMETRICALLY — particles
    crowd into overdense cells — which seeds discrete gravitational collapse of the
    crowded knots (the PF23 central-knot contraction). This sampler encodes the SAME
    BBKS-shaped density field in particle WEIGHTS instead: positions stay a jittered
    (quasi-uniform, non-crystalline) Lagrangian grid with NO crowding, and each kept
    cell's weight is ``max(1 + delta, weight_floor)`` with delta normalised to
    ``delta_rms`` over the support cells. The caller (ParticleSystem) rescales the
    weights so the TOTAL mass is exactly the EdS-critical cloud mass (mass-preserving;
    PF1 M=0==EdS depends on the total, not the split).

    Same contract as sample_grf: deterministic per seed, sphere support masks the
    UNDISPLACED grid to the uniform_sphere radius ``(box/2)/sqrt(3/5)``, positions raw
    (centre/RMS-normalisation happens in particles.py).

    Returns:
        (positions (N,3) float64 metres, weights (N,) float64 > 0, mean ~ 1).
    """
    if support not in ("sphere", "box"):
        raise ValueError(
            f"Unknown GRF support {support!r}. Valid: 'sphere' (default), 'box'."
        )
    rng = np.random.default_rng(seed)

    # 1-2. White noise -> BBKS-shaped delta_k (identical recipe to sample_grf).
    white_noise = rng.standard_normal((Ng, Ng, Ng))
    noise_k = np.fft.fftn(white_noise)
    freq = np.fft.fftfreq(Ng) * Ng
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing='ij')
    k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    Pk = _power_spectrum(k_mag, n_s=n_s, Omega_m=Omega_m, h=h)
    delta_k = noise_k * np.sqrt(Pk)

    # 3. REAL-SPACE density contrast (this sampler uses delta directly, no Psi).
    delta = np.real(np.fft.ifftn(delta_k)).ravel()

    # 4. Quasi-uniform positions: the UNDISPLACED Lagrangian grid + sub-cell jitter
    #    (jitter breaks the perfect crystal without creating crowding).
    cell_size_m = box_size_m / Ng
    lin = np.linspace(-box_size_m / 2, box_size_m / 2, Ng, endpoint=False)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    g0 = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # 5. Support mask on the undisplaced grid (same geometry contract as sample_grf).
    sphere_radius_m = (box_size_m / 2.0) / np.sqrt(3.0 / 5.0)
    if support == "sphere":
        keep = np.linalg.norm(g0, axis=1) <= sphere_radius_m
    else:
        keep = np.ones(g0.shape[0], dtype=bool)
    g0, delta = g0[keep], delta[keep]

    # 6. Normalise the contrast over the SUPPORT cells to delta_rms, then weight
    #    each cell max(1+delta, floor) — overdense cells get HEAVY particles instead
    #    of MORE particles.
    std = float(delta.std())
    if std > 0:
        delta = delta * (float(delta_rms) / std)
    weights_all = np.maximum(1.0 + delta, float(weight_floor))

    # 7. Subsample to exactly N cells (deterministic; tile+jitter edge case as grf).
    n_avail = g0.shape[0]
    if n_particles <= n_avail:
        idx = rng.choice(n_avail, size=n_particles, replace=False)
        positions = g0[idx]
        weights = weights_all[idx]
    else:
        repeats = int(np.ceil(n_particles / n_avail))
        positions = np.tile(g0, (repeats, 1))[:n_particles]
        weights = np.tile(weights_all, repeats)[:n_particles]
    positions = positions + rng.uniform(-0.3 * cell_size_m, 0.3 * cell_size_m,
                                        positions.shape)

    # 8. Radial clip (jitter can nudge an edge cell out; same rule as sample_grf).
    if support == "sphere":
        r = np.linalg.norm(positions, axis=1, keepdims=True)
        over = (r > sphere_radius_m).ravel()
        if np.any(over):
            positions[over] = positions[over] * (sphere_radius_m / r[over])
    else:
        limit = box_size_m / 2
        positions = np.clip(positions, -limit, limit)

    return positions.astype(np.float64), weights.astype(np.float64)
