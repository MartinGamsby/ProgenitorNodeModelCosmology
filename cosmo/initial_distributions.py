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
               h: float = 0.70) -> np.ndarray:
    """Generate particle positions via a Gaussian random field + Zel'dovich displacement.

    Recipe (deterministic for a fixed seed):
    1. Draw a Gaussian white-noise field of shape (Ng^3) in real space.
    2. FFT to k-space, multiply by sqrt(P(k)) to impose BBKS LCDM spectrum.
    3. IFFT to get density contrast delta(x).
    4. Compute the Zel'dovich displacement field Psi(x) from delta(x).
    5. Place N particles on a regular grid and displace each by Psi at its
       grid cell.
    6. Subsample / select N particles (all of them — the grid already has Ng^3
       points; if N < Ng^3 we randomly subsample; if N > Ng^3 we tile).
    7. Scale positions to fit inside the target sphere radius (box_size_m / 2).
       The shared RMS-normalisation in particles.py will then rescale to the
       exact target regardless, so only the SHAPE / clustering is baked in here.

    Args:
        n_particles : number of output particles N.
        box_size_m  : physical box size in metres (sets the Zel'dovich amplitude
                      so displacements are proportional to box/Ng).
        seed        : integer seed → deterministic, reproducible.
        Ng          : grid resolution (default 64; higher = more structure detail).
        n_s         : scalar spectral index (LCDM default 0.965).
        Omega_m     : matter fraction (LCDM default 0.3).
        h           : dimensionless Hubble (LCDM default 0.70).

    Returns:
        positions : (N, 3) float64 array in metres, raw (NOT centred/normalised).
                    Particles lie within [-box_size_m/2, box_size_m/2] approximately.
    """
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
    # Eulerian positions after Zel'dovich displacement
    ex = (gx + psi[..., 0]).ravel()
    ey = (gy + psi[..., 1]).ravel()
    ez = (gz + psi[..., 2]).ravel()
    all_positions = np.stack([ex, ey, ez], axis=1)  # (Ng^3, 3)

    # ------------------------------------------------------------------
    # 6. Subsample to exactly N particles
    # ------------------------------------------------------------------
    n_grid = Ng ** 3
    if n_particles <= n_grid:
        # Random without-replacement subsample (deterministic)
        idx = rng.choice(n_grid, size=n_particles, replace=False)
        positions = all_positions[idx]
    else:
        # Tile the grid and add jitter (edge case: N > Ng^3)
        repeats = int(np.ceil(n_particles / n_grid))
        tiled = np.tile(all_positions, (repeats, 1))[:n_particles]
        jitter = rng.uniform(-0.1 * cell_size_m, 0.1 * cell_size_m,
                             tiled.shape)
        positions = tiled + jitter

    # Clip to bounding box to prevent wild Zel'dovich excursions
    limit = box_size_m / 2
    positions = np.clip(positions, -limit, limit)

    return positions.astype(np.float64)
