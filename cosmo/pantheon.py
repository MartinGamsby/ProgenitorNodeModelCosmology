"""
Pantheon+SH0ES Supernova Loader

Loads the Pantheon+SH0ES distance-modulus compilation from a local vendored
file and returns redshift / distance-modulus / uncertainty arrays for use in
the Hubble-diagram comparison.

The real data file is NOT committed to this repository. Acquire it manually:
see data/pantheon_plus/README.md for source URL, citation, and placement
instructions. A small synthetic fixture for testing lives at:
    tests/fixtures/pantheon_synthetic.dat

No network access is performed at runtime.
"""

import os
import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# Default path (resolved relative to THIS module, not cwd)
# ---------------------------------------------------------------------------
_MODULE_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parent

DEFAULT_PATH: pathlib.Path = (
    _REPO_ROOT / "data" / "pantheon_plus" / "Pantheon+SH0ES.dat"
)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_pantheon(
    path: os.PathLike | str | None = None,
    z_min: float = 0.01,
    exclude_calibrators: bool = True,
) -> dict:
    """
    Load the Pantheon+SH0ES supernova Hubble diagram.

    Reads a whitespace-delimited table by column name (not by index), applies
    a minimum-redshift cut, and optionally removes Cepheid-host calibrators.

    Args:
        path:
            Path to the data file.  Defaults to DEFAULT_PATH
            (``<repo>/data/pantheon_plus/Pantheon+SH0ES.dat``).
        z_min:
            Minimum redshift to include (default 0.01).  Rows with
            ``zHD < z_min`` are dropped to avoid peculiar-velocity
            dominated SNe.
        exclude_calibrators:
            If True (default), rows with ``IS_CALIBRATOR == 1`` are
            excluded.  Set to False to include Cepheid-host calibrators.

    Returns:
        dict with keys:
            ``z``     – np.ndarray of redshifts, sorted ascending.
            ``mu``    – np.ndarray of distance moduli (magnitudes).
            ``sigma`` – np.ndarray of diagonal mu uncertainties (magnitudes).
            ``n``     – int, number of SNe after all cuts.

    Raises:
        FileNotFoundError:
            If the data file is not present.  The error message explains
            where to place the file and points to data/pantheon_plus/README.md.

    Notes:
        Column names used:
            ``zHD``, ``MU_SH0ES``, ``MU_SH0ES_ERR_DIAG``, ``IS_CALIBRATOR``
        Mapping by name means minor reordering in future releases is safe.

        Future hook: full covariance chi^2 can be added by also loading
        ``Pantheon+SH0ES_STAT+SYS.cov`` from the same directory.
        See data/pantheon_plus/README.md for details.
    """
    if path is None:
        path = DEFAULT_PATH
    path = pathlib.Path(path)

    if not path.exists():
        readme = _REPO_ROOT / "data" / "pantheon_plus" / "README.md"
        raise FileNotFoundError(
            f"Pantheon+SH0ES data file not found at:\n"
            f"  {path}\n\n"
            f"This file must be acquired manually from the PantheonPlusSH0ES "
            f"data release.\n"
            f"See the acquisition instructions in:\n"
            f"  {readme}\n"
            f"Place the downloaded 'Pantheon+SH0ES.dat' at the path above."
        )

    # Parse whitespace-delimited table by header name
    data = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")

    # Extract columns by name (case-sensitive, as in the real file header)
    z = data["zHD"].astype(float)
    mu = data["MU_SH0ES"].astype(float)
    sigma = data["MU_SH0ES_ERR_DIAG"].astype(float)
    is_cal = data["IS_CALIBRATOR"].astype(int)

    # Build boolean mask: start with all True
    mask = np.ones(len(z), dtype=bool)

    # Apply z_min cut
    mask &= z >= z_min

    # Exclude calibrators if requested
    if exclude_calibrators:
        mask &= is_cal == 0

    z = z[mask]
    mu = mu[mask]
    sigma = sigma[mask]

    # Sort by redshift (aids plotting and binning)
    order = np.argsort(z)
    z = z[order]
    mu = mu[order]
    sigma = sigma[order]

    return {"z": z, "mu": mu, "sigma": sigma, "n": int(len(z))}


# ---------------------------------------------------------------------------
# Optional: inverse-variance weighted binning for plot overlays
# ---------------------------------------------------------------------------

def bin_for_plot(
    z: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_bins: int = 20,
) -> dict:
    """
    Bin SNe Ia into equal-log-z bins for cleaner Hubble-diagram overlays.

    Uses inverse-variance weighting within each bin.  Statistics (chi^2,
    goodness-of-fit) should always be computed on the UNBINNED data.

    Args:
        z:      Redshift array (sorted ascending).
        mu:     Distance modulus array.
        sigma:  Diagonal uncertainty array.
        n_bins: Number of bins in log-z space (default 20).  Actual number
                of returned bins may be fewer if some are empty.

    Returns:
        dict with keys:
            ``z``     – bin-centre redshifts (np.ndarray).
            ``mu``    – inverse-variance weighted mean mu per bin.
            ``err``   – weighted uncertainty (1/sqrt(sum of weights)).
            ``n_sne`` – number of SNe per bin (np.ndarray, int).
    """
    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if len(z) == 0:
        empty = np.array([], dtype=float)
        return {"z": empty, "mu": empty, "err": empty, "n_sne": np.array([], dtype=int)}

    log_z_min = np.log10(z.min())
    log_z_max = np.log10(z.max())

    # Guard against degenerate range (e.g. a single unique z value)
    if log_z_min == log_z_max:
        log_z_max = log_z_min + 1e-6

    edges = np.linspace(log_z_min, log_z_max, n_bins + 1)
    log_z = np.log10(z)

    bin_z: list[float] = []
    bin_mu: list[float] = []
    bin_err: list[float] = []
    bin_n: list[int] = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (log_z >= edges[i]) & (log_z < edges[i + 1])
        else:
            # Include right edge in last bin
            mask = (log_z >= edges[i]) & (log_z <= edges[i + 1])

        if not np.any(mask):
            continue

        w = 1.0 / sigma[mask] ** 2
        w_sum = w.sum()
        mu_w = (w * mu[mask]).sum() / w_sum
        err_w = 1.0 / np.sqrt(w_sum)
        z_w = (w * z[mask]).sum() / w_sum  # weighted mean z

        bin_z.append(float(z_w))
        bin_mu.append(float(mu_w))
        bin_err.append(float(err_w))
        bin_n.append(int(np.sum(mask)))

    return {
        "z": np.array(bin_z),
        "mu": np.array(bin_mu),
        "err": np.array(bin_err),
        "n_sne": np.array(bin_n, dtype=int),
    }
