# Pantheon+SH0ES Data

## Source

**Dataset:** Pantheon+SH0ES supernova compilation (Scolnic et al. 2022; Brout et al. 2022)

**References:**
- Scolnic, D. et al. 2022, ApJ 938:113 — Pantheon+ SN Ia compilation
- Brout, D. et al. 2022, ApJ 938:110 — Pantheon+ cosmological analysis

**GitHub release:**
```
https://github.com/PantheonPlusSH0ES/DataRelease
```

**File to download:** `Pantheon+SH0ES.dat`

**Location in the release:** `Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat`

**Direct link (as of the v1 tag):**
```
https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat
```

## Acquisition instructions

1. Visit the release repository linked above.
2. Navigate to `Pantheon+_Data/4_DISTANCES_AND_COVAR/`.
3. Download `Pantheon+SH0ES.dat` (the raw file, ~several MB).
4. Place it at the path the loader expects:

```
<repo>/data/pantheon_plus/Pantheon+SH0ES.dat
```

The loader (`cosmo/pantheon.py`) reads this path by default. The file must be present for
`load_pantheon()` to work with real data; if it is absent, a clear `FileNotFoundError` is
raised pointing here.

## Column mapping

The loader reads the file by **header name** (whitespace-delimited). The columns used are:

| Column name            | Description                                                      |
|------------------------|------------------------------------------------------------------|
| `zHD`                  | Hubble-diagram redshift (CMB frame, peculiar-velocity corrected) |
| `MU_SH0ES`             | Distance modulus (mu), in magnitudes                            |
| `MU_SH0ES_ERR_DIAG`    | Diagonal uncertainty on mu (sigma), in magnitudes               |
| `IS_CALIBRATOR`        | 1 for Cepheid-host calibrators; 0 for Hubble-flow SNe           |

Column names are not hardcoded by index; the loader maps by name, so minor whitespace or
ordering changes in future releases should not break parsing.

## Filters applied by default

- `z >= 0.01` (`z_min=0.01`): removes very-low-z SNe dominated by peculiar velocities.
- `IS_CALIBRATOR == 0` (`exclude_calibrators=True`): removes Cepheid-host calibrators from
  the Hubble-flow fit. Pass `exclude_calibrators=False` to include them.

## License / provenance note

The Pantheon+SH0ES dataset is made publicly available by the survey team for scientific use.
Cite the papers above when using this data. The file is NOT committed to this repository
because of its size; it is acquired manually by the researcher.

## Optional: full covariance matrix

The full statistical + systematic covariance matrix is in:
```
Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov
```

The current loader (`v1`) uses only the diagonal `MU_SH0ES_ERR_DIAG` column. Future work can
extend to a covariant chi^2 fit by loading this matrix. A hook is left in `cosmo/pantheon.py`
as a comment.
