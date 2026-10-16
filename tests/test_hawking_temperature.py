"""
Hawking temperature of the HMEA nodes (paper sec:defenses, "dark giant" principle).

The formula is checked three independent ways (literature anchors, an independent
CODATA source, an independent derivation route), then the paper's quoted figure is
checked against it so a wrong number cannot creep back into the draft.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import scipy.constants as sc

from cosmo.constants import CosmologicalConstants as CC, hawking_temperature_K

PAPER = Path(__file__).resolve().parent.parent / "docs" / "VirializedMetaStructure.tex"
M_FIDUCIAL_KG = 5e55  # 500 M_obs, the paper's matching-condition node mass


def test_solar_mass_matches_literature():
    """Textbook anchor: T_H ≈ 6.17e-8 K × (M_sun/M) (e.g. arXiv:2410.11851)."""
    assert hawking_temperature_K(CC.M_sun) == pytest.approx(6.17e-8, abs=0, rel=5e-3)


def test_matches_independent_codata_source():
    """Same formula with scipy's CODATA constants (catches a typo in our constants)."""
    M = M_FIDUCIAL_KG
    expected = sc.hbar * sc.c**3 / (8 * np.pi * sc.G * M * sc.k)
    assert hawking_temperature_K(M) == pytest.approx(expected, abs=0, rel=1e-6)


def test_matches_surface_gravity_route():
    """T_H = ħκ/(2πck_B) with κ = c²/(2 r_s), r_s = 2GM/c² (independent algebra)."""
    M = M_FIDUCIAL_KG
    r_s = 2 * CC.G * M / CC.c**2
    kappa = CC.c**2 / (2 * r_s)
    expected = CC.hbar * kappa / (2 * np.pi * CC.c * CC.k_B)
    assert hawking_temperature_K(M) == pytest.approx(expected, abs=0, rel=1e-12)


def test_inverse_mass_scaling():
    T1 = hawking_temperature_K(1e55)
    for factor in (2.0, 10.0, 3000.0):
        assert hawking_temperature_K(1e55 * factor) == pytest.approx(T1 / factor, abs=0, rel=1e-12)


def test_cmb_equilibrium_mass_is_sub_lunar():
    """Literature anchor: a black hole as hot as the CMB has roughly a lunar mass."""
    M_moon = 7.342e22
    M_eq = hawking_temperature_K(1.0) / CC.T_CMB_K  # T ∝ 1/M
    assert M_eq == pytest.approx(4.5e22, abs=0, rel=0.02)
    assert 0.5 * M_moon < M_eq < M_moon


def test_hmea_values():
    """5e55 kg → 2.45e-33 K; heavier nodes colder; nowhere near the old 1e-83 K."""
    T = hawking_temperature_K(M_FIDUCIAL_KG)
    assert T == pytest.approx(2.454e-33, abs=0, rel=1e-3)
    assert hawking_temperature_K(9000 * CC.M_observable_kg) == pytest.approx(1.363e-34, abs=0, rel=1e-3)
    assert CC.T_CMB_K / T == pytest.approx(1.11e33, abs=0, rel=1e-2)
    assert abs(np.log10(T) - (-83)) > 40


def test_paper_quotes_correct_value():
    """The figure in sec:defenses must agree with the formula for the quoted mass."""
    if not PAPER.exists():
        pytest.skip("paper source not present")
    tex = PAPER.read_text(encoding="utf-8")
    m = re.search(
        r"T_H = [^$]*\\approx ([\d.]+) \\times 10\^\{(-?\d+)\}\$ K for "
        r"\$M_\{ext\} \\approx ([\d.]+) \\times 10\^\{(\d+)\}\$ kg, "
        r"some \$10\^\{(\d+)\}\$ times colder than the CMB",
        tex,
    )
    assert m, "Hawking-temperature sentence in sec:defenses not found / reworded"
    T_quoted = float(m[1]) * 10.0 ** int(m[2])
    M_quoted = float(m[3]) * 10.0 ** int(m[4])
    T_true = hawking_temperature_K(M_quoted)
    assert T_quoted == pytest.approx(T_true, abs=0, rel=0.05)
    assert int(m[5]) == round(np.log10(CC.T_CMB_K / T_true))
