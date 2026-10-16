"""Resume / per-cell checkpoint tests for sweep.py.

The sweep checkpoints each finished cell to the results CSV and, on a re-run,
skips cells already present. The correctness gate is the KEY-MATCH property: a
cell written to disk (all strings) must produce the SAME resume key as the live
identity that built it — otherwise a resumed run silently recomputes everything
(the bug that lost hours after a Claude restart). These tests pin that property
without running any simulation.
"""
import csv
import io
import math

import sweep


def _result_row(**over):
    """A representative result row, matching what _run_cell_fixed_S produces."""
    row = dict(M_factor=10, S_gpc=20, centerM=1.0, outer_density_ceiling=1.0,
               node_mass_amplitude=0.0, node_s_amplitude=0.0, node_mass_seed=42,
               init_distribution="uniform_sphere", node_geometry="cube26",
               chi2_dof=0.52, chi2=800.0, chi2_lcdm=0.436, chi2_eds=0.843,
               R2=0.99, n_sne_used=1500, growth_factor=3.1, growth_target=3.1,
               anchor_ok=True, runaway=False, match_avg_pct=65.0, diff_pct=35.0)
    row.update(over)
    return row


def _written_then_parsed(row):
    """Round-trip a row through the CSV exactly as the checkpoint writer does."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=sweep.SWEEP_CSV_COLS, extrasaction="ignore")
    w.writeheader()
    w.writerow(row)
    buf.seek(0)
    return sweep._parse_csv_row(next(csv.DictReader(buf)))


def test_written_cell_recognized_on_resume_fixed_S():
    """A checkpointed cell yields the same key as its live identity -> SKIPPED."""
    ident = dict(M_factor=10, S_gpc=20, centerM=1.0, outer_density_ceiling=1.0,
                 node_mass_amplitude=0.0, node_s_amplitude=0.0, node_mass_seed=42,
                 init_distribution="uniform_sphere", node_geometry="cube26")
    live_key = sweep._resume_key(ident, include_S=True)
    disk_key = sweep._resume_key(_written_then_parsed(_result_row()), include_S=True)
    assert live_key == disk_key


def test_string_typed_identity_match():
    """centerM '1.0' from disk must match 1.0 live (the silent-recompute bug)."""
    live = sweep._resume_key(
        dict(M_factor=10, S_gpc=20, centerM=1.0, outer_density_ceiling=1.0,
             node_mass_amplitude=0.0, node_s_amplitude=0.0, node_mass_seed=42,
             init_distribution="uniform_sphere", node_geometry="bcc"), include_S=True)
    disk = sweep._resume_key(
        dict(M_factor="10", S_gpc="20", centerM="1.0", outer_density_ceiling="1.0",
             node_mass_amplitude="0.0", node_s_amplitude="0.0", node_mass_seed="42",
             init_distribution="uniform_sphere", node_geometry="bcc"), include_S=True)
    assert live == disk


def test_cofit_key_ignores_S():
    """In co-fit mode S is searched (an output), so identity excludes it."""
    a = sweep._resume_key(_result_row(S_gpc=20), include_S=False)
    b = sweep._resume_key(_result_row(S_gpc=55), include_S=False)
    assert a == b


def test_fixed_S_key_distinguishes_S():
    a = sweep._resume_key(_result_row(S_gpc=20), include_S=True)
    b = sweep._resume_key(_result_row(S_gpc=30), include_S=True)
    assert a != b


def test_key_distinguishes_centerM_geometry_amplitude():
    base = sweep._resume_key(_result_row(), include_S=True)
    assert sweep._resume_key(_result_row(centerM=2.0), include_S=True) != base
    assert sweep._resume_key(_result_row(node_geometry="bcc"), include_S=True) != base
    assert sweep._resume_key(_result_row(node_mass_amplitude=0.5), include_S=True) != base
    assert sweep._resume_key(_result_row(node_mass_seed=7), include_S=True) != base


def test_all_resume_cols_are_persisted():
    """Every resume-key column must be in SWEEP_CSV_COLS or it can't be matched."""
    cols = sweep._RESUME_NUM_COLS + sweep._RESUME_TXT_COLS
    assert all(c in sweep.SWEEP_CSV_COLS for c in cols)


def test_parse_csv_row_types():
    parsed = _written_then_parsed(
        _result_row(chi2_dof=float("inf"), anchor_ok=False, runaway=True))
    assert isinstance(parsed["M_factor"], int) and parsed["M_factor"] == 10
    assert parsed["centerM"] == 1.0
    assert parsed["anchor_ok"] is False
    assert parsed["runaway"] is True
    assert math.isinf(parsed["chi2_dof"])
