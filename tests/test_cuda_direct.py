"""Tests for the WS7 GPU direct-sum kernel (cosmo.cuda_direct, force_method='cuda_direct').

The kernel-EXECUTING correctness check runs in an isolated subprocess with the
CUDA SIMULATOR enabled (NUMBA_ENABLE_CUDASIM=1) so it validates the real kernel
logic on CPU regardless of whether a physical GPU is present -- the PF22-style
falsification gate (cuda_direct must reproduce numba_direct). The lightweight
wiring tests (valid force_method, cache slug) need no CUDA backend.
"""

import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_under_cudasim(body: str) -> str:
    """Run a python snippet in a fresh process with the CUDA simulator on."""
    env = dict(os.environ)
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return proc.stdout


# --------------------------------------------------------------------------
# Kernel correctness (CUDA simulator) -- cuda_direct == numba_direct
# --------------------------------------------------------------------------

def test_cuda_direct_matches_numba_direct_under_simulator():
    """GPU kernel reproduces the CPU numba_direct accelerations to ~1e-9."""
    out = _run_under_cudasim(
        """
        import numpy as np
        from cosmo.numba_direct import NumbaDirectSolver
        from cosmo.cuda_direct import CudaDirectSolver, cuda_backend_available

        assert cuda_backend_available(), "simulator backend should be available"

        rng = np.random.default_rng(0)
        # Physically-scaled magnitudes: positions ~1e26 m, masses ~1e52 kg.
        pos = rng.normal(0.0, 1e26, size=(40, 3))
        mass = rng.uniform(0.5, 1.5, size=40) * 1e52
        G = 6.674e-11

        for soft in (0.0, 3.086e25):   # zero + ~1 Gpc softening
            cpu = NumbaDirectSolver(softening_m=soft, G=G)
            cpu.build_tree(pos, mass)
            a_cpu = cpu.calculate_all_accelerations()

            gpu = CudaDirectSolver(softening_m=soft, G=G, threads_per_block=16)
            gpu.build_tree(pos, mass)
            a_gpu = gpu.calculate_all_accelerations()

            assert a_gpu.shape == a_cpu.shape == (40, 3)
            assert np.allclose(a_gpu, a_cpu, rtol=1e-9, atol=0.0), (
                f"soft={soft}: max rel err "
                f"{np.max(np.abs(a_gpu - a_cpu) / (np.abs(a_cpu) + 1e-300))}"
            )
        print("MATCH")
        """
    )
    assert "MATCH" in out


def test_cuda_direct_empty_system_under_simulator():
    """N=0 returns a (0, 3) array, no launch error."""
    out = _run_under_cudasim(
        """
        import numpy as np
        from cosmo.cuda_direct import CudaDirectSolver
        s = CudaDirectSolver()
        s.build_tree(np.zeros((0, 3)), np.zeros(0))
        a = s.calculate_all_accelerations()
        assert a.shape == (0, 3)
        print("EMPTY_OK")
        """
    )
    assert "EMPTY_OK" in out


def test_cuda_direct_integrator_dispatch_matches_numba_under_simulator():
    """End-to-end via the integrator: calculate_total_forces() with
    force_method='cuda_direct' hits the new dispatch branch and equals the
    numba_direct total (internal gravity on GPU, everything else identical)."""
    out = _run_under_cudasim(
        """
        import numpy as np
        from cosmo.constants import CosmologicalConstants
        from cosmo.particles import ParticleSystem
        from cosmo.integrator import LeapfrogIntegrator

        c = CosmologicalConstants()
        particles = ParticleSystem(
            n_particles=50, box_size_m=20.0 * c.Gpc_to_m, total_mass_kg=10 * 1e53,
        )

        def total(fm):
            ig = LeapfrogIntegrator(
                particles, use_external_nodes=False, use_dark_energy=False,
                force_method=fm,
            )
            assert ig._active_force_method == fm
            return ig.calculate_total_forces()

        a_cpu = total('numba_direct')
        a_gpu = total('cuda_direct')
        assert a_gpu.shape == a_cpu.shape
        assert np.allclose(a_gpu, a_cpu, rtol=1e-9, atol=0.0), (
            f"max rel err {np.max(np.abs(a_gpu - a_cpu) / (np.abs(a_cpu) + 1e-300))}"
        )
        print("STEP_MATCH")
        """
    )
    assert "STEP_MATCH" in out


# --------------------------------------------------------------------------
# Wiring (no CUDA backend needed)
# --------------------------------------------------------------------------

def test_cuda_direct_is_valid_force_method():
    from cosmo.constants import SimulationParameters
    p = SimulationParameters(force_method="cuda_direct")
    assert p.force_method == "cuda_direct"


def test_invalid_force_method_still_rejected():
    from cosmo.constants import SimulationParameters
    with pytest.raises(ValueError):
        SimulationParameters(force_method="cuda_wrong")


def test_cuda_direct_cache_slug_keyed_equals_run():
    """cuda_direct gets its own cache key ('cudadirectfm'); auto stays byte-identical."""
    from cosmo.parameter_sweep import build_cache_name, SweepConfig

    args = (100.0, 30.0, 1.0, [0])  # M_factor, S_val, centerM, seeds
    auto_key = build_cache_name(SweepConfig(force_method="auto"), *args)
    cuda_key = build_cache_name(SweepConfig(force_method="cuda_direct"), *args)

    assert "cudadirectfm" in cuda_key
    assert "cudadirectfm" not in auto_key
    assert cuda_key != auto_key


def test_backend_available_reflects_env(monkeypatch):
    """cuda_backend_available() is True under the simulator flag."""
    import cosmo.cuda_direct as cd
    monkeypatch.setenv("NUMBA_ENABLE_CUDASIM", "1")
    assert cd.cuda_backend_available() is True
