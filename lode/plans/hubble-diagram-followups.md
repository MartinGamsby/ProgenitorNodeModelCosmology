# Hubble-Diagram Follow-ups

Open items related to the data-anchored Pantheon+ test
([../physics/hubble-diagram.md](../physics/hubble-diagram.md)). Human's call.

## 1. Reconcile the (M,S) discrepancy (PRIORITY — physics)
The (M,S) that best matches real SNe (Omega_Lambda_eff ~= 0.70, near
M=855,S=37.8) differs from the paper's primary N-body config
(M=9000,S=38 -> Omega_Lambda_eff ~= 7.24, a closed universe). Decide whether to
revisit the N-body match metric, the Omega_Lambda_eff normalization, or the
paper's headline config. Until resolved, the script defaults to the
data-matching config, not the paper's primary config.

## 2. Optional paper edit (NOT done — needs human approval)
A short paragraph in docs/VirializedMetaStructure.tex (Numerical Validation
section) describing the Pantheon+ background test would strengthen credibility:
model H(z) -> mu(z), overlaid on real SNe, offset marginalized, chi^2/R^2 per
model. Paper edits are the human's decision; not made in this pass.

## 3. Full covariance chi^2 (future)
v1 uses the diagonal MU_SH0ES_ERR_DIAG column only. A covariant chi^2 would load
Pantheon+SH0ES_STAT+SYS.cov (hook noted in cosmo/pantheon.py and
data/pantheon_plus/README.md).

## 4. N-body-derived d_L (future)
Currently H(z) is semi-analytic. An N-body-derived d_L is deliberate future work
(N-body a(t) only covers z~0..1.2; differentiation edge artifacts) — would
require extending the sim earlier and a robust a(t)->H(z) path.
