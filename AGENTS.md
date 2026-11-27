# Development

Read README.md, docs/DESIGN.md, NOTES.md and MISTAKES.md before changing the solver.
Use SemVer and record tested changes in CHANGELOG.md. Make focused local commits
at tested milestones. Keep original JonesRec and Hodge repositories unchanged.

Preserve the current qhsolve reconstruction strategy. Changes to blocked
elimination, pivot profiles, normalization, support, interpolation, CRT, reference
selection, denominator clearing or cache keys require independent arithmetic
regressions. Performance claims require equal-output cold end-to-end comparisons,
with equal validation and persistence, not cached stage timings.

Keep finite-data validity distinct from all-index annihilation and minimality.
A zero modular kernel proves emptiness only for the tested finite ansatz. Gaps,
exceptional images and insufficient data are not empty boxes. General module
relation spaces need not have dimension one or be principal ideals.

Use E for forward shift, Q for multiplication by q**n, and EQ=qQE. Preserve
caller-supplied Ore generator names at the adapter boundary. Record their action.
Never silently truncate rationals to integers or coerce approximate data to exact.

Defaults must respect an aggregate worker-memory budget. Caches are caller chosen;
bind reusable answers to the full mathematical observations and ordered support.
Never overwrite unrelated jobs, databases, manifests or benchmark results.
