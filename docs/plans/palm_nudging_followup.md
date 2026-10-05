# PALM nudging out-of-bounds read: follow-up

The code fix is merged (#173, closes #165); details in `docs/pypalm.md` §8.
What is left is outside the code.

## What happened

PALM's `nudge_ref` interpolates the NUDGING_DATA profiles on the LSF_DATA time
axis (`time_vert`) instead of `timenudge`. Our inert LSF_DATA had its only time
past `end_time`, so every **periodic** PALM run before #173 read
`unudge(:,0)`/`vnudge(:,0)` out of bounds into `u_init`/`v_init`, which set the
top boundary (Rayleigh damping is off, `rayleigh_damping_factor = 0`).

On macOS that memory held zeros: a **u = v = 0 lid** at the top. Measured on
the Xie & Castro case (#173), the mean u changes by +4.8 m/s at the top level,
+1.7 at 31 m and +0.6 at 29 m, and fades out below ~25 m. At the z = 2 m sensors
it changes by ≤ 0.01 m/s. Occasionally (~1 in 20 runs) the read crashed the run
with SIGBUS instead. `inflow_outflow` runs are not affected.

## To do

1. **File the upstream report** with PALM. The draft is in the #173
   description (cause, reproduction, three-line patch to use `timenudge`).
2. **Rerun past periodic PALM results that use the upper part of the domain**
   (full-state metrics, mean profiles, domain means). Results built only from
   near-ground sensors (the z = 2 m DA observations) are essentially unchanged.
3. **Snellius / DelftBlue (optional).** What the out-of-bounds memory held on
   Linux is unknown: zeros, other values or garbage. If periodic PALM results
   from the clusters matter, run one periodic case there on a commit before
   #173 with `u_init`/`v_init` printed at the top level (a debug print in a
   scratch build), or simply compare one old cluster run with a rerun on
   current `main`.

## Done when

The report is filed (link it on #165), the affected results are rerun or marked
as pre-#173, and this file moves to `docs/plans/implemented/`.
