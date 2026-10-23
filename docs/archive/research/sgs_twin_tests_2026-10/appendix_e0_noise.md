# E0: realisation noise of uDALES sensor observations (stage B)

Sources: members = B_T3_L2_s{1,2,3} windows 1-2 prior states (ddof=1 across 32 members, RMS over bins, pooled over runs/windows); T/R = truth vs rep_on_bstar, RMS(diff)/sqrt2 over 3 windows. All values m/s. Observation operator and aggregation as in pilot_analysis.py.

## Parameter spread check (members share parameters)

| run_window | inflow_angle | velocity_magnitude | pressure_gradient_magnitude | sgs_bias_b0 | sgs_bias_b1 | sgs_bias_b2 |
|---|---|---|---|---|---|---|
| s1_w1 | 1.13e-02 | 4.46e-04 | 0.00e+00 | 1.43e-03 | 9.94e-04 | 9.14e-04 |
| s1_w2 | 8.01e-05 | 3.11e-06 | 0.00e+00 | 1.23e-05 | 5.46e-06 | 6.33e-06 |
| s2_w1 | 6.83e-03 | 7.56e-04 | 0.00e+00 | 6.34e-04 | 1.23e-03 | 1.01e-03 |
| s2_w2 | 3.12e-05 | 7.24e-06 | 0.00e+00 | 2.41e-06 | 1.06e-05 | 5.25e-06 |
| s3_w1 | 8.60e-03 | 6.65e-04 | 0.00e+00 | 1.14e-03 | 1.51e-03 | 1.28e-03 |
| s3_w2 | 4.21e-05 | 4.23e-06 | 0.00e+00 | 7.83e-06 | 1.05e-05 | 6.92e-06 |

window-0 prior std (s1): inflow_angle=7.59, velocity_magnitude=1.18, pressure_gradient_magnitude=0, sgs_bias_b0=0.243, sgs_bias_b1=0.172, sgs_bias_b2=0.234

## sigma_noise by interval x height x component (members | truth-replica)

| sensors | interval s | u (mem \| T/R) | v (mem \| T/R) | both (mem \| T/R) |
|---|---|---|---|---|
| L2_z2 | 1 | 0.230 \| 0.222 | 0.196 \| 0.182 | 0.214 \| 0.203 |
| L2_z2 | 20 | 0.168 \| 0.147 | 0.110 \| 0.104 | 0.142 \| 0.127 |
| L2_z2 | 60 | 0.122 \| 0.087 | 0.065 \| 0.066 | 0.098 \| 0.077 |
| L2_z2 | 180 | 0.094 \| 0.053 | 0.040 \| 0.039 | 0.072 \| 0.046 |
| L2_z8 | 1 | 0.292 \| 0.274 | 0.217 \| 0.215 | 0.257 \| 0.246 |
| L2_z8 | 20 | 0.206 \| 0.177 | 0.102 \| 0.111 | 0.162 \| 0.148 |
| L2_z8 | 60 | 0.144 \| 0.101 | 0.061 \| 0.064 | 0.110 \| 0.084 |
| L2_z8 | 180 | 0.106 \| 0.061 | 0.037 \| 0.045 | 0.080 \| 0.054 |
| L2_z14 | 1 | 0.388 \| 0.370 | 0.335 \| 0.348 | 0.362 \| 0.359 |
| L2_z14 | 20 | 0.252 \| 0.238 | 0.144 \| 0.161 | 0.205 \| 0.203 |
| L2_z14 | 60 | 0.177 \| 0.157 | 0.080 \| 0.108 | 0.138 \| 0.135 |
| L2_z14 | 180 | 0.138 \| 0.109 | 0.051 \| 0.073 | 0.104 \| 0.093 |
| L2_all | 1 | 0.310 \| 0.295 | 0.257 \| 0.259 | 0.285 \| 0.277 |
| L2_all | 20 | 0.211 \| 0.191 | 0.120 \| 0.128 | 0.172 \| 0.162 |
| L2_all | 60 | 0.150 \| 0.119 | 0.069 \| 0.082 | 0.116 \| 0.102 |
| L2_all | 180 | 0.114 \| 0.078 | 0.043 \| 0.054 | 0.086 \| 0.067 |
| VAL0_x2_y30_z2 | 1 | 0.157 \| 0.158 | 0.394 \| 0.402 | 0.300 \| 0.305 |
| VAL0_x2_y30_z2 | 20 | 0.107 \| 0.088 | 0.258 \| 0.297 | 0.198 \| 0.219 |
| VAL0_x2_y30_z2 | 60 | 0.065 \| 0.059 | 0.156 \| 0.238 | 0.120 \| 0.173 |
| VAL0_x2_y30_z2 | 180 | 0.038 \| 0.017 | 0.101 \| 0.171 | 0.076 \| 0.122 |
| VAL1_x20_y55_z2 | 1 | 0.227 \| 0.195 | 0.140 \| 0.121 | 0.188 \| 0.162 |
| VAL1_x20_y55_z2 | 20 | 0.170 \| 0.157 | 0.097 \| 0.090 | 0.138 \| 0.128 |
| VAL1_x20_y55_z2 | 60 | 0.106 \| 0.086 | 0.065 \| 0.047 | 0.088 \| 0.070 |
| VAL1_x20_y55_z2 | 180 | 0.075 \| 0.055 | 0.049 \| 0.025 | 0.063 \| 0.042 |
| VAL2_x17_y30_z2 | 1 | 0.032 \| 0.026 | 0.121 \| 0.110 | 0.088 \| 0.080 |
| VAL2_x17_y30_z2 | 20 | 0.026 \| 0.020 | 0.058 \| 0.054 | 0.045 \| 0.041 |
| VAL2_x17_y30_z2 | 60 | 0.019 \| 0.013 | 0.036 \| 0.028 | 0.028 \| 0.022 |
| VAL2_x17_y30_z2 | 180 | 0.012 \| 0.010 | 0.026 \| 0.011 | 0.020 \| 0.011 |
| VAL3_x10_y30_z20 | 1 | 0.418 \| 0.384 | 0.373 \| 0.374 | 0.396 \| 0.379 |
| VAL3_x10_y30_z20 | 20 | 0.271 \| 0.230 | 0.154 \| 0.160 | 0.220 \| 0.198 |
| VAL3_x10_y30_z20 | 60 | 0.186 \| 0.128 | 0.087 \| 0.103 | 0.145 \| 0.116 |
| VAL3_x10_y30_z20 | 180 | 0.132 \| 0.067 | 0.052 \| 0.028 | 0.100 \| 0.051 |

## Autocorrelation time of 1 s member anomalies (s)

| group | tau_int (to first zero) | lag ACF<1/e | first-zero lag |
|---|---|---|---|
| z2_u | 12.74 | 7.0 | 91 |
| z2_v | 3.59 | 5.0 | 7 |
| z8_u | 11.52 | 7.0 | 91 |
| z8_v | 2.74 | 4.0 | 6 |
| z14_u | 10.40 | 6.0 | 91 |
| z14_v | 2.28 | 3.0 | 6 |
| L2_all_u | 11.55 | 7.0 | 91 |
| L2_all_v | 2.85 | 4.0 | 7 |
| L2_all_both | 7.43 | 5.0 | 91 |
| VAL0_z2_u | 5.15 | 6.0 | 13 |
| VAL0_z2_v | 5.30 | 6.0 | 23 |
| VAL1_z2_u | 7.55 | 9.0 | 19 |
| VAL1_z2_v | 11.02 | 7.0 | 91 |
| VAL2_z2_u | 11.99 | 12.0 | 75 |
| VAL2_z2_v | 3.32 | 4.0 | 6 |
| VAL3_z20_u | 6.34 | 5.0 | 46 |
| VAL3_z20_v | 2.00 | 3.0 | 5 |

Over-weighting of raw 1 s frames ~ tau/1 s = **7.4** (tau_int, L2 all); effective independent samples per 180 s window ~ 180/(2 tau) = **12.1**.

## Frozen sigma_r

The states are clean (no instrument noise), so the measured member/truth noise is pure realisation noise; the instrument term (0.25 m/s) is separate and added by the DA's persistent model as sqrt(0.25^2/n + sigma_r^2). Hence sigma_r = sigma_noise directly (L2 RMS over sensors and u,v, rounded to 0.005).

| interval s | sigma_noise L2 (members) | sigma_r frozen | T/R L2 pooled | z2 | z8 | z14 (frozen bands) | max/min over heights |
|---|---|---|---|---|---|
| 1 | 0.2848 | 0.285 | 0.2775 | 0.215 | 0.255 | 0.36 | 1.69 |
| 20 | 0.1718 | 0.17 | 0.1624 | 0.14 | 0.16 | 0.205 | 1.45 |
| 60 | 0.1164 | 0.115 | 0.1019 | 0.1 | 0.11 | 0.14 | 1.41 |
| 180 | 0.0863 | 0.085 | 0.0674 | 0.07 | 0.08 | 0.105 | 1.44 |

Height bands needed (>1.5x between heights): [1]

## Gate (Round 1 SGS signal: p1_cv024_b0_0 vs p2_on_bstar, L2, 3 windows)

sigma_eff = sqrt(0.25^2/n + sigma_r^2), n = frames per bin. Over-weighting f = max(tau/interval, 1); N_eff = N_obs / f.

| interval s | signal RMS | sigma_r | sigma_eff | D | obs/window | f | N_eff | D*sqrt(N_eff) | D*sqrt(N_obs) naive |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0332 | 0.285 | 0.379 | 0.088 | 6480 | 7.4 | 872 | 2.59 | 7.06 |
| 20 | 0.0309 | 0.17 | 0.179 | 0.173 | 324 | 1.0 | 324 | 3.11 | 3.11 |
| 60 | 0.0267 | 0.115 | 0.119 | 0.224 | 108 | 1.0 | 108 | 2.33 | 2.33 |
| 180 | 0.0259 | 0.085 | 0.087 | 0.297 | 36 | 1.0 | 36 | 1.78 | 1.78 |
