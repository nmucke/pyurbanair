# SGS value analysis (T3 vs T2)
Evidence from 3 seeds, not a statistical claim. Validation sensors (held out). Posterior = mean of per-window values over all 3 windows; forecast = `prior` source at windows 1-2 (forecast from the previous window's posterior). Replica floor = replica-vs-truth on the same windows (max over the 6 runs). Verdict: |T3−T2| > max(floor, seed sd) and same sign in all 3 seeds. Spectra: signed dB band ratio, |dB| used (0 = perfect). A_T3_L2_s1_filtering FAILED and is skipped.

## 1. T3 vs T2, stage A (inlet turbulence off), L2
| score | windows | orient | T2 (mean±sd) | T3 (mean±sd) | T3−T2 | rel % | replica floor | sign/seed | verdict | ≥10% CRPS |
|---|---|---|---|---|---|---|---|---|---|---|
| sensor CRPS mean_u | posterior | lower | 0.0348±0.00206 | 0.0106±0.00408 | -0.0242 | -69.6 | 6.77e-05 | --- | T3 better | YES |
| sensor CRPS mean_u | forecast(w1-2) | lower | 0.0465±0.00245 | 0.0232±0.00493 | -0.0233 | -50.2 | 9.08e-05 | --- | T3 better | YES |
| sensor CRPS mean_v | posterior | lower | 0.0257±0.00332 | 0.00936±0.00217 | -0.0163 | -63.6 | 0.0001 | --- | T3 better | YES |
| sensor CRPS mean_v | forecast(w1-2) | lower | 0.029±0.00224 | 0.0132±0.00282 | -0.0158 | -54.6 | 0.000137 | --- | T3 better | YES |
| sensor CRPS mean_magnitude | posterior | lower | 0.0327±0.00207 | 0.0109±0.00352 | -0.0217 | -66.6 | 6.4e-05 | --- | T3 better | YES |
| sensor CRPS mean_magnitude | forecast(w1-2) | lower | 0.0441±0.00266 | 0.0235±0.00452 | -0.0206 | -46.8 | 8.23e-05 | --- | T3 better | YES |
| sensor CRPS variance_u | posterior | lower | 2.32e-05±1.75e-06 | 5.93e-06±3.54e-07 | -1.73e-05 | -74.5 | 1.45e-06 | --- | T3 better | YES |
| sensor CRPS variance_u | forecast(w1-2) | lower | 1.22e-05±1.88e-07 | 1.24e-05±2.2e-07 | 2.19e-07 | 1.79 | 1.78e-08 | +++ | within noise |  |
| sensor CRPS variance_v | posterior | lower | 6.82e-05±1.03e-06 | 6.9e-06±2.65e-06 | -6.13e-05 | -89.9 | 1.88e-06 | --- | T3 better | YES |
| sensor CRPS variance_v | forecast(w1-2) | lower | 4.99e-06±7.87e-08 | 4.3e-06±1.47e-07 | -6.85e-07 | -13.7 | 3.24e-08 | --- | T3 better | YES |
| sensor CRPS variance_magnitude | posterior | lower | 6e-05±1.8e-06 | 7.46e-06±7.76e-07 | -5.25e-05 | -87.6 | 2.22e-06 | --- | T3 better | YES |
| sensor CRPS variance_magnitude | forecast(w1-2) | lower | 1.47e-05±4.46e-07 | 1.52e-05±2.79e-07 | 5.02e-07 | 3.42 | 2.36e-08 | +++ | T2 better |  |
| sensor w2_member_median u | posterior | lower | 0.0484±0.0021 | 0.0172±0.00471 | -0.0312 | -64.5 | 9.63e-05 | --- | T3 better |  |
| sensor w2_member_median u | forecast(w1-2) | lower | 0.0595±0.00243 | 0.0327±0.00495 | -0.0268 | -45 | 0.000116 | --- | T3 better |  |
| sensor w2 u | posterior | lower | 0.0485±0.00204 | 0.0173±0.00452 | -0.0311 | -64.2 | 9.63e-05 | --- | T3 better |  |
| sensor w2 u | forecast(w1-2) | lower | 0.0596±0.00229 | 0.0331±0.0051 | -0.0265 | -44.5 | 0.000116 | --- | T3 better |  |
| sensor w2_member_median v | posterior | lower | 0.0312±0.00309 | 0.0139±0.00264 | -0.0173 | -55.4 | 0.00016 | --- | T3 better |  |
| sensor w2_member_median v | forecast(w1-2) | lower | 0.0363±0.00199 | 0.0196±0.00366 | -0.0166 | -45.9 | 0.000191 | --- | T3 better |  |
| sensor w2 v | posterior | lower | 0.0315±0.00287 | 0.0146±0.00259 | -0.0169 | -53.6 | 0.00016 | --- | T3 better |  |
| sensor w2 v | forecast(w1-2) | lower | 0.0367±0.0018 | 0.0204±0.00347 | -0.0163 | -44.4 | 0.000191 | --- | T3 better |  |
| sensor w2_member_median magnitude | posterior | lower | 0.0476±0.00235 | 0.0186±0.00482 | -0.029 | -60.8 | 0.000106 | --- | T3 better |  |
| sensor w2_member_median magnitude | forecast(w1-2) | lower | 0.0597±0.00248 | 0.0346±0.00539 | -0.025 | -41.9 | 0.000117 | --- | T3 better |  |
| sensor w2 magnitude | posterior | lower | 0.0476±0.00226 | 0.0188±0.00459 | -0.0288 | -60.5 | 0.000106 | --- | T3 better |  |
| sensor w2 magnitude | forecast(w1-2) | lower | 0.0598±0.00232 | 0.0349±0.0054 | -0.0249 | -41.7 | 0.000117 | --- | T3 better |  |
| field RMSE u | posterior | lower | 0.0513±7.29e-05 | 0.0287±0.00326 | -0.0227 | -44.1 | 0.00124 | --- | T3 better |  |
| field RMSE u | forecast(w1-2) | lower | 0.0686±0.00217 | 0.0502±0.00508 | -0.0184 | -26.8 | 0.00172 | --- | T3 better |  |
| field RMSE v | posterior | lower | 0.0304±0.0032 | 0.0132±0.00225 | -0.0172 | -56.6 | 0.000323 | --- | T3 better |  |
| field RMSE v | forecast(w1-2) | lower | 0.0315±0.00187 | 0.0139±0.000895 | -0.0176 | -55.9 | 0.000439 | --- | T3 better |  |
| field RMSE tke | posterior | lower | 0.00477±0.000121 | 0.0019±0.000349 | -0.00287 | -60.3 | 0.000242 | --- | T3 better |  |
| field RMSE tke | forecast(w1-2) | lower | 0.00437±8.84e-05 | 0.00154±0.000434 | -0.00284 | -64.9 | 0.000311 | --- | T3 better |  |
| field RMSE uw | posterior | lower | 0.000756±4.15e-05 | 0.000582±9.7e-05 | -0.000174 | -23.1 | 3.41e-05 | --- | T3 better |  |
| field RMSE uw | forecast(w1-2) | lower | 0.000616±6.12e-05 | 0.000464±0.000164 | -0.000152 | -24.7 | 3.64e-05 | --+ | within noise |  |
| canopy profile RMSE u | posterior | lower | 0.00888±0.00218 | 0.00825±0.00503 | -0.000629 | -7.08 | 8.43e-05 | +-- | within noise |  |
| canopy profile RMSE u | forecast(w1-2) | lower | 0.0305±0.00555 | 0.0318±0.00541 | 0.0013 | 4.24 | 0.000115 | +++ | within noise |  |
| canopy profile RMSE tke | posterior | lower | 0.000464±1.65e-05 | 0.00013±4.78e-05 | -0.000334 | -72.1 | 1.35e-05 | --- | T3 better |  |
| canopy profile RMSE tke | forecast(w1-2) | lower | 0.000334±8.02e-06 | 8.32e-05±1.28e-05 | -0.000251 | -75.1 | 1.53e-05 | --- | T3 better |  |
| canopy profile RMSE uw | posterior | lower | 4.62e-05±2.33e-06 | 3.24e-05±6.82e-06 | -1.38e-05 | -29.9 | 2.01e-06 | --- | T3 better |  |
| canopy profile RMSE uw | forecast(w1-2) | lower | 3.35e-05±2.69e-06 | 2.92e-05±2.95e-06 | -4.21e-06 | -12.6 | 1.6e-06 | --- | T3 better |  |
| spectra near_cutoff u in_canopy |dB| | posterior | |x| lower | 8.24±0.227 | 6.37±0.633 | -1.87 | -22.6 | 0.171 | --- | T3 better |  |
| spectra near_cutoff u in_canopy |dB| | forecast(w1-2) | |x| lower | 12.3±0.233 | 8.92±0.601 | -3.35 | -27.3 | 0.255 | --- | T3 better |  |
| spectra near_cutoff u above_canopy |dB| | posterior | |x| lower | 3.1±0.987 | 3.42±1.66 | 0.318 | 10.2 | 0.096 | +-+ | within noise |  |
| spectra near_cutoff u above_canopy |dB| | forecast(w1-2) | |x| lower | 4.01±1.32 | 5.81±5.35 | 1.8 | 44.7 | 0.144 | --+ | within noise |  |
| spectra near_cutoff w in_canopy |dB| | posterior | |x| lower | 8.66±0.297 | 6.28±1.04 | -2.38 | -27.5 | 0.115 | --- | T3 better |  |
| spectra near_cutoff w in_canopy |dB| | forecast(w1-2) | |x| lower | 12.3±0.283 | 7.33±0.926 | -4.99 | -40.5 | 0.172 | --- | T3 better |  |
| spectra near_cutoff w above_canopy |dB| | posterior | |x| lower | 4.57±1.88 | 4.75±4.36 | 0.18 | 3.94 | 0.123 | --+ | within noise |  |
| spectra near_cutoff w above_canopy |dB| | forecast(w1-2) | |x| lower | 5.93±2.31 | 8.36±10.7 | 2.43 | 41 | 0.161 | --+ | within noise |  |

**Verdict A: Adds value.** Of 12 validation CRPS rows (posterior+forecast), T3 is clearly better in 10, T2 in 1, noise in 1; 10 rows meet the plan's ≥10% reduction (all 12 would be needed). On the other 34 held-out rows (W2, field, canopy, spectra) T3 is clearly better in 27, T2 in 0. Rule: Adds value needs T3-better ≥ 2x T2-better with a quarter of CRPS rows; Hurts needs T2-better > T3-better (≥3 rows).

## 1. T3 vs T2, stage B (inlet turbulence on), L2
| score | windows | orient | T2 (mean±sd) | T3 (mean±sd) | T3−T2 | rel % | replica floor | sign/seed | verdict | ≥10% CRPS |
|---|---|---|---|---|---|---|---|---|---|---|
| sensor CRPS mean_u | posterior | lower | 0.0665±0.0226 | 0.0645±0.0172 | -0.00201 | -3.02 | 0.0416 | +-- | within noise |  |
| sensor CRPS mean_u | forecast(w1-2) | lower | 0.0673±0.0291 | 0.0643±0.0239 | -0.00296 | -4.4 | 0.0344 | +-- | within noise |  |
| sensor CRPS mean_v | posterior | lower | 0.104±0.0595 | 0.107±0.0504 | 0.00223 | 2.14 | 0.0626 | +-+ | within noise |  |
| sensor CRPS mean_v | forecast(w1-2) | lower | 0.108±0.0635 | 0.11±0.0556 | 0.00138 | 1.28 | 0.0696 | +-+ | within noise |  |
| sensor CRPS mean_magnitude | posterior | lower | 0.0846±0.0282 | 0.0852±0.0168 | 0.000557 | 0.658 | 0.0639 | +-- | within noise |  |
| sensor CRPS mean_magnitude | forecast(w1-2) | lower | 0.0883±0.0372 | 0.0872±0.028 | -0.00104 | -1.18 | 0.0521 | +-- | within noise |  |
| sensor CRPS variance_u | posterior | lower | 0.0105±0.00275 | 0.0102±0.00292 | -0.000238 | -2.27 | 0.021 | --+ | within noise |  |
| sensor CRPS variance_u | forecast(w1-2) | lower | 0.0113±0.00382 | 0.0111±0.00402 | -0.000217 | -1.92 | 0.0219 | --+ | within noise |  |
| sensor CRPS variance_v | posterior | lower | 0.012±0.00312 | 0.0118±0.00294 | -0.000146 | -1.22 | 0.0228 | --+ | within noise |  |
| sensor CRPS variance_v | forecast(w1-2) | lower | 0.011±0.00208 | 0.0111±0.00238 | 5.56e-05 | 0.504 | 0.0281 | +-+ | within noise |  |
| sensor CRPS variance_magnitude | posterior | lower | 0.0131±0.00306 | 0.0126±0.0038 | -0.000548 | -4.17 | 0.0195 | --+ | within noise |  |
| sensor CRPS variance_magnitude | forecast(w1-2) | lower | 0.0136±0.00472 | 0.0131±0.00519 | -0.000473 | -3.47 | 0.0218 | --+ | within noise |  |
| sensor w2_member_median u | posterior | lower | 0.132±0.0297 | 0.127±0.0277 | -0.00506 | -3.82 | 0.0786 | --- | within noise |  |
| sensor w2_member_median u | forecast(w1-2) | lower | 0.138±0.0396 | 0.131±0.0386 | -0.00724 | -5.26 | 0.0672 | --- | within noise |  |
| sensor w2 u | posterior | lower | 0.118±0.0289 | 0.112±0.0277 | -0.00561 | -4.77 | 0.0786 | --- | within noise |  |
| sensor w2 u | forecast(w1-2) | lower | 0.12±0.0382 | 0.112±0.0377 | -0.00823 | -6.85 | 0.0672 | --- | within noise |  |
| sensor w2_member_median v | posterior | lower | 0.165±0.0738 | 0.165±0.0683 | 0.000151 | 0.0914 | 0.0919 | +-+ | within noise |  |
| sensor w2_member_median v | forecast(w1-2) | lower | 0.172±0.0761 | 0.173±0.0709 | 0.000385 | 0.224 | 0.111 | +-+ | within noise |  |
| sensor w2 v | posterior | lower | 0.155±0.082 | 0.155±0.0774 | 0.00047 | 0.303 | 0.0919 | +-+ | within noise |  |
| sensor w2 v | forecast(w1-2) | lower | 0.16±0.0836 | 0.161±0.079 | 0.00022 | 0.137 | 0.111 | +-+ | within noise |  |
| sensor w2_member_median magnitude | posterior | lower | 0.115±0.00836 | 0.118±0.009 | 0.00276 | 2.4 | 0.102 | ++- | within noise |  |
| sensor w2_member_median magnitude | forecast(w1-2) | lower | 0.119±0.0154 | 0.119±0.0124 | 0.000662 | 0.558 | 0.0971 | +-- | within noise |  |
| sensor w2 magnitude | posterior | lower | 0.0937±0.00876 | 0.0963±0.0123 | 0.00255 | 2.72 | 0.102 | ++- | within noise |  |
| sensor w2 magnitude | forecast(w1-2) | lower | 0.0955±0.0152 | 0.0954±0.0135 | -0.000171 | -0.179 | 0.0971 | +-- | within noise |  |
| field RMSE u | posterior | lower | 0.157±0.0333 | 0.155±0.039 | -0.00133 | -0.848 | 0.174 | --+ | within noise |  |
| field RMSE u | forecast(w1-2) | lower | 0.163±0.0364 | 0.161±0.0422 | -0.00175 | -1.08 | 0.159 | --+ | within noise |  |
| field RMSE v | posterior | lower | 0.186±0.107 | 0.187±0.102 | 0.00146 | 0.788 | 0.0883 | +-+ | within noise |  |
| field RMSE v | forecast(w1-2) | lower | 0.178±0.112 | 0.18±0.107 | 0.00155 | 0.869 | 0.0804 | +-+ | within noise |  |
| field RMSE tke | posterior | lower | 0.0359±0.00182 | 0.0358±0.00197 | -7.19e-05 | -0.2 | 0.0543 | -++ | within noise |  |
| field RMSE tke | forecast(w1-2) | lower | 0.0344±0.00287 | 0.034±0.00365 | -0.00041 | -1.19 | 0.0513 | -++ | within noise |  |
| field RMSE uw | posterior | lower | 0.0159±0.000855 | 0.0159±0.000805 | 3.38e-05 | 0.213 | 0.023 | +-+ | within noise |  |
| field RMSE uw | forecast(w1-2) | lower | 0.016±0.00136 | 0.016±0.00134 | 1.71e-05 | 0.107 | 0.0241 | +-+ | within noise |  |
| canopy profile RMSE u | posterior | lower | 0.103±0.0453 | 0.1±0.053 | -0.00331 | -3.2 | 0.0398 | --+ | within noise |  |
| canopy profile RMSE u | forecast(w1-2) | lower | 0.116±0.0428 | 0.113±0.0502 | -0.00322 | -2.78 | 0.0392 | --+ | within noise |  |
| canopy profile RMSE tke | posterior | lower | 0.0171±0.00242 | 0.0172±0.00248 | 0.000113 | 0.662 | 0.0268 | +++ | within noise |  |
| canopy profile RMSE tke | forecast(w1-2) | lower | 0.0169±0.0012 | 0.0163±0.0025 | -0.000565 | -3.34 | 0.024 | -++ | within noise |  |
| canopy profile RMSE uw | posterior | lower | 0.00517±0.00189 | 0.00513±0.00199 | -4.43e-05 | -0.856 | 0.00815 | -++ | within noise |  |
| canopy profile RMSE uw | forecast(w1-2) | lower | 0.00539±0.00175 | 0.00533±0.00185 | -5.72e-05 | -1.06 | 0.00942 | -++ | within noise |  |
| spectra near_cutoff u in_canopy |dB| | posterior | |x| lower | 0.998±0.161 | 0.751±0.274 | -0.248 | -24.8 | 0.526 | --+ | within noise |  |
| spectra near_cutoff u in_canopy |dB| | forecast(w1-2) | |x| lower | 0.854±0.34 | 0.49±0.172 | -0.364 | -42.6 | 0.433 | --+ | within noise |  |
| spectra near_cutoff u above_canopy |dB| | posterior | |x| lower | 0.498±0.1 | 0.673±0.431 | 0.174 | 35 | 0.405 | ++- | within noise |  |
| spectra near_cutoff u above_canopy |dB| | forecast(w1-2) | |x| lower | 0.513±0.0971 | 0.599±0.356 | 0.0857 | 16.7 | 0.371 | ++- | within noise |  |
| spectra near_cutoff w in_canopy |dB| | posterior | |x| lower | 1.24±0.0758 | 1.18±0.168 | -0.058 | -4.67 | 0.479 | --+ | within noise |  |
| spectra near_cutoff w in_canopy |dB| | forecast(w1-2) | |x| lower | 1.05±0.242 | 0.929±0.236 | -0.122 | -11.6 | 0.487 | --+ | within noise |  |
| spectra near_cutoff w above_canopy |dB| | posterior | |x| lower | 0.202±0.0886 | 0.707±0.833 | 0.505 | 250 | 0.267 | +++ | within noise |  |
| spectra near_cutoff w above_canopy |dB| | forecast(w1-2) | |x| lower | 0.232±0.136 | 0.664±0.797 | 0.432 | 186 | 0.363 | +++ | within noise |  |

**Verdict B: No added value.** Of 12 validation CRPS rows (posterior+forecast), T3 is clearly better in 0, T2 in 0, noise in 12; 0 rows meet the plan's ≥10% reduction (all 12 would be needed). On the other 34 held-out rows (W2, field, canopy, spectra) T3 is clearly better in 0, T2 in 0. Rule: Adds value needs T3-better ≥ 2x T2-better with a quarter of CRPS rows; Hurts needs T2-better > T3-better (≥3 rows).

## 2. Layout effect, stage A: T3 L1 vs L2 (seed 1; no T2_L1 exists)
| score | win | T3_L1_s1 | T3_L2_s1 | replica |
|---|---|---|---|---|
| sensor CRPS mean_magnitude | post | 0.0159 | 0.0105 | 6.4e-05 |
| sensor CRPS mean_magnitude | fcst | 0.0303 | 0.0225 | 8.23e-05 |
| sensor CRPS variance_magnitude | post | 3.87e-05 | 7.31e-06 | 2.22e-06 |
| sensor CRPS variance_magnitude | fcst | 1.41e-05 | 1.55e-05 | 2.36e-08 |
| sensor w2_member_median magnitude | post | 0.0277 | 0.017 | 0.000106 |
| sensor w2_member_median magnitude | fcst | 0.0458 | 0.0331 | 0.000117 |
| field RMSE u | post | 0.0395 | 0.0281 | 0.00124 |
| field RMSE u | fcst | 0.0561 | 0.0508 | 0.00172 |
| field RMSE tke | post | 0.00236 | 0.00152 | 0.000242 |
| field RMSE tke | fcst | 0.00189 | 0.0011 | 0.000311 |
| canopy profile RMSE u | post | 0.00613 | 0.0132 | 8.43e-05 |
| canopy profile RMSE u | fcst | 0.0282 | 0.0352 | 0.000115 |
| spectra near_cutoff u in_canopy |dB| | post | 7.92 | 7.07 | 0.171 |
| spectra near_cutoff u in_canopy |dB| | fcst | 8.01 | 9.21 | 0.255 |
| spectra near_cutoff w above_canopy |dB| | post | 11.3 | 2.08 | 0.123 |
| spectra near_cutoff w above_canopy |dB| | fcst | 24.3 | 1.83 | 0.161 |

## 2. Layout effect, stage B: T3 L1 vs L2 (seed 1; no T2_L1 exists)
| score | win | T3_L1_s1 | T3_L2_s1 | replica |
|---|---|---|---|---|
| sensor CRPS mean_magnitude | post | 0.0768 | 0.0686 | 0.0639 |
| sensor CRPS mean_magnitude | fcst | 0.0784 | 0.0605 | 0.0521 |
| sensor CRPS variance_magnitude | post | 0.00891 | 0.00929 | 0.0195 |
| sensor CRPS variance_magnitude | fcst | 0.00924 | 0.00973 | 0.0218 |
| sensor w2_member_median magnitude | post | 0.133 | 0.125 | 0.102 |
| sensor w2_member_median magnitude | fcst | 0.139 | 0.118 | 0.0971 |
| field RMSE u | post | 0.175 | 0.136 | 0.174 |
| field RMSE u | fcst | 0.176 | 0.131 | 0.159 |
| field RMSE tke | post | 0.0341 | 0.0341 | 0.0463 |
| field RMSE tke | fcst | 0.0287 | 0.0301 | 0.0423 |
| canopy profile RMSE u | post | 0.124 | 0.0648 | 0.0398 |
| canopy profile RMSE u | fcst | 0.137 | 0.0798 | 0.0392 |
| spectra near_cutoff u in_canopy |dB| | post | 0.398 | 0.444 | 0.526 |
| spectra near_cutoff u in_canopy |dB| | fcst | 0.3 | 0.299 | 0.433 |
| spectra near_cutoff w above_canopy |dB| | post | 1.56 | 1.67 | 0.265 |
| spectra near_cutoff w above_canopy |dB| | fcst | 1.48 | 1.57 | 0.24 |

## 3. T0 sanity, stage A (T0 posterior vs replica floor; T3, T2 for reference)
| score | win | T0_L2_s1 | T3_L2_s1 | T2_L2_s1 | replica |
|---|---|---|---|---|---|
| sensor CRPS mean_magnitude | post | 0.0108 | 0.0105 | 0.0349 | 2.99e-05 |
| sensor CRPS mean_magnitude | fcst | 0.0205 | 0.0225 | 0.0462 | 3.23e-05 |
| sensor CRPS variance_magnitude | post | 1.12e-05 | 7.31e-06 | 6.18e-05 | 1.52e-06 |
| sensor CRPS variance_magnitude | fcst | 1.19e-05 | 1.55e-05 | 1.43e-05 | 9.05e-08 |
| sensor w2_member_median magnitude | post | 0.0168 | 0.017 | 0.05 | 0.000163 |
| sensor w2_member_median magnitude | fcst | 0.0306 | 0.0331 | 0.0618 | 0.000159 |
| field RMSE u | post | 0.031 | 0.0281 | 0.0513 | 0.000717 |
| field RMSE u | fcst | 0.0536 | 0.0508 | 0.0697 | 0.000896 |
| field RMSE tke | post | 0.00381 | 0.00152 | 0.0049 | 0.000432 |
| field RMSE tke | fcst | 0.00425 | 0.0011 | 0.00434 | 0.000584 |
| canopy profile RMSE u | post | 0.0147 | 0.0132 | 0.0112 | 4.17e-05 |
| canopy profile RMSE u | fcst | 0.0364 | 0.0352 | 0.0342 | 4.84e-05 |
| spectra near_cutoff u in_canopy |dB| | post | 2.75 | 7.07 | 8.49 | 0.175 |
| spectra near_cutoff u in_canopy |dB| | fcst | 2.36 | 9.21 | 12.5 | 0.263 |
| spectra near_cutoff w above_canopy |dB| | post | 6.03 | 2.08 | 2.57 | 0.175 |
| spectra near_cutoff w above_canopy |dB| | fcst | 8.44 | 1.83 | 3.48 | 0.25 |

## 3. T0 sanity, stage B (T0 posterior vs replica floor; T3, T2 for reference)
| score | win | T0_L2_s1 | T3_L2_s1 | T2_L2_s1 | replica |
|---|---|---|---|---|---|
| sensor CRPS mean_magnitude | post | 0.0587 | 0.0686 | 0.0531 | 0.0622 |
| sensor CRPS mean_magnitude | fcst | 0.0517 | 0.0605 | 0.0486 | 0.0502 |
| sensor CRPS variance_magnitude | post | 0.00978 | 0.00929 | 0.0107 | 0.0204 |
| sensor CRPS variance_magnitude | fcst | 0.0104 | 0.00973 | 0.011 | 0.023 |
| sensor w2_member_median magnitude | post | 0.119 | 0.125 | 0.114 | 0.0997 |
| sensor w2_member_median magnitude | fcst | 0.113 | 0.118 | 0.111 | 0.0933 |
| field RMSE u | post | 0.14 | 0.136 | 0.14 | 0.174 |
| field RMSE u | fcst | 0.136 | 0.131 | 0.136 | 0.159 |
| field RMSE tke | post | 0.0343 | 0.0341 | 0.0344 | 0.0468 |
| field RMSE tke | fcst | 0.0301 | 0.0301 | 0.0315 | 0.0427 |
| canopy profile RMSE u | post | 0.0709 | 0.0648 | 0.0728 | 0.0399 |
| canopy profile RMSE u | fcst | 0.0855 | 0.0798 | 0.0872 | 0.0393 |
| spectra near_cutoff u in_canopy |dB| | post | 0.493 | 0.444 | 1.01 | 0.485 |
| spectra near_cutoff u in_canopy |dB| | fcst | 0.329 | 0.299 | 1.21 | 0.381 |
| spectra near_cutoff w above_canopy |dB| | post | 1.37 | 1.67 | 0.228 | 0.194 |
| spectra near_cutoff w above_canopy |dB| | fcst | 1.29 | 1.57 | 0.307 | 0.174 |

## 4. Method comparison, stage A, T3 L2 seed 1
| score | win | smoother | hybrid | replica |
|---|---|---|---|---|
| sensor CRPS mean_magnitude | post | 0.0105 | 0.124 | 6.4e-05 |
| sensor CRPS mean_magnitude | fcst | 0.0225 | n/a | 8.23e-05 |
| sensor CRPS variance_magnitude | post | 7.31e-06 | 0.000307 | 2.22e-06 |
| sensor CRPS variance_magnitude | fcst | 1.55e-05 | n/a | 2.36e-08 |
| sensor w2_member_median magnitude | post | 0.017 | 0.169 | 0.000106 |
| sensor w2_member_median magnitude | fcst | 0.0331 | n/a | 0.000117 |
| field RMSE u | post | 0.0281 | 0.306 | 0.00124 |
| field RMSE u | fcst | 0.0508 | n/a | 0.00172 |
| field RMSE tke | post | 0.00152 | 0.00428 | 0.000242 |
| field RMSE tke | fcst | 0.0011 | n/a | 0.000311 |
| canopy profile RMSE u | post | 0.0132 | 0.128 | 8.43e-05 |
| canopy profile RMSE u | fcst | 0.0352 | n/a | 0.000115 |
| spectra near_cutoff u in_canopy |dB| | post | 7.07 | 10.7 | 0.171 |
| spectra near_cutoff u in_canopy |dB| | fcst | 9.21 | n/a | 0.255 |
| spectra near_cutoff w above_canopy |dB| | post | 2.08 | 8.56 | 0.123 |
| spectra near_cutoff w above_canopy |dB| | fcst | 1.83 | n/a | 0.161 |

## 4. Method comparison, stage B, T3 L2 seed 1
| score | win | smoother | hybrid | filtering | replica |
|---|---|---|---|---|---|
| sensor CRPS mean_magnitude | post | 0.0686 | 0.129 | 0.0898 | 0.0639 |
| sensor CRPS mean_magnitude | fcst | 0.0605 | n/a | n/a | 0.0521 |
| sensor CRPS variance_magnitude | post | 0.00929 | 0.0582 | 0.0658 | 0.0195 |
| sensor CRPS variance_magnitude | fcst | 0.00973 | n/a | n/a | 0.0218 |
| sensor w2_member_median magnitude | post | 0.125 | 0.179 | 0.15 | 0.102 |
| sensor w2_member_median magnitude | fcst | 0.118 | n/a | n/a | 0.0971 |
| field RMSE u | post | 0.136 | 0.314 | 0.298 | 0.174 |
| field RMSE u | fcst | 0.131 | n/a | n/a | 0.159 |
| field RMSE tke | post | 0.0341 | 0.303 | 0.291 | 0.0463 |
| field RMSE tke | fcst | 0.0301 | n/a | n/a | 0.0423 |
| canopy profile RMSE u | post | 0.0648 | 0.203 | 0.171 | 0.0398 |
| canopy profile RMSE u | fcst | 0.0798 | n/a | n/a | 0.0392 |
| spectra near_cutoff u in_canopy |dB| | post | 0.444 | 2.07 | 1.66 | 0.526 |
| spectra near_cutoff u in_canopy |dB| | fcst | 0.299 | n/a | n/a | 0.433 |
| spectra near_cutoff w above_canopy |dB| | post | 1.67 | 4.12 | 2.99 | 0.265 |
| spectra near_cutoff w above_canopy |dB| | fcst | 1.57 | n/a | n/a | 0.24 |

## 5. Verdicts
- Stage A: **Adds value**
- Stage B: **No added value**
