# Hypothesis tests for b2 (162 audited cells)

Δ = candidate − baseline (AuditKIT `paired_bootstrap`) with a percentile-bootstrap 95 % CI of the paired mean difference (2000 resamples, seed 0). Main cells only unless stated; sweep cells (`cond@mult`) feed the matched-efficacy section.

L4 bystander circuits: b1_en, ioi_fixed, langid_en (`ioi_fixed` replaces the mis-specified built-in `ioi`, amendment A16).

Base English held-out harmful refusal is 0.917, so the pre-registered English efficacy is capped at +0.083; `efficacy_xl` (amendment A3, post hoc) is the mean held-out harmful refusal gain over hi/ar/fr/sw, where the base refuses hi 0.50, ar 0.58, fr 0.83, sw 0.58.

## Cell means by intervention × condition

| intervention | condition | n | efficacy | efficacy_xl (B2, A3) | |Δθ| | L4 bystander CAD | L4 bystander CAD − floor | L4 target CAD | L4 target CAD − floor | L3 mean |Δ| | L2 benign refusal Δ (en) | L1 neigh Δ (en) | L1 MMLU Δ (en) | WikiText ppl ratio | L5 AURC Δ | L5 never-edited top-1 error Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| steer | circuit | 3 | 0.000 | 0.215 | 24.488 | 0.196 | -0.220 | 0.415 | 0.020 | 0.188 | 0.190 | -0.100 | -0.111 | 1.168 | 0.006 | 0.047 |
| steer | inverse | 3 | 0.028 | 0.049 | 24.488 | 0.094 | -0.321 | 0.179 | -0.216 | 0.038 | 0.012 | -0.017 | -0.033 | 1.059 | 0.002 | 0.010 |
| steer | random | 5 | -0.100 | 0.104 | 24.488 | 0.159 | -0.253 | 0.288 | -0.098 | 0.109 | 0.157 | -0.057 | -0.017 | 1.122 | 0.007 | 0.040 |
| steer | safetune_rd | 3 | 0.083 | 0.340 | 24.488 | 0.368 | -0.048 | 0.583 | 0.188 | 0.644 | 0.810 | -0.772 | -0.083 | 1.495 | 0.712 | 0.783 |
| steer | taylor | 3 | 0.000 | 0.125 | 24.488 | 0.133 | -0.283 | 0.372 | -0.023 | 0.081 | 0.071 | -0.033 | -0.067 | 1.124 | 0.003 | 0.040 |
| steer | unrestricted | 3 | 0.083 | 0.312 | 24.488 | 0.166 | -0.250 | 0.525 | 0.130 | 0.263 | 0.500 | -0.067 | -0.050 | 1.182 | 0.017 | 0.080 |
| lora | circuit | 3 | 0.083 | 0.292 | 3.534 | 0.049 | -0.366 | 0.499 | 0.104 | 0.082 | 0.381 | -0.017 | -0.017 | 1.064 | -0.000 | 0.003 |
| lora | inverse | 3 | 0.083 | 0.201 | 4.020 | 0.034 | -0.382 | 0.256 | -0.139 | 0.045 | 0.369 | -0.022 | -0.033 | 1.012 | 0.000 | 0.000 |
| lora | random | 5 | 0.083 | 0.287 | 3.697 | 0.041 | -0.371 | 0.448 | 0.063 | 0.090 | 0.400 | -0.020 | -0.003 | 1.033 | -0.000 | 0.002 |
| lora | taylor | 3 | 0.083 | 0.278 | 3.599 | 0.024 | -0.392 | 0.495 | 0.100 | 0.097 | 0.548 | -0.033 | 0.028 | 1.053 | -0.000 | 0.010 |
| lora | unrestricted | 3 | 0.083 | 0.375 | 6.319 | 0.063 | -0.352 | 0.873 | 0.478 | 0.202 | 0.417 | -0.033 | -0.044 | 1.048 | 0.000 | 0.017 |
| lora | perturb_circuit | 3 | 0.000 | -0.007 | 3.535 | 0.015 | -0.400 | 0.020 | -0.375 | 0.004 | 0.000 | 0.000 | 0.011 | 1.000 | 0.000 | 0.000 |
| lora | perturb_inverse | 3 | 0.000 | 0.000 | 4.021 | 0.013 | -0.402 | 0.014 | -0.381 | 0.002 | 0.000 | 0.000 | 0.033 | 1.000 | 0.000 | 0.000 |
| lora | perturb_random | 5 | 0.000 | 0.000 | 3.697 | 0.015 | -0.398 | 0.014 | -0.372 | 0.002 | 0.000 | 0.000 | 0.023 | 1.000 | 0.000 | 0.000 |
| lora | perturb_taylor | 3 | 0.000 | -0.007 | 3.599 | 0.009 | -0.406 | 0.013 | -0.382 | 0.005 | 0.000 | 0.000 | 0.022 | 1.001 | 0.000 | 0.000 |
| lora | perturb_unrestricted | 3 | 0.000 | 0.000 | 6.320 | 0.015 | -0.400 | 0.021 | -0.374 | 0.002 | 0.000 | 0.000 | 0.011 | 1.000 | 0.000 | 0.003 |
| edit | circuit | 3 | -0.111 | -0.125 | 77.748 | 0.317 | -0.099 | 0.447 | 0.051 | 0.073 | -0.024 | -0.056 | -0.067 | 0.998 | 0.010 | 0.070 |
| edit | circuit_deep | 3 | 0.000 | 0.062 | 27.330 | 0.044 | -0.372 | 0.183 | -0.212 | 0.023 | 0.000 | -0.017 | 0.022 | 1.003 | 0.000 | 0.010 |
| edit | inverse | 3 | 0.083 | 0.146 | 39.475 | 0.050 | -0.366 | 0.210 | -0.185 | 0.034 | -0.024 | -0.006 | -0.006 | 1.001 | 0.000 | 0.003 |
| edit | random | 5 | 0.017 | 0.029 | 29.850 | 0.049 | -0.363 | 0.155 | -0.231 | 0.020 | 0.007 | -0.017 | 0.003 | 1.002 | 0.000 | 0.008 |
| edit | taylor | 3 | 0.000 | -0.007 | 29.713 | 0.027 | -0.388 | 0.091 | -0.304 | 0.023 | 0.000 | -0.017 | -0.006 | 1.000 | 0.000 | 0.000 |
| edit | unrestricted | 3 | 0.000 | -0.104 | 26.431 | 0.065 | -0.350 | 0.219 | -0.177 | 0.021 | 0.000 | -0.017 | -0.028 | 1.001 | 0.000 | 0.010 |
| edit | perturb_circuit | 3 | -0.250 | -0.222 | 77.712 | 0.438 | 0.023 | 0.625 | 0.229 | 0.186 | 0.000 | -0.122 | -0.094 | 1.016 | 0.051 | 0.153 |
| edit | perturb_circuit_deep | 3 | -0.028 | -0.014 | 27.309 | 0.028 | -0.388 | 0.046 | -0.349 | 0.012 | 0.000 | -0.006 | 0.028 | 1.001 | 0.000 | 0.000 |
| edit | perturb_inverse | 3 | 0.000 | -0.000 | 39.452 | 0.046 | -0.370 | 0.061 | -0.335 | 0.009 | 0.000 | 0.000 | 0.011 | 1.002 | 0.000 | 0.000 |
| edit | perturb_random | 5 | -0.017 | -0.017 | 29.826 | 0.040 | -0.373 | 0.054 | -0.331 | 0.013 | 0.000 | -0.010 | 0.023 | 1.003 | 0.000 | -0.002 |
| edit | perturb_taylor | 3 | 0.000 | -0.014 | 29.688 | 0.028 | -0.387 | 0.034 | -0.361 | 0.016 | 0.000 | -0.017 | 0.006 | 1.007 | 0.001 | 0.017 |
| edit | perturb_unrestricted | 3 | -0.083 | 0.000 | 26.415 | 0.069 | -0.347 | 0.081 | -0.314 | 0.014 | 0.000 | -0.011 | 0.011 | 1.002 | 0.000 | 0.007 |

L4 columns use the **cell** circuit definition (pre-registered per-type set, `audit/l4_cellset.py`; the joint-ranked numbers first reported are the `*_joint` fields of the CSV and the comparison below).

## Circuit definition sensitivity (per-type cell set vs joint-ranked set)

| intervention | condition | n | bystander CAD − floor (cell / joint) | target CAD − floor (cell / joint) | AS-NC (cell / joint) | circuit distance (cell / joint) |
|---|---|---|---|---|---|---|
| steer | circuit | 3 | -0.220 / -0.245 | 0.020 / 0.057 | 0.212 / 0.221 | 0.276 / 0.273 |
| steer | inverse | 3 | -0.321 / -0.351 | -0.216 / -0.221 | 0.199 / 0.198 | 0.156 / 0.133 |
| steer | random | 5 | -0.253 / -0.276 | -0.098 / -0.067 | 0.268 / 0.266 | 0.245 / 0.204 |
| steer | safetune_rd | 3 | -0.048 / -0.047 | 0.188 / 0.281 | 0.485 / 0.475 | 0.448 / 0.372 |
| steer | taylor | 3 | -0.283 / -0.312 | -0.023 / -0.024 | 0.194 / 0.195 | 0.211 / 0.305 |
| steer | unrestricted | 3 | -0.250 / -0.274 | 0.130 / 0.191 | 0.363 / 0.362 | 0.388 / 0.415 |
| lora | circuit | 3 | -0.366 / -0.418 | 0.104 / 0.136 | 0.076 / 0.077 | 0.236 / 0.223 |
| lora | inverse | 3 | -0.382 / -0.434 | -0.139 / -0.145 | 0.072 / 0.072 | 0.140 / 0.179 |
| lora | random | 5 | -0.371 / -0.422 | 0.063 / 0.117 | 0.068 / 0.068 | 0.248 / 0.232 |
| lora | taylor | 3 | -0.392 / -0.444 | 0.100 / 0.136 | 0.038 / 0.039 | 0.245 / 0.214 |
| lora | unrestricted | 3 | -0.352 / -0.401 | 0.478 / 0.686 | 0.122 / 0.122 | 0.447 / 0.449 |
| lora | perturb_circuit | 3 | -0.400 / -0.455 | -0.375 / -0.414 | 0.020 / 0.020 | 0.031 / 0.043 |
| lora | perturb_inverse | 3 | -0.402 / -0.457 | -0.381 / -0.419 | 0.021 / 0.021 | 0.021 / 0.000 |
| lora | perturb_random | 5 | -0.398 / -0.451 | -0.372 / -0.409 | 0.021 / 0.021 | 0.032 / 0.013 |
| lora | perturb_taylor | 3 | -0.406 / -0.462 | -0.382 / -0.420 | 0.012 / 0.012 | 0.000 / 0.022 |
| lora | perturb_unrestricted | 3 | -0.400 / -0.455 | -0.374 / -0.411 | 0.024 / 0.024 | 0.062 / 0.052 |
| edit | circuit | 3 | -0.099 / -0.099 | 0.051 / 0.102 | 1.008 / 1.017 | 0.375 / 0.255 |
| edit | circuit_deep | 3 | -0.372 / -0.420 | -0.212 / -0.229 | 0.035 / 0.034 | 0.121 / 0.063 |
| edit | inverse | 3 | -0.366 / -0.408 | -0.185 / -0.174 | 0.057 / 0.057 | 0.203 / 0.179 |
| edit | random | 5 | -0.363 / -0.405 | -0.231 / -0.236 | 0.058 / 0.057 | 0.158 / 0.099 |
| edit | taylor | 3 | -0.388 / -0.443 | -0.304 / -0.327 | 0.031 / 0.030 | 0.092 / 0.094 |
| edit | unrestricted | 3 | -0.350 / -0.401 | -0.177 / -0.194 | 0.123 / 0.123 | 0.185 / 0.160 |
| edit | perturb_circuit | 3 | 0.023 / 0.040 | 0.229 / 0.272 | 1.623 / 1.640 | 0.456 / 0.379 |
| edit | perturb_circuit_deep | 3 | -0.388 / -0.437 | -0.349 / -0.375 | 0.027 / 0.026 | 0.021 / 0.022 |
| edit | perturb_inverse | 3 | -0.370 / -0.415 | -0.335 / -0.364 | 0.056 / 0.056 | 0.082 / 0.063 |
| edit | perturb_random | 5 | -0.373 / -0.419 | -0.331 / -0.356 | 0.041 / 0.040 | 0.053 / 0.063 |
| edit | perturb_taylor | 3 | -0.387 / -0.439 | -0.361 / -0.380 | 0.027 / 0.026 | 0.021 / 0.022 |
| edit | perturb_unrestricted | 3 | -0.347 / -0.396 | -0.314 / -0.328 | 0.170 / 0.171 | 0.102 / 0.093 |

Spearman(cell, joint) of the per-cell bystander CAD excess = 0.857 (n=94); the recomputation reproduces the stored joint-set CAD to 0.0e+00 (max over cells and circuits).

## steer

- **H1** efficacy circuit/unrestricted = 0.000 [0.000, 0.000] → refuted at the 0.9 rule — vacuous: both efficacies < 0.1, the intervention does not produce the behaviour; paired per seed: Δ=-0.0833 [-0.0833, -0.0833] (CI excludes 0; n=3; means 0.0000 vs 0.0833); seed-cluster CI [-0.0833, -0.0833] (3 seeds, excludes 0); per-seed Δ -0.0833 / -0.0833 / -0.0833, paired t p=0.000
- **H1-xl (A3, mean hi/ar/fr/sw held-out harmful refusal gain)** efficacy circuit/unrestricted = 0.689 [0.667, 0.733] → refuted at the 0.9 rule; paired per seed: Δ=-0.0972 [-0.1042, -0.0833] (CI excludes 0; n=3; means 0.2153 vs 0.3125); seed-cluster CI [-0.1042, -0.0833] (3 seeds, excludes 0); per-seed Δ -0.1042 / -0.1042 / -0.0833, paired t p=0.005
- A3 efficacy_xl: circuit − random (seed × language): Δ=+0.1083 [+0.0333, +0.1833] (CI excludes 0; n=20; means 0.2125 vs 0.1042); seed-cluster CI [-0.0313, +0.2292] (3 seeds, includes 0); per-seed Δ -0.0313 / +0.1875 / +0.2292, paired t p=0.253
- A3 efficacy_xl: circuit − inverse (seed × language): Δ=+0.1667 [+0.0903, +0.2431] (CI excludes 0; n=12; means 0.2153 vs 0.0486); seed-cluster CI [+0.1042, +0.2292] (3 seeds, excludes 0); per-seed Δ +0.1042 / +0.1667 / +0.2292, paired t p=0.044
- A3 efficacy_xl: circuit − taylor (seed × language): Δ=+0.0903 [+0.0417, +0.1389] (CI excludes 0; n=12; means 0.2153 vs 0.1250); seed-cluster CI [+0.0833, +0.1042] (3 seeds, excludes 0); per-seed Δ +0.0833 / +0.0833 / +0.1042, paired t p=0.006
- A3 efficacy_xl: circuit − unrestricted (seed × language): Δ=-0.0972 [-0.1458, -0.0556] (CI excludes 0; n=12; means 0.2153 vs 0.3125); seed-cluster CI [-0.1042, -0.0833] (3 seeds, excludes 0); per-seed Δ -0.1042 / -0.1042 / -0.0833, paired t p=0.005
- A3 efficacy_xl: circuit − safetune_rd (seed × language): Δ=-0.1250 [-0.1806, -0.0764] (CI excludes 0; n=12; means 0.2153 vs 0.3403); seed-cluster CI [-0.1250, -0.1250] (3 seeds, excludes 0); per-seed Δ -0.1250 / -0.1250 / -0.1250, paired t p=0.000
- H2 l4_bystander_cad_mean: circuit − random: Δ=+0.0366 [+0.0103, +0.0624] (CI excludes 0; n=15; means 0.1961 vs 0.1595); seed-cluster CI [+0.0054, +0.0722] (3 seeds, excludes 0); per-seed Δ +0.0167 / +0.0722 / +0.0054, paired t p=0.267
  - same test on the joint-ranked definition (as first reported): Δ=+0.0359 [+0.0056, +0.0680] (CI excludes 0; n=15; means 0.2273 vs 0.1914); seed-cluster CI [-0.0040, +0.0811] (3 seeds, includes 0); per-seed Δ +0.0105 / +0.0811 / -0.0040, paired t p=0.382
- H2 l4_bystander_cad_mean: circuit − inverse: Δ=+0.1017 [+0.0705, +0.1410] (CI excludes 0; n=9; means 0.1958 vs 0.0941); seed-cluster CI [+0.0744, +0.1456] (3 seeds, excludes 0); per-seed Δ +0.0744 / +0.0852 / +0.1456, paired t p=0.044
  - same test on the joint-ranked definition (as first reported): Δ=+0.1066 [+0.0625, +0.1597] (CI excludes 0; n=9; means 0.2283 vs 0.1216); seed-cluster CI [+0.0606, +0.1698] (3 seeds, excludes 0); per-seed Δ +0.0606 / +0.0895 / +0.1698, paired t p=0.082
- ctrl l4_bystander_cad_mean: circuit − taylor: Δ=+0.0629 [+0.0488, +0.0758] (CI excludes 0; n=9; means 0.1958 vs 0.1329); seed-cluster CI [+0.0533, +0.0742] (3 seeds, excludes 0); per-seed Δ +0.0742 / +0.0533 / +0.0611, paired t p=0.009
  - same test on the joint-ranked definition (as first reported): Δ=+0.0677 [+0.0449, +0.0897] (CI excludes 0; n=9; means 0.2283 vs 0.1606); seed-cluster CI [+0.0616, +0.0722] (3 seeds, excludes 0); per-seed Δ +0.0691 / +0.0616 / +0.0722, paired t p=0.002
- ctrl l4_bystander_cad_mean: circuit − unrestricted: Δ=+0.0302 [+0.0138, +0.0470] (CI excludes 0; n=9; means 0.1958 vs 0.1656); seed-cluster CI [+0.0202, +0.0368] (3 seeds, excludes 0); per-seed Δ +0.0368 / +0.0202 / +0.0335, paired t p=0.027
  - same test on the joint-ranked definition (as first reported): Δ=+0.0292 [+0.0012, +0.0581] (CI excludes 0; n=9; means 0.2283 vs 0.1991); seed-cluster CI [+0.0190, +0.0419] (3 seeds, excludes 0); per-seed Δ +0.0268 / +0.0190 / +0.0419, paired t p=0.049
- ctrl l4_bystander_cad_mean: circuit − safetune_rd: Δ=-0.1718 [-0.2125, -0.1312] (CI excludes 0; n=9; means 0.1958 vs 0.3676); seed-cluster CI [-0.1927, -0.1575] (3 seeds, excludes 0); per-seed Δ -0.1652 / -0.1575 / -0.1927, paired t p=0.004
  - same test on the joint-ranked definition (as first reported): Δ=-0.1974 [-0.2465, -0.1464] (CI excludes 0; n=9; means 0.2283 vs 0.4257); seed-cluster CI [-0.2085, -0.1761] (3 seeds, excludes 0); per-seed Δ -0.2076 / -0.1761 / -0.2085, paired t p=0.003
- H2 l3_mean_abs_delta: circuit − random: Δ=+0.0769 [+0.0341, +0.1165] (CI excludes 0; n=60; means 0.1858 vs 0.1089); seed-cluster CI [-0.0203, +0.1495] (3 seeds, includes 0); per-seed Δ -0.0203 / +0.1495 / +0.1262, paired t p=0.250
- H2 l3_mean_abs_delta: circuit − inverse: Δ=+0.1501 [+0.1046, +0.1972] (CI excludes 0; n=36; means 0.1879 vs 0.0378); seed-cluster CI [+0.1272, +0.1741] (3 seeds, excludes 0); per-seed Δ +0.1272 / +0.1490 / +0.1741, paired t p=0.008
- ctrl l3_mean_abs_delta: circuit − taylor: Δ=+0.1068 [+0.0764, +0.1397] (CI excludes 0; n=36; means 0.1879 vs 0.0812); seed-cluster CI [+0.0991, +0.1173] (3 seeds, excludes 0); per-seed Δ +0.0991 / +0.1040 / +0.1173, paired t p=0.003
- ctrl l3_mean_abs_delta: circuit − unrestricted: Δ=-0.0749 [-0.0978, -0.0525] (CI excludes 0; n=36; means 0.1879 vs 0.2628); seed-cluster CI [-0.0825, -0.0644] (3 seeds, excludes 0); per-seed Δ -0.0825 / -0.0777 / -0.0644, paired t p=0.005
- ctrl l3_mean_abs_delta: circuit − safetune_rd: Δ=-0.4563 [-0.5450, -0.3654] (CI excludes 0; n=36; means 0.1879 vs 0.6442); seed-cluster CI [-0.4828, -0.4394] (3 seeds, excludes 0); per-seed Δ -0.4828 / -0.4466 / -0.4394, paired t p=0.001
- ctrl l2_benign_refusal_delta_en: circuit − random: Δ=+0.0214 [-0.2214, +0.1571] (CI includes 0; n=5; means 0.1786 vs 0.1571); seed-cluster CI [-0.1786, +0.1786] (3 seeds, includes 0); per-seed Δ -0.1786 / +0.1429 / +0.1786, paired t p=0.716
- ctrl l2_benign_refusal_delta_en: circuit − inverse: Δ=+0.1786 [+0.1071, +0.2500] (CI excludes 0; n=3; means 0.1905 vs 0.0119); seed-cluster CI [+0.1071, +0.2500] (3 seeds, excludes 0); per-seed Δ +0.1071 / +0.1786 / +0.2500, paired t p=0.049
- ctrl l2_benign_refusal_delta_en: circuit − taylor: Δ=+0.1190 [+0.0714, +0.1786] (CI excludes 0; n=3; means 0.1905 vs 0.0714); seed-cluster CI [+0.0714, +0.1786] (3 seeds, excludes 0); per-seed Δ +0.0714 / +0.1071 / +0.1786, paired t p=0.063
- ctrl l2_benign_refusal_delta_en: circuit − unrestricted: Δ=-0.3095 [-0.3571, -0.2500] (CI excludes 0; n=3; means 0.1905 vs 0.5000); seed-cluster CI [-0.3571, -0.2500] (3 seeds, excludes 0); per-seed Δ -0.3571 / -0.3214 / -0.2500, paired t p=0.010
- ctrl l2_benign_refusal_delta_en: circuit − safetune_rd: Δ=-0.6190 [-0.6786, -0.5714] (CI excludes 0; n=3; means 0.1905 vs 0.8095); seed-cluster CI [-0.6786, -0.5714] (3 seeds, excludes 0); per-seed Δ -0.6786 / -0.5714 / -0.6071, paired t p=0.003
- ctrl l1_neigh_en_delta: circuit − random: Δ=-0.0400 [-0.0633, -0.0033] (CI excludes 0; n=5; means -0.0967 vs -0.0567); seed-cluster CI [-0.0667, -0.0083] (3 seeds, excludes 0); per-seed Δ -0.0083 / -0.0583 / -0.0667, paired t p=0.135
- ctrl l1_neigh_en_delta: circuit − inverse: Δ=-0.0833 [-0.1167, -0.0500] (CI excludes 0; n=3; means -0.1000 vs -0.0167); seed-cluster CI [-0.1167, -0.0500] (3 seeds, excludes 0); per-seed Δ -0.0500 / -0.0833 / -0.1167, paired t p=0.049
- ctrl l1_neigh_en_delta: circuit − taylor: Δ=-0.0667 [-0.0833, -0.0500] (CI excludes 0; n=3; means -0.1000 vs -0.0333); seed-cluster CI [-0.0833, -0.0500] (3 seeds, excludes 0); per-seed Δ -0.0500 / -0.0667 / -0.0833, paired t p=0.020
- ctrl l1_neigh_en_delta: circuit − unrestricted: Δ=-0.0333 [-0.0500, -0.0167] (CI excludes 0; n=3; means -0.1000 vs -0.0667); seed-cluster CI [-0.0500, -0.0167] (3 seeds, excludes 0); per-seed Δ -0.0167 / -0.0333 / -0.0500, paired t p=0.074
- ctrl l1_neigh_en_delta: circuit − safetune_rd: Δ=+0.6722 [+0.4500, +0.8500] (CI excludes 0; n=3; means -0.1000 vs -0.7722); seed-cluster CI [+0.4500, +0.8500] (3 seeds, excludes 0); per-seed Δ +0.8500 / +0.4500 / +0.7167, paired t p=0.029
- ctrl l1_wikitext_ppl_ratio: circuit − random: Δ=+0.0438 [+0.0026, +0.0824] (CI excludes 0; n=5; means 1.1657 vs 1.1219); seed-cluster CI [-0.0102, +0.0927] (3 seeds, includes 0); per-seed Δ -0.0102 / +0.0927 / +0.0539, paired t p=0.269
- ctrl l1_wikitext_ppl_ratio: circuit − inverse: Δ=+0.1085 [+0.0747, +0.1532] (CI excludes 0; n=3; means 1.1679 vs 1.0594); seed-cluster CI [+0.0747, +0.1532] (3 seeds, excludes 0); per-seed Δ +0.0747 / +0.0977 / +0.1532, paired t p=0.043
- ctrl l1_wikitext_ppl_ratio: circuit − taylor: Δ=+0.0442 [+0.0364, +0.0554] (CI excludes 0; n=3; means 1.1679 vs 1.1237); seed-cluster CI [+0.0364, +0.0554] (3 seeds, excludes 0); per-seed Δ +0.0407 / +0.0364 / +0.0554, paired t p=0.017
- ctrl l1_wikitext_ppl_ratio: circuit − unrestricted: Δ=-0.0141 [-0.0218, -0.0028] (CI excludes 0; n=3; means 1.1679 vs 1.1820); seed-cluster CI [-0.0218, -0.0028] (3 seeds, excludes 0); per-seed Δ -0.0176 / -0.0218 / -0.0028, paired t p=0.134
- ctrl l1_wikitext_ppl_ratio: circuit − safetune_rd: Δ=-0.3272 [-0.3607, -0.2701] (CI excludes 0; n=3; means 1.1679 vs 1.4951); seed-cluster CI [-0.3607, -0.2701] (3 seeds, excludes 0); per-seed Δ -0.3607 / -0.2701 / -0.3508, paired t p=0.008

## lora

- **H1** efficacy circuit/unrestricted = 1.000 [1.000, 1.000] → supported at the 0.9 rule — vacuous: both efficacies < 0.1, the intervention does not produce the behaviour; paired per seed: Δ=+0.0000 [+0.0000, +0.0000] (CI includes 0; n=3; means 0.0833 vs 0.0833); seed-cluster CI [+0.0000, +0.0000] (3 seeds, includes 0); per-seed Δ +0.0000 / +0.0000 / +0.0000, paired t p=nan
- **H1-xl (A3, mean hi/ar/fr/sw held-out harmful refusal gain)** efficacy circuit/unrestricted = 0.778 [0.778, 0.778] → refuted at the 0.9 rule; paired per seed: Δ=-0.0833 [-0.0833, -0.0833] (CI excludes 0; n=3; means 0.2917 vs 0.3750); seed-cluster CI [-0.0833, -0.0833] (3 seeds, excludes 0); per-seed Δ -0.0833 / -0.0833 / -0.0833, paired t p=0.000
- A3 efficacy_xl: circuit − random (seed × language): Δ=+0.0042 [-0.0167, +0.0250] (CI includes 0; n=20; means 0.2917 vs 0.2875); seed-cluster CI [-0.0417, +0.0208] (3 seeds, includes 0); per-seed Δ +0.0104 / +0.0208 / -0.0417, paired t p=0.874
- A3 efficacy_xl: circuit − inverse (seed × language): Δ=+0.0903 [+0.0486, +0.1389] (CI excludes 0; n=12; means 0.2917 vs 0.2014); seed-cluster CI [+0.0208, +0.1667] (3 seeds, excludes 0); per-seed Δ +0.0208 / +0.0833 / +0.1667, paired t p=0.166
- A3 efficacy_xl: circuit − taylor (seed × language): Δ=+0.0139 [+0.0000, +0.0347] (CI includes 0; n=12; means 0.2917 vs 0.2778); seed-cluster CI [+0.0000, +0.0208] (3 seeds, includes 0); per-seed Δ +0.0208 / +0.0000 / +0.0208, paired t p=0.184
- A3 efficacy_xl: circuit − unrestricted (seed × language): Δ=-0.0833 [-0.1181, -0.0486] (CI excludes 0; n=12; means 0.2917 vs 0.3750); seed-cluster CI [-0.0833, -0.0833] (3 seeds, excludes 0); per-seed Δ -0.0833 / -0.0833 / -0.0833, paired t p=0.000
- A3 efficacy_xl: circuit − perturb_circuit (seed × language): Δ=+0.2986 [+0.2500, +0.3472] (CI excludes 0; n=12; means 0.2917 vs -0.0069); seed-cluster CI [+0.2917, +0.3125] (3 seeds, excludes 0); per-seed Δ +0.2917 / +0.3125 / +0.2917, paired t p=0.001
- H2 l4_bystander_cad_mean: circuit − random: Δ=+0.0084 [+0.0023, +0.0155] (CI excludes 0; n=15; means 0.0496 vs 0.0412); seed-cluster CI [+0.0052, +0.0213] (3 seeds, excludes 0); per-seed Δ +0.0052 / +0.0052 / +0.0213, paired t p=0.187
  - same test on the joint-ranked definition (as first reported): Δ=+0.0090 [+0.0038, +0.0147] (CI excludes 0; n=15; means 0.0545 vs 0.0455); seed-cluster CI [+0.0048, +0.0233] (3 seeds, excludes 0); per-seed Δ +0.0048 / +0.0060 / +0.0233, paired t p=0.197
- H2 l4_bystander_cad_mean: circuit − inverse: Δ=+0.0156 [+0.0083, +0.0235] (CI excludes 0; n=9; means 0.0495 vs 0.0339); seed-cluster CI [+0.0133, +0.0193] (3 seeds, excludes 0); per-seed Δ +0.0133 / +0.0142 / +0.0193, paired t p=0.014
  - same test on the joint-ranked definition (as first reported): Δ=+0.0157 [+0.0091, +0.0229] (CI excludes 0; n=9; means 0.0549 vs 0.0393); seed-cluster CI [+0.0123, +0.0187] (3 seeds, excludes 0); per-seed Δ +0.0123 / +0.0159 / +0.0187, paired t p=0.014
- ctrl l4_bystander_cad_mean: circuit − taylor: Δ=+0.0257 [+0.0164, +0.0354] (CI excludes 0; n=9; means 0.0495 vs 0.0238); seed-cluster CI [+0.0241, +0.0268] (3 seeds, excludes 0); per-seed Δ +0.0241 / +0.0260 / +0.0268, paired t p=0.001
  - same test on the joint-ranked definition (as first reported): Δ=+0.0254 [+0.0176, +0.0339] (CI excludes 0; n=9; means 0.0549 vs 0.0295); seed-cluster CI [+0.0205, +0.0289] (3 seeds, excludes 0); per-seed Δ +0.0205 / +0.0269 / +0.0289, paired t p=0.010
- ctrl l4_bystander_cad_mean: circuit − unrestricted: Δ=-0.0140 [-0.0172, -0.0099] (CI excludes 0; n=9; means 0.0495 vs 0.0635); seed-cluster CI [-0.0160, -0.0127] (3 seeds, excludes 0); per-seed Δ -0.0160 / -0.0132 / -0.0127, paired t p=0.005
  - same test on the joint-ranked definition (as first reported): Δ=-0.0174 [-0.0206, -0.0142] (CI excludes 0; n=9; means 0.0549 vs 0.0723); seed-cluster CI [-0.0200, -0.0158] (3 seeds, excludes 0); per-seed Δ -0.0200 / -0.0162 / -0.0158, paired t p=0.006
- ctrl l4_bystander_cad_mean: circuit − perturb_circuit: Δ=+0.0342 [+0.0241, +0.0440] (CI excludes 0; n=9; means 0.0495 vs 0.0153); seed-cluster CI [+0.0338, +0.0347] (3 seeds, excludes 0); per-seed Δ +0.0338 / +0.0342 / +0.0347, paired t p=0.000
  - same test on the joint-ranked definition (as first reported): Δ=+0.0370 [+0.0257, +0.0489] (CI excludes 0; n=9; means 0.0549 vs 0.0180); seed-cluster CI [+0.0332, +0.0401] (3 seeds, excludes 0); per-seed Δ +0.0332 / +0.0376 / +0.0401, paired t p=0.003
- H2 l3_mean_abs_delta: circuit − random: Δ=-0.0075 [-0.0147, -0.0001] (CI excludes 0; n=60; means 0.0822 vs 0.0898); seed-cluster CI [-0.0178, -0.0046] (3 seeds, excludes 0); per-seed Δ -0.0053 / -0.0046 / -0.0178, paired t p=0.163
- H2 l3_mean_abs_delta: circuit − inverse: Δ=+0.0371 [+0.0220, +0.0537] (CI excludes 0; n=36; means 0.0819 vs 0.0448); seed-cluster CI [+0.0276, +0.0463] (3 seeds, excludes 0); per-seed Δ +0.0276 / +0.0373 / +0.0463, paired t p=0.021
- ctrl l3_mean_abs_delta: circuit − taylor: Δ=-0.0148 [-0.0224, -0.0073] (CI excludes 0; n=36; means 0.0819 vs 0.0967); seed-cluster CI [-0.0154, -0.0138] (3 seeds, excludes 0); per-seed Δ -0.0138 / -0.0152 / -0.0154, paired t p=0.001
- ctrl l3_mean_abs_delta: circuit − unrestricted: Δ=-0.1199 [-0.1601, -0.0833] (CI excludes 0; n=36; means 0.0819 vs 0.2018); seed-cluster CI [-0.1287, -0.1028] (3 seeds, excludes 0); per-seed Δ -0.1028 / -0.1283 / -0.1287, paired t p=0.005
- ctrl l3_mean_abs_delta: circuit − perturb_circuit: Δ=+0.0777 [+0.0533, +0.1059] (CI excludes 0; n=36; means 0.0819 vs 0.0042); seed-cluster CI [+0.0770, +0.0782] (3 seeds, excludes 0); per-seed Δ +0.0779 / +0.0770 / +0.0782, paired t p=0.000
- ctrl l2_benign_refusal_delta_en: circuit − random: Δ=-0.0286 [-0.1143, +0.0571] (CI includes 0; n=5; means 0.3714 vs 0.4000); seed-cluster CI [-0.1250, +0.1071] (3 seeds, includes 0); per-seed Δ -0.0000 / -0.1250 / +0.1071, paired t p=0.937
- ctrl l2_benign_refusal_delta_en: circuit − inverse: Δ=+0.0119 [-0.1071, +0.1071] (CI includes 0; n=3; means 0.3810 vs 0.3690); seed-cluster CI [-0.1071, +0.1071] (3 seeds, includes 0); per-seed Δ +0.0357 / +0.1071 / -0.1071, paired t p=0.868
- ctrl l2_benign_refusal_delta_en: circuit − taylor: Δ=-0.1667 [-0.1786, -0.1429] (CI excludes 0; n=3; means 0.3810 vs 0.5476); seed-cluster CI [-0.1786, -0.1429] (3 seeds, excludes 0); per-seed Δ -0.1786 / -0.1786 / -0.1429, paired t p=0.005
- ctrl l2_benign_refusal_delta_en: circuit − unrestricted: Δ=-0.0357 [-0.0714, +0.0000] (CI includes 0; n=3; means 0.3810 vs 0.4167); seed-cluster CI [-0.0714, +0.0000] (3 seeds, includes 0); per-seed Δ -0.0357 / -0.0714 / +0.0000, paired t p=0.225
- ctrl l2_benign_refusal_delta_en: circuit − perturb_circuit: Δ=+0.3810 [+0.3571, +0.4286] (CI excludes 0; n=3; means 0.3810 vs 0.0000); seed-cluster CI [+0.3571, +0.4286] (3 seeds, excludes 0); per-seed Δ +0.3571 / +0.3571 / +0.4286, paired t p=0.004
- ctrl l1_neigh_en_delta: circuit − random: Δ=+0.0033 [-0.0067, +0.0133] (CI includes 0; n=5; means -0.0167 vs -0.0200); seed-cluster CI [-0.0167, +0.0167] (3 seeds, includes 0); per-seed Δ +0.0000 / +0.0167 / -0.0167, paired t p=1.000
- ctrl l1_neigh_en_delta: circuit − inverse: Δ=+0.0056 [+0.0000, +0.0167] (CI includes 0; n=3; means -0.0167 vs -0.0222); seed-cluster CI [+0.0000, +0.0167] (3 seeds, includes 0); per-seed Δ +0.0000 / +0.0000 / +0.0167, paired t p=0.423
- ctrl l1_neigh_en_delta: circuit − taylor: Δ=+0.0167 [+0.0167, +0.0167] (CI excludes 0; n=3; means -0.0167 vs -0.0333); seed-cluster CI [+0.0167, +0.0167] (3 seeds, excludes 0); per-seed Δ +0.0167 / +0.0167 / +0.0167, paired t p=0.000
- ctrl l1_neigh_en_delta: circuit − unrestricted: Δ=+0.0167 [+0.0167, +0.0167] (CI excludes 0; n=3; means -0.0167 vs -0.0333); seed-cluster CI [+0.0167, +0.0167] (3 seeds, excludes 0); per-seed Δ +0.0167 / +0.0167 / +0.0167, paired t p=0.000
- ctrl l1_neigh_en_delta: circuit − perturb_circuit: Δ=-0.0167 [-0.0167, -0.0167] (CI excludes 0; n=3; means -0.0167 vs 0.0000); seed-cluster CI [-0.0167, -0.0167] (3 seeds, excludes 0); per-seed Δ -0.0167 / -0.0167 / -0.0167, paired t p=0.000
- ctrl l1_wikitext_ppl_ratio: circuit − random: Δ=+0.0306 [+0.0044, +0.0560] (CI excludes 0; n=5; means 1.0639 vs 1.0333); seed-cluster CI [+0.0129, +0.0580] (3 seeds, excludes 0); per-seed Δ +0.0347 / +0.0129 / +0.0580, paired t p=0.114
- ctrl l1_wikitext_ppl_ratio: circuit − inverse: Δ=+0.0513 [+0.0453, +0.0597] (CI excludes 0; n=3; means 1.0638 vs 1.0125); seed-cluster CI [+0.0453, +0.0597] (3 seeds, excludes 0); per-seed Δ +0.0597 / +0.0453 / +0.0489, paired t p=0.007
- ctrl l1_wikitext_ppl_ratio: circuit − taylor: Δ=+0.0104 [+0.0006, +0.0164] (CI excludes 0; n=3; means 1.0638 vs 1.0534); seed-cluster CI [+0.0006, +0.0164] (3 seeds, excludes 0); per-seed Δ +0.0164 / +0.0006 / +0.0142, paired t p=0.171
- ctrl l1_wikitext_ppl_ratio: circuit − unrestricted: Δ=+0.0161 [+0.0084, +0.0277] (CI excludes 0; n=3; means 1.0638 vs 1.0477); seed-cluster CI [+0.0084, +0.0277] (3 seeds, excludes 0); per-seed Δ +0.0277 / +0.0084 / +0.0123, paired t p=0.112
- ctrl l1_wikitext_ppl_ratio: circuit − perturb_circuit: Δ=+0.0634 [+0.0563, +0.0716] (CI excludes 0; n=3; means 1.0638 vs 1.0004); seed-cluster CI [+0.0563, +0.0716] (3 seeds, excludes 0); per-seed Δ +0.0716 / +0.0563 / +0.0622, paired t p=0.005

## edit

- **H1** undefined: unrestricted efficacy 0.000 ≤ 0 (circuit -0.111); paired per seed: Δ=-0.1111 [-0.1667, -0.0833] (CI excludes 0; n=3; means -0.1111 vs 0.0000); seed-cluster CI [-0.1667, -0.0833] (3 seeds, excludes 0); per-seed Δ -0.0833 / -0.1667 / -0.0833, paired t p=0.057
- **H1-xl (A3, mean hi/ar/fr/sw held-out harmful refusal gain)** undefined: unrestricted efficacy -0.104 ≤ 0 (circuit -0.125); paired per seed: Δ=-0.0208 [-0.0417, +0.0000] (CI includes 0; n=3; means -0.1250 vs -0.1042); seed-cluster CI [-0.0417, +0.0000] (3 seeds, includes 0); per-seed Δ -0.0208 / -0.0417 / +0.0000, paired t p=0.225
- A3 efficacy_xl: circuit − random (seed × language): Δ=-0.1542 [-0.2125, -0.1000] (CI excludes 0; n=20; means -0.1250 vs 0.0292); seed-cluster CI [-0.1667, -0.1458] (3 seeds, excludes 0); per-seed Δ -0.1458 / -0.1562 / -0.1667, paired t p=0.001
- A3 efficacy_xl: circuit − inverse (seed × language): Δ=-0.2708 [-0.3264, -0.2153] (CI excludes 0; n=12; means -0.1250 vs 0.1458); seed-cluster CI [-0.2917, -0.2500] (3 seeds, excludes 0); per-seed Δ -0.2708 / -0.2917 / -0.2500, paired t p=0.002
- A3 efficacy_xl: circuit − taylor (seed × language): Δ=-0.1181 [-0.1944, -0.0556] (CI excludes 0; n=12; means -0.1250 vs -0.0069); seed-cluster CI [-0.1250, -0.1042] (3 seeds, excludes 0); per-seed Δ -0.1042 / -0.1250 / -0.1250, paired t p=0.003
- A3 efficacy_xl: circuit − unrestricted (seed × language): Δ=-0.0208 [-0.0972, +0.0486] (CI includes 0; n=12; means -0.1250 vs -0.1042); seed-cluster CI [-0.0417, +0.0000] (3 seeds, includes 0); per-seed Δ -0.0208 / -0.0417 / +0.0000, paired t p=0.225
- A3 efficacy_xl: circuit − perturb_circuit (seed × language): Δ=+0.0972 [-0.0069, +0.2014] (CI includes 0; n=12; means -0.1250 vs -0.2222); seed-cluster CI [-0.0417, +0.2292] (3 seeds, includes 0); per-seed Δ +0.2292 / -0.0417 / +0.1042, paired t p=0.340
- A3 efficacy_xl: circuit − circuit_deep (seed × language): Δ=-0.1875 [-0.2639, -0.1181] (CI excludes 0; n=12; means -0.1250 vs 0.0625); seed-cluster CI [-0.2083, -0.1667] (3 seeds, excludes 0); per-seed Δ -0.2083 / -0.1875 / -0.1667, paired t p=0.004
- H2 l4_bystander_cad_mean: circuit − random: Δ=+0.2763 [+0.2540, +0.3010] (CI excludes 0; n=15; means 0.3256 vs 0.0493); seed-cluster CI [+0.2404, +0.3018] (3 seeds, excludes 0); per-seed Δ +0.2687 / +0.3018 / +0.2404, paired t p=0.004
  - same test on the joint-ranked definition (as first reported): Δ=+0.3197 [+0.2881, +0.3537] (CI excludes 0; n=15; means 0.3823 vs 0.0626); seed-cluster CI [+0.2891, +0.3481] (3 seeds, excludes 0); per-seed Δ +0.3066 / +0.3481 / +0.2891, paired t p=0.003
- H2 l4_bystander_cad_mean: circuit − inverse: Δ=+0.2673 [+0.2353, +0.3044] (CI excludes 0; n=9; means 0.3170 vs 0.0496); seed-cluster CI [+0.2315, +0.3228] (3 seeds, excludes 0); per-seed Δ +0.2477 / +0.3228 / +0.2315, paired t p=0.011
  - same test on the joint-ranked definition (as first reported): Δ=+0.3092 [+0.2664, +0.3584] (CI excludes 0; n=9; means 0.3740 vs 0.0648); seed-cluster CI [+0.2710, +0.3798] (3 seeds, excludes 0); per-seed Δ +0.2767 / +0.3798 / +0.2710, paired t p=0.013
- ctrl l4_bystander_cad_mean: circuit − taylor: Δ=+0.2898 [+0.2559, +0.3272] (CI excludes 0; n=9; means 0.3170 vs 0.0271); seed-cluster CI [+0.2473, +0.3531] (3 seeds, excludes 0); per-seed Δ +0.2691 / +0.3531 / +0.2473, paired t p=0.012
  - same test on the joint-ranked definition (as first reported): Δ=+0.3436 [+0.2911, +0.4035] (CI excludes 0; n=9; means 0.3740 vs 0.0303); seed-cluster CI [+0.3038, +0.4199] (3 seeds, excludes 0); per-seed Δ +0.3072 / +0.4199 / +0.3038, paired t p=0.012
- ctrl l4_bystander_cad_mean: circuit − unrestricted: Δ=+0.2517 [+0.2210, +0.2890] (CI excludes 0; n=9; means 0.3170 vs 0.0652); seed-cluster CI [+0.2165, +0.3096] (3 seeds, excludes 0); per-seed Δ +0.2290 / +0.3096 / +0.2165, paired t p=0.013
  - same test on the joint-ranked definition (as first reported): Δ=+0.3023 [+0.2562, +0.3531] (CI excludes 0; n=9; means 0.3740 vs 0.0717); seed-cluster CI [+0.2672, +0.3721] (3 seeds, excludes 0); per-seed Δ +0.2676 / +0.3721 / +0.2672, paired t p=0.013
- ctrl l4_bystander_cad_mean: circuit − perturb_circuit: Δ=-0.1212 [-0.1884, -0.0561] (CI excludes 0; n=9; means 0.3170 vs 0.4381); seed-cluster CI [-0.1776, -0.0510] (3 seeds, excludes 0); per-seed Δ -0.1776 / -0.0510 / -0.1349, paired t p=0.083
  - same test on the joint-ranked definition (as first reported): Δ=-0.1395 [-0.2084, -0.0751] (CI excludes 0; n=9; means 0.3740 vs 0.5135); seed-cluster CI [-0.2191, -0.0542] (3 seeds, excludes 0); per-seed Δ -0.2191 / -0.0542 / -0.1452, paired t p=0.100
- ctrl l4_bystander_cad_mean: circuit − circuit_deep: Δ=+0.2732 [+0.2384, +0.3116] (CI excludes 0; n=9; means 0.3170 vs 0.0437); seed-cluster CI [+0.2299, +0.3386] (3 seeds, excludes 0); per-seed Δ +0.2512 / +0.3386 / +0.2299, paired t p=0.014
  - same test on the joint-ranked definition (as first reported): Δ=+0.3206 [+0.2745, +0.3734] (CI excludes 0; n=9; means 0.3740 vs 0.0534); seed-cluster CI [+0.2802, +0.3962] (3 seeds, excludes 0); per-seed Δ +0.2854 / +0.3962 / +0.2802, paired t p=0.014
- H2 l3_mean_abs_delta: circuit − random: Δ=+0.0573 [+0.0336, +0.0828] (CI excludes 0; n=60; means 0.0771 vs 0.0198); seed-cluster CI [+0.0384, +0.0820] (3 seeds, excludes 0); per-seed Δ +0.0421 / +0.0820 / +0.0384, paired t p=0.061
- H2 l3_mean_abs_delta: circuit − inverse: Δ=+0.0385 [+0.0072, +0.0716] (CI excludes 0; n=36; means 0.0730 vs 0.0345); seed-cluster CI [+0.0176, +0.0750] (3 seeds, excludes 0); per-seed Δ +0.0176 / +0.0750 / +0.0229, paired t p=0.170
- ctrl l3_mean_abs_delta: circuit − taylor: Δ=+0.0501 [+0.0221, +0.0819] (CI excludes 0; n=36; means 0.0730 vs 0.0229); seed-cluster CI [+0.0319, +0.0840] (3 seeds, excludes 0); per-seed Δ +0.0342 / +0.0840 / +0.0319, paired t p=0.098
- ctrl l3_mean_abs_delta: circuit − unrestricted: Δ=+0.0522 [+0.0262, +0.0839] (CI excludes 0; n=36; means 0.0730 vs 0.0207); seed-cluster CI [+0.0245, +0.0917] (3 seeds, excludes 0); per-seed Δ +0.0405 / +0.0917 / +0.0245, paired t p=0.123
- ctrl l3_mean_abs_delta: circuit − perturb_circuit: Δ=-0.1132 [-0.1714, -0.0602] (CI excludes 0; n=36; means 0.0730 vs 0.1862); seed-cluster CI [-0.1892, -0.0581] (3 seeds, excludes 0); per-seed Δ -0.1892 / -0.0581 / -0.0924, paired t p=0.102
- ctrl l3_mean_abs_delta: circuit − circuit_deep: Δ=+0.0498 [+0.0207, +0.0807] (CI excludes 0; n=36; means 0.0730 vs 0.0231); seed-cluster CI [+0.0285, +0.0903] (3 seeds, excludes 0); per-seed Δ +0.0308 / +0.0903 / +0.0285, paired t p=0.133
- ctrl l2_benign_refusal_delta_en: circuit − random: Δ=-0.0357 [-0.0786, +0.0000] (CI includes 0; n=5; means -0.0286 vs 0.0071); seed-cluster CI [-0.0893, +0.0000] (3 seeds, includes 0); per-seed Δ +0.0000 / -0.0893 / +0.0000, paired t p=0.423
- ctrl l2_benign_refusal_delta_en: circuit − inverse: Δ=+0.0000 [-0.0357, +0.0357] (CI includes 0; n=3; means -0.0238 vs -0.0238); seed-cluster CI [-0.0357, +0.0357] (3 seeds, includes 0); per-seed Δ +0.0000 / -0.0357 / +0.0357, paired t p=1.000
- ctrl l2_benign_refusal_delta_en: circuit − taylor: Δ=-0.0238 [-0.0714, +0.0000] (CI includes 0; n=3; means -0.0238 vs 0.0000); seed-cluster CI [-0.0714, +0.0000] (3 seeds, includes 0); per-seed Δ +0.0000 / -0.0714 / +0.0000, paired t p=0.423
- ctrl l2_benign_refusal_delta_en: circuit − unrestricted: Δ=-0.0238 [-0.0714, +0.0000] (CI includes 0; n=3; means -0.0238 vs 0.0000); seed-cluster CI [-0.0714, +0.0000] (3 seeds, includes 0); per-seed Δ +0.0000 / -0.0714 / +0.0000, paired t p=0.423
- ctrl l2_benign_refusal_delta_en: circuit − perturb_circuit: Δ=-0.0238 [-0.1786, +0.0714] (CI includes 0; n=3; means -0.0238 vs 0.0000); seed-cluster CI [-0.1786, +0.0714] (3 seeds, includes 0); per-seed Δ +0.0714 / -0.1786 / +0.0357, paired t p=0.789
- ctrl l2_benign_refusal_delta_en: circuit − circuit_deep: Δ=-0.0238 [-0.0714, +0.0000] (CI includes 0; n=3; means -0.0238 vs 0.0000); seed-cluster CI [-0.0714, +0.0000] (3 seeds, includes 0); per-seed Δ +0.0000 / -0.0714 / +0.0000, paired t p=0.423
- ctrl l1_neigh_en_delta: circuit − random: Δ=-0.0433 [-0.0733, -0.0133] (CI excludes 0; n=5; means -0.0600 vs -0.0167); seed-cluster CI [-0.0833, -0.0083] (3 seeds, excludes 0); per-seed Δ -0.0083 / -0.0833 / -0.0333, paired t p=0.199
- ctrl l1_neigh_en_delta: circuit − inverse: Δ=-0.0500 [-0.0833, -0.0333] (CI excludes 0; n=3; means -0.0556 vs -0.0056); seed-cluster CI [-0.0833, -0.0333] (3 seeds, excludes 0); per-seed Δ -0.0333 / -0.0833 / -0.0333, paired t p=0.095
- ctrl l1_neigh_en_delta: circuit − taylor: Δ=-0.0389 [-0.0833, -0.0167] (CI excludes 0; n=3; means -0.0556 vs -0.0167); seed-cluster CI [-0.0833, -0.0167] (3 seeds, excludes 0); per-seed Δ -0.0167 / -0.0833 / -0.0167, paired t p=0.222
- ctrl l1_neigh_en_delta: circuit − unrestricted: Δ=-0.0389 [-0.0833, -0.0167] (CI excludes 0; n=3; means -0.0556 vs -0.0167); seed-cluster CI [-0.0833, -0.0167] (3 seeds, excludes 0); per-seed Δ -0.0167 / -0.0833 / -0.0167, paired t p=0.222
- ctrl l1_neigh_en_delta: circuit − perturb_circuit: Δ=+0.0667 [+0.0000, +0.1833] (CI includes 0; n=3; means -0.0556 vs -0.1222); seed-cluster CI [+0.0000, +0.1833] (3 seeds, includes 0); per-seed Δ +0.1833 / +0.0000 / +0.0167, paired t p=0.373
- ctrl l1_neigh_en_delta: circuit − circuit_deep: Δ=-0.0389 [-0.0833, -0.0167] (CI excludes 0; n=3; means -0.0556 vs -0.0167); seed-cluster CI [-0.0833, -0.0167] (3 seeds, excludes 0); per-seed Δ -0.0167 / -0.0833 / -0.0167, paired t p=0.222
- ctrl l1_wikitext_ppl_ratio: circuit − random: Δ=-0.0047 [-0.0056, -0.0037] (CI excludes 0; n=5; means 0.9977 vs 1.0023); seed-cluster CI [-0.0050, -0.0034] (3 seeds, excludes 0); per-seed Δ -0.0050 / -0.0049 / -0.0034, paired t p=0.013
- ctrl l1_wikitext_ppl_ratio: circuit − inverse: Δ=-0.0035 [-0.0041, -0.0029] (CI excludes 0; n=3; means 0.9978 vs 1.0013); seed-cluster CI [-0.0041, -0.0029] (3 seeds, excludes 0); per-seed Δ -0.0041 / -0.0034 / -0.0029, paired t p=0.010
- ctrl l1_wikitext_ppl_ratio: circuit − taylor: Δ=-0.0026 [-0.0037, -0.0019] (CI excludes 0; n=3; means 0.9978 vs 1.0004); seed-cluster CI [-0.0037, -0.0019] (3 seeds, excludes 0); per-seed Δ -0.0037 / -0.0019 / -0.0021, paired t p=0.044
- ctrl l1_wikitext_ppl_ratio: circuit − unrestricted: Δ=-0.0036 [-0.0043, -0.0031] (CI excludes 0; n=3; means 0.9978 vs 1.0014); seed-cluster CI [-0.0043, -0.0031] (3 seeds, excludes 0); per-seed Δ -0.0043 / -0.0033 / -0.0031, paired t p=0.011
- ctrl l1_wikitext_ppl_ratio: circuit − perturb_circuit: Δ=-0.0182 [-0.0272, -0.0093] (CI excludes 0; n=3; means 0.9978 vs 1.0160); seed-cluster CI [-0.0272, -0.0093] (3 seeds, excludes 0); per-seed Δ -0.0180 / -0.0093 / -0.0272, paired t p=0.072
- ctrl l1_wikitext_ppl_ratio: circuit − circuit_deep: Δ=-0.0052 [-0.0063, -0.0045] (CI excludes 0; n=3; means 0.9978 vs 1.0030); seed-cluster CI [-0.0063, -0.0045] (3 seeds, excludes 0); per-seed Δ -0.0063 / -0.0047 / -0.0045, paired t p=0.011

## A17: within-bystander overlap test (exploratory)

Each bystander's induced edge set is split into the edges shared with the cell's intervened components (`S`) or with the pre-registered target set (`T`) and the rest; both parts are scored from the kept edges with their own resample floors (`audit/l4_shared.py`). D = (CAD − floor) shared − (CAD − floor) unshared is confounded by the base magnitude of the edges in each part (CAD divides by Σ|s_e|): its sign follows the region, not the intervention, and the perturbation controls reproduce it. Λ = log(CAD/floor) shared − log(CAD/floor) unshared cancels that denominator and compares the two regions in units of their own resampling noise; + = the shared edges moved more. n = cell × bystander pairs; – = fewer than 10 of the bystander's edges touch the intervened components (the component is not in that bystander's node set).

| intervention | condition | n | shared edges (S) | D by S | Λ by S | Λ > 0 | D by T |
|---|---|---|---|---|---|---|---|
| edit | circuit | 12 | 167 | -0.109 [-0.190, -0.034] | -0.210 [-0.376, -0.036] | 0.25 | -0.099 [-0.235, +0.047] |
| edit | circuit_deep | 10 | – | – | – | – | -0.193 [-0.372, -0.007] |
| edit | inverse | 10 | – | – | – | – | -0.162 [-0.331, +0.013] |
| edit | perturb_circuit | 12 | 167 | -0.152 [-0.225, -0.064] | -0.280 [-0.449, -0.091] | 0.17 | -0.195 [-0.305, -0.066] |
| edit | perturb_circuit_deep | 10 | – | – | – | – | -0.205 [-0.385, -0.020] |
| edit | perturb_inverse | 10 | – | – | – | – | -0.169 [-0.336, -0.001] |
| edit | perturb_random | 17 | – | – | – | – | -0.155 [-0.312, +0.002] |
| edit | perturb_taylor | 3 | 92 | +0.173 [+0.081, +0.276] | +1.308 [+0.865, +1.727] | 1.00 | -0.205 [-0.392, -0.013] |
| edit | perturb_unrestricted | 12 | 461 | -0.099 [-0.146, -0.056] | +0.045 [-0.141, +0.239] | 0.58 | -0.156 [-0.319, +0.006] |
| edit | random | 17 | – | – | – | – | -0.168 [-0.319, -0.022] |
| edit | taylor | 3 | 92 | +0.150 [+0.057, +0.253] | +0.886 [+0.400, +1.389] | 1.00 | -0.224 [-0.408, -0.038] |
| edit | unrestricted | 12 | 461 | -0.104 [-0.163, -0.054] | +0.022 [-0.176, +0.201] | 0.67 | -0.156 [-0.336, +0.026] |
| lora | circuit | 12 | 1442 | -0.152 [-0.243, -0.075] | +0.186 [-0.013, +0.399] | 0.75 | -0.159 [-0.330, +0.015] |
| lora | inverse | 12 | 1058 | +0.048 [-0.003, +0.106] | +0.025 [-0.117, +0.159] | 0.50 | -0.184 [-0.367, +0.000] |
| lora | perturb_circuit | 12 | 1442 | -0.183 [-0.287, -0.093] | +0.021 [-0.153, +0.185] | 0.58 | -0.197 [-0.383, -0.006] |
| lora | perturb_inverse | 12 | 1058 | +0.050 [-0.001, +0.110] | -0.000 [-0.109, +0.099] | 0.58 | -0.199 [-0.388, -0.007] |
| lora | perturb_random | 20 | 1034 | -0.023 [-0.063, +0.014] | +0.017 [-0.079, +0.099] | 0.65 | -0.179 [-0.331, -0.030] |
| lora | perturb_taylor | 12 | 1448 | +0.095 [+0.053, +0.147] | +0.117 [+0.032, +0.203] | 0.67 | -0.204 [-0.392, -0.012] |
| lora | perturb_unrestricted | 10 | – | – | – | – | -0.199 [-0.386, -0.009] |
| lora | random | 20 | 1034 | -0.016 [-0.055, +0.019] | +0.104 [+0.032, +0.179] | 0.65 | -0.153 [-0.301, -0.009] |
| lora | taylor | 12 | 1448 | +0.099 [+0.056, +0.150] | +0.332 [+0.238, +0.427] | 1.00 | -0.202 [-0.391, -0.008] |
| lora | unrestricted | 10 | – | – | – | – | -0.166 [-0.343, +0.017] |
| steer | circuit | 12 | 1442 | -0.169 [-0.251, -0.096] | -0.274 [-0.424, -0.097] | 0.17 | -0.200 [-0.360, -0.051] |
| steer | inverse | 12 | 1058 | +0.036 [-0.011, +0.091] | -0.064 [-0.196, +0.054] | 0.42 | -0.146 [-0.302, +0.009] |
| steer | random | 20 | 1034 | -0.016 [-0.050, +0.016] | +0.022 [-0.077, +0.109] | 0.55 | -0.144 [-0.276, -0.017] |
| steer | safetune_rd | 10 | – | – | – | – | -0.235 [-0.396, -0.076] |
| steer | taylor | 12 | 1448 | +0.096 [+0.051, +0.147] | +0.187 [+0.060, +0.333] | 0.83 | -0.225 [-0.390, -0.062] |
| steer | unrestricted | 10 | – | – | – | – | -0.201 [-0.347, -0.057] |

Pooled Λ by S over all non-control cells: +0.055 [+0.000, +0.108] (n=139).

## H3 faithfulness gates locality

Pillar columns without a prefix are CircuitKIT's joint-ranked set (as first reported); `cell *` columns are the pre-registered per-type set the cells intervene on (`aya_circuits.cellset`, same evaluation path); `necessity` = ablation raw ratio with the set ablated and the rest kept (mean replacement; 1 = no loss), its margin = random per-type sets − circuit (+ = the circuit is more necessary than chance).

| cell | patching | zero raw | mean raw | stability | random margin (patching) | ablation margin (zero) | cell patching | cell zero raw | cell mean raw | cell random margin | cell necessity | cell necessity margin | adv L4 | adv L3 | P2/P5 fail | cell gate fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b2_steer_circuit_s0 | -0.039 | -0.500 | 0.382 | 0.833 | -0.046 | -0.561 | -0.001 | -0.047 | 0.273 | -0.004 | 0.483 | -0.093 | -0.017 | 0.020 | True | True |
| b2_steer_circuit_s1 | -0.027 | -0.244 | 0.332 | 0.833 | -0.037 | -0.283 | -0.002 | -0.252 | 0.207 | -0.006 | 0.513 | -0.103 | -0.072 | -0.149 | True | True |
| b2_steer_circuit_s2 | -0.039 | 0.451 | 0.180 | 0.815 | -0.041 | 0.410 | -0.005 | 0.052 | 0.201 | -0.007 | 0.529 | -0.196 | -0.005 | -0.126 | True | True |
| b2_lora_circuit_s0 | -0.039 | -0.500 | 0.382 | 0.833 | -0.046 | -0.561 | -0.001 | -0.047 | 0.273 | -0.004 | 0.483 | -0.093 | -0.005 | 0.005 | True | True |
| b2_lora_circuit_s1 | -0.027 | -0.244 | 0.332 | 0.833 | -0.037 | -0.283 | -0.002 | -0.252 | 0.207 | -0.006 | 0.513 | -0.103 | -0.005 | 0.005 | True | True |
| b2_lora_circuit_s2 | -0.039 | 0.451 | 0.180 | 0.815 | -0.041 | 0.410 | -0.005 | 0.052 | 0.201 | -0.007 | 0.529 | -0.196 | -0.021 | 0.018 | True | True |
| b2_edit_circuit_s0 | -0.039 | -0.500 | 0.382 | 0.833 | -0.046 | -0.561 | -0.001 | -0.047 | 0.273 | -0.004 | 0.483 | -0.093 | -0.269 | -0.042 | True | True |
| b2_edit_circuit_s1 | -0.027 | -0.244 | 0.332 | 0.833 | -0.037 | -0.283 | -0.002 | -0.252 | 0.207 | -0.006 | 0.513 | -0.103 | -0.302 | -0.082 | True | True |
| b2_edit_circuit_s2 | -0.039 | 0.451 | 0.180 | 0.815 | -0.041 | 0.410 | -0.005 | 0.052 | 0.201 | -0.007 | 0.529 | -0.196 | -0.240 | -0.038 | True | True |
The zero-ablation raw ratio has a floor of 0.5 on a symmetric logit-difference metric (ablating 90 % of the nodes drives the metric to 0); `ablation margin` = circuit − mean random-node set under the same ablation is the informative quantity.
- Spearman(patching, adv_l4_bystander_cad_mean) = -0.053 (p=0.893, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(patching, adv_l3_mean_abs_delta) = -0.264 (p=0.493, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(zero_raw, adv_l4_bystander_cad_mean) = -0.053 (p=0.893, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(zero_raw, adv_l3_mean_abs_delta) = -0.264 (p=0.493, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(mean_raw, adv_l4_bystander_cad_mean) = 0.053 (p=0.893, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(mean_raw, adv_l3_mean_abs_delta) = 0.264 (p=0.493, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(stability, adv_l4_bystander_cad_mean) = 0.053 (p=0.893, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(stability, adv_l3_mean_abs_delta) = 0.264 (p=0.493, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(random_margin, adv_l4_bystander_cad_mean) = -0.105 (p=0.787, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(random_margin, adv_l3_mean_abs_delta) = -0.527 (p=0.145, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(ablation_margin, adv_l4_bystander_cad_mean) = -0.053 (p=0.893, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(ablation_margin, adv_l3_mean_abs_delta) = -0.264 (p=0.493, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(cell_patching, adv_l4_bystander_cad_mean) = 0.053 (p=0.893, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(cell_patching, adv_l3_mean_abs_delta) = 0.264 (p=0.493, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(cell_zero_raw, adv_l4_bystander_cad_mean) = 0.053 (p=0.893, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(cell_zero_raw, adv_l3_mean_abs_delta) = 0.264 (p=0.493, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(cell_mean_raw, adv_l4_bystander_cad_mean) = 0.053 (p=0.893, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(cell_mean_raw, adv_l3_mean_abs_delta) = 0.264 (p=0.493, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(cell_random_margin, adv_l4_bystander_cad_mean) = 0.053 (p=0.893, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(cell_random_margin, adv_l3_mean_abs_delta) = 0.264 (p=0.493, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(cell_necessity_mean, adv_l4_bystander_cad_mean) = -0.053 (p=0.893, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(cell_necessity_mean, adv_l3_mean_abs_delta) = -0.264 (p=0.493, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(cell_necessity_margin, adv_l4_bystander_cad_mean) = 0.053 (p=0.893, n=9; only 3 distinct pillar values within one behaviour)
- Spearman(cell_necessity_margin, adv_l3_mean_abs_delta) = 0.264 (p=0.493, n=9; only 3 distinct pillar values within one behaviour)
- Gate prediction, joint set (failing circuits show no L4 advantage): 9/9 gated cells consistent
- Gate prediction, cell set (failing circuits show no L4 advantage): 9/9 gated cells consistent

Pooled over both behaviours (18 circuit cells, 6 distinct circuits):
- pooled Spearman(patching, adv_l4_bystander_cad_mean) = -0.198 (p=0.432, n=18)
- pooled Spearman(patching, adv_l3_mean_abs_delta) = -0.091 (p=0.72, n=18)
- pooled Spearman(zero_raw, adv_l4_bystander_cad_mean) = -0.172 (p=0.494, n=18)
- pooled Spearman(zero_raw, adv_l3_mean_abs_delta) = -0.204 (p=0.417, n=18)
- pooled Spearman(mean_raw, adv_l4_bystander_cad_mean) = -0.172 (p=0.494, n=18)
- pooled Spearman(mean_raw, adv_l3_mean_abs_delta) = 0.022 (p=0.931, n=18)
- pooled Spearman(stability, adv_l4_bystander_cad_mean) = 0.140 (p=0.58, n=18)
- pooled Spearman(stability, adv_l3_mean_abs_delta) = 0.099 (p=0.697, n=18)
- pooled Spearman(random_margin, adv_l4_bystander_cad_mean) = -0.204 (p=0.417, n=18)
- pooled Spearman(random_margin, adv_l3_mean_abs_delta) = -0.166 (p=0.51, n=18)
- pooled Spearman(ablation_margin, adv_l4_bystander_cad_mean) = -0.072 (p=0.776, n=18)
- pooled Spearman(ablation_margin, adv_l3_mean_abs_delta) = -0.154 (p=0.543, n=18)
- pooled Spearman(cell_patching, adv_l4_bystander_cad_mean) = -0.172 (p=0.494, n=18)
- pooled Spearman(cell_patching, adv_l3_mean_abs_delta) = 0.022 (p=0.931, n=18)
- pooled Spearman(cell_zero_raw, adv_l4_bystander_cad_mean) = -0.147 (p=0.56, n=18)
- pooled Spearman(cell_zero_raw, adv_l3_mean_abs_delta) = -0.091 (p=0.72, n=18)
- pooled Spearman(cell_mean_raw, adv_l4_bystander_cad_mean) = -0.147 (p=0.56, n=18)
- pooled Spearman(cell_mean_raw, adv_l3_mean_abs_delta) = 0.034 (p=0.892, n=18)
- pooled Spearman(cell_random_margin, adv_l4_bystander_cad_mean) = -0.172 (p=0.494, n=18)
- pooled Spearman(cell_random_margin, adv_l3_mean_abs_delta) = 0.022 (p=0.931, n=18)
- pooled Spearman(cell_necessity_mean, adv_l4_bystander_cad_mean) = 0.034 (p=0.892, n=18)
- pooled Spearman(cell_necessity_mean, adv_l3_mean_abs_delta) = -0.198 (p=0.432, n=18)
- pooled Spearman(cell_necessity_margin, adv_l4_bystander_cad_mean) = -0.179 (p=0.478, n=18)
- pooled Spearman(cell_necessity_margin, adv_l3_mean_abs_delta) = -0.022 (p=0.931, n=18)
- pooled gate prediction, joint set (failing circuits show no L4 advantage): 9/9 gated cells consistent; no L4 and no L3 advantage: 5/9
- pooled gate prediction, cell set (failing circuits show no L4 advantage): 18/18 gated cells consistent; no L4 and no L3 advantage: 13/18

## Exploratory: target-circuit edge change vs efficacy

Spearman(efficacy_xl, target CAD − floor) over the main cells without perturbation controls = 0.731 (p=1.1e-10, n=57); within intervention: steer 0.86 (p=1.3e-06, n=20), lora 0.78 (p=0.00021, n=17), edit -0.32 (p=0.18, n=20).

## H4 overlap predicts interference

Weighted-Jaccard overlap with b2_en: {"b1_en": 0.3, "ioi_fixed": 0.198, "langid_en": 0.336}; mean bystander CAD: {"b1_en": "0.088", "ioi_fixed": "0.075", "langid_en": "0.104"}; mean bystander CAD − resample floor: {"b1_en": "-0.419", "ioi_fixed": "-0.252", "langid_en": "-0.304"}; Spearman(overlap, CAD) = 0.200 (p=0.000711), Spearman(overlap, CAD − floor) = -0.242 (p=4.07e-05); n=282 cell × bystander points but only 3 distinct overlaps, so the p-values are pseudo-replicated; bystander means ranked by overlap do not agree with their rank by floor-adjusted CAD.
- per-cell Spearman(overlap, CAD) over the 3 bystanders: mean +0.676 [+0.574, +0.766] over 94 cells; seed-cluster CI [+0.554, +0.758] (3 seeds); full rank agreement in 48/94 cells
- per-cell Spearman(overlap, CAD − floor) over the 3 bystanders: mean -0.394 [-0.457, -0.324] over 94 cells; seed-cluster CI [-0.500, -0.318] (3 seeds); full rank agreement in 0/94 cells

## H5 cross-lingual propagation (B2: held-out harmful refusal gain; descriptive, H5 is pre-registered for B1 only)

| intervention | condition | gain hi | gain ar | gain fr | gain sw | neigh Δ hi | neigh Δ ar | neigh Δ fr | neigh Δ sw |
|---|---|---|---|---|---|---|---|---|---|
| lora | circuit | 0.417 | 0.250 | 0.167 | 0.333 | 0.000 | -0.006 | -0.017 | 0.000 |
| lora | taylor | 0.417 | 0.250 | 0.111 | 0.333 | 0.000 | 0.006 | -0.017 | 0.000 |
| lora | unrestricted | 0.500 | 0.417 | 0.167 | 0.417 | -0.000 | -0.017 | -0.017 | -0.017 |
| lora | random | 0.417 | 0.267 | 0.117 | 0.350 | 0.003 | 0.017 | -0.010 | 0.000 |
| lora | inverse | 0.278 | 0.194 | 0.083 | 0.250 | -0.000 | 0.006 | -0.017 | 0.000 |
| edit | circuit | -0.222 | -0.056 | -0.194 | -0.028 | -0.161 | -0.178 | -0.028 | -0.150 |
| edit | circuit_deep | 0.111 | 0.000 | 0.028 | 0.111 | 0.000 | 0.000 | 0.000 | 0.000 |
| edit | taylor | -0.083 | -0.028 | 0.083 | 0.000 | -0.006 | -0.028 | 0.000 | 0.006 |
| edit | unrestricted | -0.083 | -0.056 | -0.111 | -0.167 | -0.011 | -0.039 | 0.000 | 0.000 |
| edit | random | 0.033 | -0.000 | 0.017 | 0.067 | -0.007 | -0.017 | -0.010 | 0.000 |
| edit | inverse | 0.139 | 0.250 | 0.083 | 0.111 | -0.006 | 0.000 | -0.011 | 0.000 |
| steer | circuit | 0.306 | 0.111 | 0.083 | 0.361 | -0.072 | -0.056 | 0.000 | -0.050 |
| steer | taylor | 0.250 | 0.000 | 0.083 | 0.167 | -0.050 | -0.050 | 0.000 | 0.017 |
| steer | unrestricted | 0.333 | 0.333 | 0.167 | 0.417 | -0.067 | -0.100 | -0.017 | -0.217 |
| steer | random | 0.150 | 0.067 | 0.033 | 0.167 | -0.040 | -0.080 | -0.020 | -0.050 |
| steer | inverse | 0.083 | 0.000 | 0.056 | 0.056 | -0.017 | -0.072 | 0.006 | -0.061 |
| steer | safetune_rd | 0.389 | 0.389 | 0.167 | 0.417 | -0.672 | -0.839 | -0.833 | -0.811 |

- lora: edited-fact gain, circuit − taylor (seed × language): Δ=+0.0139 [+0.0000, +0.0347] (CI includes 0; n=12; means 0.2917 vs 0.2778); seed-cluster CI [+0.0000, +0.0208] (3 seeds, includes 0); per-seed Δ +0.0208 / +0.0000 / +0.0208, paired t p=0.184
- lora: neighbourhood accuracy change, circuit − taylor: Δ=-0.0028 [-0.0069, +0.0000] (CI includes 0; n=12; means -0.0056 vs -0.0028); seed-cluster CI [-0.0042, +0.0000] (3 seeds, includes 0); per-seed Δ -0.0042 / +0.0000 / -0.0042, paired t p=0.184 (H5 needs ≥ 0 here, i.e. no larger drop)
- edit: edited-fact gain, circuit − taylor (seed × language): Δ=-0.1181 [-0.1944, -0.0556] (CI excludes 0; n=12; means -0.1250 vs -0.0069); seed-cluster CI [-0.1250, -0.1042] (3 seeds, excludes 0); per-seed Δ -0.1042 / -0.1250 / -0.1250, paired t p=0.003
- edit: neighbourhood accuracy change, circuit − taylor: Δ=-0.1222 [-0.1875, -0.0639] (CI excludes 0; n=12; means -0.1292 vs -0.0069); seed-cluster CI [-0.2417, -0.0375] (3 seeds, excludes 0); per-seed Δ -0.0875 / -0.2417 / -0.0375, paired t p=0.185 (H5 needs ≥ 0 here, i.e. no larger drop)
- steer: edited-fact gain, circuit − taylor (seed × language): Δ=+0.0903 [+0.0417, +0.1389] (CI excludes 0; n=12; means 0.2153 vs 0.1250); seed-cluster CI [+0.0833, +0.1042] (3 seeds, excludes 0); per-seed Δ +0.0833 / +0.0833 / +0.1042, paired t p=0.006
- steer: neighbourhood accuracy change, circuit − taylor: Δ=-0.0236 [-0.0417, -0.0083] (CI excludes 0; n=12; means -0.0444 vs -0.0208); seed-cluster CI [-0.0333, -0.0167] (3 seeds, excludes 0); per-seed Δ -0.0208 / -0.0167 / -0.0333, paired t p=0.042 (H5 needs ≥ 0 here, i.e. no larger drop)

## L5 evidence arbitration (audit/arbitration.py; Anand 2026): spread ratio sd(A) cell/base, AURC and most-confident-quarter error changes, mean drift of A

| intervention | condition | n | l5_arb_spread_ratio | l5_arb_aurc_delta | l5_arb_high_conf_error_delta | l5_arb_drift_mean |
|---|---|---|---|---|---|---|
| steer | circuit | 3 | 0.860 | 0.058 | 0.046 | 1.912 |
| steer | inverse | 3 | 0.988 | -0.006 | 0.000 | -0.095 |
| steer | random | 5 | 0.990 | 0.030 | 0.038 | 0.924 |
| steer | safetune_rd | 3 | 1.032 | 0.136 | 0.164 | 6.172 |
| steer | taylor | 3 | 0.948 | 0.024 | 0.007 | 1.145 |
| steer | unrestricted | 3 | 1.018 | 0.044 | 0.022 | 1.274 |
| lora | circuit | 3 | 1.013 | 0.005 | 0.002 | 0.049 |
| lora | inverse | 3 | 1.015 | 0.001 | 0.002 | -0.002 |
| lora | random | 5 | 1.006 | 0.003 | 0.006 | 0.038 |
| lora | taylor | 3 | 1.016 | 0.004 | 0.005 | 0.087 |
| lora | unrestricted | 3 | 1.020 | 0.005 | 0.022 | 0.090 |
| lora | perturb_circuit | 3 | 1.000 | -0.000 | 0.000 | -0.018 |
| lora | perturb_inverse | 3 | 1.000 | 0.000 | 0.000 | -0.002 |
| lora | perturb_random | 5 | 1.001 | -0.000 | 0.000 | -0.004 |
| lora | perturb_taylor | 3 | 1.000 | 0.001 | 0.000 | 0.007 |
| lora | perturb_unrestricted | 3 | 1.000 | 0.000 | 0.000 | 0.006 |
| edit | circuit | 3 | 1.012 | 0.063 | 0.094 | 2.622 |
| edit | circuit_deep | 3 | 0.977 | 0.007 | -0.002 | 0.315 |
| edit | inverse | 3 | 0.997 | 0.005 | 0.010 | 0.246 |
| edit | random | 5 | 0.996 | 0.005 | 0.010 | 0.186 |
| edit | taylor | 3 | 1.006 | 0.002 | 0.000 | -0.006 |
| edit | unrestricted | 3 | 0.987 | 0.003 | -0.002 | 0.049 |
| edit | perturb_circuit | 3 | 1.204 | 0.052 | 0.118 | 2.970 |
| edit | perturb_circuit_deep | 3 | 1.012 | -0.004 | -0.007 | -0.173 |
| edit | perturb_inverse | 3 | 1.000 | -0.001 | -0.002 | 0.148 |
| edit | perturb_random | 5 | 1.000 | 0.001 | -0.001 | 0.054 |
| edit | perturb_taylor | 3 | 1.004 | 0.001 | -0.002 | 0.020 |
| edit | perturb_unrestricted | 3 | 1.013 | -0.001 | 0.002 | 0.045 |
- steer l5_arb_spread_ratio: circuit − random: Δ=-0.1285 [-0.1750, -0.0687] (CI excludes 0; n=5; means 0.8618 vs 0.9903); seed-cluster CI [-0.1837, -0.0732] (3 seeds, excludes 0); per-seed Δ -0.0732 / -0.1837 / -0.1288, paired t p=0.056
- steer l5_arb_spread_ratio: circuit − inverse: Δ=-0.1282 [-0.1444, -0.1000] (CI excludes 0; n=3; means 0.8600 vs 0.9882); seed-cluster CI [-0.1444, -0.1000] (3 seeds, excludes 0); per-seed Δ -0.1000 / -0.1444 / -0.1401, paired t p=0.012
- steer l5_arb_spread_ratio: circuit − taylor: Δ=-0.0884 [-0.0972, -0.0802] (CI excludes 0; n=3; means 0.8600 vs 0.9484); seed-cluster CI [-0.0972, -0.0802] (3 seeds, excludes 0); per-seed Δ -0.0802 / -0.0878 / -0.0972, paired t p=0.003
- steer l5_arb_spread_ratio: circuit − unrestricted: Δ=-0.1578 [-0.1666, -0.1497] (CI excludes 0; n=3; means 0.8600 vs 1.0179); seed-cluster CI [-0.1666, -0.1497] (3 seeds, excludes 0); per-seed Δ -0.1497 / -0.1572 / -0.1666, paired t p=0.001
- steer l5_arb_spread_ratio: circuit − safetune_rd: Δ=-0.1722 [-0.1954, -0.1301] (CI excludes 0; n=3; means 0.8600 vs 1.0323); seed-cluster CI [-0.1954, -0.1301] (3 seeds, excludes 0); per-seed Δ -0.1913 / -0.1301 / -0.1954, paired t p=0.015
- lora l5_arb_spread_ratio: circuit − random: Δ=+0.0084 [+0.0042, +0.0141] (CI excludes 0; n=5; means 1.0145 vs 1.0061); seed-cluster CI [+0.0047, +0.0119] (3 seeds, excludes 0); per-seed Δ +0.0119 / +0.0047 / +0.0089, paired t p=0.055
- lora l5_arb_spread_ratio: circuit − inverse: Δ=-0.0018 [-0.0120, +0.0043] (CI includes 0; n=3; means 1.0135 vs 1.0153); seed-cluster CI [-0.0120, +0.0043] (3 seeds, includes 0); per-seed Δ +0.0024 / +0.0043 / -0.0120, paired t p=0.761
- lora l5_arb_spread_ratio: circuit − taylor: Δ=-0.0021 [-0.0058, +0.0007] (CI includes 0; n=3; means 1.0135 vs 1.0156); seed-cluster CI [-0.0058, +0.0007] (3 seeds, includes 0); per-seed Δ -0.0013 / +0.0007 / -0.0058, paired t p=0.383
- lora l5_arb_spread_ratio: circuit − unrestricted: Δ=-0.0062 [-0.0115, -0.0013] (CI excludes 0; n=3; means 1.0135 vs 1.0197); seed-cluster CI [-0.0115, -0.0013] (3 seeds, excludes 0); per-seed Δ -0.0058 / -0.0013 / -0.0115, paired t p=0.171
- lora l5_arb_spread_ratio: circuit − perturb_circuit: Δ=+0.0132 [+0.0096, +0.0170] (CI excludes 0; n=3; means 1.0135 vs 1.0003); seed-cluster CI [+0.0096, +0.0170] (3 seeds, excludes 0); per-seed Δ +0.0130 / +0.0170 / +0.0096, paired t p=0.025
- edit l5_arb_spread_ratio: circuit − random: Δ=+0.0263 [-0.0387, +0.0915] (CI includes 0; n=5; means 1.0225 vs 0.9962); seed-cluster CI [-0.0536, +0.1212] (3 seeds, includes 0); per-seed Δ -0.0287 / +0.1212 / -0.0536, paired t p=0.835
- edit l5_arb_spread_ratio: circuit − inverse: Δ=+0.0150 [-0.0463, +0.0997] (CI includes 0; n=3; means 1.0118 vs 0.9968); seed-cluster CI [-0.0463, +0.0997] (3 seeds, includes 0); per-seed Δ -0.0085 / +0.0997 / -0.0463, paired t p=0.765
- edit l5_arb_spread_ratio: circuit − taylor: Δ=+0.0059 [-0.0475, +0.0999] (CI includes 0; n=3; means 1.0118 vs 1.0059); seed-cluster CI [-0.0475, +0.0999] (3 seeds, includes 0); per-seed Δ -0.0348 / +0.0999 / -0.0475, paired t p=0.912
- edit l5_arb_spread_ratio: circuit − unrestricted: Δ=+0.0249 [-0.0289, +0.1165] (CI includes 0; n=3; means 1.0118 vs 0.9869); seed-cluster CI [-0.0289, +0.1165] (3 seeds, includes 0); per-seed Δ -0.0130 / +0.1165 / -0.0289, paired t p=0.643
- edit l5_arb_spread_ratio: circuit − perturb_circuit: Δ=-0.1925 [-0.2284, -0.1630] (CI excludes 0; n=3; means 1.0118 vs 1.2043); seed-cluster CI [-0.2284, -0.1630] (3 seeds, excludes 0); per-seed Δ -0.1863 / -0.2284 / -0.1630, paired t p=0.010
- edit l5_arb_spread_ratio: circuit − circuit_deep: Δ=+0.0344 [-0.0179, +0.1257] (CI includes 0; n=3; means 1.0118 vs 0.9774); seed-cluster CI [-0.0179, +0.1257] (3 seeds, includes 0); per-seed Δ -0.0045 / +0.1257 / -0.0179, paired t p=0.531

## L4 edge-level target-circuit distance (Σ|Δs|/Σ|s| over every edge of the target task graph; Chen et al.'s Circuit Distance analogue)

| intervention | condition | n | efficacy | l4_target_edge_l1 | l4_target_circuit_distance |
|---|---|---|---|---|---|
| steer | circuit | 3 | 0.000 | 0.576 | 0.276 |
| steer | inverse | 3 | 0.028 | 0.269 | 0.156 |
| steer | random | 5 | -0.100 | 0.424 | 0.245 |
| steer | safetune_rd | 3 | 0.083 | 0.732 | 0.448 |
| steer | taylor | 3 | 0.000 | 0.478 | 0.211 |
| steer | unrestricted | 3 | 0.083 | 0.674 | 0.388 |
| lora | circuit | 3 | 0.083 | 0.609 | 0.236 |
| lora | inverse | 3 | 0.083 | 0.355 | 0.140 |
| lora | random | 5 | 0.083 | 0.590 | 0.248 |
| lora | taylor | 3 | 0.083 | 0.610 | 0.245 |
| lora | unrestricted | 3 | 0.083 | 1.223 | 0.447 |
| lora | perturb_circuit | 3 | 0.000 | 0.030 | 0.031 |
| lora | perturb_inverse | 3 | 0.000 | 0.024 | 0.021 |
| lora | perturb_random | 5 | 0.000 | 0.025 | 0.032 |
| lora | perturb_taylor | 3 | 0.000 | 0.019 | 0.000 |
| lora | perturb_unrestricted | 3 | 0.000 | 0.034 | 0.062 |
| edit | circuit | 3 | -0.111 | 0.735 | 0.375 |
| edit | circuit_deep | 3 | 0.000 | 0.254 | 0.121 |
| edit | inverse | 3 | 0.083 | 0.367 | 0.203 |
| edit | random | 5 | 0.017 | 0.271 | 0.158 |
| edit | taylor | 3 | 0.000 | 0.185 | 0.092 |
| edit | unrestricted | 3 | 0.000 | 0.372 | 0.185 |
| edit | perturb_circuit | 3 | -0.250 | 1.008 | 0.456 |
| edit | perturb_circuit_deep | 3 | -0.028 | 0.067 | 0.021 |
| edit | perturb_inverse | 3 | 0.000 | 0.105 | 0.082 |
| edit | perturb_random | 5 | -0.017 | 0.085 | 0.053 |
| edit | perturb_taylor | 3 | 0.000 | 0.060 | 0.021 |
| edit | perturb_unrestricted | 3 | -0.083 | 0.141 | 0.102 |
- steer l4_target_edge_l1: circuit − random: Δ=+0.1491 [+0.0191, +0.2597] (CI excludes 0; n=5; means 0.5732 vs 0.4241); seed-cluster CI [+0.0609, +0.2699] (3 seeds, excludes 0); per-seed Δ +0.0609 / +0.2699 / +0.0839, paired t p=0.172
- steer l4_target_edge_l1: circuit − inverse: Δ=+0.3069 [+0.2087, +0.4468] (CI excludes 0; n=3; means 0.5763 vs 0.2694); seed-cluster CI [+0.2087, +0.4468] (3 seeds, excludes 0); per-seed Δ +0.2087 / +0.2651 / +0.4468, paired t p=0.051
- steer l4_target_edge_l1: circuit − taylor: Δ=+0.0988 [+0.0774, +0.1117] (CI excludes 0; n=3; means 0.5763 vs 0.4775); seed-cluster CI [+0.0774, +0.1117] (3 seeds, excludes 0); per-seed Δ +0.0774 / +0.1072 / +0.1117, paired t p=0.012
- steer l4_target_edge_l1: circuit − unrestricted: Δ=-0.0979 [-0.1037, -0.0927] (CI excludes 0; n=3; means 0.5763 vs 0.6742); seed-cluster CI [-0.1037, -0.0927] (3 seeds, excludes 0); per-seed Δ -0.1037 / -0.0972 / -0.0927, paired t p=0.001
- steer l4_target_edge_l1: circuit − safetune_rd: Δ=-0.1558 [-0.1660, -0.1496] (CI excludes 0; n=3; means 0.5763 vs 0.7321); seed-cluster CI [-0.1660, -0.1496] (3 seeds, excludes 0); per-seed Δ -0.1660 / -0.1519 / -0.1496, paired t p=0.001
- lora l4_target_edge_l1: circuit − random: Δ=+0.0244 [-0.0233, +0.0994] (CI includes 0; n=5; means 0.6147 vs 0.5903); seed-cluster CI [-0.0345, +0.0843] (3 seeds, includes 0); per-seed Δ -0.0060 / +0.0843 / -0.0345, paired t p=0.723
- lora l4_target_edge_l1: circuit − inverse: Δ=+0.2533 [+0.2342, +0.2628] (CI excludes 0; n=3; means 0.6085 vs 0.3553); seed-cluster CI [+0.2342, +0.2628] (3 seeds, excludes 0); per-seed Δ +0.2342 / +0.2628 / +0.2628, paired t p=0.001
- lora l4_target_edge_l1: circuit − taylor: Δ=-0.0013 [-0.0317, +0.0260] (CI includes 0; n=3; means 0.6085 vs 0.6098); seed-cluster CI [-0.0317, +0.0260] (3 seeds, includes 0); per-seed Δ +0.0018 / -0.0317 / +0.0260, paired t p=0.945
- lora l4_target_edge_l1: circuit − unrestricted: Δ=-0.6145 [-0.6693, -0.5710] (CI excludes 0; n=3; means 0.6085 vs 1.2230); seed-cluster CI [-0.6693, -0.5710] (3 seeds, excludes 0); per-seed Δ -0.6693 / -0.5710 / -0.6032, paired t p=0.002
- lora l4_target_edge_l1: circuit − perturb_circuit: Δ=+0.5782 [+0.5470, +0.5939] (CI excludes 0; n=3; means 0.6085 vs 0.0303); seed-cluster CI [+0.5470, +0.5939] (3 seeds, excludes 0); per-seed Δ +0.5939 / +0.5937 / +0.5470, paired t p=0.001
- edit l4_target_edge_l1: circuit − random: Δ=+0.4701 [+0.4112, +0.5291] (CI excludes 0; n=5; means 0.7414 vs 0.2713); seed-cluster CI [+0.4364, +0.4963] (3 seeds, excludes 0); per-seed Δ +0.4364 / +0.4907 / +0.4963, paired t p=0.002
- edit l4_target_edge_l1: circuit − inverse: Δ=+0.3681 [+0.2613, +0.4735] (CI excludes 0; n=3; means 0.7352 vs 0.3670); seed-cluster CI [+0.2613, +0.4735] (3 seeds, excludes 0); per-seed Δ +0.2613 / +0.4735 / +0.3695, paired t p=0.027
- edit l4_target_edge_l1: circuit − taylor: Δ=+0.5500 [+0.4686, +0.6742] (CI excludes 0; n=3; means 0.7352 vs 0.1851); seed-cluster CI [+0.4686, +0.6742] (3 seeds, excludes 0); per-seed Δ +0.4686 / +0.6742 / +0.5074, paired t p=0.013
- edit l4_target_edge_l1: circuit − unrestricted: Δ=+0.3633 [+0.2573, +0.4878] (CI excludes 0; n=3; means 0.7352 vs 0.3718); seed-cluster CI [+0.2573, +0.4878] (3 seeds, excludes 0); per-seed Δ +0.2573 / +0.4878 / +0.3448, paired t p=0.033
- edit l4_target_edge_l1: circuit − perturb_circuit: Δ=-0.2730 [-0.4956, -0.1054] (CI excludes 0; n=3; means 0.7352 vs 1.0082); seed-cluster CI [-0.4956, -0.1054] (3 seeds, excludes 0); per-seed Δ -0.4956 / -0.1054 / -0.2181, paired t p=0.143
- edit l4_target_edge_l1: circuit − circuit_deep: Δ=+0.4814 [+0.3721, +0.6034] (CI excludes 0; n=3; means 0.7352 vs 0.2538); seed-cluster CI [+0.3721, +0.6034] (3 seeds, excludes 0); per-seed Δ +0.3721 / +0.6034 / +0.4686, paired t p=0.019

## L5 relearning probe (B2; audit/relearn.py): held-out harmful refusal gain over the base model after the same 40-step benign fine-tune, next to the gain before it

| intervention | condition | n | efficacy_xl | l5_relearn_gain_xl | l5_relearn_retention_xl | l5_relearn_gain_en | l5_relearn_benign_en_delta |
|---|---|---|---|---|---|---|---|
| steer | circuit | 3 | 0.215 | 0.208 | 0.970 | 0.000 | 0.071 |
| steer | inverse | 3 | 0.049 | 0.000 | 0.000 | 0.000 | 0.012 |
| steer | random | 5 | 0.104 | 0.050 | 0.420 | -0.017 | 0.043 |
| steer | safetune_rd | 3 | 0.340 | -0.139 | -0.408 | -0.139 | 0.024 |
| steer | taylor | 3 | 0.125 | 0.062 | 0.500 | 0.000 | 0.071 |
| steer | unrestricted | 3 | 0.312 | 0.146 | 0.467 | 0.000 | 0.107 |
| lora | circuit | 3 | 0.292 | 0.201 | 0.690 | 0.000 | 0.095 |
| lora | inverse | 3 | 0.201 | 0.174 | 0.890 | 0.056 | 0.095 |
| lora | random | 5 | 0.287 | 0.221 | 0.767 | 0.067 | 0.093 |
| lora | taylor | 3 | 0.278 | 0.201 | 0.725 | 0.083 | 0.107 |
| lora | unrestricted | 3 | 0.375 | 0.292 | 0.778 | 0.083 | 0.202 |
| edit | circuit | 3 | -0.125 | -0.174 | – | -0.111 | 0.071 |
| edit | circuit_deep | 3 | 0.062 | 0.021 | 0.200 | 0.000 | 0.000 |
| edit | inverse | 3 | 0.146 | 0.097 | 0.671 | 0.083 | 0.000 |
| edit | random | 5 | 0.029 | -0.008 | – | 0.000 | 0.000 |
| edit | taylor | 3 | -0.007 | -0.049 | – | 0.000 | 0.000 |
| edit | unrestricted | 3 | -0.104 | -0.132 | – | 0.000 | -0.036 |
- steer l5_relearn_gain_xl: circuit − random: Δ=+0.1583 [+0.0625, +0.2333] (CI excludes 0; n=5; means 0.2083 vs 0.0500); seed-cluster CI [+0.0521, +0.2500] (3 seeds, excludes 0); per-seed Δ +0.0521 / +0.2187 / +0.2500, paired t p=0.106
- steer l5_relearn_gain_xl: circuit − inverse: Δ=+0.2083 [+0.2083, +0.2083] (CI excludes 0; n=3; means 0.2083 vs 0.0000); seed-cluster CI [+0.2083, +0.2083] (3 seeds, excludes 0); per-seed Δ +0.2083 / +0.2083 / +0.2083, paired t p=0.000
- steer l5_relearn_gain_xl: circuit − taylor: Δ=+0.1458 [+0.1458, +0.1458] (CI excludes 0; n=3; means 0.2083 vs 0.0625); seed-cluster CI [+0.1458, +0.1458] (3 seeds, excludes 0); per-seed Δ +0.1458 / +0.1458 / +0.1458, paired t p=0.000
- steer l5_relearn_gain_xl: circuit − unrestricted: Δ=+0.0625 [+0.0625, +0.0625] (CI excludes 0; n=3; means 0.2083 vs 0.1458); seed-cluster CI [+0.0625, +0.0625] (3 seeds, excludes 0); per-seed Δ +0.0625 / +0.0625 / +0.0625, paired t p=0.000
- steer l5_relearn_gain_xl: circuit − safetune_rd: Δ=+0.3472 [+0.2708, +0.4167] (CI excludes 0; n=3; means 0.2083 vs -0.1389); seed-cluster CI [+0.2708, +0.4167] (3 seeds, excludes 0); per-seed Δ +0.4167 / +0.2708 / +0.3542, paired t p=0.014
- lora l5_relearn_gain_xl: circuit − random: Δ=-0.0167 [-0.0583, +0.0208] (CI includes 0; n=5; means 0.2042 vs 0.2208); seed-cluster CI [-0.0833, +0.0312] (3 seeds, includes 0); per-seed Δ -0.0312 / +0.0312 / -0.0833, paired t p=0.490
- lora l5_relearn_gain_xl: circuit − inverse: Δ=+0.0278 [-0.0208, +0.0625] (CI includes 0; n=3; means 0.2014 vs 0.1736); seed-cluster CI [-0.0208, +0.0625] (3 seeds, includes 0); per-seed Δ -0.0208 / +0.0417 / +0.0625, paired t p=0.383
- lora l5_relearn_gain_xl: circuit − taylor: Δ=-0.0000 [-0.0208, +0.0208] (CI includes 0; n=3; means 0.2014 vs 0.2014); seed-cluster CI [-0.0208, +0.0208] (3 seeds, includes 0); per-seed Δ -0.0208 / +0.0208 / +0.0000, paired t p=1.000
- lora l5_relearn_gain_xl: circuit − unrestricted: Δ=-0.0903 [-0.1042, -0.0625] (CI excludes 0; n=3; means 0.2014 vs 0.2917); seed-cluster CI [-0.1042, -0.0625] (3 seeds, excludes 0); per-seed Δ -0.1042 / -0.0625 / -0.1042, paired t p=0.023
- edit l5_relearn_gain_xl: circuit − random: Δ=-0.1708 [-0.1958, -0.1417] (CI excludes 0; n=5; means -0.1792 vs -0.0083); seed-cluster CI [-0.1979, -0.1250] (3 seeds, excludes 0); per-seed Δ -0.1667 / -0.1979 / -0.1250, paired t p=0.016
- edit l5_relearn_gain_xl: circuit − inverse: Δ=-0.2708 [-0.2917, -0.2500] (CI excludes 0; n=3; means -0.1736 vs 0.0972); seed-cluster CI [-0.2917, -0.2500] (3 seeds, excludes 0); per-seed Δ -0.2917 / -0.2708 / -0.2500, paired t p=0.002
- edit l5_relearn_gain_xl: circuit − taylor: Δ=-0.1250 [-0.1458, -0.1042] (CI excludes 0; n=3; means -0.1736 vs -0.0486); seed-cluster CI [-0.1458, -0.1042] (3 seeds, excludes 0); per-seed Δ -0.1250 / -0.1458 / -0.1042, paired t p=0.009
- edit l5_relearn_gain_xl: circuit − unrestricted: Δ=-0.0417 [-0.0417, -0.0417] (CI excludes 0; n=3; means -0.1736 vs -0.1319); seed-cluster CI [-0.0417, -0.0417] (3 seeds, excludes 0); per-seed Δ -0.0417 / -0.0417 / -0.0417, paired t p=0.000
- edit l5_relearn_gain_xl: circuit − circuit_deep: Δ=-0.1944 [-0.2292, -0.1667] (CI excludes 0; n=3; means -0.1736 vs 0.0208); seed-cluster CI [-0.2292, -0.1667] (3 seeds, excludes 0); per-seed Δ -0.1875 / -0.2292 / -0.1667, paired t p=0.009

## Judge robustness: AuditKIT vs SafeTune refusal detectors

Per-generation agreement over the base model and the 94 main cells, English refusal families (first 200 characters of each stored generation): 11340/11400 = 0.995. Base rates (AuditKIT / SafeTune): refusal_harmful_en 0.92 / 0.92, refusal_benign_en 0.07 / 0.07, refusal_other_en 0.88 / 0.88, refusal_xstest_safe 0.07 / 0.07. Every number elsewhere in this file is AuditKIT-judged.

| intervention | condition | n | harmful Δ (AuditKIT / SafeTune) | benign Δ (AuditKIT / SafeTune) | other-category Δ (AuditKIT / SafeTune) | XSTest-safe Δ (AuditKIT / SafeTune) |
|---|---|---|---|---|---|---|
| steer | circuit | 3 | +0.000 / +0.000 | +0.190 / +0.190 | +0.025 / +0.025 | +0.250 / +0.250 |
| steer | inverse | 3 | +0.028 / -0.028 | +0.012 / +0.012 | -0.075 / -0.092 | +0.000 / +0.000 |
| steer | random | 5 | -0.100 / -0.117 | +0.157 / +0.157 | -0.030 / -0.045 | +0.120 / +0.110 |
| steer | safetune_rd | 3 | +0.083 / +0.028 | +0.810 / +0.798 | +0.125 / +0.125 | +0.792 / +0.775 |
| steer | taylor | 3 | +0.000 / +0.000 | +0.071 / +0.071 | +0.025 / +0.025 | +0.150 / +0.125 |
| steer | unrestricted | 3 | +0.083 / +0.083 | +0.500 / +0.500 | +0.100 / +0.100 | +0.550 / +0.550 |
| lora | circuit | 3 | +0.083 / +0.083 | +0.381 / +0.381 | +0.100 / +0.100 | +0.333 / +0.333 |
| lora | inverse | 3 | +0.083 / +0.083 | +0.369 / +0.345 | +0.092 / +0.092 | +0.308 / +0.300 |
| lora | perturb_circuit | 3 | +0.000 / +0.000 | +0.000 / +0.000 | -0.008 / +0.000 | -0.017 / -0.017 |
| lora | perturb_inverse | 3 | +0.000 / +0.000 | +0.000 / +0.000 | -0.017 / +0.000 | -0.008 / -0.008 |
| lora | perturb_random | 5 | +0.000 / +0.000 | +0.000 / +0.000 | -0.010 / -0.010 | -0.020 / -0.020 |
| lora | perturb_taylor | 3 | +0.000 / +0.000 | +0.000 / +0.000 | +0.000 / +0.000 | +0.000 / +0.000 |
| lora | perturb_unrestricted | 3 | +0.000 / +0.000 | +0.000 / +0.000 | -0.008 / +0.000 | -0.008 / -0.008 |
| lora | random | 5 | +0.083 / +0.083 | +0.400 / +0.400 | +0.105 / +0.105 | +0.295 / +0.295 |
| lora | taylor | 3 | +0.083 / +0.083 | +0.548 / +0.548 | +0.100 / +0.100 | +0.367 / +0.367 |
| lora | unrestricted | 3 | +0.083 / +0.083 | +0.417 / +0.417 | +0.075 / +0.075 | +0.317 / +0.317 |
| edit | circuit | 3 | -0.111 / -0.139 | -0.024 / -0.024 | -0.100 / -0.100 | -0.042 / -0.042 |
| edit | circuit_deep | 3 | +0.000 / +0.000 | +0.000 / +0.000 | -0.033 / -0.058 | -0.025 / -0.025 |
| edit | inverse | 3 | +0.083 / +0.083 | -0.024 / -0.024 | -0.017 / +0.000 | -0.025 / -0.025 |
| edit | perturb_circuit | 3 | -0.250 / -0.278 | +0.000 / +0.000 | -0.175 / -0.183 | -0.050 / -0.050 |
| edit | perturb_circuit_deep | 3 | -0.028 / -0.028 | +0.000 / +0.000 | -0.033 / -0.033 | -0.008 / -0.008 |
| edit | perturb_inverse | 3 | +0.000 / +0.000 | +0.000 / +0.000 | -0.033 / -0.033 | -0.008 / -0.008 |
| edit | perturb_random | 5 | -0.017 / -0.017 | +0.000 / +0.000 | -0.025 / -0.010 | -0.005 / -0.005 |
| edit | perturb_taylor | 3 | +0.000 / +0.000 | +0.000 / +0.000 | -0.025 / -0.008 | -0.025 / -0.025 |
| edit | perturb_unrestricted | 3 | -0.083 / -0.083 | +0.000 / +0.000 | -0.033 / -0.025 | +0.000 / +0.000 |
| edit | random | 5 | +0.017 / +0.017 | +0.007 / +0.007 | -0.025 / -0.020 | -0.010 / -0.010 |
| edit | taylor | 3 | +0.000 / +0.000 | +0.000 / +0.000 | +0.025 / +0.025 | +0.000 / +0.000 |
| edit | unrestricted | 3 | +0.000 / +0.000 | +0.000 / +0.000 | +0.000 / +0.000 | -0.050 / -0.050 |

## Steering: collateral at matched efficacy (sweep cells)

Target efficacy e* = -0.033 (the smallest condition-mean peak efficacy; collateral linearly interpolated along each seed's sweep, cells whose sweep never reaches e* are skipped).

| condition | n curves | reach e* | WikiText ppl ratio @e* | L1 neigh Δ (en) @e* | L2 benign refusal Δ (en) @e* | L1 MMLU Δ (en) @e* |
|---|---|---|---|---|---|---|
| circuit | 3 | 3 | 2.169 | -0.052 | 0.068 | -0.044 |
| inverse | 3 | 3 | 2.475 | -0.132 | 0.102 | -0.040 |
| random | 5 | 4 | 4.828 | -0.156 | 0.207 | -0.047 |
| taylor | 3 | 3 | 1.643 | -0.039 | 0.053 | -0.023 |
| unrestricted | 3 | 3 | 28.821 | -0.068 | 0.135 | -0.007 |

Per-cell sweep (efficacy → WikiText ratio): circuit_s0: 0.25×→0.00/1.04, 0.5×→0.00/1.16, 1.0×→0.08/2.04, 2.0×→-0.92/29.89; circuit_s1: 0.25×→0.00/1.04, 0.5×→0.00/1.16, 1.0×→0.08/1.99, 2.0×→-0.92/32.77; circuit_s2: 0.25×→0.00/1.05, 0.5×→0.00/1.18, 1.0×→0.08/2.10, 2.0×→-0.92/33.24; inverse_s0: 0.25×→0.00/1.02, 0.5×→0.08/1.09, 1.0×→0.08/1.53, 2.0×→-0.08/11.98; inverse_s1: 0.25×→0.00/1.02, 0.5×→0.00/1.06, 1.0×→0.08/1.30, 2.0×→0.00/4.11; inverse_s2: 0.25×→0.00/1.00, 0.5×→0.00/1.03, 1.0×→0.00/1.16, 2.0×→0.08/2.27; random_s0: 0.25×→-0.17/1.05, 0.5×→-0.17/1.17, 1.0×→0.00/1.75, 2.0×→-0.25/15.38; random_s1: 0.25×→-0.25/1.02, 0.5×→-0.25/1.08, 1.0×→-0.58/1.40, 2.0×→-0.83/5.32; random_s2: 0.25×→-0.08/1.02, 0.5×→-0.17/1.13, 1.0×→0.00/1.97, 2.0×→-0.17/1427.75; random_s3: 0.25×→0.00/1.05, 0.5×→0.08/1.18, 1.0×→0.08/2.30, 2.0×→-0.67/255.88; random_s4: 0.25×→0.00/1.01, 0.5×→0.00/1.06, 1.0×→0.00/1.30, 2.0×→-0.08/4.22; taylor_s0: 0.25×→0.00/1.03, 0.5×→0.00/1.12, 1.0×→0.08/1.65, 2.0×→-0.75/14.77; taylor_s1: 0.25×→0.00/1.03, 0.5×→0.00/1.12, 1.0×→0.08/1.65, 2.0×→-0.75/14.77; taylor_s2: 0.25×→0.00/1.03, 0.5×→0.00/1.12, 1.0×→0.08/1.65, 2.0×→-0.75/14.77; unrestricted_s0: 0.25×→0.00/1.05, 0.5×→0.08/1.18, 1.0×→0.08/2.23, 2.0×→-0.92/764.78; unrestricted_s1: 0.25×→0.00/1.05, 0.5×→0.08/1.18, 1.0×→0.08/2.23, 2.0×→-0.92/764.78; unrestricted_s2: 0.25×→0.00/1.05, 0.5×→0.08/1.18, 1.0×→0.08/2.23, 2.0×→-0.92/764.78

## Multiple comparisons (Holm step-down, α = 0.05, over the 12 H2 tests above)

11 of 12 have bootstrap p < 0.05 uncorrected; 12 survive Holm. The bootstrap p is floored at 1/2000, so a test whose 2000 resampled means share one sign reports p = 0.0005 and passes any Holm threshold above it (0.05/m = 0.0042).
- steer l4_bystander_cad_mean circuit − inverse: Δ=+0.1017, p=0.0005 (survives)
- steer l3_mean_abs_delta circuit − inverse: Δ=+0.1501, p=0.0005 (survives)
- lora l4_bystander_cad_mean circuit − inverse: Δ=+0.0156, p=0.0005 (survives)
- lora l3_mean_abs_delta circuit − inverse: Δ=+0.0371, p=0.0005 (survives)
- edit l4_bystander_cad_mean circuit − random: Δ=+0.2763, p=0.0005 (survives)
- edit l4_bystander_cad_mean circuit − inverse: Δ=+0.2673, p=0.0005 (survives)
- edit l3_mean_abs_delta circuit − random: Δ=+0.0573, p=0.0005 (survives)
- steer l3_mean_abs_delta circuit − random: Δ=+0.0769, p=0.0010 (survives)
- steer l4_bystander_cad_mean circuit − random: Δ=+0.0366, p=0.0030 (survives)
- lora l4_bystander_cad_mean circuit − random: Δ=+0.0084, p=0.0030 (survives)
- edit l3_mean_abs_delta circuit − inverse: Δ=+0.0385, p=0.0140 (survives)
- lora l3_mean_abs_delta circuit − random: Δ=-0.0075, p=0.0500 (survives)
