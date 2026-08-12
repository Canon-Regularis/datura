# Every comparison, corrected for how many there are

46 comparisons across base_10k, base_5k, wide_10k, context_10k, context_shuffled_10k. Each configuration reports these with an uncorrected p value, which is the right number to read for one comparison on its own and the wrong one to read across a table of them.

`q_value` is the Benjamini-Hochberg adjusted figure, controlling the share of resolved comparisons that are noise. `rejected` marks the ones that survive it at 0.05. For reference, dividing the threshold across every comparison instead puts the bar at 1.09e-03, which is the stricter question of whether any one of them is a false positive.

| config               | family                       | model                              | floor                                |   margin |   p_value |   q_value | rejected   |   agreeing |   folds |
|:---------------------|:-----------------------------|:-----------------------------------|:-------------------------------------|---------:|----------:|----------:|:-----------|-----------:|--------:|
| wide_10k             | species                      | probe                              | logbook                              |  -0.5359 |  5.02e-20 |  2.31e-18 | True       |         50 |      50 |
| wide_10k             | species                      | xgboost+probe                      | logbook                              |  -0.5153 |  1.91e-18 |  4.39e-17 | True       |         50 |      50 |
| wide_10k             | species                      | xgboost                            | logbook                              |  -0.5479 |  5.55e-15 |  8.51e-14 | True       |         50 |      50 |
| wide_10k             | species                      | logbook                            | metadata                             |   0.3985 |  1.54e-08 |  1.77e-07 | True       |         50 |      50 |
| base_10k             | species                      | xgboost                            | logbook                              |  -0.2448 |  2.37e-08 |  2.18e-07 | True       |         50 |      50 |
| base_10k             | species                      | probe                              | logbook                              |  -0.2581 |  1.11e-06 |  8.48e-06 | True       |         50 |      50 |
| base_5k              | species                      | probe                              | logbook                              |  -0.2759 |  2.94e-06 |  1.93e-05 | True       |         50 |      50 |
| base_10k             | species                      | xgboost+probe                      | logbook                              |  -0.2324 |  7.44e-06 |  4.28e-05 | True       |         50 |      50 |
| base_5k              | species                      | xgboost+probe                      | logbook                              |  -0.2386 |  0.00012  |  0.00055  | True       |         50 |      50 |
| context_10k          | species                      | xgboost                            | logbook                              |  -0.6602 |  0.000108 |  0.00055  | True       |          5 |       5 |
| base_5k              | species                      | xgboost                            | logbook                              |  -0.2497 |  0.000246 |  0.00103  | True       |         50 |      50 |
| base_10k             | species                      | cnn_small                          | logbook                              |  -0.2876 |  0.000319 |  0.00122  | True       |         50 |      50 |
| wide_10k             | species                      | xgboost                            | metadata                             |  -0.1494 |  0.00134  |  0.00469  | True       |         48 |      50 |
| context_10k          | species                      | xgboost+probe                      | logbook                              |  -0.5412 |  0.00143  |  0.00469  | True       |          5 |       5 |
| context_10k          | species                      | probe                              | logbook                              |  -0.5371 |  0.00153  |  0.0047   | True       |          5 |       5 |
| base_5k              | species                      | probe                              | metadata                             |  -0.1764 |  0.00177  |  0.00508  | True       |         47 |      50 |
| base_5k              | species                      | xgboost                            | metadata                             |  -0.1502 |  0.0119   |  0.0323   | True       |         46 |      50 |
| base_5k              | species                      | xgboost+probe                      | metadata                             |  -0.1391 |  0.0189   |  0.0482   | True       |         45 |      50 |
| base_10k             | species                      | cnn                                | logbook                              |  -0.2863 |  0.0254   |  0.0616   | False      |          5 |       5 |
| wide_10k             | species                      | probe                              | metadata                             |  -0.1375 |  0.047    |  0.108    | False      |         43 |      50 |
| wide_10k             | species                      | xgboost+probe                      | metadata                             |  -0.1168 |  0.0541   |  0.119    | False      |         43 |      50 |
| context_shuffled_10k | species                      | xgboost                            | metadata                             |  -0.3426 |  0.0571   |  0.119    | False      |          5 |       5 |
| base_10k             | species                      | logbook                            | metadata                             |   0.1294 |  0.0667   |  0.133    | False      |         44 |      50 |
| context_10k          | species                      | logbook                            | metadata                             |   0.3603 |  0.0771   |  0.148    | False      |          5 |       5 |
| context_10k          | species                      | xgboost                            | metadata                             |  -0.2999 |  0.0824   |  0.152    | False      |          5 |       5 |
| context_shuffled_10k | species                      | xgboost                            | logbook                              |  -0.3716 |  0.107    |  0.189    | False      |          4 |       5 |
| base_10k             | calltype_spermwhale_coda     | calltype_spermwhale_coda_cnn_small | calltype_spermwhale_coda_context     |  -0.1776 |  0.127    |  0.209    | False      |          5 |       5 |
| base_5k              | species                      | cnn_small                          | logbook                              |  -0.2369 |  0.124    |  0.209    | False      |          5 |       5 |
| base_10k             | calltype_killerwhale_chirp   | calltype_killerwhale_chirp         | calltype_killerwhale_chirp_context   |  -0.0632 |  0.135    |  0.213    | False      |         34 |      50 |
| base_10k             | calltype_spermwhale_coda     | calltype_spermwhale_coda           | calltype_spermwhale_coda_context     |  -0.094  |  0.144    |  0.213    | False      |         39 |      50 |
| base_10k             | species                      | probe                              | metadata                             |  -0.1287 |  0.148    |  0.213    | False      |         39 |      50 |
| base_5k              | species                      | logbook                            | metadata                             |   0.0995 |  0.144    |  0.213    | False      |         39 |      50 |
| base_5k              | species                      | cnn_small                          | metadata                             |  -0.1705 |  0.167    |  0.232    | False      |          5 |       5 |
| base_10k             | calltype_killerwhale_click   | calltype_killerwhale_click         | calltype_killerwhale_click_context   |   0.1155 |  0.198    |  0.258    | False      |         37 |      50 |
| base_10k             | species                      | cnn_small                          | metadata                             |  -0.1582 |  0.198    |  0.258    | False      |         40 |      50 |
| base_10k             | calltype_killerwhale_whistle | calltype_killerwhale_whistle       | calltype_killerwhale_whistle_context |   0.0435 |  0.202    |  0.258    | False      |         36 |      50 |
| base_10k             | species                      | xgboost                            | metadata                             |  -0.1154 |  0.239    |  0.297    | False      |         38 |      50 |
| base_10k             | species                      | xgboost+probe                      | metadata                             |  -0.1031 |  0.286    |  0.347    | False      |         38 |      50 |
| context_10k          | species                      | xgboost+probe                      | metadata                             |  -0.181  |  0.323    |  0.381    | False      |          4 |       5 |
| context_10k          | species                      | probe                              | metadata                             |  -0.1769 |  0.357    |  0.41     | False      |          4 |       5 |
| base_10k             | species                      | cnn                                | metadata                             |  -0.1702 |  0.387    |  0.434    | False      |          4 |       5 |
| base_10k             | calltype_killerwhale_call    | calltype_killerwhale_call          | calltype_killerwhale_call_context    |  -0.0773 |  0.406    |  0.445    | False      |         30 |      50 |
| base_10k             | calltype_spermwhale_click    | calltype_spermwhale_click          | calltype_spermwhale_click_context    |  -0.0317 |  0.582    |  0.622    | False      |         29 |      50 |
| base_10k             | calltype_killerwhale_squeal  | calltype_killerwhale_squeal        | calltype_killerwhale_squeal_context  |   0.0361 |  0.733    |  0.749    | False      |         32 |      50 |
| context_shuffled_10k | species                      | logbook                            | metadata                             |   0.029  |  0.725    |  0.749    | False      |          3 |       5 |
| base_10k             | calltype_spermwhale_whistle  | calltype_spermwhale_whistle        | calltype_spermwhale_whistle_context  |  -0.0335 |  0.841    |  0.841    | False      |         31 |      50 |

## What survives

18 of 46 comparisons survive the correction, and the smallest of them clears the divided threshold of 1.09e-03 as well.

- `probe` is 0.536 below `logbook` in wide_10k, at q = 2.31e-18 over 50 folds.
- `xgboost+probe` is 0.515 below `logbook` in wide_10k, at q = 4.39e-17 over 50 folds.
- `xgboost` is 0.548 below `logbook` in wide_10k, at q = 8.51e-14 over 50 folds.
- `logbook` is 0.398 above `metadata` in wide_10k, at q = 1.77e-07 over 50 folds.
- `xgboost` is 0.245 below `logbook` in base_10k, at q = 2.18e-07 over 50 folds.
- `probe` is 0.258 below `logbook` in base_10k, at q = 8.48e-06 over 50 folds.
- `probe` is 0.276 below `logbook` in base_5k, at q = 1.93e-05 over 50 folds.
- `xgboost+probe` is 0.232 below `logbook` in base_10k, at q = 4.28e-05 over 50 folds.
- `xgboost+probe` is 0.239 below `logbook` in base_5k, at q = 5.50e-04 over 50 folds.
- `xgboost` is 0.660 below `logbook` in context_10k, at q = 5.50e-04 over 5 folds.
- `xgboost` is 0.250 below `logbook` in base_5k, at q = 1.03e-03 over 50 folds.
- `cnn_small` is 0.288 below `logbook` in base_10k, at q = 1.22e-03 over 50 folds.
- `xgboost` is 0.149 below `metadata` in wide_10k, at q = 4.69e-03 over 50 folds.
- `xgboost+probe` is 0.541 below `logbook` in context_10k, at q = 4.69e-03 over 5 folds.
- `probe` is 0.537 below `logbook` in context_10k, at q = 4.70e-03 over 5 folds.
- `probe` is 0.176 below `metadata` in base_5k, at q = 5.08e-03 over 50 folds.
- `xgboost` is 0.150 below `metadata` in base_5k, at q = 3.23e-02 over 50 folds.
- `xgboost+probe` is 0.139 below `metadata` in base_5k, at q = 4.82e-02 over 50 folds.
