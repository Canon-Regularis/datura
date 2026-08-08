# Every comparison, corrected for how many there are

30 comparisons across base_10k, base_5k, wide_10k. Each configuration reports these with an uncorrected p value, which is the right number to read for one comparison on its own and the wrong one to read across a table of them.

`q_value` is the Benjamini-Hochberg adjusted figure, controlling the share of resolved comparisons that are noise. `rejected` marks the ones that survive it at 0.05. For reference, dividing the threshold across every comparison instead puts the bar at 1.67e-03, which is the stricter question of whether any one of them is a false positive.

| config   | family                       | model                              | floor                                |   margin |   p_value |   q_value | rejected   |   agreeing |   folds |
|:---------|:-----------------------------|:-----------------------------------|:-------------------------------------|---------:|----------:|----------:|:-----------|-----------:|--------:|
| wide_10k | species                      | xgboost                            | logbook                              |  -0.4946 |  3.42e-16 |  1.03e-14 | True       |         50 |      50 |
| wide_10k | species                      | probe                              | logbook                              |  -0.4827 |  3.8e-15  |  5.7e-14  | True       |         50 |      50 |
| wide_10k | species                      | logbook                            | metadata                             |   0.3452 |  4.64e-09 |  4.64e-08 | True       |         50 |      50 |
| base_10k | species                      | xgboost                            | logbook                              |  -0.245  |  1.66e-08 |  1.25e-07 | True       |         50 |      50 |
| base_10k | species                      | probe                              | logbook                              |  -0.2583 |  1.3e-06  |  7.8e-06  | True       |         50 |      50 |
| base_5k  | species                      | probe                              | logbook                              |  -0.2737 |  4e-06    |  2e-05    | True       |         50 |      50 |
| base_5k  | species                      | xgboost                            | logbook                              |  -0.2474 |  0.000247 |  0.00106  | True       |         50 |      50 |
| base_10k | species                      | cnn_small                          | logbook                              |  -0.2878 |  0.000285 |  0.00107  | True       |         50 |      50 |
| wide_10k | species                      | xgboost                            | metadata                             |  -0.1494 |  0.00134  |  0.00447  | True       |         48 |      50 |
| base_5k  | species                      | probe                              | metadata                             |  -0.1764 |  0.00177  |  0.0053   | True       |         47 |      50 |
| base_5k  | species                      | xgboost                            | metadata                             |  -0.1502 |  0.0119   |  0.0326   | True       |         46 |      50 |
| base_10k | calltype_spermwhale_coda     | calltype_spermwhale_coda           | calltype_spermwhale_coda_context     |  -0.1622 |  0.0234   |  0.0585   | False      |         46 |      50 |
| base_10k | species                      | cnn                                | logbook                              |  -0.2848 |  0.0259   |  0.0598   | False      |          5 |       5 |
| wide_10k | species                      | probe                              | metadata                             |  -0.1375 |  0.047    |  0.101    | False      |         43 |      50 |
| base_10k | species                      | logbook                            | metadata                             |   0.1296 |  0.0724   |  0.145    | False      |         45 |      50 |
| base_10k | calltype_killerwhale_click   | calltype_killerwhale_click         | calltype_killerwhale_click_context   |   0.1249 |  0.134    |  0.245    | False      |         37 |      50 |
| base_10k | species                      | probe                              | metadata                             |  -0.1287 |  0.148    |  0.245    | False      |         39 |      50 |
| base_5k  | species                      | cnn_small                          | logbook                              |  -0.2313 |  0.142    |  0.245    | False      |          5 |       5 |
| base_5k  | species                      | logbook                            | metadata                             |   0.0972 |  0.155    |  0.245    | False      |         40 |      50 |
| base_5k  | species                      | cnn_small                          | metadata                             |  -0.1705 |  0.167    |  0.25     | False      |          5 |       5 |
| base_10k | calltype_spermwhale_coda     | calltype_spermwhale_coda_cnn_small | calltype_spermwhale_coda_context     |  -0.2242 |  0.187    |  0.258    | False      |          5 |       5 |
| base_10k | calltype_killerwhale_whistle | calltype_killerwhale_whistle       | calltype_killerwhale_whistle_context |   0.0446 |  0.192    |  0.258    | False      |         36 |      50 |
| base_10k | species                      | cnn_small                          | metadata                             |  -0.1582 |  0.198    |  0.258    | False      |         40 |      50 |
| base_10k | species                      | xgboost                            | metadata                             |  -0.1154 |  0.239    |  0.299    | False      |         38 |      50 |
| base_10k | calltype_killerwhale_chirp   | calltype_killerwhale_chirp         | calltype_killerwhale_chirp_context   |  -0.0515 |  0.281    |  0.338    | False      |         32 |      50 |
| base_10k | species                      | cnn                                | metadata                             |  -0.1702 |  0.387    |  0.446    | False      |          4 |       5 |
| base_10k | calltype_spermwhale_click    | calltype_spermwhale_click          | calltype_spermwhale_click_context    |  -0.0301 |  0.606    |  0.673    | False      |         25 |      50 |
| base_10k | calltype_spermwhale_whistle  | calltype_spermwhale_whistle        | calltype_spermwhale_whistle_context  |   0.0387 |  0.8      |  0.846    | False      |         26 |      50 |
| base_10k | calltype_killerwhale_call    | calltype_killerwhale_call          | calltype_killerwhale_call_context    |   0.0332 |  0.818    |  0.846    | False      |         29 |      50 |
| base_10k | calltype_killerwhale_squeal  | calltype_killerwhale_squeal        | calltype_killerwhale_squeal_context  |   0.0056 |  0.961    |  0.961    | False      |         29 |      50 |

## What survives

11 of 30 comparisons survive the correction, and the smallest of them clears the divided threshold of 1.67e-03 as well.

- `xgboost` is 0.495 below `logbook` in wide_10k, at q = 1.03e-14 over 50 folds.
- `probe` is 0.483 below `logbook` in wide_10k, at q = 5.70e-14 over 50 folds.
- `logbook` is 0.345 above `metadata` in wide_10k, at q = 4.64e-08 over 50 folds.
- `xgboost` is 0.245 below `logbook` in base_10k, at q = 1.25e-07 over 50 folds.
- `probe` is 0.258 below `logbook` in base_10k, at q = 7.80e-06 over 50 folds.
- `probe` is 0.274 below `logbook` in base_5k, at q = 2.00e-05 over 50 folds.
- `xgboost` is 0.247 below `logbook` in base_5k, at q = 1.06e-03 over 50 folds.
- `cnn_small` is 0.288 below `logbook` in base_10k, at q = 1.07e-03 over 50 folds.
- `xgboost` is 0.149 below `metadata` in wide_10k, at q = 4.47e-03 over 50 folds.
- `probe` is 0.176 below `metadata` in base_5k, at q = 5.30e-03 over 50 folds.
- `xgboost` is 0.150 below `metadata` in base_5k, at q = 3.26e-02 over 50 folds.
