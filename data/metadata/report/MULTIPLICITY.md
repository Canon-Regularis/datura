# Every comparison, corrected for how many there are

46 comparisons across base_10k, base_5k, wide_10k, context_10k, context_shuffled_10k. Each configuration reports these with an uncorrected p value, which is the right number to read for one comparison on its own and the wrong one to read across a table of them.

`q_value` is the Benjamini-Hochberg adjusted figure, controlling the share of resolved comparisons that are noise. `rejected` marks the ones that survive it at 0.05. For reference, dividing the threshold across every comparison instead puts the bar at 1.09e-03, which is the stricter question of whether any one of them is a false positive.

| config               | family                       | model                              | floor                                |   margin |   p_value |   q_value | rejected   |   agreeing |   folds |
|:---------------------|:-----------------------------|:-----------------------------------|:-------------------------------------|---------:|----------:|----------:|:-----------|-----------:|--------:|
| context_10k          | species                      | xgboost                            | logbook                              |  -0.6698 |  6.3e-24  |  2.9e-22  | True       |         50 |      50 |
| wide_10k             | species                      | xgboost+probe                      | logbook                              |  -0.462  |  1.23e-16 |  2.83e-15 | True       |         50 |      50 |
| wide_10k             | species                      | xgboost                            | logbook                              |  -0.4946 |  3.42e-16 |  5.25e-15 | True       |         50 |      50 |
| wide_10k             | species                      | probe                              | logbook                              |  -0.4827 |  3.8e-15  |  4.37e-14 | True       |         50 |      50 |
| context_10k          | species                      | xgboost+probe                      | logbook                              |  -0.5508 |  1.58e-13 |  1.45e-12 | True       |         50 |      50 |
| context_10k          | species                      | probe                              | logbook                              |  -0.5467 |  2.68e-13 |  2.06e-12 | True       |         50 |      50 |
| wide_10k             | species                      | logbook                            | metadata                             |   0.3452 |  4.64e-09 |  3.05e-08 | True       |         50 |      50 |
| base_10k             | species                      | xgboost                            | logbook                              |  -0.245  |  1.66e-08 |  9.56e-08 | True       |         50 |      50 |
| base_10k             | species                      | probe                              | logbook                              |  -0.2583 |  1.3e-06  |  6.64e-06 | True       |         50 |      50 |
| base_5k              | species                      | probe                              | logbook                              |  -0.2737 |  4e-06    |  1.84e-05 | True       |         50 |      50 |
| base_10k             | species                      | xgboost+probe                      | logbook                              |  -0.2327 |  6.39e-06 |  2.67e-05 | True       |         50 |      50 |
| base_5k              | species                      | xgboost+probe                      | logbook                              |  -0.2363 |  0.000155 |  0.000592 | True       |         50 |      50 |
| base_5k              | species                      | xgboost                            | logbook                              |  -0.2474 |  0.000247 |  0.000874 | True       |         50 |      50 |
| base_10k             | species                      | cnn_small                          | logbook                              |  -0.2878 |  0.000285 |  0.000938 | True       |         50 |      50 |
| context_shuffled_10k | species                      | xgboost                            | logbook                              |  -0.3611 |  0.000666 |  0.00204  | True       |         50 |      50 |
| wide_10k             | species                      | xgboost                            | metadata                             |  -0.1494 |  0.00134  |  0.00386  | True       |         48 |      50 |
| context_10k          | species                      | logbook                            | metadata                             |   0.3699 |  0.00158  |  0.00427  | True       |         50 |      50 |
| base_5k              | species                      | probe                              | metadata                             |  -0.1764 |  0.00177  |  0.00444  | True       |         47 |      50 |
| context_10k          | species                      | xgboost                            | metadata                             |  -0.2999 |  0.00183  |  0.00444  | True       |         50 |      50 |
| base_5k              | species                      | xgboost                            | metadata                             |  -0.1502 |  0.0119   |  0.0275   | True       |         46 |      50 |
| base_5k              | species                      | xgboost+probe                      | metadata                             |  -0.1391 |  0.0189   |  0.0413   | True       |         45 |      50 |
| base_10k             | calltype_spermwhale_coda     | calltype_spermwhale_coda           | calltype_spermwhale_coda_context     |  -0.1622 |  0.0234   |  0.0489   | True       |         46 |      50 |
| base_10k             | species                      | cnn                                | logbook                              |  -0.2848 |  0.0259   |  0.0518   | False      |          5 |       5 |
| context_shuffled_10k | species                      | xgboost                            | metadata                             |  -0.2569 |  0.0299   |  0.0573   | False      |         40 |      50 |
| wide_10k             | species                      | probe                              | metadata                             |  -0.1375 |  0.047    |  0.0865   | False      |         43 |      50 |
| wide_10k             | species                      | xgboost+probe                      | metadata                             |  -0.1168 |  0.0541   |  0.0958   | False      |         43 |      50 |
| base_10k             | species                      | logbook                            | metadata                             |   0.1296 |  0.0724   |  0.123    | False      |         45 |      50 |
| context_10k          | species                      | xgboost+probe                      | metadata                             |  -0.181  |  0.114    |  0.188    | False      |         40 |      50 |
| base_10k             | calltype_killerwhale_click   | calltype_killerwhale_click         | calltype_killerwhale_click_context   |   0.1249 |  0.134    |  0.205    | False      |         37 |      50 |
| context_shuffled_10k | species                      | logbook                            | metadata                             |   0.1042 |  0.133    |  0.205    | False      |         40 |      50 |
| base_5k              | species                      | cnn_small                          | logbook                              |  -0.2313 |  0.142    |  0.206    | False      |          5 |       5 |
| context_10k          | species                      | probe                              | metadata                             |  -0.1769 |  0.143    |  0.206    | False      |         40 |      50 |
| base_10k             | species                      | probe                              | metadata                             |  -0.1287 |  0.148    |  0.207    | False      |         39 |      50 |
| base_5k              | species                      | logbook                            | metadata                             |   0.0972 |  0.155    |  0.21     | False      |         40 |      50 |
| base_5k              | species                      | cnn_small                          | metadata                             |  -0.1705 |  0.167    |  0.219    | False      |          5 |       5 |
| base_10k             | calltype_spermwhale_coda     | calltype_spermwhale_coda_cnn_small | calltype_spermwhale_coda_context     |  -0.2242 |  0.187    |  0.239    | False      |          5 |       5 |
| base_10k             | calltype_killerwhale_whistle | calltype_killerwhale_whistle       | calltype_killerwhale_whistle_context |   0.0446 |  0.192    |  0.239    | False      |         36 |      50 |
| base_10k             | species                      | cnn_small                          | metadata                             |  -0.1582 |  0.198    |  0.239    | False      |         40 |      50 |
| base_10k             | species                      | xgboost                            | metadata                             |  -0.1154 |  0.239    |  0.282    | False      |         38 |      50 |
| base_10k             | calltype_killerwhale_chirp   | calltype_killerwhale_chirp         | calltype_killerwhale_chirp_context   |  -0.0515 |  0.281    |  0.321    | False      |         32 |      50 |
| base_10k             | species                      | xgboost+probe                      | metadata                             |  -0.1031 |  0.286    |  0.321    | False      |         38 |      50 |
| base_10k             | species                      | cnn                                | metadata                             |  -0.1702 |  0.387    |  0.424    | False      |          4 |       5 |
| base_10k             | calltype_spermwhale_click    | calltype_spermwhale_click          | calltype_spermwhale_click_context    |  -0.0301 |  0.606    |  0.648    | False      |         25 |      50 |
| base_10k             | calltype_spermwhale_whistle  | calltype_spermwhale_whistle        | calltype_spermwhale_whistle_context  |   0.0387 |  0.8      |  0.836    | False      |         26 |      50 |
| base_10k             | calltype_killerwhale_call    | calltype_killerwhale_call          | calltype_killerwhale_call_context    |   0.0332 |  0.818    |  0.836    | False      |         29 |      50 |
| base_10k             | calltype_killerwhale_squeal  | calltype_killerwhale_squeal        | calltype_killerwhale_squeal_context  |   0.0056 |  0.961    |  0.961    | False      |         29 |      50 |

## What survives

22 of 46 comparisons survive the correction, and the smallest of them clears the divided threshold of 1.09e-03 as well.

- `xgboost` is 0.670 below `logbook` in context_10k, at q = 2.90e-22 over 50 folds.
- `xgboost+probe` is 0.462 below `logbook` in wide_10k, at q = 2.83e-15 over 50 folds.
- `xgboost` is 0.495 below `logbook` in wide_10k, at q = 5.25e-15 over 50 folds.
- `probe` is 0.483 below `logbook` in wide_10k, at q = 4.37e-14 over 50 folds.
- `xgboost+probe` is 0.551 below `logbook` in context_10k, at q = 1.45e-12 over 50 folds.
- `probe` is 0.547 below `logbook` in context_10k, at q = 2.06e-12 over 50 folds.
- `logbook` is 0.345 above `metadata` in wide_10k, at q = 3.05e-08 over 50 folds.
- `xgboost` is 0.245 below `logbook` in base_10k, at q = 9.56e-08 over 50 folds.
- `probe` is 0.258 below `logbook` in base_10k, at q = 6.64e-06 over 50 folds.
- `probe` is 0.274 below `logbook` in base_5k, at q = 1.84e-05 over 50 folds.
- `xgboost+probe` is 0.233 below `logbook` in base_10k, at q = 2.67e-05 over 50 folds.
- `xgboost+probe` is 0.236 below `logbook` in base_5k, at q = 5.92e-04 over 50 folds.
- `xgboost` is 0.247 below `logbook` in base_5k, at q = 8.74e-04 over 50 folds.
- `cnn_small` is 0.288 below `logbook` in base_10k, at q = 9.38e-04 over 50 folds.
- `xgboost` is 0.361 below `logbook` in context_shuffled_10k, at q = 2.04e-03 over 50 folds.
- `xgboost` is 0.149 below `metadata` in wide_10k, at q = 3.86e-03 over 50 folds.
- `logbook` is 0.370 above `metadata` in context_10k, at q = 4.27e-03 over 50 folds.
- `probe` is 0.176 below `metadata` in base_5k, at q = 4.44e-03 over 50 folds.
- `xgboost` is 0.300 below `metadata` in context_10k, at q = 4.44e-03 over 50 folds.
- `xgboost` is 0.150 below `metadata` in base_5k, at q = 2.75e-02 over 50 folds.
- `xgboost+probe` is 0.139 below `metadata` in base_5k, at q = 4.13e-02 over 50 folds.
- `calltype_spermwhale_coda` is 0.162 below `calltype_spermwhale_coda_context` in base_10k, at q = 4.89e-02 over 50 folds.
