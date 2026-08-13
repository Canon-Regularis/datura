# Every comparison, corrected for how many there are

70 comparisons across base_10k, base_5k, context_10k, context_shuffled_10k, context_shuffled_wide_10k, context_wide_10k, wide_10k. Each configuration reports these with an uncorrected p value, which is the right number to read for one comparison on its own and the wrong one to read across a table of them.

`q_value` is the Benjamini-Hochberg adjusted figure, controlling the share of resolved comparisons that are noise. `rejected` marks the ones that survive it at 0.05. For reference, dividing the threshold across every comparison instead puts the bar at 7.14e-04, which is the stricter question of whether any one of them is a false positive.

| config                    | family                       | model                              | floor                                |   margin |   p_value |   q_value | rejected   |   agreeing |   folds |
|:--------------------------|:-----------------------------|:-----------------------------------|:-------------------------------------|---------:|----------:|----------:|:-----------|-----------:|--------:|
| wide_10k                  | species                      | probe                              | logbook                              |  -0.5359 |  5.02e-20 |  3.52e-18 | True       |         50 |      50 |
| wide_10k                  | species                      | xgboost+probe                      | logbook                              |  -0.5153 |  1.91e-18 |  6.67e-17 | True       |         50 |      50 |
| wide_10k                  | species                      | xgboost                            | logbook                              |  -0.5479 |  5.55e-15 |  1.29e-13 | True       |         50 |      50 |
| wide_10k                  | species                      | logbook                            | metadata                             |   0.3985 |  1.54e-08 |  2.7e-07  | True       |         50 |      50 |
| base_10k                  | species                      | xgboost                            | logbook                              |  -0.2448 |  2.37e-08 |  3.32e-07 | True       |         50 |      50 |
| base_10k                  | species                      | probe                              | logbook                              |  -0.2581 |  1.11e-06 |  1.29e-05 | True       |         50 |      50 |
| base_5k                   | species                      | probe                              | logbook                              |  -0.2759 |  2.94e-06 |  2.94e-05 | True       |         50 |      50 |
| base_10k                  | species                      | xgboost+probe                      | logbook                              |  -0.2324 |  7.44e-06 |  6.51e-05 | True       |         50 |      50 |
| base_5k                   | species                      | xgboost+probe                      | logbook                              |  -0.2386 |  0.00012  |  0.000837 | True       |         50 |      50 |
| context_10k               | species                      | xgboost                            | logbook                              |  -0.6602 |  0.000108 |  0.000837 | True       |          5 |       5 |
| base_5k                   | species                      | xgboost                            | logbook                              |  -0.2497 |  0.000246 |  0.00157  | True       |         50 |      50 |
| base_10k                  | species                      | cnn_small                          | logbook                              |  -0.2876 |  0.000319 |  0.00186  | True       |         50 |      50 |
| base_10k                  | species                      | xgboost_centred                    | logbook                              |  -0.1747 |  0.000582 |  0.00314  | True       |         50 |      50 |
| context_10k               | species                      | xgboost+probe                      | logbook                              |  -0.5412 |  0.00143  |  0.00666  | True       |          5 |       5 |
| wide_10k                  | species                      | xgboost                            | metadata                             |  -0.1494 |  0.00134  |  0.00666  | True       |         48 |      50 |
| context_10k               | species                      | probe                              | logbook                              |  -0.5371 |  0.00153  |  0.0067   | True       |          5 |       5 |
| base_5k                   | species                      | probe                              | metadata                             |  -0.1764 |  0.00177  |  0.00728  | True       |         47 |      50 |
| context_wide_10k          | species                      | probe                              | logbook                              |  -0.6902 |  0.00224  |  0.00871  | True       |          4 |       4 |
| context_shuffled_wide_10k | species                      | xgboost+probe                      | logbook                              |  -0.521  |  0.00249  |  0.00917  | True       |          4 |       4 |
| context_shuffled_wide_10k | species                      | xgboost                            | logbook                              |  -0.5705 |  0.00322  |  0.0105   | True       |          4 |       4 |
| context_wide_10k          | species                      | xgboost+probe                      | logbook                              |  -0.6802 |  0.00302  |  0.0105   | True       |          4 |       4 |
| context_wide_10k          | species                      | xgboost                            | logbook                              |  -0.7154 |  0.00331  |  0.0105   | True       |          4 |       4 |
| context_shuffled_wide_10k | species                      | probe                              | logbook                              |  -0.5419 |  0.00401  |  0.0122   | True       |          4 |       4 |
| base_5k                   | species                      | xgboost                            | metadata                             |  -0.1502 |  0.0119   |  0.0348   | True       |         46 |      50 |
| context_10k               | species                      | xgboost_centred                    | logbook                              |  -0.4358 |  0.0157   |  0.0423   | True       |          5 |       5 |
| context_shuffled_10k      | species                      | probe                              | metadata                             |  -0.3039 |  0.0155   |  0.0423   | True       |          5 |       5 |
| base_5k                   | species                      | xgboost+probe                      | metadata                             |  -0.1391 |  0.0189   |  0.0489   | True       |         45 |      50 |
| base_10k                  | species                      | cnn                                | logbook                              |  -0.2863 |  0.0254   |  0.0636   | False      |          5 |       5 |
| context_wide_10k          | species                      | logbook                            | metadata                             |   0.4794 |  0.0316   |  0.0763   | False      |          4 |       4 |
| context_wide_10k          | species                      | xgboost                            | metadata                             |  -0.236  |  0.0341   |  0.0796   | False      |          4 |       4 |
| wide_10k                  | species                      | probe                              | metadata                             |  -0.1375 |  0.047    |  0.106    | False      |         43 |      50 |
| wide_10k                  | species                      | xgboost+probe                      | metadata                             |  -0.1168 |  0.0541   |  0.118    | False      |         43 |      50 |
| base_10k                  | species                      | logbook                            | metadata                             |   0.1294 |  0.0667   |  0.12     | False      |         44 |      50 |
| context_shuffled_10k      | species                      | xgboost                            | metadata                             |  -0.3426 |  0.0571   |  0.12     | False      |          5 |       5 |
| context_shuffled_10k      | species                      | probe                              | logbook                              |  -0.3329 |  0.0623   |  0.12     | False      |          5 |       5 |
| context_shuffled_10k      | species                      | xgboost+probe                      | metadata                             |  -0.2631 |  0.067    |  0.12     | False      |          5 |       5 |
| context_shuffled_wide_10k | species                      | logbook                            | metadata                             |   0.3528 |  0.0628   |  0.12     | False      |          4 |       4 |
| context_wide_10k          | species                      | xgboost+probe                      | metadata                             |  -0.2008 |  0.0612   |  0.12     | False      |          4 |       4 |
| context_wide_10k          | species                      | probe                              | metadata                             |  -0.2108 |  0.0644   |  0.12     | False      |          4 |       4 |
| context_10k               | species                      | logbook                            | metadata                             |   0.3603 |  0.0771   |  0.135    | False      |          5 |       5 |
| context_10k               | species                      | xgboost                            | metadata                             |  -0.2999 |  0.0824   |  0.141    | False      |          5 |       5 |
| context_shuffled_10k      | species                      | xgboost                            | logbook                              |  -0.3716 |  0.107    |  0.178    | False      |          4 |       5 |
| context_shuffled_10k      | species                      | xgboost_centred                    | metadata                             |  -0.2409 |  0.117    |  0.19     | False      |          4 |       5 |
| base_5k                   | species                      | cnn_small                          | logbook                              |  -0.2369 |  0.124    |  0.196    | False      |          5 |       5 |
| base_10k                  | calltype_spermwhale_coda     | calltype_spermwhale_coda_cnn_small | calltype_spermwhale_coda_context     |  -0.1776 |  0.127    |  0.198    | False      |          5 |       5 |
| base_10k                  | calltype_killerwhale_chirp   | calltype_killerwhale_chirp         | calltype_killerwhale_chirp_context   |  -0.0632 |  0.135    |  0.203    | False      |         34 |      50 |
| context_shuffled_10k      | species                      | xgboost+probe                      | logbook                              |  -0.2922 |  0.136    |  0.203    | False      |          4 |       5 |
| base_10k                  | calltype_spermwhale_coda     | calltype_spermwhale_coda           | calltype_spermwhale_coda_context     |  -0.094  |  0.144    |  0.206    | False      |         39 |      50 |
| base_5k                   | species                      | logbook                            | metadata                             |   0.0995 |  0.144    |  0.206    | False      |         39 |      50 |
| base_10k                  | species                      | probe                              | metadata                             |  -0.1287 |  0.148    |  0.208    | False      |         39 |      50 |
| base_5k                   | species                      | cnn_small                          | metadata                             |  -0.1705 |  0.167    |  0.228    | False      |          5 |       5 |
| context_shuffled_10k      | species                      | xgboost_centred                    | logbook                              |  -0.27   |  0.17     |  0.228    | False      |          4 |       5 |
| base_10k                  | calltype_killerwhale_click   | calltype_killerwhale_click         | calltype_killerwhale_click_context   |   0.1155 |  0.198    |  0.256    | False      |         37 |      50 |
| base_10k                  | species                      | cnn_small                          | metadata                             |  -0.1582 |  0.198    |  0.256    | False      |         40 |      50 |
| base_10k                  | calltype_killerwhale_whistle | calltype_killerwhale_whistle       | calltype_killerwhale_whistle_context |   0.0435 |  0.202    |  0.257    | False      |         36 |      50 |
| context_shuffled_wide_10k | species                      | xgboost                            | metadata                             |  -0.2177 |  0.213    |  0.266    | False      |          4 |       4 |
| context_shuffled_wide_10k | species                      | probe                              | metadata                             |  -0.1891 |  0.219    |  0.269    | False      |          4 |       4 |
| base_10k                  | species                      | xgboost                            | metadata                             |  -0.1154 |  0.239    |  0.289    | False      |         38 |      50 |
| context_shuffled_wide_10k | species                      | xgboost+probe                      | metadata                             |  -0.1682 |  0.268    |  0.318    | False      |          4 |       4 |
| base_10k                  | species                      | xgboost+probe                      | metadata                             |  -0.1031 |  0.286    |  0.334    | False      |         38 |      50 |
| context_10k               | species                      | xgboost+probe                      | metadata                             |  -0.181  |  0.323    |  0.371    | False      |          4 |       5 |
| context_10k               | species                      | probe                              | metadata                             |  -0.1769 |  0.357    |  0.403    | False      |          4 |       5 |
| base_10k                  | species                      | cnn                                | metadata                             |  -0.1702 |  0.387    |  0.43     | False      |          4 |       5 |
| base_10k                  | calltype_killerwhale_call    | calltype_killerwhale_call          | calltype_killerwhale_call_context    |  -0.0773 |  0.406    |  0.444    | False      |         30 |      50 |
| base_10k                  | calltype_spermwhale_click    | calltype_spermwhale_click          | calltype_spermwhale_click_context    |  -0.0317 |  0.582    |  0.626    | False      |         29 |      50 |
| base_10k                  | species                      | xgboost_centred                    | metadata                             |  -0.0453 |  0.641    |  0.68     | False      |         36 |      50 |
| base_10k                  | calltype_killerwhale_squeal  | calltype_killerwhale_squeal        | calltype_killerwhale_squeal_context  |   0.0361 |  0.733    |  0.744    | False      |         32 |      50 |
| context_10k               | species                      | xgboost_centred                    | metadata                             |  -0.0755 |  0.716    |  0.744    | False      |          3 |       5 |
| context_shuffled_10k      | species                      | logbook                            | metadata                             |   0.029  |  0.725    |  0.744    | False      |          3 |       5 |
| base_10k                  | calltype_spermwhale_whistle  | calltype_spermwhale_whistle        | calltype_spermwhale_whistle_context  |  -0.0335 |  0.841    |  0.841    | False      |         31 |      50 |

## What survives

27 of 70 comparisons survive the correction, and the smallest of them clears the divided threshold of 7.14e-04 as well.

- `probe` is 0.536 below `logbook` in wide_10k, at q = 3.52e-18 over 50 folds.
- `xgboost+probe` is 0.515 below `logbook` in wide_10k, at q = 6.67e-17 over 50 folds.
- `xgboost` is 0.548 below `logbook` in wide_10k, at q = 1.29e-13 over 50 folds.
- `logbook` is 0.398 above `metadata` in wide_10k, at q = 2.70e-07 over 50 folds.
- `xgboost` is 0.245 below `logbook` in base_10k, at q = 3.32e-07 over 50 folds.
- `probe` is 0.258 below `logbook` in base_10k, at q = 1.29e-05 over 50 folds.
- `probe` is 0.276 below `logbook` in base_5k, at q = 2.94e-05 over 50 folds.
- `xgboost+probe` is 0.232 below `logbook` in base_10k, at q = 6.51e-05 over 50 folds.
- `xgboost+probe` is 0.239 below `logbook` in base_5k, at q = 8.37e-04 over 50 folds.
- `xgboost` is 0.660 below `logbook` in context_10k, at q = 8.37e-04 over 5 folds.
- `xgboost` is 0.250 below `logbook` in base_5k, at q = 1.57e-03 over 50 folds.
- `cnn_small` is 0.288 below `logbook` in base_10k, at q = 1.86e-03 over 50 folds.
- `xgboost_centred` is 0.175 below `logbook` in base_10k, at q = 3.14e-03 over 50 folds.
- `xgboost+probe` is 0.541 below `logbook` in context_10k, at q = 6.66e-03 over 5 folds.
- `xgboost` is 0.149 below `metadata` in wide_10k, at q = 6.66e-03 over 50 folds.
- `probe` is 0.537 below `logbook` in context_10k, at q = 6.70e-03 over 5 folds.
- `probe` is 0.176 below `metadata` in base_5k, at q = 7.28e-03 over 50 folds.
- `probe` is 0.690 below `logbook` in context_wide_10k, at q = 8.71e-03 over 4 folds.
- `xgboost+probe` is 0.521 below `logbook` in context_shuffled_wide_10k, at q = 9.17e-03 over 4 folds.
- `xgboost` is 0.571 below `logbook` in context_shuffled_wide_10k, at q = 1.05e-02 over 4 folds.
- `xgboost+probe` is 0.680 below `logbook` in context_wide_10k, at q = 1.05e-02 over 4 folds.
- `xgboost` is 0.715 below `logbook` in context_wide_10k, at q = 1.05e-02 over 4 folds.
- `probe` is 0.542 below `logbook` in context_shuffled_wide_10k, at q = 1.22e-02 over 4 folds.
- `xgboost` is 0.150 below `metadata` in base_5k, at q = 3.48e-02 over 50 folds.
- `xgboost_centred` is 0.436 below `logbook` in context_10k, at q = 4.23e-02 over 5 folds.
- `probe` is 0.304 below `metadata` in context_shuffled_10k, at q = 4.23e-02 over 5 folds.
- `xgboost+probe` is 0.139 below `metadata` in base_5k, at q = 4.89e-02 over 50 folds.
