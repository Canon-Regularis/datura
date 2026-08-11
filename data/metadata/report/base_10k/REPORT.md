# Results: base_10k

Species: HumpbackWhale, SpermWhale, KillerWhale  
Common band: 0 to 5000 Hz at 10000 Hz  
Folds: 5 per split, grouped by tape  
Windows: 2.0 s, hop 1.0 s, at most 16 per clip  
Families: 9, each a set of models and the control they were measured against

Every p value in this document is uncorrected for the number of comparisons reported.
`MULTIPLICITY.md` beside this file adjusts across every comparison in every
configuration at once, which is the number to read before calling one of them a
finding.

## Every comparison

Columns beside the margin say what the design resolves. `folds` counts every fold of every repeat, so a run of ten repeats over five folds shows 50. `low` and `high` bound the paired difference at 95%, and `p_value` is the corrected resampled test, which accounts for the training data those folds share. `agreeing` counts the folds that pointed the same way as the mean, and is worth reading where the p value settles nothing.

| family                       | model                              | floor                                |   margin |     low |    high |   p_value |   agreeing |   folds |
|:-----------------------------|:-----------------------------------|:-------------------------------------|---------:|--------:|--------:|----------:|-----------:|--------:|
| species                      | xgboost                            | logbook                              |  -0.245  | -0.3181 | -0.172  |  1.66e-08 |         50 |      50 |
| species                      | probe                              | logbook                              |  -0.2583 | -0.3525 | -0.1642 |  1.3e-06  |         50 |      50 |
| species                      | xgboost+probe                      | logbook                              |  -0.2327 | -0.3251 | -0.1402 |  6.39e-06 |         50 |      50 |
| species                      | cnn_small                          | logbook                              |  -0.2878 | -0.4358 | -0.1398 |  0.000285 |         50 |      50 |
| calltype_spermwhale_coda     | calltype_spermwhale_coda           | calltype_spermwhale_coda_context     |  -0.1622 | -0.3015 | -0.0229 |  0.0234   |         46 |      50 |
| species                      | cnn                                | logbook                              |  -0.2848 | -0.5136 | -0.056  |  0.0259   |          5 |       5 |
| species                      | logbook                            | metadata                             |   0.1296 | -0.0123 |  0.2715 |  0.0724   |         45 |      50 |
| calltype_killerwhale_click   | calltype_killerwhale_click         | calltype_killerwhale_click_context   |   0.1249 | -0.0398 |  0.2897 |  0.134    |         37 |      50 |
| species                      | probe                              | metadata                             |  -0.1287 | -0.3049 |  0.0474 |  0.148    |         39 |      50 |
| calltype_spermwhale_coda     | calltype_spermwhale_coda_cnn_small | calltype_spermwhale_coda_context     |  -0.2242 | -0.6154 |  0.1669 |  0.187    |          5 |       5 |
| calltype_killerwhale_whistle | calltype_killerwhale_whistle       | calltype_killerwhale_whistle_context |   0.0446 | -0.0232 |  0.1124 |  0.192    |         36 |      50 |
| species                      | cnn_small                          | metadata                             |  -0.1582 | -0.4016 |  0.0853 |  0.198    |         40 |      50 |
| species                      | xgboost                            | metadata                             |  -0.1154 | -0.3101 |  0.0793 |  0.239    |         38 |      50 |
| calltype_killerwhale_chirp   | calltype_killerwhale_chirp         | calltype_killerwhale_chirp_context   |  -0.0515 | -0.1466 |  0.0435 |  0.281    |         32 |      50 |
| species                      | xgboost+probe                      | metadata                             |  -0.1031 | -0.2952 |  0.0891 |  0.286    |         38 |      50 |
| species                      | cnn                                | metadata                             |  -0.1702 | -0.6573 |  0.3169 |  0.387    |          4 |       5 |
| calltype_spermwhale_click    | calltype_spermwhale_click          | calltype_spermwhale_click_context    |  -0.0301 | -0.1463 |  0.0862 |  0.606    |         25 |      50 |
| calltype_spermwhale_whistle  | calltype_spermwhale_whistle        | calltype_spermwhale_whistle_context  |   0.0387 | -0.2676 |  0.3451 |  0.8      |         26 |      50 |
| calltype_killerwhale_call    | calltype_killerwhale_call          | calltype_killerwhale_call_context    |   0.0332 | -0.2549 |  0.3213 |  0.818    |         29 |      50 |
| calltype_killerwhale_squeal  | calltype_killerwhale_squeal        | calltype_killerwhale_squeal_context  |   0.0056 | -0.2236 |  0.2348 |  0.961    |         29 |      50 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. It is a floor rather than the floor, and the table after this one measures
against the highest any model that hears nothing reaches.

| model         |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds | family   |
|:--------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|:---------|
| logbook       | 0.9974 |    0.8678 |   0.1296 | -0.0123 | 0.2715 |    0.0724 |         45 |      50 | species  |
| xgboost+probe | 0.7647 |    0.8678 |  -0.1031 | -0.2952 | 0.0891 |    0.286  |         38 |      50 | species  |
| xgboost       | 0.7524 |    0.8678 |  -0.1154 | -0.3101 | 0.0793 |    0.239  |         38 |      50 | species  |
| probe         | 0.7391 |    0.8678 |  -0.1287 | -0.3049 | 0.0474 |    0.148  |         39 |      50 | species  |
| cnn_small     | 0.7096 |    0.8678 |  -0.1582 | -0.4016 | 0.0853 |    0.198  |         40 |      50 | species  |
| cnn           | 0.7135 |    0.8837 |  -0.1702 | -0.6573 | 0.3169 |    0.387  |          4 |       5 | species  |

### Margin over logbook, the strongest model that hears no audio

`logbook` also sees the site, the coordinates, the noise conditions and the
collection code the field note opens with. None of that is the animal, so this is the
number an audio result has to clear before it is evidence about whales.

| model         |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:--------------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| xgboost+probe | 0.7647 |    0.9974 |  -0.2327 | -0.3251 | -0.1402 |  6.39e-06 |         50 |      50 | species  |
| xgboost       | 0.7524 |    0.9974 |  -0.245  | -0.3181 | -0.172  |  1.66e-08 |         50 |      50 | species  |
| probe         | 0.7391 |    0.9974 |  -0.2583 | -0.3525 | -0.1642 |  1.3e-06  |         50 |      50 | species  |
| cnn           | 0.7135 |    0.9983 |  -0.2848 | -0.5136 | -0.056  |  0.0259   |          5 |       5 | species  |
| cnn_small     | 0.7096 |    0.9974 |  -0.2878 | -0.4358 | -0.1398 |  0.000285 |         50 |      50 | species  |

### Every model, with the range the recordings support

The interval comes from resampling whole tapes with replacement. Cuts from one tape
are near duplicates, so resampling clips would count the same recording many times
and produce an interval several times too narrow.

| model         |   estimate |    low |   high |   tapes |
|:--------------|-----------:|-------:|-------:|--------:|
| xgboost       |     0.7265 | 0.6191 | 0.8387 |     134 |
| cnn           |     0.7064 | 0.5737 | 0.8671 |     134 |
| cnn_small     |     0.6849 | 0.5497 | 0.8535 |     134 |
| logbook       |     0.9987 | 0.9935 | 1      |     134 |
| probe         |     0.7512 | 0.6313 | 0.8358 |     134 |
| xgboost+probe |     0.7802 | 0.6634 | 0.8682 |     134 |
| metadata      |     0.9343 | 0.7962 | 0.9832 |     134 |

### Spread across folds

| model         |   mean |    std |
|:--------------|-------:|-------:|
| xgboost       | 0.7524 | 0.0698 |
| cnn           | 0.7135 | 0.123  |
| cnn_small     | 0.7096 | 0.1418 |
| logbook       | 0.9974 | 0.0045 |
| probe         | 0.7391 | 0.0902 |
| xgboost+probe | 0.7647 | 0.0889 |
| metadata      | 0.8678 | 0.1344 |

### Per species recall

| model         |   HumpbackWhale |   SpermWhale |   KillerWhale |
|:--------------|----------------:|-------------:|--------------:|
| xgboost       |          0.5734 |       0.8688 |        0.8854 |
| cnn           |          0.4956 |       0.8828 |        0.9007 |
| cnn_small     |          0.5292 |       0.8602 |        0.8741 |
| logbook       |          0.9952 |       0.9955 |        0.9996 |
| probe         |          0.6466 |       0.8131 |        0.8588 |
| xgboost+probe |          0.6358 |       0.8655 |        0.8824 |
| metadata      |          0.6804 |       0.9889 |        0.9443 |

One of these recordings carries more than one of the classes above, across HumpbackWhale and SpermWhale. Grouping keeps each tape whole, so none of them crosses a fold boundary, and they still contribute to two recalls apiece: the classes sharing a tape are not scored on independent evidence.

### With and without the giveaway

Test clips split by what their native sample rate or their collection code does to the species. A value
used by one species names it before any audio is heard. A value used by several does
not, and those rows are where audio has to earn its result. A clip carrying no value
at all is a third case, and it is neither of the other two.

Read `classes_scored` against `classes_total` before the score. A slice can hold
fewer species than the task does, and it is scored over the ones it holds. Averaging
in a class that cannot appear scores it zero and divides by it anyway, which caps the
column and reads as a collapse the predictions do not contain.

| giveaway           | model         | subset                                 |   clips |   classes_scored |   classes_total |   folds |   macro_f1_mean |   macro_f1_std |
|:-------------------|:--------------|:---------------------------------------|--------:|-----------------:|----------------:|--------:|----------------:|---------------:|
| native sample rate | xgboost       | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.6611 |         0.1356 |
| native sample rate | xgboost       | native sample rate unique to a species |    3313 |                3 |               3 |      50 |          0.705  |         0.1305 |
| collection code    | xgboost       | collection code not recorded           |     359 |                2 |               3 |      43 |          0.6165 |         0.2831 |
| collection code    | xgboost       | collection code unique to a species    |    3801 |                3 |               3 |      50 |          0.7503 |         0.072  |
| native sample rate | cnn           | native sample rate shared by species   |     847 |                3 |               3 |       5 |          0.7068 |         0.1369 |
| native sample rate | cnn           | native sample rate unique to a species |    3313 |                3 |               3 |       5 |          0.6436 |         0.1201 |
| collection code    | cnn           | collection code not recorded           |     359 |                2 |               3 |       4 |          0.5815 |         0.1652 |
| collection code    | cnn           | collection code unique to a species    |    3801 |                3 |               3 |       5 |          0.7111 |         0.1245 |
| native sample rate | cnn_small     | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.6713 |         0.1512 |
| native sample rate | cnn_small     | native sample rate unique to a species |    3313 |                3 |               3 |      50 |          0.6712 |         0.1708 |
| collection code    | cnn_small     | collection code not recorded           |     359 |                2 |               3 |      43 |          0.6091 |         0.2279 |
| collection code    | cnn_small     | collection code unique to a species    |    3801 |                3 |               3 |      50 |          0.7075 |         0.1446 |
| native sample rate | logbook       | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.8636 |         0.1655 |
| native sample rate | logbook       | native sample rate unique to a species |    3313 |                3 |               3 |      50 |          0.932  |         0.1341 |
| collection code    | logbook       | collection code not recorded           |     359 |                2 |               3 |      43 |          0.6384 |         0.2945 |
| collection code    | logbook       | collection code unique to a species    |    3801 |                3 |               3 |      50 |          0.9977 |         0.0048 |
| native sample rate | probe         | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.7084 |         0.1286 |
| native sample rate | probe         | native sample rate unique to a species |    3313 |                3 |               3 |      50 |          0.679  |         0.1312 |
| collection code    | probe         | collection code not recorded           |     359 |                2 |               3 |      43 |          0.5139 |         0.2721 |
| collection code    | probe         | collection code unique to a species    |    3801 |                3 |               3 |      50 |          0.7462 |         0.093  |
| native sample rate | xgboost+probe | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.719  |         0.1396 |
| native sample rate | xgboost+probe | native sample rate unique to a species |    3313 |                3 |               3 |      50 |          0.6991 |         0.1326 |
| collection code    | xgboost+probe | collection code not recorded           |     359 |                2 |               3 |      43 |          0.6121 |         0.2744 |
| collection code    | xgboost+probe | collection code unique to a species    |    3801 |                3 |               3 |      50 |          0.7656 |         0.0887 |
| native sample rate | metadata      | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.6828 |         0.1895 |
| native sample rate | metadata      | native sample rate unique to a species |    3313 |                3 |               3 |      50 |          0.8777 |         0.1888 |
| collection code    | metadata      | collection code not recorded           |     359 |                2 |               3 |      43 |          0.6977 |         0.2474 |
| collection code    | metadata      | collection code unique to a species    |    3801 |                3 |               3 |      50 |          0.8668 |         0.1349 |

### Accuracy against coverage

Predictions ranked by the probability of the class the model chose, then cut at a
threshold. `coverage` is the share kept, and the row at 1.0 is the score reported
everywhere else. Nothing is refitted: this reads the held out probabilities the
cross validation already wrote.

| model         |   coverage |   threshold |   predictions |   clips |   accuracy |   macro_f1 |
|:--------------|-----------:|------------:|--------------:|--------:|-----------:|-----------:|
| xgboost       |     1      |      0.3346 |         41600 |    4160 |     0.8298 |     0.7502 |
| xgboost       |     0.95   |      0.4271 |         39520 |    4145 |     0.8472 |     0.7646 |
| xgboost       |     0.9    |      0.4665 |         37441 |    4116 |     0.8614 |     0.774  |
| xgboost       |     0.8    |      0.5297 |         33280 |    4024 |     0.8841 |     0.7839 |
| xgboost       |     0.7    |      0.5913 |         29120 |    3814 |     0.903  |     0.7933 |
| xgboost       |     0.6    |      0.6508 |         24960 |    3495 |     0.9151 |     0.7889 |
| xgboost       |     0.5    |      0.6991 |         20800 |    3219 |     0.9203 |     0.7834 |
| xgboost       |     0.4    |      0.7423 |         16640 |    2959 |     0.924  |     0.7861 |
| xgboost       |     0.3    |      0.7888 |         12480 |    2518 |     0.9297 |     0.7922 |
| xgboost       |     0.2    |      0.8468 |          8321 |    1881 |     0.9368 |     0.7532 |
| xgboost       |     0.1    |      0.8801 |          4160 |    1169 |     0.9519 |     0.7085 |
| cnn           |     1      |      0.3731 |          4160 |    4160 |     0.8243 |     0.7064 |
| cnn           |     0.95   |      0.6    |          3952 |    3952 |     0.8494 |     0.7259 |
| cnn           |     0.9    |      0.7408 |          3744 |    3744 |     0.8771 |     0.7533 |
| cnn           |     0.8    |      0.9355 |          3328 |    3328 |     0.9231 |     0.8074 |
| cnn           |     0.7    |      0.99   |          2912 |    2912 |     0.954  |     0.8467 |
| cnn           |     0.6    |      0.9989 |          2496 |    2496 |     0.9776 |     0.8769 |
| cnn           |     0.5    |      0.9999 |          2080 |    2080 |     0.9885 |     0.9064 |
| cnn           |     0.4    |      1      |          1664 |    1664 |     0.994  |     0.9099 |
| cnn           |     0.3    |      1      |          1248 |    1248 |     0.9952 |     0.9443 |
| cnn           |     0.2012 |      1      |           837 |     837 |     1      |     0.6667 |
| cnn           |     0.195  |      1      |           811 |     811 |     1      |     0.6667 |
| cnn_small     |     1      |      0.3415 |         41600 |    4160 |     0.8025 |     0.6952 |
| cnn_small     |     0.95   |      0.5935 |         39520 |    4159 |     0.8239 |     0.7113 |
| cnn_small     |     0.9    |      0.7145 |         37440 |    4153 |     0.8439 |     0.7264 |
| cnn_small     |     0.8    |      0.8781 |         33280 |    4099 |     0.8819 |     0.7587 |
| cnn_small     |     0.7    |      0.9544 |         29120 |    3975 |     0.914  |     0.7941 |
| cnn_small     |     0.6    |      0.9836 |         24960 |    3759 |     0.9431 |     0.8404 |
| cnn_small     |     0.5    |      0.9948 |         20800 |    3404 |     0.9666 |     0.8937 |
| cnn_small     |     0.4    |      0.9984 |         16640 |    3038 |     0.981  |     0.936  |
| cnn_small     |     0.3    |      0.9996 |         12480 |    2603 |     0.9885 |     0.9626 |
| cnn_small     |     0.2    |      0.9999 |          8321 |    2048 |     0.9919 |     0.9759 |
| cnn_small     |     0.1003 |      1      |          4172 |    1302 |     0.9919 |     0.9764 |
| probe         |     1      |      0.3385 |         41600 |    4160 |     0.8181 |     0.749  |
| probe         |     0.95   |      0.5204 |         39520 |    4138 |     0.8386 |     0.7689 |
| probe         |     0.9    |      0.5959 |         37440 |    4071 |     0.8579 |     0.7871 |
| probe         |     0.8    |      0.7348 |         33280 |    3815 |     0.8913 |     0.8144 |
| probe         |     0.7    |      0.8509 |         29120 |    3469 |     0.923  |     0.8491 |
| probe         |     0.6    |      0.9239 |         24960 |    3106 |     0.9435 |     0.8776 |
| probe         |     0.5    |      0.9628 |         20800 |    2711 |     0.9584 |     0.8995 |
| probe         |     0.4    |      0.9832 |         16640 |    2296 |     0.9687 |     0.911  |
| probe         |     0.3    |      0.9933 |         12480 |    1845 |     0.9753 |     0.9179 |
| probe         |     0.2    |      0.9979 |          8320 |    1296 |     0.9819 |     0.912  |
| probe         |     0.1    |      0.9996 |          4160 |     734 |     0.9882 |     0.9158 |
| xgboost+probe |     1      |      0.3382 |         41600 |    4160 |     0.8418 |     0.7729 |
| xgboost+probe |     0.95   |      0.4635 |         39520 |    4132 |     0.8624 |     0.7925 |
| xgboost+probe |     0.9    |      0.5076 |         37440 |    4063 |     0.8827 |     0.8136 |
| xgboost+probe |     0.8    |      0.5958 |         33280 |    3782 |     0.9228 |     0.861  |
| xgboost+probe |     0.7    |      0.6865 |         29120 |    3391 |     0.9471 |     0.884  |
| xgboost+probe |     0.6    |      0.7545 |         24960 |    3121 |     0.9583 |     0.8816 |
| xgboost+probe |     0.5    |      0.8024 |         20800 |    2823 |     0.9643 |     0.8772 |
| xgboost+probe |     0.4    |      0.8404 |         16640 |    2514 |     0.9677 |     0.869  |
| xgboost+probe |     0.3    |      0.8679 |         12480 |    2290 |     0.9699 |     0.8651 |
| xgboost+probe |     0.2    |      0.8975 |          8320 |    1750 |     0.9701 |     0.842  |
| xgboost+probe |     0.1    |      0.93   |          4160 |    1134 |     0.9803 |     0.7593 |

### Figures

- `model_comparison.png`
- `per_class_recall.png`
- `coverage.png`
- `ambiguity_native_sample_rate.png`
- `ambiguity_collection_code.png`
- `confusion_xgboost.png`
- `feature_importance_xgboost.png`
- `confusion_cnn.png`
- `training_history_cnn.png`
- `confusion_cnn_small.png`
- `training_history_cnn_small.png`
- `occlusion.png`
- `confusion_logbook.png`
- `feature_importance_logbook.png`
- `confusion_probe.png`
- `training_history_probe.png`
- `confusion_xgboost+probe.png`
- `confusion_metadata.png`
- `feature_importance_metadata.png`

Every figure has a CSV of the same name beside it, or in the model directory.

## KillerWhale, call

Against `calltype_killerwhale_call_context`, which sees everything written down about the recording
and none of the recording. That includes clip duration, which matters here more than
anywhere else: a note is written against a whole cut, so a longer cut is more likely
to carry any given call whatever the animal was doing.

| model                     |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds | family                    |
|:--------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|:--------------------------|
| calltype_killerwhale_call |  0.695 |    0.6618 |   0.0332 | -0.2549 | 0.3213 |     0.818 |         29 |      50 | calltype_killerwhale_call |

| model                             |   estimate |    low |   high |   tapes |
|:----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_call         |     0.7016 | 0.5506 | 0.8309 |      65 |
| calltype_killerwhale_call_context |     0.647  | 0.5023 | 0.7769 |      65 |

## KillerWhale, chirp

Against `calltype_killerwhale_chirp_context`, which sees everything written down about the recording
and none of the recording. That includes clip duration, which matters here more than
anywhere else: a note is written against a whole cut, so a longer cut is more likely
to carry any given call whatever the animal was doing.

| model                      |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds | family                     |
|:---------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|:---------------------------|
| calltype_killerwhale_chirp | 0.5556 |    0.6071 |  -0.0515 | -0.1466 | 0.0435 |     0.281 |         32 |      50 | calltype_killerwhale_chirp |

| model                              |   estimate |    low |   high |   tapes |
|:-----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_chirp         |     0.5865 | 0.525  | 0.6463 |      65 |
| calltype_killerwhale_chirp_context |     0.6208 | 0.4956 | 0.7126 |      65 |

## KillerWhale, click

Against `calltype_killerwhale_click_context`, which sees everything written down about the recording
and none of the recording. That includes clip duration, which matters here more than
anywhere else: a note is written against a whole cut, so a longer cut is more likely
to carry any given call whatever the animal was doing.

| model                      |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds | family                     |
|:---------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|:---------------------------|
| calltype_killerwhale_click |  0.597 |    0.4721 |   0.1249 | -0.0398 | 0.2897 |     0.134 |         37 |      50 | calltype_killerwhale_click |

| model                              |   estimate |    low |   high |   tapes |
|:-----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_click         |     0.6029 | 0.4501 | 0.724  |      65 |
| calltype_killerwhale_click_context |     0.4833 | 0.3926 | 0.5927 |      65 |

## KillerWhale, squeal

Against `calltype_killerwhale_squeal_context`, which sees everything written down about the recording
and none of the recording. That includes clip duration, which matters here more than
anywhere else: a note is written against a whole cut, so a longer cut is more likely
to carry any given call whatever the animal was doing.

| model                       |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds | family                      |
|:----------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|:----------------------------|
| calltype_killerwhale_squeal |  0.769 |    0.7634 |   0.0056 | -0.2236 | 0.2348 |     0.961 |         29 |      50 | calltype_killerwhale_squeal |

| model                               |   estimate |    low |   high |   tapes |
|:------------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_squeal         |     0.7524 | 0.5217 | 0.9223 |      65 |
| calltype_killerwhale_squeal_context |     0.7523 | 0.5623 | 0.9075 |      65 |

## KillerWhale, whistle

Against `calltype_killerwhale_whistle_context`, which sees everything written down about the recording
and none of the recording. That includes clip duration, which matters here more than
anywhere else: a note is written against a whole cut, so a longer cut is more likely
to carry any given call whatever the animal was doing.

| model                        |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds | family                       |
|:-----------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|:-----------------------------|
| calltype_killerwhale_whistle | 0.5671 |    0.5225 |   0.0446 | -0.0232 | 0.1124 |     0.192 |         36 |      50 | calltype_killerwhale_whistle |

| model                                |   estimate |    low |   high |   tapes |
|:-------------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_whistle         |     0.6038 | 0.5077 | 0.643  |      65 |
| calltype_killerwhale_whistle_context |     0.5392 | 0.4706 | 0.5971 |      65 |

## SpermWhale, click

Against `calltype_spermwhale_click_context`, which sees everything written down about the recording
and none of the recording. That includes clip duration, which matters here more than
anywhere else: a note is written against a whole cut, so a longer cut is more likely
to carry any given call whatever the animal was doing.

| model                     |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds | family                    |
|:--------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|:--------------------------|
| calltype_spermwhale_click | 0.7097 |    0.7397 |  -0.0301 | -0.1463 | 0.0862 |     0.606 |         25 |      50 | calltype_spermwhale_click |

| model                             |   estimate |    low |   high |   tapes |
|:----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_spermwhale_click         |     0.7378 | 0.6021 |  0.829 |      58 |
| calltype_spermwhale_click_context |     0.7474 | 0.6078 |  0.836 |      58 |

## SpermWhale, coda

Against `calltype_spermwhale_coda_context`, which sees everything written down about the recording
and none of the recording. That includes clip duration, which matters here more than
anywhere else: a note is written against a whole cut, so a longer cut is more likely
to carry any given call whatever the animal was doing.

| model                              |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family                   |
|:-----------------------------------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:-------------------------|
| calltype_spermwhale_coda           | 0.577  |    0.7392 |  -0.1622 | -0.3015 | -0.0229 |    0.0234 |         46 |      50 | calltype_spermwhale_coda |
| calltype_spermwhale_coda_cnn_small | 0.4533 |    0.6775 |  -0.2242 | -0.6154 |  0.1669 |    0.187  |          5 |       5 | calltype_spermwhale_coda |

| model                              |   estimate |    low |   high |   tapes |
|:-----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_spermwhale_coda           |     0.5522 | 0.4563 | 0.6293 |      38 |
| calltype_spermwhale_coda_cnn_small |     0.4721 | 0.355  | 0.5768 |      38 |
| calltype_spermwhale_coda_context   |     0.6613 | 0.4983 | 0.8122 |      38 |

## SpermWhale, whistle

Against `calltype_spermwhale_whistle_context`, which sees everything written down about the recording
and none of the recording. That includes clip duration, which matters here more than
anywhere else: a note is written against a whole cut, so a longer cut is more likely
to carry any given call whatever the animal was doing.

| model                       |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds | family                      |
|:----------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|:----------------------------|
| calltype_spermwhale_whistle | 0.7129 |    0.6741 |   0.0387 | -0.2676 | 0.3451 |       0.8 |         26 |      50 | calltype_spermwhale_whistle |

| model                               |   estimate |    low |   high |   tapes |
|:------------------------------------|-----------:|-------:|-------:|--------:|
| calltype_spermwhale_whistle         |     0.782  | 0.4868 | 0.8965 |      58 |
| calltype_spermwhale_whistle_context |     0.6649 | 0.4505 | 0.8857 |      58 |
