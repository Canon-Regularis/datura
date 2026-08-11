# Results: base_5k

Species: HumpbackWhale, SpermWhale, KillerWhale  
Common band: 0 to 2560 Hz at 5120 Hz  
Folds: 5 per split, grouped by tape  
Windows: 2.0 s, hop 1.0 s, at most 16 per clip  
Families: 1, each a set of models and the control they were measured against

Every p value in this document is uncorrected for the number of comparisons reported.
`MULTIPLICITY.md` beside this file adjusts across every comparison in every
configuration at once, which is the number to read before calling one of them a
finding.

## Every comparison

Columns beside the margin say what the design resolves. `folds` counts every fold of every repeat, so a run of ten repeats over five folds shows 50. `low` and `high` bound the paired difference at 95%, and `p_value` is the corrected resampled test, which accounts for the training data those folds share. `agreeing` counts the folds that pointed the same way as the mean, and is worth reading where the p value settles nothing.

| family   | model         | floor    |   margin |     low |    high |   p_value |   agreeing |   folds |
|:---------|:--------------|:---------|---------:|--------:|--------:|----------:|-----------:|--------:|
| species  | probe         | logbook  |  -0.2737 | -0.3796 | -0.1677 |  4e-06    |         50 |      50 |
| species  | xgboost+probe | logbook  |  -0.2363 | -0.3521 | -0.1205 |  0.000155 |         50 |      50 |
| species  | xgboost       | logbook  |  -0.2474 | -0.3732 | -0.1217 |  0.000247 |         50 |      50 |
| species  | probe         | metadata |  -0.1764 | -0.2836 | -0.0692 |  0.00177  |         47 |      50 |
| species  | xgboost       | metadata |  -0.1502 | -0.2658 | -0.0346 |  0.0119   |         46 |      50 |
| species  | xgboost+probe | metadata |  -0.1391 | -0.2541 | -0.024  |  0.0189   |         45 |      50 |
| species  | cnn_small     | logbook  |  -0.2313 | -0.5832 |  0.1205 |  0.142    |          5 |       5 |
| species  | logbook       | metadata |   0.0972 | -0.038  |  0.2325 |  0.155    |         40 |      50 |
| species  | cnn_small     | metadata |  -0.1705 | -0.4511 |  0.11   |  0.167    |          5 |       5 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. It is a floor rather than the floor, and the table after this one measures
against the highest any model that hears nothing reaches.

| model         |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:--------------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| logbook       | 0.9958 |    0.8986 |   0.0972 | -0.038  |  0.2325 |   0.155   |         40 |      50 | species  |
| xgboost+probe | 0.7595 |    0.8986 |  -0.1391 | -0.2541 | -0.024  |   0.0189  |         45 |      50 | species  |
| xgboost       | 0.7484 |    0.8986 |  -0.1502 | -0.2658 | -0.0346 |   0.0119  |         46 |      50 | species  |
| cnn_small     | 0.7563 |    0.9268 |  -0.1705 | -0.4511 |  0.11   |   0.167   |          5 |       5 | species  |
| probe         | 0.7221 |    0.8986 |  -0.1764 | -0.2836 | -0.0692 |   0.00177 |         47 |      50 | species  |

### Margin over logbook, the strongest model that hears no audio

`logbook` also sees the site, the coordinates, the noise conditions and the
collection code the field note opens with. None of that is the animal, so this is the
number an audio result has to clear before it is evidence about whales.

| model         |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:--------------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| cnn_small     | 0.7563 |    0.9876 |  -0.2313 | -0.5832 |  0.1205 |  0.142    |          5 |       5 | species  |
| xgboost+probe | 0.7595 |    0.9958 |  -0.2363 | -0.3521 | -0.1205 |  0.000155 |         50 |      50 | species  |
| xgboost       | 0.7484 |    0.9958 |  -0.2474 | -0.3732 | -0.1217 |  0.000247 |         50 |      50 | species  |
| probe         | 0.7221 |    0.9958 |  -0.2737 | -0.3796 | -0.1677 |  4e-06    |         50 |      50 | species  |

### Every model, with the range the recordings support

The interval comes from resampling whole tapes with replacement. Cuts from one tape
are near duplicates, so resampling clips would count the same recording many times
and produce an interval several times too narrow.

| model         |   estimate |    low |   high |   tapes |
|:--------------|-----------:|-------:|-------:|--------:|
| xgboost       |     0.7763 | 0.6538 | 0.8854 |     136 |
| cnn_small     |     0.7366 | 0.5971 | 0.9018 |     136 |
| logbook       |     0.9889 | 0.9711 | 0.9993 |     136 |
| probe         |     0.754  | 0.6426 | 0.836  |     136 |
| xgboost+probe |     0.7925 | 0.6732 | 0.8807 |     136 |
| metadata      |     0.9224 | 0.8172 | 0.991  |     136 |

### Spread across folds

| model         |   mean |    std |
|:--------------|-------:|-------:|
| xgboost       | 0.7484 | 0.1203 |
| cnn_small     | 0.7563 | 0.1758 |
| logbook       | 0.9958 | 0.0088 |
| probe         | 0.7221 | 0.1    |
| xgboost+probe | 0.7595 | 0.1094 |
| metadata      | 0.8986 | 0.128  |

### Per species recall

| model         |   HumpbackWhale |   SpermWhale |   KillerWhale |
|:--------------|----------------:|-------------:|--------------:|
| xgboost       |          0.6    |       0.852  |        0.8607 |
| cnn_small     |          0.6691 |       0.8777 |        0.855  |
| logbook       |          0.9948 |       0.9929 |        0.9997 |
| probe         |          0.7448 |       0.7556 |        0.7901 |
| xgboost+probe |          0.7568 |       0.8155 |        0.8251 |
| metadata      |          0.866  |       0.9694 |        0.8849 |

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
| native sample rate | xgboost       | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.6117 |         0.1527 |
| native sample rate | xgboost       | native sample rate unique to a species |    3370 |                3 |               3 |      50 |          0.7507 |         0.1781 |
| collection code    | xgboost       | collection code not recorded           |     359 |                2 |               3 |      47 |          0.5819 |         0.2583 |
| collection code    | xgboost       | collection code unique to a species    |    3858 |                3 |               3 |      50 |          0.7411 |         0.1234 |
| native sample rate | cnn_small     | native sample rate shared by species   |     847 |                3 |               3 |       5 |          0.669  |         0.1968 |
| native sample rate | cnn_small     | native sample rate unique to a species |    3370 |                3 |               3 |       5 |          0.7434 |         0.2192 |
| collection code    | cnn_small     | collection code not recorded           |     359 |                2 |               3 |       5 |          0.4898 |         0.0143 |
| collection code    | cnn_small     | collection code unique to a species    |    3858 |                3 |               3 |       5 |          0.7641 |         0.1839 |
| native sample rate | logbook       | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.8533 |         0.1632 |
| native sample rate | logbook       | native sample rate unique to a species |    3370 |                3 |               3 |      50 |          0.9989 |         0.0059 |
| collection code    | logbook       | collection code not recorded           |     359 |                2 |               3 |      47 |          0.5847 |         0.2399 |
| collection code    | logbook       | collection code unique to a species    |    3858 |                3 |               3 |      50 |          0.9961 |         0.0089 |
| native sample rate | probe         | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.6718 |         0.1191 |
| native sample rate | probe         | native sample rate unique to a species |    3370 |                3 |               3 |      50 |          0.6846 |         0.1457 |
| collection code    | probe         | collection code not recorded           |     359 |                2 |               3 |      47 |          0.3953 |         0.189  |
| collection code    | probe         | collection code unique to a species    |    3858 |                3 |               3 |      50 |          0.7346 |         0.1035 |
| native sample rate | xgboost+probe | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.6936 |         0.1171 |
| native sample rate | xgboost+probe | native sample rate unique to a species |    3370 |                3 |               3 |      50 |          0.7229 |         0.1579 |
| collection code    | xgboost+probe | collection code not recorded           |     359 |                2 |               3 |      47 |          0.4933 |         0.1884 |
| collection code    | xgboost+probe | collection code unique to a species    |    3858 |                3 |               3 |      50 |          0.7646 |         0.1102 |
| native sample rate | metadata      | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.741  |         0.1782 |
| native sample rate | metadata      | native sample rate unique to a species |    3370 |                3 |               3 |      50 |          0.9254 |         0.1208 |
| collection code    | metadata      | collection code not recorded           |     359 |                2 |               3 |      47 |          0.6489 |         0.2311 |
| collection code    | metadata      | collection code unique to a species    |    3858 |                3 |               3 |      50 |          0.8974 |         0.1284 |

### Accuracy against coverage

Predictions ranked by the probability of the class the model chose, then cut at a
threshold. `coverage` is the share kept, and the row at 1.0 is the score reported
everywhere else. Nothing is refitted: this reads the held out probabilities the
cross validation already wrote.

| model         |   coverage |   threshold |   predictions |   clips |   accuracy |   macro_f1 |
|:--------------|-----------:|------------:|--------------:|--------:|-----------:|-----------:|
| xgboost       |     1      |      0.3338 |         42170 |    4217 |     0.8175 |     0.7498 |
| xgboost       |     0.95   |      0.4279 |         40061 |    4212 |     0.8361 |     0.7696 |
| xgboost       |     0.9    |      0.469  |         37953 |    4187 |     0.8512 |     0.7849 |
| xgboost       |     0.8    |      0.5365 |         33736 |    4018 |     0.8727 |     0.804  |
| xgboost       |     0.7    |      0.5858 |         29519 |    3839 |     0.8835 |     0.8134 |
| xgboost       |     0.6    |      0.613  |         25302 |    3702 |     0.8814 |     0.8122 |
| xgboost       |     0.5    |      0.6527 |         21085 |    3510 |     0.879  |     0.8063 |
| xgboost       |     0.4    |      0.7028 |         16868 |    2903 |     0.8774 |     0.7974 |
| xgboost       |     0.3    |      0.7689 |         12651 |    2423 |     0.8883 |     0.7889 |
| xgboost       |     0.2    |      0.8193 |          8434 |    1881 |     0.8941 |     0.797  |
| xgboost       |     0.1    |      0.8741 |          4217 |    1206 |     0.9134 |     0.8664 |
| cnn_small     |     1      |      0.387  |          4217 |    4217 |     0.8074 |     0.7366 |
| cnn_small     |     0.95   |      0.63   |          4006 |    4006 |     0.8278 |     0.7538 |
| cnn_small     |     0.8999 |      0.7554 |          3795 |    3795 |     0.8472 |     0.7676 |
| cnn_small     |     0.7999 |      0.9101 |          3373 |    3373 |     0.8639 |     0.7752 |
| cnn_small     |     0.7    |      0.9685 |          2952 |    2952 |     0.8777 |     0.7739 |
| cnn_small     |     0.6    |      0.9868 |          2530 |    2530 |     0.8889 |     0.7744 |
| cnn_small     |     0.5001 |      0.994  |          2109 |    2109 |     0.9    |     0.7858 |
| cnn_small     |     0.4    |      0.9977 |          1687 |    1687 |     0.9146 |     0.793  |
| cnn_small     |     0.3    |      0.9994 |          1265 |    1265 |     0.9383 |     0.8234 |
| cnn_small     |     0.2001 |      0.9999 |           844 |     844 |     0.9716 |     0.8703 |
| cnn_small     |     0.1005 |      1      |           424 |     424 |     1      |     1      |
| probe         |     1      |      0.3408 |         42170 |    4217 |     0.7719 |     0.7193 |
| probe         |     0.95   |      0.5185 |         40061 |    4199 |     0.7877 |     0.7325 |
| probe         |     0.9    |      0.5815 |         37953 |    4153 |     0.8023 |     0.7437 |
| probe         |     0.8    |      0.6959 |         33736 |    3957 |     0.8305 |     0.761  |
| probe         |     0.7    |      0.7998 |         29519 |    3648 |     0.8595 |     0.7778 |
| probe         |     0.6    |      0.8814 |         25302 |    3272 |     0.8859 |     0.791  |
| probe         |     0.5    |      0.9381 |         21085 |    2847 |     0.9116 |     0.8    |
| probe         |     0.4    |      0.9726 |         16868 |    2362 |     0.9365 |     0.8001 |
| probe         |     0.3    |      0.9902 |         12651 |    1899 |     0.9599 |     0.8303 |
| probe         |     0.2    |      0.9973 |          8434 |    1404 |     0.974  |     0.821  |
| probe         |     0.1    |      0.9996 |          4217 |     790 |     0.9893 |     0.8005 |
| xgboost+probe |     1      |      0.3356 |         42170 |    4217 |     0.8077 |     0.7553 |
| xgboost+probe |     0.95   |      0.4691 |         40061 |    4189 |     0.8243 |     0.7701 |
| xgboost+probe |     0.9    |      0.5096 |         37953 |    4140 |     0.8434 |     0.7881 |
| xgboost+probe |     0.8    |      0.5796 |         33736 |    3930 |     0.8808 |     0.82   |
| xgboost+probe |     0.7    |      0.6546 |         29519 |    3573 |     0.9053 |     0.83   |
| xgboost+probe |     0.6    |      0.7233 |         25302 |    3205 |     0.9235 |     0.8348 |
| xgboost+probe |     0.5    |      0.7742 |         21085 |    2868 |     0.9345 |     0.8341 |
| xgboost+probe |     0.4    |      0.804  |         16868 |    2655 |     0.9372 |     0.8269 |
| xgboost+probe |     0.3    |      0.8302 |         12651 |    2370 |     0.9349 |     0.8269 |
| xgboost+probe |     0.2    |      0.8781 |          8434 |    1637 |     0.9478 |     0.8448 |
| xgboost+probe |     0.1    |      0.9223 |          4217 |    1085 |     0.9642 |     0.8927 |

### Figures

- `model_comparison.png`
- `per_class_recall.png`
- `coverage.png`
- `ambiguity_native_sample_rate.png`
- `ambiguity_collection_code.png`
- `confusion_xgboost.png`
- `feature_importance_xgboost.png`
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
