# Results: context_wide_10k

Species: HumpbackWhale, SpermWhale, KillerWhale, Long_FinnedPilotWhale, NorthernRightWhale, SpinnerDolphin, Short_Finned(Pacific)PilotWhale, Beluga_WhiteWhale, WeddellSeal, Walrus, StripedDolphin  
Common band: 0 to 5000 Hz at 10000 Hz  
Folds: 4 per split, grouped by tape  
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
| species  | probe         | logbook  |  -0.6902 | -0.9138 | -0.4667 |   0.00224 |          4 |       4 |
| species  | xgboost+probe | logbook  |  -0.6802 | -0.9243 | -0.4361 |   0.00302 |          4 |       4 |
| species  | xgboost       | logbook  |  -0.7154 | -0.9802 | -0.4506 |   0.00331 |          4 |       4 |
| species  | logbook       | metadata |   0.4794 |  0.0799 |  0.879  |   0.0316  |          4 |       4 |
| species  | xgboost       | metadata |  -0.236  | -0.4386 | -0.0334 |   0.0341  |          4 |       4 |
| species  | xgboost+probe | metadata |  -0.2008 | -0.4192 |  0.0177 |   0.0612  |          4 |       4 |
| species  | probe         | metadata |  -0.2108 | -0.4451 |  0.0235 |   0.0644  |          4 |       4 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. It is a floor rather than the floor, and the table after this one measures
against the highest any model that hears nothing reaches.

| model         |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:--------------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| logbook       | 0.8754 |    0.3959 |   0.4794 |  0.0799 |  0.879  |    0.0316 |          4 |       4 | species  |
| xgboost+probe | 0.1952 |    0.3959 |  -0.2008 | -0.4192 |  0.0177 |    0.0612 |          4 |       4 | species  |
| probe         | 0.1851 |    0.3959 |  -0.2108 | -0.4451 |  0.0235 |    0.0644 |          4 |       4 | species  |
| xgboost       | 0.1599 |    0.3959 |  -0.236  | -0.4386 | -0.0334 |    0.0341 |          4 |       4 | species  |

### Margin over logbook, the strongest model that hears no audio

`logbook` also sees the site, the coordinates, the noise conditions and the
collection code the field note opens with. None of that is the animal, so this is the
number an audio result has to clear before it is evidence about whales.

| model         |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:--------------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| xgboost+probe | 0.1952 |    0.8754 |  -0.6802 | -0.9243 | -0.4361 |   0.00302 |          4 |       4 | species  |
| probe         | 0.1851 |    0.8754 |  -0.6902 | -0.9138 | -0.4667 |   0.00224 |          4 |       4 | species  |
| xgboost       | 0.1599 |    0.8754 |  -0.7154 | -0.9802 | -0.4506 |   0.00331 |          4 |       4 | species  |

### Every model, with the range the recordings support

The interval comes from resampling whole groups with replacement, where the group is
whatever this configuration's folds held out and the `unit` column names it. Cuts from
one recording are near duplicates, so resampling clips would count the same recording
many times and produce an interval several times too narrow. Resampling tapes under a
fold rule that holds out places is the same mistake one level up, and it reported an
interval 59% narrower than the design supports.

| model         |   estimate |    low |   high |   groups | unit   |
|:--------------|-----------:|-------:|-------:|---------:|:-------|
| xgboost       |     0.1597 | 0.1006 | 0.2084 |       68 | place  |
| logbook       |     0.8884 | 0.7742 | 0.962  |       68 | place  |
| probe         |     0.1958 | 0.1329 | 0.2521 |       68 | place  |
| xgboost+probe |     0.2019 | 0.1407 | 0.2595 |       68 | place  |
| metadata      |     0.3808 | 0.2584 | 0.5013 |       68 | place  |

### Spread across folds

| model         |   mean |    std |
|:--------------|-------:|-------:|
| xgboost       | 0.1599 | 0.0326 |
| logbook       | 0.8754 | 0.0953 |
| probe         | 0.191  | 0.0188 |
| xgboost+probe | 0.1952 | 0.0241 |
| metadata      | 0.3959 | 0.0884 |

### Per species recall

| model         |   HumpbackWhale |   SpermWhale |   KillerWhale |   Long_FinnedPilotWhale |   NorthernRightWhale |   SpinnerDolphin |   Short_Finned(Pacific)PilotWhale |   Beluga_WhiteWhale |   WeddellSeal |   Walrus |   StripedDolphin |
|:--------------|----------------:|-------------:|--------------:|------------------------:|---------------------:|-----------------:|----------------------------------:|--------------------:|--------------:|---------:|-----------------:|
| xgboost       |          0.1146 |       0.1415 |        0.1533 |                  0.5272 |               0.2367 |           0.2125 |                            0.061  |              0.3457 |        0.5525 |   0.4414 |           0.028  |
| logbook       |          1      |       0.8898 |        0.8225 |                  0.9956 |               1      |           0.2937 |                            0.7672 |              1      |        1      |   0.9138 |           1      |
| probe         |          0.1799 |       0.1704 |        0.0848 |                  0.3189 |               0.2627 |           0.3465 |                            0.172  |              0.3951 |        0.5507 |   0.5458 |           0.3894 |
| xgboost+probe |          0.1376 |       0.1862 |        0.0822 |                  0.3328 |               0.2924 |           0.3915 |                            0.143  |              0.456  |        0.5686 |   0.5199 |           0.3589 |
| metadata      |          0.2302 |       0.2978 |        0.6722 |                  0.5876 |               0.4263 |           0.5    |                            0      |              0.8155 |        0.8002 |   0.4712 |           0.5976 |

8 of these recordings carry more than one of the classes above, across HumpbackWhale, Long_FinnedPilotWhale, SpermWhale and StripedDolphin. Grouping keeps each tape whole, so none of them crosses a fold boundary, and they still contribute to two recalls apiece: the classes sharing a tape are not scored on independent evidence.

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
| native sample rate | xgboost       | native sample rate shared by species   |    4863 |               11 |              11 |       4 |          0.1809 |         0.0622 |
| native sample rate | xgboost       | native sample rate unique to a species |    2482 |                8 |              11 |       4 |          0.0759 |         0.0725 |
| collection code    | xgboost       | collection code not recorded           |     361 |                3 |              11 |       3 |          0.0516 |         0.0793 |
| collection code    | xgboost       | collection code shared by species      |    2289 |                4 |              11 |       4 |          0.2127 |         0.0861 |
| collection code    | xgboost       | collection code unique to a species    |    4695 |               10 |              11 |       4 |          0.154  |         0.0295 |
| native sample rate | logbook       | native sample rate shared by species   |    4863 |               11 |              11 |       4 |          0.8518 |         0.1369 |
| native sample rate | logbook       | native sample rate unique to a species |    2482 |                8 |              11 |       4 |          0.481  |         0.1762 |
| collection code    | logbook       | collection code not recorded           |     361 |                3 |              11 |       3 |          0.1523 |         0.1685 |
| collection code    | logbook       | collection code shared by species      |    2289 |                4 |              11 |       4 |          0.6608 |         0.1095 |
| collection code    | logbook       | collection code unique to a species    |    4695 |               10 |              11 |       4 |          0.8262 |         0.0788 |
| native sample rate | probe         | native sample rate shared by species   |    4863 |               11 |              11 |      40 |          0.2258 |         0.0189 |
| native sample rate | probe         | native sample rate unique to a species |    2482 |                8 |              11 |      40 |          0.0548 |         0.025  |
| collection code    | probe         | collection code not recorded           |     361 |                3 |              11 |      30 |          0.0149 |         0.012  |
| collection code    | probe         | collection code shared by species      |    2289 |                4 |              11 |      40 |          0.2138 |         0.067  |
| collection code    | probe         | collection code unique to a species    |    4695 |               10 |              11 |      40 |          0.1916 |         0.0486 |
| native sample rate | xgboost+probe | native sample rate shared by species   |    4863 |               11 |              11 |       4 |          0.225  |         0.0121 |
| native sample rate | xgboost+probe | native sample rate unique to a species |    2482 |                8 |              11 |       4 |          0.0736 |         0.0447 |
| collection code    | xgboost+probe | collection code not recorded           |     361 |                3 |              11 |       3 |          0.0142 |         0.0145 |
| collection code    | xgboost+probe | collection code shared by species      |    2289 |                4 |              11 |       4 |          0.222  |         0.0809 |
| collection code    | xgboost+probe | collection code unique to a species    |    4695 |               10 |              11 |       4 |          0.193  |         0.0579 |
| native sample rate | metadata      | native sample rate shared by species   |    4863 |               11 |              11 |       4 |          0.3647 |         0.1238 |
| native sample rate | metadata      | native sample rate unique to a species |    2482 |                8 |              11 |       4 |          0.2537 |         0.1744 |
| collection code    | metadata      | collection code not recorded           |     361 |                3 |              11 |       3 |          0.3333 |         0      |
| collection code    | metadata      | collection code shared by species      |    2289 |                4 |              11 |       4 |          0.3211 |         0.2234 |
| collection code    | metadata      | collection code unique to a species    |    4695 |               10 |              11 |       4 |          0.3851 |         0.0401 |

### Accuracy against coverage

Predictions ranked by the probability of the class the model chose, then cut at a
threshold. `coverage` is the share kept, and the row at 1.0 is the score reported
everywhere else. Nothing is refitted: this reads the held out probabilities the
cross validation already wrote.

| model         |   coverage |   threshold |   predictions |   clips |   accuracy |   macro_f1 |
|:--------------|-----------:|------------:|--------------:|--------:|-----------:|-----------:|
| xgboost       |     1      |      0.0923 |          7345 |    7345 |     0.2138 |     0.1597 |
| xgboost       |     0.9499 |      0.1052 |          6977 |    6977 |     0.2184 |     0.1625 |
| xgboost       |     0.8999 |      0.1114 |          6610 |    6610 |     0.2204 |     0.1618 |
| xgboost       |     0.8    |      0.1212 |          5876 |    5876 |     0.2132 |     0.1635 |
| xgboost       |     0.6999 |      0.1332 |          5141 |    5141 |     0.2116 |     0.1636 |
| xgboost       |     0.6    |      0.1425 |          4407 |    4407 |     0.179  |     0.1536 |
| xgboost       |     0.5001 |      0.1577 |          3673 |    3673 |     0.1704 |     0.154  |
| xgboost       |     0.4    |      0.1772 |          2938 |    2938 |     0.1658 |     0.1518 |
| xgboost       |     0.3001 |      0.204  |          2204 |    2204 |     0.1937 |     0.1809 |
| xgboost       |     0.2    |      0.2602 |          1469 |    1469 |     0.1981 |     0.1797 |
| xgboost       |     0.1001 |      0.3637 |           735 |     735 |     0.1197 |     0.147  |
| probe         |     1      |      0.1259 |         73450 |    7345 |     0.1883 |     0.2018 |
| probe         |     0.95   |      0.2261 |         69780 |    7138 |     0.192  |     0.2047 |
| probe         |     0.9    |      0.2544 |         66107 |    6871 |     0.1984 |     0.2089 |
| probe         |     0.8    |      0.3086 |         58761 |    6199 |     0.2103 |     0.2193 |
| probe         |     0.7001 |      0.3592 |         51421 |    5478 |     0.2226 |     0.2305 |
| probe         |     0.6001 |      0.4102 |         44079 |    4743 |     0.2337 |     0.2428 |
| probe         |     0.5001 |      0.4619 |         36734 |    4002 |     0.2468 |     0.2573 |
| probe         |     0.4001 |      0.524  |         29386 |    3249 |     0.26   |     0.2715 |
| probe         |     0.3001 |      0.6059 |         22041 |    2476 |     0.2693 |     0.2871 |
| probe         |     0.2001 |      0.7149 |         14699 |    1676 |     0.268  |     0.3031 |
| probe         |     0.1001 |      0.8696 |          7350 |     815 |     0.2429 |     0.2914 |
| xgboost+probe |     1      |      0.108  |          7345 |    7345 |     0.182  |     0.2019 |
| xgboost+probe |     0.9499 |      0.1565 |          6977 |    6977 |     0.1855 |     0.2045 |
| xgboost+probe |     0.8999 |      0.1695 |          6610 |    6610 |     0.1915 |     0.2084 |
| xgboost+probe |     0.8    |      0.1996 |          5876 |    5876 |     0.2017 |     0.2154 |
| xgboost+probe |     0.6999 |      0.2287 |          5141 |    5141 |     0.2153 |     0.2293 |
| xgboost+probe |     0.6    |      0.2587 |          4407 |    4407 |     0.2262 |     0.2422 |
| xgboost+probe |     0.5001 |      0.2901 |          3673 |    3673 |     0.2374 |     0.2545 |
| xgboost+probe |     0.4    |      0.3225 |          2938 |    2938 |     0.2495 |     0.2684 |
| xgboost+probe |     0.3001 |      0.366  |          2204 |    2204 |     0.2627 |     0.283  |
| xgboost+probe |     0.2    |      0.4225 |          1469 |    1469 |     0.2689 |     0.2964 |
| xgboost+probe |     0.1001 |      0.518  |           735 |     735 |     0.2327 |     0.266  |

### Figures

- `model_comparison.png`
- `per_class_recall.png`
- `coverage.png`
- `ambiguity_native_sample_rate.png`
- `ambiguity_collection_code.png`
- `confusion_xgboost.png`
- `feature_importance_xgboost.png`
- `confusion_logbook.png`
- `feature_importance_logbook.png`
- `confusion_probe.png`
- `training_history_probe.png`
- `confusion_xgboost+probe.png`
- `confusion_metadata.png`
- `feature_importance_metadata.png`

Every figure has a CSV of the same name beside it, or in the model directory.
