# Results: context_shuffled_wide_10k

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
| species  | xgboost+probe | logbook  |  -0.521  | -0.6959 | -0.3461 |   0.00249 |          4 |       4 |
| species  | xgboost       | logbook  |  -0.5705 | -0.7797 | -0.3613 |   0.00322 |          4 |       4 |
| species  | probe         | logbook  |  -0.5419 | -0.7562 | -0.3275 |   0.00401 |          4 |       4 |
| species  | logbook       | metadata |   0.3528 | -0.0352 |  0.7408 |   0.0628  |          4 |       4 |
| species  | xgboost       | metadata |  -0.2177 | -0.6568 |  0.2214 |   0.213   |          4 |       4 |
| species  | probe         | metadata |  -0.1891 | -0.577  |  0.1988 |   0.219   |          4 |       4 |
| species  | xgboost+probe | metadata |  -0.1682 | -0.5632 |  0.2268 |   0.268   |          4 |       4 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. It is a floor rather than the floor, and the table after this one measures
against the highest any model that hears nothing reaches.

| model         |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds | family   |
|:--------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|:---------|
| logbook       | 0.9257 |    0.5729 |   0.3528 | -0.0352 | 0.7408 |    0.0628 |          4 |       4 | species  |
| xgboost+probe | 0.4047 |    0.5729 |  -0.1682 | -0.5632 | 0.2268 |    0.268  |          4 |       4 | species  |
| probe         | 0.3838 |    0.5729 |  -0.1891 | -0.577  | 0.1988 |    0.219  |          4 |       4 | species  |
| xgboost       | 0.3552 |    0.5729 |  -0.2177 | -0.6568 | 0.2214 |    0.213  |          4 |       4 | species  |

### Margin over logbook, the strongest model that hears no audio

`logbook` also sees the site, the coordinates, the noise conditions and the
collection code the field note opens with. None of that is the animal, so this is the
number an audio result has to clear before it is evidence about whales.

| model         |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:--------------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| xgboost+probe | 0.4047 |    0.9257 |  -0.521  | -0.6959 | -0.3461 |   0.00249 |          4 |       4 | species  |
| probe         | 0.3838 |    0.9257 |  -0.5419 | -0.7562 | -0.3275 |   0.00401 |          4 |       4 | species  |
| xgboost       | 0.3552 |    0.9257 |  -0.5705 | -0.7797 | -0.3613 |   0.00322 |          4 |       4 | species  |

### Every model, with the range the recordings support

The interval comes from resampling whole groups with replacement, where the group is
whatever this configuration's folds held out and the `unit` column names it. Cuts from
one recording are near duplicates, so resampling clips would count the same recording
many times and produce an interval several times too narrow. Resampling tapes under a
fold rule that holds out places is the same mistake one level up, and it reported an
interval 59% narrower than the design supports.

| model         |   estimate |    low |   high |   groups | unit           |
|:--------------|-----------:|-------:|-------:|---------:|:---------------|
| xgboost       |     0.3731 | 0.2821 | 0.4325 |       68 | place_shuffled |
| logbook       |     0.9663 | 0.8614 | 0.9823 |       68 | place_shuffled |
| probe         |     0.4002 | 0.3122 | 0.4601 |       68 | place_shuffled |
| xgboost+probe |     0.4291 | 0.3316 | 0.4868 |       68 | place_shuffled |
| metadata      |     0.5673 | 0.4172 | 0.7145 |       68 | place_shuffled |

### Spread across folds

| model         |   mean |    std |
|:--------------|-------:|-------:|
| xgboost       | 0.3552 | 0.0951 |
| logbook       | 0.9257 | 0.0683 |
| probe         | 0.3839 | 0.103  |
| xgboost+probe | 0.4047 | 0.1061 |
| metadata      | 0.5729 | 0.2239 |

### Per species recall

| model         |   HumpbackWhale |   SpermWhale |   KillerWhale |   Long_FinnedPilotWhale |   NorthernRightWhale |   SpinnerDolphin |   Short_Finned(Pacific)PilotWhale |   Beluga_WhiteWhale |   WeddellSeal |   Walrus |   StripedDolphin |
|:--------------|----------------:|-------------:|--------------:|------------------------:|---------------------:|-----------------:|----------------------------------:|--------------------:|--------------:|---------:|-----------------:|
| xgboost       |          0.0976 |       0.631  |        0.7733 |                  0.4086 |               0.6917 |           0.5362 |                            0.0224 |              0.4697 |        0.795  |   0.5029 |           0.127  |
| logbook       |          1      |       0.8509 |        0.7987 |                  0.9977 |               1      |           0.7611 |                            0.8771 |              1      |        1      |   0.9167 |           1      |
| probe         |          0.2763 |       0.44   |        0.79   |                  0.2222 |               0.6489 |           0.7057 |                            0.2271 |              0.5631 |        0.6888 |   0.6056 |           0.4019 |
| xgboost+probe |          0.1554 |       0.571  |        0.8402 |                  0.3251 |               0.7063 |           0.6701 |                            0.141  |              0.5825 |        0.7606 |   0.6022 |           0.3581 |
| metadata      |          0.4774 |       0.4575 |        0.8898 |                  0.5776 |               0.9169 |           0.7761 |                            0.3523 |              0.9053 |        0.9875 |   0.4967 |           0.374  |

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
| native sample rate | xgboost       | native sample rate shared by species   |    4863 |               11 |              11 |       4 |          0.3142 |         0.0729 |
| native sample rate | xgboost       | native sample rate unique to a species |    2482 |                8 |              11 |       4 |          0.2773 |         0.0691 |
| collection code    | xgboost       | collection code not recorded           |     361 |                3 |              11 |       3 |          0.3511 |         0.1952 |
| collection code    | xgboost       | collection code shared by species      |    2289 |                4 |              11 |       4 |          0.2689 |         0.073  |
| collection code    | xgboost       | collection code unique to a species    |    4695 |               10 |              11 |       4 |          0.3569 |         0.1178 |
| native sample rate | logbook       | native sample rate shared by species   |    4863 |               11 |              11 |       4 |          0.9212 |         0.0745 |
| native sample rate | logbook       | native sample rate unique to a species |    2482 |                8 |              11 |       4 |          0.5059 |         0.1184 |
| collection code    | logbook       | collection code not recorded           |     361 |                3 |              11 |       3 |          0.1997 |         0.1199 |
| collection code    | logbook       | collection code shared by species      |    2289 |                4 |              11 |       4 |          0.7142 |         0.0377 |
| collection code    | logbook       | collection code unique to a species    |    4695 |               10 |              11 |       4 |          0.8203 |         0.0967 |
| native sample rate | probe         | native sample rate shared by species   |    4863 |               11 |              11 |      40 |          0.3414 |         0.0702 |
| native sample rate | probe         | native sample rate unique to a species |    2482 |                8 |              11 |      40 |          0.3363 |         0.107  |
| collection code    | probe         | collection code not recorded           |     361 |                3 |              11 |      30 |          0.276  |         0.0224 |
| collection code    | probe         | collection code shared by species      |    2289 |                4 |              11 |      40 |          0.3016 |         0.0403 |
| collection code    | probe         | collection code unique to a species    |    4695 |               10 |              11 |      40 |          0.3969 |         0.1333 |
| native sample rate | xgboost+probe | native sample rate shared by species   |    4863 |               11 |              11 |       4 |          0.3623 |         0.0836 |
| native sample rate | xgboost+probe | native sample rate unique to a species |    2482 |                8 |              11 |       4 |          0.3413 |         0.1069 |
| collection code    | xgboost+probe | collection code not recorded           |     361 |                3 |              11 |       3 |          0.2662 |         0.0862 |
| collection code    | xgboost+probe | collection code shared by species      |    2289 |                4 |              11 |       4 |          0.3124 |         0.0927 |
| collection code    | xgboost+probe | collection code unique to a species    |    4695 |               10 |              11 |       4 |          0.4106 |         0.1551 |
| native sample rate | metadata      | native sample rate shared by species   |    4863 |               11 |              11 |       4 |          0.5217 |         0.1912 |
| native sample rate | metadata      | native sample rate unique to a species |    2482 |                8 |              11 |       4 |          0.3618 |         0.1939 |
| collection code    | metadata      | collection code not recorded           |     361 |                3 |              11 |       3 |          0.2775 |         0.096  |
| collection code    | metadata      | collection code shared by species      |    2289 |                4 |              11 |       4 |          0.3852 |         0.2117 |
| collection code    | metadata      | collection code unique to a species    |    4695 |               10 |              11 |       4 |          0.5132 |         0.2144 |

### Accuracy against coverage

Predictions ranked by the probability of the class the model chose, then cut at a
threshold. `coverage` is the share kept, and the row at 1.0 is the score reported
everywhere else. Nothing is refitted: this reads the held out probabilities the
cross validation already wrote.

| model         |   coverage |   threshold |   predictions |   clips |   accuracy |   macro_f1 |
|:--------------|-----------:|------------:|--------------:|--------:|-----------:|-----------:|
| xgboost       |     1      |      0.116  |          7345 |    7345 |     0.4931 |     0.3731 |
| xgboost       |     0.9499 |      0.1932 |          6977 |    6977 |     0.5095 |     0.3813 |
| xgboost       |     0.8999 |      0.2266 |          6610 |    6610 |     0.5263 |     0.3924 |
| xgboost       |     0.8    |      0.2784 |          5876 |    5876 |     0.5608 |     0.4146 |
| xgboost       |     0.6999 |      0.3291 |          5141 |    5141 |     0.5939 |     0.4298 |
| xgboost       |     0.6    |      0.3904 |          4407 |    4407 |     0.6304 |     0.4413 |
| xgboost       |     0.5001 |      0.4686 |          3673 |    3673 |     0.6836 |     0.4553 |
| xgboost       |     0.4    |      0.5634 |          2938 |    2938 |     0.7434 |     0.4623 |
| xgboost       |     0.3001 |      0.698  |          2204 |    2204 |     0.8122 |     0.5114 |
| xgboost       |     0.2    |      0.8203 |          1469 |    1469 |     0.8611 |     0.5875 |
| xgboost       |     0.1001 |      0.9408 |           735 |     735 |     0.9878 |     0.3818 |
| probe         |     1      |      0.1437 |         73450 |    7345 |     0.4955 |     0.4004 |
| probe         |     0.9501 |      0.2874 |         69784 |    6983 |     0.5125 |     0.4135 |
| probe         |     0.9001 |      0.335  |         66114 |    6619 |     0.5262 |     0.4233 |
| probe         |     0.8001 |      0.4072 |         58764 |    5884 |     0.5578 |     0.4467 |
| probe         |     0.7001 |      0.4808 |         51422 |    5154 |     0.5912 |     0.4719 |
| probe         |     0.6001 |      0.5524 |         44078 |    4419 |     0.626  |     0.4934 |
| probe         |     0.5    |      0.6345 |         36726 |    3683 |     0.661  |     0.5158 |
| probe         |     0.4001 |      0.727  |         29386 |    2944 |     0.6997 |     0.5319 |
| probe         |     0.3    |      0.8246 |         22036 |    2211 |     0.7524 |     0.5667 |
| probe         |     0.2001 |      0.9107 |         14696 |    1474 |     0.7862 |     0.5507 |
| probe         |     0.1    |      0.9753 |          7346 |     740 |     0.8524 |     0.4933 |
| xgboost+probe |     1      |      0.1343 |          7345 |    7345 |     0.5372 |     0.4291 |
| xgboost+probe |     0.9499 |      0.2274 |          6977 |    6977 |     0.5554 |     0.4429 |
| xgboost+probe |     0.8999 |      0.2621 |          6610 |    6610 |     0.5722 |     0.4544 |
| xgboost+probe |     0.8    |      0.3171 |          5876 |    5876 |     0.6031 |     0.4765 |
| xgboost+probe |     0.6999 |      0.3676 |          5141 |    5141 |     0.6382 |     0.5001 |
| xgboost+probe |     0.6    |      0.4222 |          4407 |    4407 |     0.6821 |     0.5244 |
| xgboost+probe |     0.5001 |      0.4749 |          3673 |    3673 |     0.7245 |     0.5523 |
| xgboost+probe |     0.4    |      0.5307 |          2938 |    2938 |     0.7805 |     0.5741 |
| xgboost+probe |     0.3001 |      0.6263 |          2204 |    2204 |     0.8616 |     0.6271 |
| xgboost+probe |     0.2    |      0.7711 |          1469 |    1469 |     0.9312 |     0.5343 |
| xgboost+probe |     0.1001 |      0.9262 |           735 |     735 |     0.985  |     0.4471 |

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
