# Results: context_10k

Species: HumpbackWhale, SpermWhale, KillerWhale  
Common band: 0 to 5000 Hz at 10000 Hz  
Folds: 5 per split, grouped by tape  
Windows: 2.0 s, hop 1.0 s, at most 16 per clip  
Families: 1, each a set of models and the control they were measured against

Every p value in this document is uncorrected for the number of comparisons reported.
`MULTIPLICITY.md` beside this file adjusts across every comparison in every
configuration at once, which is the number to read before calling one of them a
finding.

## Every comparison

Columns beside the margin say what the design resolves. `folds` counts every fold of every repeat, so a run of ten repeats over five folds shows 50. `low` and `high` bound the paired difference at 95%, and `p_value` is the corrected resampled test, which accounts for the training data those folds share. `agreeing` counts the folds that pointed the same way as the mean, and is worth reading where the p value settles nothing.

| family   | model           | floor    |   margin |     low |    high |   p_value |   agreeing |   folds |
|:---------|:----------------|:---------|---------:|--------:|--------:|----------:|-----------:|--------:|
| species  | xgboost         | logbook  |  -0.6602 | -0.7804 | -0.54   |  0.000108 |          5 |       5 |
| species  | xgboost+probe   | logbook  |  -0.5412 | -0.7329 | -0.3496 |  0.00143  |          5 |       5 |
| species  | probe           | logbook  |  -0.5371 | -0.7309 | -0.3434 |  0.00153  |          5 |       5 |
| species  | xgboost_centred | logbook  |  -0.4358 | -0.7359 | -0.1357 |  0.0157   |          5 |       5 |
| species  | logbook         | metadata |   0.3603 | -0.0625 |  0.783  |  0.0771   |          5 |       5 |
| species  | xgboost         | metadata |  -0.2999 | -0.661  |  0.0611 |  0.0824   |          5 |       5 |
| species  | xgboost+probe   | metadata |  -0.181  | -0.6273 |  0.2654 |  0.323    |          4 |       5 |
| species  | probe           | metadata |  -0.1769 | -0.6486 |  0.2948 |  0.357    |          4 |       5 |
| species  | xgboost_centred | metadata |  -0.0755 | -0.6128 |  0.4617 |  0.716    |          3 |       5 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. It is a floor rather than the floor, and the table after this one measures
against the highest any model that hears nothing reaches.

| model           |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds | family   |
|:----------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|:---------|
| logbook         | 0.9815 |    0.6212 |   0.3603 | -0.0625 | 0.783  |    0.0771 |          5 |       5 | species  |
| xgboost_centred | 0.5457 |    0.6212 |  -0.0755 | -0.6128 | 0.4617 |    0.716  |          3 |       5 | species  |
| probe           | 0.4444 |    0.6212 |  -0.1769 | -0.6486 | 0.2948 |    0.357  |          4 |       5 | species  |
| xgboost+probe   | 0.4403 |    0.6212 |  -0.181  | -0.6273 | 0.2654 |    0.323  |          4 |       5 | species  |
| xgboost         | 0.3213 |    0.6212 |  -0.2999 | -0.661  | 0.0611 |    0.0824 |          5 |       5 | species  |

### Margin over logbook, the strongest model that hears no audio

`logbook` also sees the site, the coordinates, the noise conditions and the
collection code the field note opens with. None of that is the animal, so this is the
number an audio result has to clear before it is evidence about whales.

| model           |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:----------------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| xgboost_centred | 0.5457 |    0.9815 |  -0.4358 | -0.7359 | -0.1357 |  0.0157   |          5 |       5 | species  |
| probe           | 0.4444 |    0.9815 |  -0.5371 | -0.7309 | -0.3434 |  0.00153  |          5 |       5 | species  |
| xgboost+probe   | 0.4403 |    0.9815 |  -0.5412 | -0.7329 | -0.3496 |  0.00143  |          5 |       5 | species  |
| xgboost         | 0.3213 |    0.9815 |  -0.6602 | -0.7804 | -0.54   |  0.000108 |          5 |       5 | species  |

### Every model, with the range the recordings support

The interval comes from resampling whole groups with replacement, where the group is
whatever this configuration's folds held out and the `unit` column names it. Cuts from
one recording are near duplicates, so resampling clips would count the same recording
many times and produce an interval several times too narrow. Resampling tapes under a
fold rule that holds out places is the same mistake one level up, and it reported an
interval 59% narrower than the design supports.

| model           |   estimate |    low |   high |   groups | unit   |
|:----------------|-----------:|-------:|-------:|---------:|:-------|
| xgboost         |     0.317  | 0.2117 | 0.4877 |       24 | place  |
| logbook         |     0.9878 | 0.9438 | 0.9989 |       24 | place  |
| probe           |     0.4446 | 0.3163 | 0.5816 |       24 | place  |
| xgboost_centred |     0.5431 | 0.4334 | 0.6557 |       24 | place  |
| xgboost+probe   |     0.4308 | 0.3061 | 0.5799 |       24 | place  |
| metadata        |     0.6059 | 0.3554 | 0.9346 |       24 | place  |

### Spread across folds

| model           |   mean |    std |
|:----------------|-------:|-------:|
| xgboost         | 0.3213 | 0.0655 |
| logbook         | 0.9815 | 0.0184 |
| probe           | 0.4444 | 0.0965 |
| xgboost_centred | 0.5457 | 0.1431 |
| xgboost+probe   | 0.4403 | 0.0959 |
| metadata        | 0.6212 | 0.2179 |

### Per species recall

| model           |   HumpbackWhale |   SpermWhale |   KillerWhale |
|:----------------|----------------:|-------------:|--------------:|
| xgboost         |          0.4094 |       0.5948 |        0.4078 |
| logbook         |          1      |       0.9241 |        0.9983 |
| probe           |          0.6117 |       0.5889 |        0.4767 |
| xgboost_centred |          0.3031 |       0.8071 |        0.7052 |
| xgboost+probe   |          0.6544 |       0.6001 |        0.4812 |
| metadata        |          0.8334 |       0.9669 |        0.4116 |

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

| giveaway           | model           | subset                                 |   clips |   classes_scored |   classes_total |   folds |   macro_f1_mean |   macro_f1_std |
|:-------------------|:----------------|:---------------------------------------|--------:|-----------------:|----------------:|--------:|----------------:|---------------:|
| native sample rate | xgboost         | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.3597 |         0.1515 |
| native sample rate | xgboost         | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.2609 |         0.1615 |
| collection code    | xgboost         | collection code not recorded           |     359 |                2 |               3 |      30 |          0.4838 |         0.3526 |
| collection code    | xgboost         | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.3517 |         0.089  |
| native sample rate | logbook         | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.6598 |         0.2131 |
| native sample rate | logbook         | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.847  |         0.1671 |
| collection code    | logbook         | collection code not recorded           |     359 |                2 |               3 |      30 |          0.54   |         0.3665 |
| collection code    | logbook         | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.9898 |         0.0152 |
| native sample rate | probe           | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.476  |         0.1148 |
| native sample rate | probe           | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.3561 |         0.1018 |
| collection code    | probe           | collection code not recorded           |     359 |                2 |               3 |      30 |          0.2766 |         0.1086 |
| collection code    | probe           | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.459  |         0.1036 |
| native sample rate | xgboost_centred | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.4538 |         0.1577 |
| native sample rate | xgboost_centred | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.4993 |         0.1354 |
| collection code    | xgboost_centred | collection code not recorded           |     359 |                2 |               3 |      30 |          0.3356 |         0.1183 |
| collection code    | xgboost_centred | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.5672 |         0.1437 |
| native sample rate | xgboost+probe   | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.4797 |         0.1164 |
| native sample rate | xgboost+probe   | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.3545 |         0.1158 |
| collection code    | xgboost+probe   | collection code not recorded           |     359 |                2 |               3 |      30 |          0.2744 |         0.1115 |
| collection code    | xgboost+probe   | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.4581 |         0.1053 |
| native sample rate | metadata        | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.3244 |         0.2061 |
| native sample rate | metadata        | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.6358 |         0.3602 |
| collection code    | metadata        | collection code not recorded           |     359 |                2 |               3 |      30 |          0.6664 |         0.2399 |
| collection code    | metadata        | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.6098 |         0.217  |

### Accuracy against coverage

Predictions ranked by the probability of the class the model chose, then cut at a
threshold. `coverage` is the share kept, and the row at 1.0 is the score reported
everywhere else. Nothing is refitted: this reads the held out probabilities the
cross validation already wrote.

| model           |   coverage |   threshold |   predictions |   clips |   accuracy |   macro_f1 |
|:----------------|-----------:|------------:|--------------:|--------:|-----------:|-----------:|
| xgboost         |     1      |      0.3335 |         40880 |    4088 |     0.3598 |     0.317  |
| xgboost         |     0.9501 |      0.3348 |         38840 |    3884 |     0.3682 |     0.3231 |
| xgboost         |     0.9002 |      0.3431 |         36800 |    3680 |     0.3674 |     0.3187 |
| xgboost         |     0.8001 |      0.3549 |         32710 |    3271 |     0.3662 |     0.3168 |
| xgboost         |     0.7013 |      0.3648 |         28670 |    2867 |     0.354  |     0.3076 |
| xgboost         |     0.6013 |      0.3769 |         24580 |    2458 |     0.3775 |     0.3299 |
| xgboost         |     0.5    |      0.3867 |         20440 |    2044 |     0.385  |     0.341  |
| xgboost         |     0.4002 |      0.3966 |         16360 |    1636 |     0.4309 |     0.3849 |
| xgboost         |     0.3001 |      0.4081 |         12270 |    1227 |     0.4719 |     0.4282 |
| xgboost         |     0.2001 |      0.4466 |          8180 |     818 |     0.4951 |     0.4429 |
| xgboost         |     0.1    |      0.4988 |          4090 |     409 |     0.4719 |     0.4356 |
| probe           |     1      |      0.3386 |         40880 |    4088 |     0.4738 |     0.4446 |
| probe           |     0.9501 |      0.4052 |         38840 |    3884 |     0.4779 |     0.4503 |
| probe           |     0.9002 |      0.4356 |         36800 |    3680 |     0.4837 |     0.459  |
| probe           |     0.8001 |      0.4779 |         32710 |    3271 |     0.4907 |     0.4722 |
| probe           |     0.7001 |      0.5164 |         28620 |    2862 |     0.501  |     0.4871 |
| probe           |     0.6    |      0.5654 |         24530 |    2453 |     0.5137 |     0.504  |
| probe           |     0.5    |      0.6248 |         20440 |    2044 |     0.5279 |     0.5219 |
| probe           |     0.4002 |      0.6972 |         16360 |    1636 |     0.5336 |     0.528  |
| probe           |     0.3001 |      0.774  |         12270 |    1227 |     0.533  |     0.529  |
| probe           |     0.2001 |      0.8554 |          8180 |     818 |     0.5244 |     0.522  |
| probe           |     0.1    |      0.9316 |          4090 |     409 |     0.5575 |     0.5432 |
| xgboost_centred |     1      |      0.3348 |         40880 |    4088 |     0.627  |     0.5431 |
| xgboost_centred |     0.9501 |      0.3498 |         38840 |    3884 |     0.6367 |     0.5549 |
| xgboost_centred |     0.9058 |      0.3545 |         37030 |    3703 |     0.6238 |     0.5506 |
| xgboost_centred |     0.8001 |      0.4025 |         32710 |    3271 |     0.6172 |     0.5577 |
| xgboost_centred |     0.7001 |      0.4661 |         28620 |    2862 |     0.6363 |     0.5797 |
| xgboost_centred |     0.6    |      0.5169 |         24530 |    2453 |     0.6563 |     0.5994 |
| xgboost_centred |     0.5    |      0.5512 |         20440 |    2044 |     0.6678 |     0.6165 |
| xgboost_centred |     0.4002 |      0.6433 |         16360 |    1636 |     0.7048 |     0.6381 |
| xgboost_centred |     0.3001 |      0.7493 |         12270 |    1227 |     0.7702 |     0.6748 |
| xgboost_centred |     0.2001 |      0.8574 |          8180 |     818 |     0.846  |     0.7067 |
| xgboost_centred |     0.1    |      0.9414 |          4090 |     409 |     0.9022 |     0.759  |
| xgboost+probe   |     1      |      0.3387 |         40880 |    4088 |     0.4609 |     0.4308 |
| xgboost+probe   |     0.9501 |      0.3712 |         38840 |    3884 |     0.4658 |     0.4375 |
| xgboost+probe   |     0.9002 |      0.3864 |         36800 |    3680 |     0.4726 |     0.4463 |
| xgboost+probe   |     0.8001 |      0.4077 |         32710 |    3271 |     0.4864 |     0.4646 |
| xgboost+probe   |     0.7001 |      0.4291 |         28620 |    2862 |     0.5028 |     0.4851 |
| xgboost+probe   |     0.6    |      0.4543 |         24530 |    2453 |     0.5165 |     0.5016 |
| xgboost+probe   |     0.5    |      0.4862 |         20440 |    2044 |     0.5377 |     0.5232 |
| xgboost+probe   |     0.4002 |      0.5241 |         16360 |    1636 |     0.5538 |     0.5373 |
| xgboost+probe   |     0.3001 |      0.5671 |         12270 |    1227 |     0.5623 |     0.5407 |
| xgboost+probe   |     0.2001 |      0.6054 |          8180 |     818 |     0.5917 |     0.5542 |
| xgboost+probe   |     0.1    |      0.6546 |          4090 |     409 |     0.6504 |     0.5503 |

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
- `confusion_xgboost_centred.png`
- `feature_importance_xgboost_centred.png`
- `confusion_xgboost+probe.png`
- `confusion_metadata.png`
- `feature_importance_metadata.png`

Every figure has a CSV of the same name beside it, or in the model directory.
