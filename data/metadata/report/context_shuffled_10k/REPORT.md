# Results: context_shuffled_10k

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
| species  | probe           | metadata |  -0.3039 | -0.5125 | -0.0953 |    0.0155 |          5 |       5 |
| species  | xgboost         | metadata |  -0.3426 | -0.7018 |  0.0166 |    0.0571 |          5 |       5 |
| species  | probe           | logbook  |  -0.3329 | -0.6933 |  0.0274 |    0.0623 |          5 |       5 |
| species  | xgboost+probe   | metadata |  -0.2631 | -0.5557 |  0.0295 |    0.067  |          5 |       5 |
| species  | xgboost         | logbook  |  -0.3716 | -0.8694 |  0.1262 |    0.107  |          4 |       5 |
| species  | xgboost_centred | metadata |  -0.2409 | -0.5761 |  0.0942 |    0.117  |          4 |       5 |
| species  | xgboost+probe   | logbook  |  -0.2922 | -0.7284 |  0.1441 |    0.136  |          4 |       5 |
| species  | xgboost_centred | logbook  |  -0.27   | -0.7178 |  0.1779 |    0.17   |          4 |       5 |
| species  | logbook         | metadata |   0.029  | -0.1844 |  0.2425 |    0.725  |          3 |       5 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. It is a floor rather than the floor, and the table after this one measures
against the highest any model that hears nothing reaches.

| model           |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:----------------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| logbook         | 0.9272 |    0.8982 |   0.029  | -0.1844 |  0.2425 |    0.725  |          3 |       5 | species  |
| xgboost_centred | 0.6572 |    0.8982 |  -0.2409 | -0.5761 |  0.0942 |    0.117  |          4 |       5 | species  |
| xgboost+probe   | 0.635  |    0.8982 |  -0.2631 | -0.5557 |  0.0295 |    0.067  |          5 |       5 | species  |
| probe           | 0.5942 |    0.8982 |  -0.3039 | -0.5125 | -0.0953 |    0.0155 |          5 |       5 | species  |
| xgboost         | 0.5556 |    0.8982 |  -0.3426 | -0.7018 |  0.0166 |    0.0571 |          5 |       5 | species  |

### Margin over logbook, the strongest model that hears no audio

`logbook` also sees the site, the coordinates, the noise conditions and the
collection code the field note opens with. None of that is the animal, so this is the
number an audio result has to clear before it is evidence about whales.

| model           |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds | family   |
|:----------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|:---------|
| xgboost_centred | 0.6572 |    0.9272 |  -0.27   | -0.7178 | 0.1779 |    0.17   |          4 |       5 | species  |
| xgboost+probe   | 0.635  |    0.9272 |  -0.2922 | -0.7284 | 0.1441 |    0.136  |          4 |       5 | species  |
| probe           | 0.5942 |    0.9272 |  -0.3329 | -0.6933 | 0.0274 |    0.0623 |          5 |       5 | species  |
| xgboost         | 0.5556 |    0.9272 |  -0.3716 | -0.8694 | 0.1262 |    0.107  |          4 |       5 | species  |

### Every model, with the range the recordings support

The interval comes from resampling whole groups with replacement, where the group is
whatever this configuration's folds held out and the `unit` column names it. Cuts from
one recording are near duplicates, so resampling clips would count the same recording
many times and produce an interval several times too narrow. Resampling tapes under a
fold rule that holds out places is the same mistake one level up, and it reported an
interval 59% narrower than the design supports.

| model           |   estimate |    low |   high |   groups | unit           |
|:----------------|-----------:|-------:|-------:|---------:|:---------------|
| xgboost         |     0.6005 | 0.387  | 0.7507 |       24 | place_shuffled |
| logbook         |     0.9652 | 0.8281 | 0.9964 |       24 | place_shuffled |
| probe           |     0.6699 | 0.4737 | 0.7819 |       24 | place_shuffled |
| xgboost_centred |     0.7436 | 0.5542 | 0.8349 |       24 | place_shuffled |
| xgboost+probe   |     0.6645 | 0.4797 | 0.8141 |       24 | place_shuffled |
| metadata        |     0.9246 | 0.7481 | 0.9771 |       24 | place_shuffled |

### Spread across folds

| model           |   mean |    std |
|:----------------|-------:|-------:|
| xgboost         | 0.5556 | 0.1621 |
| logbook         | 0.9272 | 0.1108 |
| probe           | 0.5942 | 0.1008 |
| xgboost_centred | 0.6572 | 0.1222 |
| xgboost+probe   | 0.635  | 0.1567 |
| metadata        | 0.8982 | 0.0994 |

### Per species recall

| model           |   HumpbackWhale |   SpermWhale |   KillerWhale |
|:----------------|----------------:|-------------:|--------------:|
| xgboost         |          0.768  |       0.8022 |        0.6324 |
| logbook         |          0.9963 |       0.9908 |        0.879  |
| probe           |          0.7699 |       0.6718 |        0.7798 |
| xgboost_centred |          0.8634 |       0.8843 |        0.7305 |
| xgboost+probe   |          0.8523 |       0.7565 |        0.7814 |
| metadata        |          0.9585 |       0.9565 |        0.9084 |

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
| native sample rate | xgboost         | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.4916 |         0.2016 |
| native sample rate | xgboost         | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.5351 |         0.1776 |
| collection code    | xgboost         | collection code not recorded           |     359 |                2 |               3 |      30 |          0.3976 |         0.1455 |
| collection code    | xgboost         | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.5908 |         0.2057 |
| native sample rate | logbook         | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.6634 |         0.2267 |
| native sample rate | logbook         | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.8572 |         0.173  |
| collection code    | logbook         | collection code not recorded           |     359 |                2 |               3 |      30 |          0.5    |         0.4152 |
| collection code    | logbook         | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.985  |         0.0184 |
| native sample rate | probe           | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.5615 |         0.2275 |
| native sample rate | probe           | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.4859 |         0.0672 |
| collection code    | probe           | collection code not recorded           |     359 |                2 |               3 |      30 |          0.2759 |         0.2408 |
| collection code    | probe           | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.6411 |         0.1531 |
| native sample rate | xgboost_centred | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.5044 |         0.2069 |
| native sample rate | xgboost_centred | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.6622 |         0.086  |
| collection code    | xgboost_centred | collection code not recorded           |     359 |                2 |               3 |      30 |          0.4611 |         0.0117 |
| collection code    | xgboost_centred | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.6641 |         0.1268 |
| native sample rate | xgboost+probe   | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.5545 |         0.2196 |
| native sample rate | xgboost+probe   | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.5376 |         0.0902 |
| collection code    | xgboost+probe   | collection code not recorded           |     359 |                2 |               3 |      30 |          0.3461 |         0.2424 |
| collection code    | xgboost+probe   | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.6824 |         0.2008 |
| native sample rate | metadata        | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.6244 |         0.1805 |
| native sample rate | metadata        | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.8539 |         0.1305 |
| collection code    | metadata        | collection code not recorded           |     359 |                2 |               3 |      30 |          0.6118 |         0.2874 |
| collection code    | metadata        | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.9211 |         0.0844 |

### Accuracy against coverage

Predictions ranked by the probability of the class the model chose, then cut at a
threshold. `coverage` is the share kept, and the row at 1.0 is the score reported
everywhere else. Nothing is refitted: this reads the held out probabilities the
cross validation already wrote.

| model           |   coverage |   threshold |   predictions |   clips |   accuracy |   macro_f1 |
|:----------------|-----------:|------------:|--------------:|--------:|-----------:|-----------:|
| xgboost         |     1      |      0.336  |         40880 |    4088 |     0.6869 |     0.6005 |
| xgboost         |     0.9501 |      0.4342 |         38840 |    3884 |     0.7039 |     0.6145 |
| xgboost         |     0.9002 |      0.4915 |         36800 |    3680 |     0.7177 |     0.6259 |
| xgboost         |     0.8001 |      0.5665 |         32710 |    3271 |     0.7487 |     0.6495 |
| xgboost         |     0.7001 |      0.6505 |         28620 |    2862 |     0.7631 |     0.6431 |
| xgboost         |     0.6    |      0.7125 |         24530 |    2453 |     0.7709 |     0.6275 |
| xgboost         |     0.5    |      0.7679 |         20440 |    2044 |     0.7696 |     0.5734 |
| xgboost         |     0.4002 |      0.8561 |         16360 |    1636 |     0.794  |     0.5428 |
| xgboost         |     0.3001 |      0.9641 |         12270 |    1227 |     0.8101 |     0.4516 |
| xgboost         |     0.2001 |      0.996  |          8180 |     818 |     0.8998 |     0.7379 |
| xgboost         |     0.1    |      0.9995 |          4090 |     409 |     0.9951 |     0.6242 |
| probe           |     1      |      0.3575 |         40880 |    4088 |     0.7268 |     0.6699 |
| probe           |     0.9501 |      0.5084 |         38840 |    3884 |     0.7446 |     0.6827 |
| probe           |     0.9002 |      0.5663 |         36800 |    3680 |     0.7611 |     0.6946 |
| probe           |     0.8001 |      0.6719 |         32710 |    3271 |     0.7903 |     0.7185 |
| probe           |     0.7001 |      0.7654 |         28620 |    2862 |     0.8166 |     0.7369 |
| probe           |     0.6    |      0.8478 |         24530 |    2453 |     0.8398 |     0.7535 |
| probe           |     0.5    |      0.9029 |         20440 |    2044 |     0.8523 |     0.7595 |
| probe           |     0.4002 |      0.9442 |         16360 |    1636 |     0.8637 |     0.7708 |
| probe           |     0.3001 |      0.9725 |         12270 |    1227 |     0.8769 |     0.7783 |
| probe           |     0.2001 |      0.9905 |          8180 |     818 |     0.89   |     0.8054 |
| probe           |     0.1    |      0.9977 |          4090 |     409 |     0.9413 |     0.8767 |
| xgboost_centred |     1      |      0.3356 |         40880 |    4088 |     0.8021 |     0.7436 |
| xgboost_centred |     0.9501 |      0.4921 |         38840 |    3884 |     0.8177 |     0.7553 |
| xgboost_centred |     0.9002 |      0.5352 |         36800 |    3680 |     0.8323 |     0.7667 |
| xgboost_centred |     0.8001 |      0.6322 |         32710 |    3271 |     0.8606 |     0.7915 |
| xgboost_centred |     0.7001 |      0.7136 |         28620 |    2862 |     0.8868 |     0.8138 |
| xgboost_centred |     0.6    |      0.7912 |         24530 |    2453 |     0.9034 |     0.8267 |
| xgboost_centred |     0.5    |      0.8596 |         20440 |    2044 |     0.9256 |     0.8553 |
| xgboost_centred |     0.4002 |      0.9132 |         16360 |    1636 |     0.9419 |     0.8733 |
| xgboost_centred |     0.3001 |      0.952  |         12270 |    1227 |     0.965  |     0.9137 |
| xgboost_centred |     0.2001 |      0.978  |          8180 |     818 |     0.9768 |     0.9261 |
| xgboost_centred |     0.1    |      0.9932 |          4090 |     409 |     0.9902 |     0.9511 |
| xgboost+probe   |     1      |      0.3465 |         40880 |    4088 |     0.7334 |     0.6645 |
| xgboost+probe   |     0.9501 |      0.4659 |         38840 |    3884 |     0.7515 |     0.6746 |
| xgboost+probe   |     0.9002 |      0.5124 |         36800 |    3680 |     0.766  |     0.6832 |
| xgboost+probe   |     0.8001 |      0.5792 |         32710 |    3271 |     0.8001 |     0.7092 |
| xgboost+probe   |     0.7001 |      0.641  |         28620 |    2862 |     0.8319 |     0.7396 |
| xgboost+probe   |     0.6    |      0.7117 |         24530 |    2453 |     0.8516 |     0.7592 |
| xgboost+probe   |     0.5    |      0.7799 |         20440 |    2044 |     0.8596 |     0.7465 |
| xgboost+probe   |     0.4002 |      0.8411 |         16360 |    1636 |     0.8753 |     0.7437 |
| xgboost+probe   |     0.3001 |      0.8999 |         12270 |    1227 |     0.8704 |     0.6932 |
| xgboost+probe   |     0.2001 |      0.9658 |          8180 |     818 |     0.8961 |     0.6711 |
| xgboost+probe   |     0.1    |      0.9947 |          4090 |     409 |     0.9462 |     0.7615 |

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
