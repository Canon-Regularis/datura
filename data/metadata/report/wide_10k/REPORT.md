# Results: wide_10k

Species: HumpbackWhale, SpermWhale, KillerWhale, Long_FinnedPilotWhale, NorthernRightWhale, SpinnerDolphin, Short_Finned(Pacific)PilotWhale, Beluga_WhiteWhale, WeddellSeal, Walrus, StripedDolphin  
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

| family   | model         | floor    |   margin |     low |    high |   p_value |   agreeing |   folds |
|:---------|:--------------|:---------|---------:|--------:|--------:|----------:|-----------:|--------:|
| species  | probe         | logbook  |  -0.5359 | -0.6074 | -0.4645 |  5.02e-20 |         50 |      50 |
| species  | xgboost+probe | logbook  |  -0.5153 | -0.5906 | -0.44   |  1.91e-18 |         50 |      50 |
| species  | xgboost       | logbook  |  -0.5479 | -0.647  | -0.4487 |  5.55e-15 |         50 |      50 |
| species  | logbook       | metadata |   0.3985 |  0.2801 |  0.5169 |  1.54e-08 |         50 |      50 |
| species  | xgboost       | metadata |  -0.1494 | -0.2377 | -0.0611 |  0.00134  |         48 |      50 |
| species  | probe         | metadata |  -0.1375 | -0.273  | -0.0019 |  0.047    |         43 |      50 |
| species  | xgboost+probe | metadata |  -0.1168 | -0.2357 |  0.0022 |  0.0541   |         43 |      50 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. It is a floor rather than the floor, and the table after this one measures
against the highest any model that hears nothing reaches.

| model         |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:--------------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| logbook       | 0.9731 |    0.5746 |   0.3985 |  0.2801 |  0.5169 |  1.54e-08 |         50 |      50 | species  |
| xgboost+probe | 0.4578 |    0.5746 |  -0.1168 | -0.2357 |  0.0022 |  0.0541   |         43 |      50 | species  |
| probe         | 0.4372 |    0.5746 |  -0.1375 | -0.273  | -0.0019 |  0.047    |         43 |      50 | species  |
| xgboost       | 0.4252 |    0.5746 |  -0.1494 | -0.2377 | -0.0611 |  0.00134  |         48 |      50 | species  |

### Margin over logbook, the strongest model that hears no audio

`logbook` also sees the site, the coordinates, the noise conditions and the
collection code the field note opens with. None of that is the animal, so this is the
number an audio result has to clear before it is evidence about whales.

| model         |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:--------------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| xgboost+probe | 0.4578 |    0.9731 |  -0.5153 | -0.5906 | -0.44   |  1.91e-18 |         50 |      50 | species  |
| probe         | 0.4372 |    0.9731 |  -0.5359 | -0.6074 | -0.4645 |  5.02e-20 |         50 |      50 | species  |
| xgboost       | 0.4252 |    0.9731 |  -0.5479 | -0.647  | -0.4487 |  5.55e-15 |         50 |      50 | species  |

### Every model, with the range the recordings support

The interval comes from resampling whole groups with replacement, where the group is
whatever this configuration's folds held out and the `unit` column names it. Cuts from
one recording are near duplicates, so resampling clips would count the same recording
many times and produce an interval several times too narrow. Resampling tapes under a
fold rule that holds out places is the same mistake one level up, and it reported an
interval 59% narrower than the design supports.

| model         |   estimate |    low |   high |   groups | unit    |
|:--------------|-----------:|-------:|-------:|---------:|:--------|
| xgboost       |     0.3951 | 0.3184 | 0.4692 |      228 | tape_id |
| logbook       |     0.975  | 0.9392 | 0.9878 |      228 | tape_id |
| probe         |     0.4669 | 0.3771 | 0.5335 |      228 | tape_id |
| xgboost+probe |     0.4836 | 0.384  | 0.5551 |      228 | tape_id |
| metadata      |     0.5837 | 0.4802 | 0.6708 |      228 | tape_id |

### Spread across folds

| model         |   mean |    std |
|:--------------|-------:|-------:|
| xgboost       | 0.4252 | 0.1009 |
| logbook       | 0.9731 | 0.0224 |
| probe         | 0.4372 | 0.0732 |
| xgboost+probe | 0.4578 | 0.0792 |
| metadata      | 0.5746 | 0.1185 |

### Per species recall

| model         |   HumpbackWhale |   SpermWhale |   KillerWhale |   Long_FinnedPilotWhale |   NorthernRightWhale |   SpinnerDolphin |   Short_Finned(Pacific)PilotWhale |   Beluga_WhiteWhale |   WeddellSeal |   Walrus |   StripedDolphin |
|:--------------|----------------:|-------------:|--------------:|------------------------:|---------------------:|-----------------:|----------------------------------:|--------------------:|--------------:|---------:|-----------------:|
| xgboost       |          0.2532 |       0.5467 |        0.8391 |                  0.5384 |               0.5965 |           0.7481 |                            0.0457 |              0.47   |        0.6978 |   0.6013 |           0.1723 |
| logbook       |          1      |       0.9175 |        0.9997 |                  0.9889 |               1      |           0.9178 |                            0.913  |              1      |        1      |   0.9286 |           1      |
| probe         |          0.3257 |       0.5044 |        0.7777 |                  0.5186 |               0.69   |           0.7986 |                            0.0725 |              0.4694 |        0.703  |   0.6461 |           0.3033 |
| xgboost+probe |          0.3292 |       0.5366 |        0.8431 |                  0.5345 |               0.6968 |           0.8161 |                            0.0683 |              0.5273 |        0.7531 |   0.6765 |           0.2704 |
| metadata      |          0.4859 |       0.5719 |        0.7587 |                  0.7397 |               0.8493 |           0.9362 |                            0.0701 |              0.8522 |        0.8642 |   0.6244 |           0.2739 |

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
| native sample rate | xgboost       | native sample rate shared by species   |    5241 |               11 |              11 |      50 |          0.4195 |         0.1057 |
| native sample rate | xgboost       | native sample rate unique to a species |    2482 |                8 |              11 |      50 |          0.261  |         0.0924 |
| collection code    | xgboost       | collection code not recorded           |     361 |                3 |              11 |      46 |          0.365  |         0.1818 |
| collection code    | xgboost       | collection code shared by species      |    2581 |                4 |              11 |      50 |          0.3083 |         0.0742 |
| collection code    | xgboost       | collection code unique to a species    |    4781 |               10 |              11 |      50 |          0.4187 |         0.1015 |
| native sample rate | logbook       | native sample rate shared by species   |    5241 |               11 |              11 |      50 |          0.9521 |         0.0487 |
| native sample rate | logbook       | native sample rate unique to a species |    2482 |                8 |              11 |      50 |          0.491  |         0.11   |
| collection code    | logbook       | collection code not recorded           |     361 |                3 |              11 |      46 |          0.4198 |         0.224  |
| collection code    | logbook       | collection code shared by species      |    2581 |                4 |              11 |      50 |          0.7228 |         0.0206 |
| collection code    | logbook       | collection code unique to a species    |    4781 |               10 |              11 |      50 |          0.8394 |         0.0521 |
| native sample rate | probe         | native sample rate shared by species   |    5241 |               11 |              11 |      50 |          0.437  |         0.0885 |
| native sample rate | probe         | native sample rate unique to a species |    2482 |                8 |              11 |      50 |          0.2783 |         0.1007 |
| collection code    | probe         | collection code not recorded           |     361 |                3 |              11 |      46 |          0.3105 |         0.1586 |
| collection code    | probe         | collection code shared by species      |    2581 |                4 |              11 |      50 |          0.3257 |         0.0658 |
| collection code    | probe         | collection code unique to a species    |    4781 |               10 |              11 |      50 |          0.4375 |         0.0739 |
| native sample rate | xgboost+probe | native sample rate shared by species   |    5241 |               11 |              11 |      50 |          0.4505 |         0.0951 |
| native sample rate | xgboost+probe | native sample rate unique to a species |    2482 |                8 |              11 |      50 |          0.3055 |         0.1046 |
| collection code    | xgboost+probe | collection code not recorded           |     361 |                3 |              11 |      46 |          0.3712 |         0.1734 |
| collection code    | xgboost+probe | collection code shared by species      |    2581 |                4 |              11 |      50 |          0.3254 |         0.0641 |
| collection code    | xgboost+probe | collection code unique to a species    |    4781 |               10 |              11 |      50 |          0.4548 |         0.0767 |
| native sample rate | metadata      | native sample rate shared by species   |    5241 |               11 |              11 |      50 |          0.545  |         0.1312 |
| native sample rate | metadata      | native sample rate unique to a species |    2482 |                8 |              11 |      50 |          0.3729 |         0.1433 |
| collection code    | metadata      | collection code not recorded           |     361 |                3 |              11 |      46 |          0.4029 |         0.2138 |
| collection code    | metadata      | collection code shared by species      |    2581 |                4 |              11 |      50 |          0.3684 |         0.0674 |
| collection code    | metadata      | collection code unique to a species    |    4781 |               10 |              11 |      50 |          0.541  |         0.1144 |

### Accuracy against coverage

Predictions ranked by the probability of the class the model chose, then cut at a
threshold. `coverage` is the share kept, and the row at 1.0 is the score reported
everywhere else. Nothing is refitted: this reads the held out probabilities the
cross validation already wrote.

| model         |   coverage |   threshold |   predictions |   clips |   accuracy |   macro_f1 |
|:--------------|-----------:|------------:|--------------:|--------:|-----------:|-----------:|
| xgboost       |       1    |      0.095  |         77230 |    7723 |     0.5682 |     0.4201 |
| xgboost       |       0.95 |      0.2099 |         73368 |    7713 |     0.5892 |     0.4328 |
| xgboost       |       0.9  |      0.2511 |         69507 |    7684 |     0.6085 |     0.4434 |
| xgboost       |       0.8  |      0.3268 |         61784 |    7504 |     0.6504 |     0.4633 |
| xgboost       |       0.7  |      0.4072 |         54061 |    7094 |     0.7008 |     0.4819 |
| xgboost       |       0.6  |      0.5085 |         46338 |    6339 |     0.7614 |     0.5091 |
| xgboost       |       0.5  |      0.6372 |         38615 |    5387 |     0.8275 |     0.5485 |
| xgboost       |       0.4  |      0.762  |         30892 |    4421 |     0.8834 |     0.5971 |
| xgboost       |       0.3  |      0.8455 |         23169 |    3677 |     0.9128 |     0.6143 |
| xgboost       |       0.2  |      0.9089 |         15446 |    2835 |     0.9392 |     0.6196 |
| xgboost       |       0.1  |      0.956  |          7723 |    1911 |     0.9648 |     0.6384 |
| probe         |       1    |      0.1615 |         77230 |    7723 |     0.5685 |     0.4409 |
| probe         |       0.95 |      0.335  |         73368 |    7694 |     0.5886 |     0.4563 |
| probe         |       0.9  |      0.3955 |         69507 |    7605 |     0.608  |     0.4708 |
| probe         |       0.8  |      0.4917 |         61784 |    7332 |     0.6484 |     0.5009 |
| probe         |       0.7  |      0.584  |         54061 |    6895 |     0.6907 |     0.5324 |
| probe         |       0.6  |      0.6851 |         46338 |    6224 |     0.7343 |     0.5611 |
| probe         |       0.5  |      0.7892 |         38615 |    5377 |     0.7783 |     0.5925 |
| probe         |       0.4  |      0.8734 |         30892 |    4498 |     0.8205 |     0.6175 |
| probe         |       0.3  |      0.9362 |         23169 |    3530 |     0.8635 |     0.6543 |
| probe         |       0.2  |      0.9749 |         15446 |    2490 |     0.8997 |     0.6752 |
| probe         |       0.1  |      0.9941 |          7723 |    1337 |     0.9433 |     0.6806 |
| xgboost+probe |       1    |      0.138  |         77230 |    7723 |     0.5978 |     0.462  |
| xgboost+probe |       0.95 |      0.258  |         73368 |    7685 |     0.62   |     0.4797 |
| xgboost+probe |       0.9  |      0.3026 |         69507 |    7619 |     0.6424 |     0.4972 |
| xgboost+probe |       0.8  |      0.3762 |         61784 |    7304 |     0.6903 |     0.5355 |
| xgboost+probe |       0.7  |      0.4473 |         54061 |    6730 |     0.7444 |     0.5822 |
| xgboost+probe |       0.6  |      0.5184 |         46338 |    5916 |     0.8018 |     0.6382 |
| xgboost+probe |       0.5  |      0.6067 |         38615 |    5105 |     0.8553 |     0.6677 |
| xgboost+probe |       0.4  |      0.7201 |         30892 |    4099 |     0.9108 |     0.6754 |
| xgboost+probe |       0.3  |      0.8353 |         23169 |    3194 |     0.9513 |     0.6991 |
| xgboost+probe |       0.2  |      0.9107 |         15446 |    2404 |     0.973  |     0.6728 |
| xgboost+probe |       0.1  |      0.9597 |          7723 |    1488 |     0.9873 |     0.6158 |

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
