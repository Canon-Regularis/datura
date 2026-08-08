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

| family   | model   | floor    |   margin |     low |    high |   p_value |   agreeing |   folds |
|:---------|:--------|:---------|---------:|--------:|--------:|----------:|-----------:|--------:|
| species  | xgboost | logbook  |  -0.4946 | -0.5775 | -0.4118 |  3.42e-16 |         50 |      50 |
| species  | probe   | logbook  |  -0.4827 | -0.5691 | -0.3962 |  3.8e-15  |         50 |      50 |
| species  | logbook | metadata |   0.3452 |  0.2475 |  0.4429 |  4.64e-09 |         50 |      50 |
| species  | xgboost | metadata |  -0.1494 | -0.2377 | -0.0611 |  0.00134  |         48 |      50 |
| species  | probe   | metadata |  -0.1375 | -0.273  | -0.0019 |  0.047    |         43 |      50 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. It is a floor rather than the floor, and the table after this one measures
against the highest any model that hears nothing reaches.

| model   |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:--------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| logbook | 0.9198 |    0.5746 |   0.3452 |  0.2475 |  0.4429 |  4.64e-09 |         50 |      50 | species  |
| probe   | 0.4372 |    0.5746 |  -0.1375 | -0.273  | -0.0019 |  0.047    |         43 |      50 | species  |
| xgboost | 0.4252 |    0.5746 |  -0.1494 | -0.2377 | -0.0611 |  0.00134  |         48 |      50 | species  |

### Margin over logbook, the strongest model that hears no audio

`logbook` also sees the site, the coordinates, the noise conditions and the
collection code the field note opens with. None of that is the animal, so this is the
number an audio result has to clear before it is evidence about whales.

| model   |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:--------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| probe   | 0.4372 |    0.9198 |  -0.4827 | -0.5691 | -0.3962 |  3.8e-15  |         50 |      50 | species  |
| xgboost | 0.4252 |    0.9198 |  -0.4946 | -0.5775 | -0.4118 |  3.42e-16 |         50 |      50 | species  |

### Every model, with the range the recordings support

The interval comes from resampling whole tapes with replacement. Cuts from one tape
are near duplicates, so resampling clips would count the same recording many times
and produce an interval several times too narrow.

| model    |   estimate |    low |   high |   tapes |
|:---------|-----------:|-------:|-------:|--------:|
| xgboost  |     0.3951 | 0.3184 | 0.4692 |     228 |
| logbook  |     0.9442 | 0.8822 | 0.9692 |     228 |
| probe    |     0.4669 | 0.3771 | 0.5335 |     228 |
| metadata |     0.5837 | 0.4802 | 0.6708 |     228 |

### Spread across folds

| model    |   mean |    std |
|:---------|-------:|-------:|
| xgboost  | 0.4252 | 0.1009 |
| logbook  | 0.9198 | 0.0863 |
| probe    | 0.4372 | 0.0732 |
| metadata | 0.5746 | 0.1185 |

### Per species recall

| model    |   HumpbackWhale |   SpermWhale |   KillerWhale |   Long_FinnedPilotWhale |   NorthernRightWhale |   SpinnerDolphin |   Short_Finned(Pacific)PilotWhale |   Beluga_WhiteWhale |   WeddellSeal |   Walrus |   StripedDolphin |
|:---------|----------------:|-------------:|--------------:|------------------------:|---------------------:|-----------------:|----------------------------------:|--------------------:|--------------:|---------:|-----------------:|
| xgboost  |          0.2532 |       0.5467 |        0.8391 |                  0.5384 |               0.5965 |           0.7481 |                            0.0457 |              0.47   |        0.6978 |   0.6013 |           0.1723 |
| logbook  |          1      |       0.9139 |        0.981  |                  0.9626 |               1      |           0.8759 |                            0.8    |              0.8683 |        1      |   0.9929 |           0.7648 |
| probe    |          0.3257 |       0.5044 |        0.7777 |                  0.5186 |               0.69   |           0.7986 |                            0.0725 |              0.4694 |        0.703  |   0.6461 |           0.3033 |
| metadata |          0.4859 |       0.5719 |        0.7587 |                  0.7397 |               0.8493 |           0.9362 |                            0.0701 |              0.8522 |        0.8642 |   0.6244 |           0.2739 |

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

| giveaway           | model    | subset                                 |   clips |   classes_scored |   classes_total |   folds |   macro_f1_mean |   macro_f1_std |
|:-------------------|:---------|:---------------------------------------|--------:|-----------------:|----------------:|--------:|----------------:|---------------:|
| native sample rate | xgboost  | native sample rate shared by species   |    5241 |               11 |              11 |      50 |          0.4195 |         0.1057 |
| native sample rate | xgboost  | native sample rate unique to a species |    2482 |                8 |              11 |      50 |          0.261  |         0.0924 |
| collection code    | xgboost  | collection code not recorded           |     361 |                3 |              11 |      46 |          0.365  |         0.1818 |
| collection code    | xgboost  | collection code shared by species      |    2581 |                4 |              11 |      50 |          0.3083 |         0.0742 |
| collection code    | xgboost  | collection code unique to a species    |    4781 |               10 |              11 |      50 |          0.4187 |         0.1015 |
| native sample rate | logbook  | native sample rate shared by species   |    5241 |               11 |              11 |      50 |          0.8992 |         0.0937 |
| native sample rate | logbook  | native sample rate unique to a species |    2482 |                8 |              11 |      50 |          0.4864 |         0.1154 |
| collection code    | logbook  | collection code not recorded           |     361 |                3 |              11 |      46 |          0.4336 |         0.2082 |
| collection code    | logbook  | collection code shared by species      |    2581 |                4 |              11 |      50 |          0.6656 |         0.1211 |
| collection code    | logbook  | collection code unique to a species    |    4781 |               10 |              11 |      50 |          0.7929 |         0.0758 |
| native sample rate | probe    | native sample rate shared by species   |    5241 |               11 |              11 |      50 |          0.437  |         0.0885 |
| native sample rate | probe    | native sample rate unique to a species |    2482 |                8 |              11 |      50 |          0.2783 |         0.1007 |
| collection code    | probe    | collection code not recorded           |     361 |                3 |              11 |      46 |          0.3105 |         0.1586 |
| collection code    | probe    | collection code shared by species      |    2581 |                4 |              11 |      50 |          0.3257 |         0.0658 |
| collection code    | probe    | collection code unique to a species    |    4781 |               10 |              11 |      50 |          0.4375 |         0.0739 |
| native sample rate | metadata | native sample rate shared by species   |    5241 |               11 |              11 |      50 |          0.545  |         0.1312 |
| native sample rate | metadata | native sample rate unique to a species |    2482 |                8 |              11 |      50 |          0.3729 |         0.1433 |
| collection code    | metadata | collection code not recorded           |     361 |                3 |              11 |      46 |          0.4029 |         0.2138 |
| collection code    | metadata | collection code shared by species      |    2581 |                4 |              11 |      50 |          0.3684 |         0.0674 |
| collection code    | metadata | collection code unique to a species    |    4781 |               10 |              11 |      50 |          0.541  |         0.1144 |

### Figures

- `model_comparison.png`
- `per_class_recall.png`
- `ambiguity_native_sample_rate.png`
- `ambiguity_collection_code.png`
- `confusion_xgboost.png`
- `feature_importance_xgboost.png`
- `confusion_logbook.png`
- `feature_importance_logbook.png`
- `confusion_probe.png`
- `training_history_probe.png`
- `confusion_metadata.png`
- `feature_importance_metadata.png`

Every figure has a CSV of the same name beside it, or in the model directory.
