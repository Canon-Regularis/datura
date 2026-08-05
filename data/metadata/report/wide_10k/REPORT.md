# Results: wide_10k

Species: HumpbackWhale, SpermWhale, KillerWhale, Long_FinnedPilotWhale, NorthernRightWhale, SpinnerDolphin, Short_Finned(Pacific)PilotWhale, Beluga_WhiteWhale, WeddellSeal, Walrus, StripedDolphin  
Common band: 0 to 5000 Hz at 10000 Hz  
Folds: 5 per split, grouped by tape  
Windows: 2.0 s, hop 1.0 s, at most 16 per clip  
Families: 1, each a set of models and the control they were measured against

## Every comparison

Columns beside the margin say what the design resolves. `folds` counts every fold of every repeat, so a run of ten repeats over five folds shows 50. `low` and `high` bound the paired difference at 95%, and `p_value` is the corrected resampled test, which accounts for the training data those folds share. `agreeing` counts the folds that pointed the same way as the mean, and is worth reading where the p value settles nothing.

| family   | model   |   margin |     low |    high |   p_value |   agreeing |   folds |
|:---------|:--------|---------:|--------:|--------:|----------:|-----------:|--------:|
| species  | logbook |   0.3427 |  0.2455 |  0.4399 |    0      |         50 |      50 |
| species  | xgboost |  -0.1534 | -0.2396 | -0.0672 |    0.0008 |         48 |      50 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. Its score is the floor an audio model has to clear.

| model   |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds |
|:--------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|
| logbook | 0.9201 |    0.5774 |   0.3427 |  0.2455 |  0.4399 |    0      |         50 |      50 |
| xgboost | 0.424  |    0.5774 |  -0.1534 | -0.2396 | -0.0672 |    0.0008 |         48 |      50 |

### Margin over logbook, the strongest model that hears no audio

`logbook` also sees the site, the coordinates, the noise conditions and the
collection code the field note opens with. None of that is the animal, so this is the
number an audio result has to clear before it is evidence about whales.

| model   |   mean |   control |   margin |    low |    high |   p_value |   agreeing |   folds |
|:--------|-------:|----------:|---------:|-------:|--------:|----------:|-----------:|--------:|
| xgboost |  0.424 |    0.9201 |  -0.4961 | -0.576 | -0.4163 |         0 |         50 |      50 |

### Every model, with the range the recordings support

The interval comes from resampling whole tapes with replacement. Cuts from one tape
are near duplicates, so resampling clips would count the same recording many times
and produce an interval several times too narrow.

| model    |   estimate |    low |   high |   tapes |
|:---------|-----------:|-------:|-------:|--------:|
| xgboost  |     0.4347 | 0.3402 | 0.5115 |     228 |
| logbook  |     0.9409 | 0.8733 | 0.9681 |     228 |
| metadata |     0.5844 | 0.4816 | 0.6723 |     228 |

### Spread across folds

| model    |   mean |    std |
|:---------|-------:|-------:|
| xgboost  | 0.424  | 0.1006 |
| logbook  | 0.9201 | 0.0844 |
| metadata | 0.5774 | 0.1195 |

### Per species recall

| model    |   HumpbackWhale |   SpermWhale |   KillerWhale |   Long_FinnedPilotWhale |   NorthernRightWhale |   SpinnerDolphin |   Short_Finned(Pacific)PilotWhale |   Beluga_WhiteWhale |   WeddellSeal |   Walrus |   StripedDolphin |
|:---------|----------------:|-------------:|--------------:|------------------------:|---------------------:|-----------------:|----------------------------------:|--------------------:|--------------:|---------:|-----------------:|
| xgboost  |          0.2604 |       0.5438 |        0.8397 |                  0.535  |               0.5926 |           0.7473 |                            0.0364 |              0.4737 |        0.6877 |   0.5857 |           0.1788 |
| logbook  |          1      |       0.9132 |        0.981  |                  0.9629 |               1      |           0.8756 |                            0.8    |              0.8683 |        1      |   0.9929 |           0.7621 |
| metadata |          0.4948 |       0.5711 |        0.7536 |                  0.7414 |               0.8535 |           0.9362 |                            0.0707 |              0.8505 |        0.8652 |   0.6238 |           0.2877 |

8 of these recordings carry more than one of the classes above, across HumpbackWhale, Long_FinnedPilotWhale, SpermWhale and StripedDolphin. Grouping keeps each tape whole, so none of them crosses a fold boundary, and they still contribute to two recalls apiece: the classes sharing a tape are not scored on independent evidence.

### With and without the giveaway

Test clips split by whether their native sample rate or their collection code is used by one species or
by several. On the shared subset the recording cannot identify the species by itself,
so those rows are where audio has to earn its result.

| giveaway           | model    | subset                                 |   clips |   macro_f1_mean |   macro_f1_std |
|:-------------------|:---------|:---------------------------------------|--------:|----------------:|---------------:|
| native sample rate | xgboost  | native sample rate unique to a species |    2482 |          0.1894 |         0.0651 |
| native sample rate | xgboost  | native sample rate shared by species   |    5241 |          0.4175 |         0.1048 |
| collection code    | xgboost  | collection code unique to a species    |    4781 |          0.3803 |         0.0931 |
| collection code    | xgboost  | collection code shared by species      |    2942 |          0.166  |         0.0382 |
| native sample rate | logbook  | native sample rate unique to a species |    2482 |          0.3544 |         0.0838 |
| native sample rate | logbook  | native sample rate shared by species   |    5241 |          0.8993 |         0.0918 |
| collection code    | logbook  | collection code unique to a species    |    4781 |          0.7208 |         0.0685 |
| collection code    | logbook  | collection code shared by species      |    2942 |          0.3106 |         0.0483 |
| native sample rate | metadata | native sample rate unique to a species |    2482 |          0.271  |         0.1044 |
| native sample rate | metadata | native sample rate shared by species   |    5241 |          0.5489 |         0.1331 |
| collection code    | metadata | collection code unique to a species    |    4781 |          0.495  |         0.1053 |
| collection code    | metadata | collection code shared by species      |    2942 |          0.1992 |         0.053  |

### Figures

- `model_comparison.png`
- `per_class_recall.png`
- `ambiguity_native_sample_rate.png`
- `ambiguity_collection_code.png`
- `confusion_xgboost.png`
- `feature_importance_xgboost.png`
- `confusion_logbook.png`
- `feature_importance_logbook.png`
- `confusion_metadata.png`
- `feature_importance_metadata.png`

Every figure has a CSV of the same name beside it, or in the model directory.
