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

| family   | model   | floor    |   margin |     low |    high |   p_value |   agreeing |   folds |
|:---------|:--------|:---------|---------:|--------:|--------:|----------:|-----------:|--------:|
| species  | xgboost | logbook  |  -0.3611 | -0.5607 | -0.1615 |  0.000666 |         50 |      50 |
| species  | xgboost | metadata |  -0.2569 | -0.4878 | -0.0261 |  0.0299   |         40 |      50 |
| species  | logbook | metadata |   0.1042 | -0.0327 |  0.241  |  0.133    |         40 |      50 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. It is a floor rather than the floor, and the table after this one measures
against the highest any model that hears nothing reaches.

| model   |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:--------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| logbook | 0.9305 |    0.8263 |   0.1042 | -0.0327 |  0.241  |    0.133  |         40 |      50 | species  |
| xgboost | 0.5694 |    0.8263 |  -0.2569 | -0.4878 | -0.0261 |    0.0299 |         40 |      50 | species  |

### Margin over logbook, the strongest model that hears no audio

`logbook` also sees the site, the coordinates, the noise conditions and the
collection code the field note opens with. None of that is the animal, so this is the
number an audio result has to clear before it is evidence about whales.

| model   |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:--------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| xgboost | 0.5694 |    0.9305 |  -0.3611 | -0.5607 | -0.1615 |  0.000666 |         50 |      50 | species  |

### Every model, with the range the recordings support

The interval comes from resampling whole tapes with replacement. Cuts from one tape
are near duplicates, so resampling clips would count the same recording many times
and produce an interval several times too narrow.

| model    |   estimate |    low |   high |   tapes |
|:---------|-----------:|-------:|-------:|--------:|
| xgboost  |     0.6201 | 0.4935 | 0.772  |     129 |
| logbook  |     0.9565 | 0.8367 | 0.9978 |     129 |
| metadata |     0.9207 | 0.8241 | 0.9838 |     129 |

### Spread across folds

| model    |   mean |    std |
|:---------|-------:|-------:|
| xgboost  | 0.5694 | 0.1437 |
| logbook  | 0.9305 | 0.1138 |
| metadata | 0.8263 | 0.1619 |

### Per species recall

| model    |   HumpbackWhale |   SpermWhale |   KillerWhale |
|:---------|----------------:|-------------:|--------------:|
| xgboost  |          0.7392 |       0.8233 |        0.5898 |
| logbook  |          0.9988 |       0.9652 |        0.8681 |
| metadata |          0.777  |       0.9315 |        0.9086 |

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

| giveaway           | model    | subset                                 |   clips |   classes_scored |   classes_total |   folds |   macro_f1_mean |   macro_f1_std |
|:-------------------|:---------|:---------------------------------------|--------:|-----------------:|----------------:|--------:|----------------:|---------------:|
| native sample rate | xgboost  | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.5434 |         0.1925 |
| native sample rate | xgboost  | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.5025 |         0.1605 |
| collection code    | xgboost  | collection code not recorded           |     359 |                2 |               3 |      30 |          0.3589 |         0.2187 |
| collection code    | xgboost  | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.6203 |         0.1944 |
| native sample rate | logbook  | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.772  |         0.2887 |
| native sample rate | logbook  | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.7944 |         0.2654 |
| collection code    | logbook  | collection code not recorded           |     359 |                2 |               3 |      30 |          0.5    |         0.4152 |
| collection code    | logbook  | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.9886 |         0.0231 |
| native sample rate | metadata | native sample rate shared by species   |     834 |                3 |               3 |      50 |          0.6289 |         0.2605 |
| native sample rate | metadata | native sample rate unique to a species |    3254 |                3 |               3 |      50 |          0.7863 |         0.2252 |
| collection code    | metadata | collection code not recorded           |     359 |                2 |               3 |      30 |          0.6118 |         0.2874 |
| collection code    | metadata | collection code unique to a species    |    3729 |                3 |               3 |      50 |          0.8444 |         0.1405 |

### Accuracy against coverage

Predictions ranked by the probability of the class the model chose, then cut at a
threshold. `coverage` is the share kept, and the row at 1.0 is the score reported
everywhere else. Nothing is refitted: this reads the held out probabilities the
cross validation already wrote.

| model   |   coverage |   threshold |   predictions |   clips |   accuracy |   macro_f1 |
|:--------|-----------:|------------:|--------------:|--------:|-----------:|-----------:|
| xgboost |     1      |      0.3336 |         40880 |    4088 |     0.7023 |     0.6218 |
| xgboost |     0.9502 |      0.421  |         38845 |    3920 |     0.7276 |     0.6452 |
| xgboost |     0.9001 |      0.4705 |         36795 |    3752 |     0.741  |     0.6564 |
| xgboost |     0.8    |      0.541  |         32705 |    3336 |     0.7633 |     0.6773 |
| xgboost |     0.7002 |      0.5982 |         28625 |    2923 |     0.771  |     0.6684 |
| xgboost |     0.6002 |      0.6632 |         24535 |    2508 |     0.7781 |     0.6394 |
| xgboost |     0.5    |      0.7236 |         20440 |    2105 |     0.7737 |     0.5694 |
| xgboost |     0.4002 |      0.8434 |         16360 |    1636 |     0.7848 |     0.4823 |
| xgboost |     0.3001 |      0.9629 |         12270 |    1227 |     0.8077 |     0.3953 |
| xgboost |     0.2001 |      0.9955 |          8180 |     818 |     0.8863 |     0.5102 |
| xgboost |     0.1    |      0.9995 |          4090 |     409 |     0.9927 |     0.3321 |

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
- `confusion_metadata.png`
- `feature_importance_metadata.png`

Every figure has a CSV of the same name beside it, or in the model directory.
