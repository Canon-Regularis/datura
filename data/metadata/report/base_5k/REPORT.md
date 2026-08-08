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

| family   | model     | floor    |   margin |     low |    high |   p_value |   agreeing |   folds |
|:---------|:----------|:---------|---------:|--------:|--------:|----------:|-----------:|--------:|
| species  | probe     | logbook  |  -0.2737 | -0.3796 | -0.1677 |  4e-06    |         50 |      50 |
| species  | xgboost   | logbook  |  -0.2474 | -0.3732 | -0.1217 |  0.000247 |         50 |      50 |
| species  | probe     | metadata |  -0.1764 | -0.2836 | -0.0692 |  0.00177  |         47 |      50 |
| species  | xgboost   | metadata |  -0.1502 | -0.2658 | -0.0346 |  0.0119   |         46 |      50 |
| species  | cnn_small | logbook  |  -0.2313 | -0.5832 |  0.1205 |  0.142    |          5 |       5 |
| species  | logbook   | metadata |   0.0972 | -0.038  |  0.2325 |  0.155    |         40 |      50 |
| species  | cnn_small | metadata |  -0.1705 | -0.4511 |  0.11   |  0.167    |          5 |       5 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. It is a floor rather than the floor, and the table after this one measures
against the highest any model that hears nothing reaches.

| model     |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:----------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| logbook   | 0.9958 |    0.8986 |   0.0972 | -0.038  |  0.2325 |   0.155   |         40 |      50 | species  |
| xgboost   | 0.7484 |    0.8986 |  -0.1502 | -0.2658 | -0.0346 |   0.0119  |         46 |      50 | species  |
| cnn_small | 0.7563 |    0.9268 |  -0.1705 | -0.4511 |  0.11   |   0.167   |          5 |       5 | species  |
| probe     | 0.7221 |    0.8986 |  -0.1764 | -0.2836 | -0.0692 |   0.00177 |         47 |      50 | species  |

### Margin over logbook, the strongest model that hears no audio

`logbook` also sees the site, the coordinates, the noise conditions and the
collection code the field note opens with. None of that is the animal, so this is the
number an audio result has to clear before it is evidence about whales.

| model     |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds | family   |
|:----------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|:---------|
| cnn_small | 0.7563 |    0.9876 |  -0.2313 | -0.5832 |  0.1205 |  0.142    |          5 |       5 | species  |
| xgboost   | 0.7484 |    0.9958 |  -0.2474 | -0.3732 | -0.1217 |  0.000247 |         50 |      50 | species  |
| probe     | 0.7221 |    0.9958 |  -0.2737 | -0.3796 | -0.1677 |  4e-06    |         50 |      50 | species  |

### Every model, with the range the recordings support

The interval comes from resampling whole tapes with replacement. Cuts from one tape
are near duplicates, so resampling clips would count the same recording many times
and produce an interval several times too narrow.

| model     |   estimate |    low |   high |   tapes |
|:----------|-----------:|-------:|-------:|--------:|
| xgboost   |     0.7763 | 0.6538 | 0.8854 |     136 |
| cnn_small |     0.7366 | 0.5971 | 0.9018 |     136 |
| logbook   |     0.9889 | 0.9711 | 0.9993 |     136 |
| probe     |     0.754  | 0.6426 | 0.836  |     136 |
| metadata  |     0.9224 | 0.8172 | 0.991  |     136 |

### Spread across folds

| model     |   mean |    std |
|:----------|-------:|-------:|
| xgboost   | 0.7484 | 0.1203 |
| cnn_small | 0.7563 | 0.1758 |
| logbook   | 0.9958 | 0.0088 |
| probe     | 0.7221 | 0.1    |
| metadata  | 0.8986 | 0.128  |

### Per species recall

| model     |   HumpbackWhale |   SpermWhale |   KillerWhale |
|:----------|----------------:|-------------:|--------------:|
| xgboost   |          0.6    |       0.852  |        0.8607 |
| cnn_small |          0.6691 |       0.8777 |        0.855  |
| logbook   |          0.9948 |       0.9929 |        0.9997 |
| probe     |          0.7448 |       0.7556 |        0.7901 |
| metadata  |          0.866  |       0.9694 |        0.8849 |

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

| giveaway           | model     | subset                                 |   clips |   classes_scored |   classes_total |   folds |   macro_f1_mean |   macro_f1_std |
|:-------------------|:----------|:---------------------------------------|--------:|-----------------:|----------------:|--------:|----------------:|---------------:|
| native sample rate | xgboost   | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.6117 |         0.1527 |
| native sample rate | xgboost   | native sample rate unique to a species |    3370 |                3 |               3 |      50 |          0.7507 |         0.1781 |
| collection code    | xgboost   | collection code not recorded           |     359 |                2 |               3 |      47 |          0.5819 |         0.2583 |
| collection code    | xgboost   | collection code unique to a species    |    3858 |                3 |               3 |      50 |          0.7411 |         0.1234 |
| native sample rate | cnn_small | native sample rate shared by species   |     847 |                3 |               3 |       5 |          0.669  |         0.1968 |
| native sample rate | cnn_small | native sample rate unique to a species |    3370 |                3 |               3 |       5 |          0.7434 |         0.2192 |
| collection code    | cnn_small | collection code not recorded           |     359 |                2 |               3 |       5 |          0.4898 |         0.0143 |
| collection code    | cnn_small | collection code unique to a species    |    3858 |                3 |               3 |       5 |          0.7641 |         0.1839 |
| native sample rate | logbook   | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.8533 |         0.1632 |
| native sample rate | logbook   | native sample rate unique to a species |    3370 |                3 |               3 |      50 |          0.9989 |         0.0059 |
| collection code    | logbook   | collection code not recorded           |     359 |                2 |               3 |      47 |          0.5847 |         0.2399 |
| collection code    | logbook   | collection code unique to a species    |    3858 |                3 |               3 |      50 |          0.9961 |         0.0089 |
| native sample rate | probe     | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.6718 |         0.1191 |
| native sample rate | probe     | native sample rate unique to a species |    3370 |                3 |               3 |      50 |          0.6846 |         0.1457 |
| collection code    | probe     | collection code not recorded           |     359 |                2 |               3 |      47 |          0.3953 |         0.189  |
| collection code    | probe     | collection code unique to a species    |    3858 |                3 |               3 |      50 |          0.7346 |         0.1035 |
| native sample rate | metadata  | native sample rate shared by species   |     847 |                3 |               3 |      50 |          0.741  |         0.1782 |
| native sample rate | metadata  | native sample rate unique to a species |    3370 |                3 |               3 |      50 |          0.9254 |         0.1208 |
| collection code    | metadata  | collection code not recorded           |     359 |                2 |               3 |      47 |          0.6489 |         0.2311 |
| collection code    | metadata  | collection code unique to a species    |    3858 |                3 |               3 |      50 |          0.8974 |         0.1284 |

### Figures

- `model_comparison.png`
- `per_class_recall.png`
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
- `confusion_metadata.png`
- `feature_importance_metadata.png`

Every figure has a CSV of the same name beside it, or in the model directory.
