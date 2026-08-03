# Results: base_5k

Species: HumpbackWhale, SpermWhale, KillerWhale  
Common band: 0 to 2560 Hz at 5120 Hz  
Folds: 5 per split, grouped by tape  
Windows: 2.0 s, hop 1.0 s, at most 16 per clip  
Families: 1, each a set of models and the control they were measured against

## Every comparison

Columns beside the margin say what the design resolves. `folds` counts every fold of every repeat, so a run of ten repeats over five folds shows 50. `low` and `high` bound the paired difference at 95%, and `p_value` is the corrected resampled test, which accounts for the training data those folds share. `agreeing` counts the folds that pointed the same way as the mean, and is worth reading where the p value settles nothing.

| family   | model     |   margin |     low |    high |   p_value |   agreeing |   folds |
|:---------|:----------|---------:|--------:|--------:|----------:|-----------:|--------:|
| species  | xgboost   |  -0.1518 | -0.2658 | -0.0379 |    0.0101 |         45 |      50 |
| species  | cnn_small |  -0.173  | -0.4889 |  0.1429 |    0.203  |          5 |       5 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. Its score is the floor an audio model has to clear.

| model     |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds |
|:----------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|
| xgboost   | 0.7471 |    0.899  |  -0.1518 | -0.2658 | -0.0379 |    0.0101 |         45 |      50 |
| cnn_small | 0.7538 |    0.9268 |  -0.173  | -0.4889 |  0.1429 |    0.203  |          5 |       5 |

### Every model, with the range the recordings support

The interval comes from resampling whole tapes with replacement. Cuts from one tape
are near duplicates, so resampling clips would count the same recording many times
and produce an interval several times too narrow.

| model     |   estimate |    low |   high |   tapes |
|:----------|-----------:|-------:|-------:|--------:|
| xgboost   |     0.7787 | 0.657  | 0.8845 |     136 |
| cnn_small |     0.7307 | 0.5831 | 0.8967 |     136 |
| metadata  |     0.9224 | 0.8172 | 0.991  |     136 |

### Spread across folds

| model     |   mean |    std |
|:----------|-------:|-------:|
| xgboost   | 0.7471 | 0.1201 |
| cnn_small | 0.7538 | 0.1937 |
| metadata  | 0.899  | 0.1261 |

### Per species recall

| model     |   HumpbackWhale |   SpermWhale |   KillerWhale |
|:----------|----------------:|-------------:|--------------:|
| xgboost   |          0.599  |       0.8517 |        0.8592 |
| cnn_small |          0.6757 |       0.9191 |        0.83   |
| metadata  |          0.8697 |       0.9696 |        0.8839 |

### With and without the equipment giveaway

Test clips split by whether their native sample rate is used by one species or by
several. On the shared rate subset the recording cannot identify the species by
itself, so that column is where audio has to earn its result.

| model     | subset                   |   clips |   macro_f1_mean |   macro_f1_std |
|:----------|:-------------------------|--------:|----------------:|---------------:|
| xgboost   | rate unique to a species |    3370 |          0.7497 |         0.1771 |
| xgboost   | rate shared by species   |     847 |          0.6113 |         0.1568 |
| cnn_small | rate unique to a species |    3370 |          0.7484 |         0.2432 |
| cnn_small | rate shared by species   |     847 |          0.6385 |         0.2115 |
| metadata  | rate unique to a species |    3370 |          0.9253 |         0.1208 |
| metadata  | rate shared by species   |     847 |          0.739  |         0.1783 |

### Figures

- `model_comparison.png`
- `per_class_recall.png`
- `ambiguity_breakdown.png`
- `confusion_xgboost.png`
- `feature_importance_xgboost.png`
- `confusion_cnn_small.png`
- `training_history_cnn_small.png`
- `occlusion.png`
- `confusion_metadata.png`
- `feature_importance_metadata.png`

Every figure has a CSV of the same name beside it, or in the model directory.
