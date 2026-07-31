# Results: base_5k

Species: HumpbackWhale, SpermWhale, KillerWhale  
Common band: 0 to 2560 Hz at 5120 Hz  
Folds: 5, grouped by tape  
Windows: 2.0 s, hop 1.0 s, at most 16 per clip

## Margin over the metadata control

The control sees native sample rate, year, clip duration and file size, and no audio.
Its score is the floor an audio model has to clear.

| model     |   mean |    std |   control |   margin |
|:----------|-------:|-------:|----------:|---------:|
| xgboost   | 0.7747 | 0.1155 |    0.9268 |  -0.1521 |
| cnn_small | 0.7538 | 0.1937 |    0.9268 |  -0.173  |

## All models

| model     |   mean |    std |
|:----------|-------:|-------:|
| xgboost   | 0.7747 | 0.1155 |
| cnn_small | 0.7538 | 0.1937 |
| metadata  | 0.9268 | 0.1135 |

## Per species recall

| model     |   HumpbackWhale |   SpermWhale |   KillerWhale |
|:----------|----------------:|-------------:|--------------:|
| xgboost   |          0.597  |       0.9162 |          0.87 |
| cnn_small |          0.6757 |       0.9191 |          0.83 |
| metadata  |          0.963  |       0.9872 |          0.88 |

## With and without the equipment giveaway

Test clips split by whether their native sample rate is used by one species or
several. On the shared-rate subset the recording cannot identify the species by
itself, so that column is where audio has to earn its result.

| model     | subset                   |   clips |   macro_f1_mean |   macro_f1_std |
|:----------|:-------------------------|--------:|----------------:|---------------:|
| xgboost   | rate unique to a species |    3370 |          0.789  |         0.2039 |
| xgboost   | rate shared by species   |     847 |          0.6619 |         0.2042 |
| cnn_small | rate unique to a species |    3370 |          0.7484 |         0.2432 |
| cnn_small | rate shared by species   |     847 |          0.6385 |         0.2115 |
| metadata  | rate unique to a species |    3370 |          0.9395 |         0.1344 |
| metadata  | rate shared by species   |     847 |          0.7715 |         0.2251 |

## Figures

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

Every figure has a CSV of the same name beside it or in the model directory.
