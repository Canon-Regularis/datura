# Results: base_10k

Species: HumpbackWhale, SpermWhale, KillerWhale  
Common band: 0 to 5000 Hz at 10000 Hz  
Folds: 5, grouped by tape  
Windows: 2.0 s, hop 1.0 s, at most 16 per clip

## Margin over the metadata control

The control sees native sample rate, year, clip duration and file size, and no audio.
Its score is the floor an audio model has to clear.

| model     |   mean |    std |   control |   margin |
|:----------|-------:|-------:|----------:|---------:|
| cnn_small | 0.7438 | 0.1645 |    0.8837 |  -0.1399 |
| xgboost   | 0.7255 | 0.0497 |    0.8837 |  -0.1582 |
| cnn       | 0.6924 | 0.1338 |    0.8837 |  -0.1913 |

## All models

| model     |   mean |    std |
|:----------|-------:|-------:|
| xgboost   | 0.7255 | 0.0497 |
| cnn       | 0.6924 | 0.1338 |
| cnn_small | 0.7438 | 0.1645 |
| metadata  | 0.8837 | 0.139  |

## Per species recall

| model     |   HumpbackWhale |   SpermWhale |   KillerWhale |
|:----------|----------------:|-------------:|--------------:|
| xgboost   |          0.5535 |       0.8433 |        0.8729 |
| cnn       |          0.5295 |       0.8128 |        0.8853 |
| cnn_small |          0.6221 |       0.8768 |        0.8796 |
| metadata  |          0.6636 |       0.9793 |        0.9927 |

## With and without the equipment giveaway

Test clips split by whether their native sample rate is used by one species or
several. On the shared-rate subset the recording cannot identify the species by
itself, so that column is where audio has to earn its result.

| model     | subset                   |   clips |   macro_f1_mean |   macro_f1_std |
|:----------|:-------------------------|--------:|----------------:|---------------:|
| xgboost   | rate unique to a species |    3313 |          0.6714 |         0.1034 |
| xgboost   | rate shared by species   |     847 |          0.6295 |         0.1067 |
| cnn       | rate unique to a species |    3313 |          0.6237 |         0.1388 |
| cnn       | rate shared by species   |     847 |          0.6631 |         0.1415 |
| cnn_small | rate unique to a species |    3313 |          0.6903 |         0.1903 |
| cnn_small | rate shared by species   |     847 |          0.6776 |         0.1643 |
| metadata  | rate unique to a species |    3313 |          0.8928 |         0.1426 |
| metadata  | rate shared by species   |     847 |          0.6924 |         0.2255 |

## Figures

- `model_comparison.png`
- `per_class_recall.png`
- `ambiguity_breakdown.png`
- `confusion_xgboost.png`
- `feature_importance_xgboost.png`
- `confusion_cnn.png`
- `training_history_cnn.png`
- `confusion_cnn_small.png`
- `training_history_cnn_small.png`
- `occlusion.png`
- `confusion_metadata.png`
- `feature_importance_metadata.png`

Every figure has a CSV of the same name beside it or in the model directory.
