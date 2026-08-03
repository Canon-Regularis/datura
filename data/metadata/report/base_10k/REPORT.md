# Results: base_10k

Species: HumpbackWhale, SpermWhale, KillerWhale  
Common band: 0 to 5000 Hz at 10000 Hz  
Folds: 5 per split, grouped by tape  
Windows: 2.0 s, hop 1.0 s, at most 16 per clip  
Families: 9, each a set of models and the control they were measured against

## Every comparison

Columns beside the margin say what the design resolves. `folds` counts every fold of every repeat, so a run of ten repeats over five folds shows 50. `low` and `high` bound the paired difference at 95%, and `p_value` is the corrected resampled test, which accounts for the training data those folds share. `agreeing` counts the folds that pointed the same way as the mean, and is worth reading where the p value settles nothing.

| family                       | model                              |   margin |     low |   high |   p_value |   agreeing |   folds |
|:-----------------------------|:-----------------------------------|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_killerwhale_whistle | calltype_killerwhale_whistle       |   0.1011 |  0.0149 | 0.1872 |    0.0225 |         41 |      50 |
| calltype_spermwhale_click    | calltype_spermwhale_click          |   0.1409 |  0.0039 | 0.2779 |    0.0441 |         45 |      50 |
| calltype_spermwhale_whistle  | calltype_spermwhale_whistle        |   0.204  | -0.0878 | 0.4958 |    0.1664 |         35 |      50 |
| calltype_killerwhale_click   | calltype_killerwhale_click         |   0.1157 | -0.0585 | 0.2899 |    0.1882 |         36 |      50 |
| calltype_killerwhale_squeal  | calltype_killerwhale_squeal        |   0.1158 | -0.0802 | 0.3119 |    0.2408 |         35 |      50 |
| species                      | xgboost                            |  -0.1117 | -0.3034 | 0.0799 |    0.2471 |         38 |      50 |
| species                      | cnn                                |  -0.1913 | -0.6758 | 0.2933 |    0.3346 |          4 |       5 |
| species                      | cnn_small                          |  -0.1399 | -0.6702 | 0.3904 |    0.5046 |          4 |       5 |
| calltype_killerwhale_call    | calltype_killerwhale_call          |  -0.0795 | -0.3246 | 0.1657 |    0.518  |         30 |      50 |
| calltype_spermwhale_coda     | calltype_spermwhale_coda_cnn_small |   0.1113 | -0.3904 | 0.613  |    0.5713 |          3 |       5 |
| calltype_spermwhale_coda     | calltype_spermwhale_coda           |   0.0418 | -0.185  | 0.2686 |    0.7128 |         24 |      50 |
| calltype_killerwhale_chirp   | calltype_killerwhale_chirp         |   0.0099 | -0.0856 | 0.1054 |    0.8353 |         21 |      50 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. Its score is the floor an audio model has to clear.

| model     |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:----------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| xgboost   | 0.7567 |    0.8684 |  -0.1117 | -0.3034 | 0.0799 |    0.2471 |         38 |      50 |
| cnn_small | 0.7438 |    0.8837 |  -0.1399 | -0.6702 | 0.3904 |    0.5046 |          4 |       5 |
| cnn       | 0.6924 |    0.8837 |  -0.1913 | -0.6758 | 0.2933 |    0.3346 |          4 |       5 |

### Every model, with the range the recordings support

The interval comes from resampling whole tapes with replacement. Cuts from one tape
are near duplicates, so resampling clips would count the same recording many times
and produce an interval several times too narrow.

| model     |   estimate |    low |   high |   tapes |
|:----------|-----------:|-------:|-------:|--------:|
| xgboost   |     0.724  | 0.6144 | 0.8421 |     134 |
| cnn       |     0.674  | 0.5317 | 0.8426 |     134 |
| cnn_small |     0.7214 | 0.5726 | 0.9003 |     134 |
| metadata  |     0.9343 | 0.7962 | 0.9832 |     134 |

### Spread across folds

| model     |   mean |    std |
|:----------|-------:|-------:|
| xgboost   | 0.7567 | 0.0672 |
| cnn       | 0.6924 | 0.1338 |
| cnn_small | 0.7438 | 0.1645 |
| metadata  | 0.8684 | 0.1356 |

### Per species recall

| model     |   HumpbackWhale |   SpermWhale |   KillerWhale |
|:----------|----------------:|-------------:|--------------:|
| xgboost   |          0.5815 |       0.8724 |        0.8858 |
| cnn       |          0.5295 |       0.8128 |        0.8853 |
| cnn_small |          0.6221 |       0.8768 |        0.8796 |
| metadata  |          0.6723 |       0.9889 |        0.9497 |

### With and without the equipment giveaway

Test clips split by whether their native sample rate is used by one species or by
several. On the shared rate subset the recording cannot identify the species by
itself, so that column is where audio has to earn its result.

| model     | subset                   |   clips |   macro_f1_mean |   macro_f1_std |
|:----------|:-------------------------|--------:|----------------:|---------------:|
| xgboost   | rate unique to a species |    3313 |          0.7105 |         0.1323 |
| xgboost   | rate shared by species   |     847 |          0.6635 |         0.1366 |
| cnn       | rate unique to a species |    3313 |          0.6237 |         0.1388 |
| cnn       | rate shared by species   |     847 |          0.6631 |         0.1415 |
| cnn_small | rate unique to a species |    3313 |          0.6903 |         0.1903 |
| cnn_small | rate shared by species   |     847 |          0.6776 |         0.1643 |
| metadata  | rate unique to a species |    3313 |          0.8781 |         0.1889 |
| metadata  | rate shared by species   |     847 |          0.6888 |         0.1913 |

### Figures

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

Every figure has a CSV of the same name beside it, or in the model directory.

## KillerWhale, call

Against `calltype_killerwhale_call_context`, which sees the site, the coordinates and the noise
conditions, and no audio.

| model                     |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:--------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_killerwhale_call | 0.6892 |    0.7686 |  -0.0795 | -0.3246 | 0.1657 |     0.518 |         30 |      50 |

| model                             |   estimate |    low |   high |   tapes |
|:----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_call         |     0.7035 | 0.5525 | 0.8348 |      65 |
| calltype_killerwhale_call_context |     0.7775 | 0.6538 | 0.8774 |      65 |

## KillerWhale, chirp

Against `calltype_killerwhale_chirp_context`, which sees the site, the coordinates and the noise
conditions, and no audio.

| model                      |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:---------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_killerwhale_chirp | 0.5575 |    0.5475 |   0.0099 | -0.0856 | 0.1054 |    0.8353 |         21 |      50 |

| model                              |   estimate |    low |   high |   tapes |
|:-----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_chirp         |     0.5966 | 0.5222 | 0.6437 |      65 |
| calltype_killerwhale_chirp_context |     0.5638 | 0.4612 | 0.6422 |      65 |

## KillerWhale, click

Against `calltype_killerwhale_click_context`, which sees the site, the coordinates and the noise
conditions, and no audio.

| model                      |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:---------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_killerwhale_click | 0.6007 |     0.485 |   0.1157 | -0.0585 | 0.2899 |    0.1882 |         36 |      50 |

| model                              |   estimate |   low |   high |   tapes |
|:-----------------------------------|-----------:|------:|-------:|--------:|
| calltype_killerwhale_click         |     0.5839 | 0.442 | 0.7024 |      65 |
| calltype_killerwhale_click_context |     0.4784 | 0.386 | 0.5988 |      65 |

## KillerWhale, squeal

Against `calltype_killerwhale_squeal_context`, which sees the site, the coordinates and the noise
conditions, and no audio.

| model                       |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:----------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_killerwhale_squeal | 0.7685 |    0.6526 |   0.1158 | -0.0802 | 0.3119 |    0.2408 |         35 |      50 |

| model                               |   estimate |    low |   high |   tapes |
|:------------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_squeal         |     0.7507 | 0.5202 | 0.9214 |      65 |
| calltype_killerwhale_squeal_context |     0.5928 | 0.394  | 0.7941 |      65 |

## KillerWhale, whistle

Against `calltype_killerwhale_whistle_context`, which sees the site, the coordinates and the noise
conditions, and no audio.

| model                        |   mean |   control |   margin |    low |   high |   p_value |   agreeing |   folds |
|:-----------------------------|-------:|----------:|---------:|-------:|-------:|----------:|-----------:|--------:|
| calltype_killerwhale_whistle | 0.5642 |    0.4632 |   0.1011 | 0.0149 | 0.1872 |    0.0225 |         41 |      50 |

| model                                |   estimate |    low |   high |   tapes |
|:-------------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_whistle         |     0.5949 | 0.5076 | 0.632  |      65 |
| calltype_killerwhale_whistle_context |     0.4711 | 0.3729 | 0.5448 |      65 |

## SpermWhale, click

Against `calltype_spermwhale_click_context`, which sees the site, the coordinates and the noise
conditions, and no audio.

| model                     |   mean |   control |   margin |    low |   high |   p_value |   agreeing |   folds |
|:--------------------------|-------:|----------:|---------:|-------:|-------:|----------:|-----------:|--------:|
| calltype_spermwhale_click | 0.7053 |    0.5645 |   0.1409 | 0.0039 | 0.2779 |    0.0441 |         45 |      50 |

| model                             |   estimate |    low |   high |   tapes |
|:----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_spermwhale_click         |     0.7346 | 0.6019 | 0.8258 |      58 |
| calltype_spermwhale_click_context |     0.5716 | 0.4147 | 0.7097 |      58 |

## SpermWhale, coda

Against `calltype_spermwhale_coda_context`, which sees the site, the coordinates and the noise
conditions, and no audio.

| model                              |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:-----------------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_spermwhale_coda_cnn_small | 0.5339 |    0.4226 |   0.1113 | -0.3904 | 0.613  |    0.5713 |          3 |       5 |
| calltype_spermwhale_coda           | 0.5702 |    0.5285 |   0.0418 | -0.185  | 0.2686 |    0.7128 |         24 |      50 |

| model                              |   estimate |    low |   high |   tapes |
|:-----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_spermwhale_coda           |     0.5324 | 0.4345 | 0.6134 |      38 |
| calltype_spermwhale_coda_cnn_small |     0.5218 | 0.3691 | 0.6698 |      38 |
| calltype_spermwhale_coda_context   |     0.4396 | 0.276  | 0.6204 |      38 |

## SpermWhale, whistle

Against `calltype_spermwhale_whistle_context`, which sees the site, the coordinates and the noise
conditions, and no audio.

| model                       |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:----------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_spermwhale_whistle | 0.7117 |    0.5077 |    0.204 | -0.0878 | 0.4958 |    0.1664 |         35 |      50 |

| model                               |   estimate |    low |   high |   tapes |
|:------------------------------------|-----------:|-------:|-------:|--------:|
| calltype_spermwhale_whistle         |     0.782  | 0.4868 | 0.8965 |      58 |
| calltype_spermwhale_whistle_context |     0.5201 | 0.399  | 0.74   |      58 |
