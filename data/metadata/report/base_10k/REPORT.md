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
| species                      | logbook                            |   0.129  | -0.0142 | 0.2723 |    0.0764 |         44 |      50 |
| calltype_killerwhale_click   | calltype_killerwhale_click         |   0.1444 | -0.0175 | 0.3062 |    0.0793 |         40 |      50 |
| calltype_spermwhale_click    | calltype_spermwhale_click          |   0.1252 | -0.0172 | 0.2677 |    0.0835 |         43 |      50 |
| calltype_spermwhale_whistle  | calltype_spermwhale_whistle        |   0.1852 | -0.1035 | 0.474  |    0.2034 |         29 |      50 |
| calltype_killerwhale_squeal  | calltype_killerwhale_squeal        |   0.1161 | -0.0799 | 0.3121 |    0.2398 |         36 |      50 |
| species                      | xgboost                            |  -0.1117 | -0.3034 | 0.0799 |    0.2471 |         38 |      50 |
| species                      | cnn                                |  -0.1913 | -0.6758 | 0.2933 |    0.3346 |          4 |       5 |
| calltype_spermwhale_coda     | calltype_spermwhale_coda_cnn_small |   0.1318 | -0.2158 | 0.4793 |    0.3519 |          4 |       5 |
| species                      | cnn_small                          |  -0.1399 | -0.6702 | 0.3904 |    0.5046 |          4 |       5 |
| calltype_spermwhale_coda     | calltype_spermwhale_coda           |   0.0257 | -0.1823 | 0.2338 |    0.8047 |         23 |      50 |
| calltype_killerwhale_chirp   | calltype_killerwhale_chirp         |   0.01   | -0.0855 | 0.1055 |    0.8346 |         21 |      50 |
| calltype_killerwhale_call    | calltype_killerwhale_call          |  -0.0229 | -0.3377 | 0.2919 |    0.8842 |         29 |      50 |

## Species

### Margin over the metadata control

The control sees native sample rate, year, clip duration and file size; it sees no
audio. Its score is the floor an audio model has to clear.

| model     |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:----------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| logbook   | 0.9974 |    0.8684 |   0.129  | -0.0142 | 0.2723 |    0.0764 |         44 |      50 |
| xgboost   | 0.7567 |    0.8684 |  -0.1117 | -0.3034 | 0.0799 |    0.2471 |         38 |      50 |
| cnn_small | 0.7438 |    0.8837 |  -0.1399 | -0.6702 | 0.3904 |    0.5046 |          4 |       5 |
| cnn       | 0.6924 |    0.8837 |  -0.1913 | -0.6758 | 0.2933 |    0.3346 |          4 |       5 |

### Margin over logbook, the strongest model that hears no audio

`logbook` also sees the site, the coordinates, the noise conditions and the
collection code the field note opens with. None of that is the animal, so this is the
number an audio result has to clear before it is evidence about whales.

| model     |   mean |   control |   margin |     low |    high |   p_value |   agreeing |   folds |
|:----------|-------:|----------:|---------:|--------:|--------:|----------:|-----------:|--------:|
| xgboost   | 0.7567 |    0.9974 |  -0.2408 | -0.3109 | -0.1706 |    0      |         50 |      50 |
| cnn_small | 0.7438 |    0.9983 |  -0.2545 | -0.5636 |  0.0546 |    0.0842 |          5 |       5 |
| cnn       | 0.6924 |    0.9983 |  -0.3059 | -0.5571 | -0.0548 |    0.0277 |          5 |       5 |

### Every model, with the range the recordings support

The interval comes from resampling whole tapes with replacement. Cuts from one tape
are near duplicates, so resampling clips would count the same recording many times
and produce an interval several times too narrow.

| model     |   estimate |    low |   high |   tapes |
|:----------|-----------:|-------:|-------:|--------:|
| xgboost   |     0.724  | 0.6144 | 0.8421 |     134 |
| cnn       |     0.674  | 0.5317 | 0.8426 |     134 |
| cnn_small |     0.7214 | 0.5726 | 0.9003 |     134 |
| logbook   |     0.9987 | 0.9935 | 1      |     134 |
| metadata  |     0.9343 | 0.7962 | 0.9832 |     134 |

### Spread across folds

| model     |   mean |    std |
|:----------|-------:|-------:|
| xgboost   | 0.7567 | 0.0672 |
| cnn       | 0.6924 | 0.1338 |
| cnn_small | 0.7438 | 0.1645 |
| logbook   | 0.9974 | 0.0041 |
| metadata  | 0.8684 | 0.1356 |

### Per species recall

| model     |   HumpbackWhale |   SpermWhale |   KillerWhale |
|:----------|----------------:|-------------:|--------------:|
| xgboost   |          0.5815 |       0.8724 |        0.8858 |
| cnn       |          0.5295 |       0.8128 |        0.8853 |
| cnn_small |          0.6221 |       0.8768 |        0.8796 |
| logbook   |          0.9952 |       0.9956 |        0.9996 |
| metadata  |          0.6723 |       0.9889 |        0.9497 |

One of these recordings carries more than one of the classes above, across HumpbackWhale and SpermWhale. Grouping keeps each tape whole, so none of them crosses a fold boundary, and they still contribute to two recalls apiece: the classes sharing a tape are not scored on independent evidence.

### With and without the giveaway

Test clips split by whether their native sample rate or their collection code is used by one species or
by several. On the shared subset the recording cannot identify the species by itself,
so those rows are where audio has to earn its result.

| giveaway           | model     | subset                                 |   clips |   macro_f1_mean |   macro_f1_std |
|:-------------------|:----------|:---------------------------------------|--------:|----------------:|---------------:|
| native sample rate | xgboost   | native sample rate unique to a species |    3313 |          0.7105 |         0.1323 |
| native sample rate | xgboost   | native sample rate shared by species   |     847 |          0.6635 |         0.1366 |
| collection code    | xgboost   | collection code unique to a species    |    3801 |          0.7544 |         0.0691 |
| collection code    | xgboost   | collection code shared by species      |     359 |          0.4108 |         0.1881 |
| native sample rate | cnn       | native sample rate unique to a species |    3313 |          0.6237 |         0.1388 |
| native sample rate | cnn       | native sample rate shared by species   |     847 |          0.6631 |         0.1415 |
| collection code    | cnn       | collection code unique to a species    |    3801 |          0.6883 |         0.1353 |
| collection code    | cnn       | collection code shared by species      |     359 |          0.4043 |         0.1426 |
| native sample rate | cnn_small | native sample rate unique to a species |    3313 |          0.6903 |         0.1903 |
| native sample rate | cnn_small | native sample rate shared by species   |     847 |          0.6776 |         0.1643 |
| collection code    | cnn_small | collection code unique to a species    |    3801 |          0.7438 |         0.169  |
| collection code    | cnn_small | collection code shared by species      |     359 |          0.3618 |         0.0577 |
| native sample rate | logbook   | native sample rate unique to a species |    3313 |          0.932  |         0.1341 |
| native sample rate | logbook   | native sample rate shared by species   |     847 |          0.8636 |         0.1655 |
| collection code    | logbook   | collection code unique to a species    |    3801 |          0.9978 |         0.0044 |
| collection code    | logbook   | collection code shared by species      |     359 |          0.4256 |         0.1964 |
| native sample rate | metadata  | native sample rate unique to a species |    3313 |          0.8781 |         0.1889 |
| native sample rate | metadata  | native sample rate shared by species   |     847 |          0.6888 |         0.1913 |
| collection code    | metadata  | collection code unique to a species    |    3801 |          0.8676 |         0.1361 |
| collection code    | metadata  | collection code shared by species      |     359 |          0.4651 |         0.1649 |

### Figures

- `model_comparison.png`
- `per_class_recall.png`
- `ambiguity_native_sample_rate.png`
- `ambiguity_collection_code.png`
- `confusion_xgboost.png`
- `feature_importance_xgboost.png`
- `confusion_cnn.png`
- `training_history_cnn.png`
- `confusion_cnn_small.png`
- `training_history_cnn_small.png`
- `occlusion.png`
- `confusion_logbook.png`
- `feature_importance_logbook.png`
- `confusion_metadata.png`
- `feature_importance_metadata.png`

Every figure has a CSV of the same name beside it, or in the model directory.

## KillerWhale, call

Against `calltype_killerwhale_call_context`, which sees the site, the coordinates, the collection
the cut came from and the noise conditions, and no audio.

| model                     |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:--------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_killerwhale_call | 0.6892 |    0.7121 |  -0.0229 | -0.3377 | 0.2919 |    0.8842 |         29 |      50 |

| model                             |   estimate |    low |   high |   tapes |
|:----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_call         |     0.7035 | 0.5525 | 0.8348 |      65 |
| calltype_killerwhale_call_context |     0.7082 | 0.5652 | 0.8274 |      65 |

## KillerWhale, chirp

Against `calltype_killerwhale_chirp_context`, which sees the site, the coordinates, the collection
the cut came from and the noise conditions, and no audio.

| model                      |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:---------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_killerwhale_chirp | 0.5575 |    0.5475 |     0.01 | -0.0855 | 0.1055 |    0.8346 |         21 |      50 |

| model                              |   estimate |    low |   high |   tapes |
|:-----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_chirp         |     0.5966 | 0.5222 | 0.6437 |      65 |
| calltype_killerwhale_chirp_context |     0.5638 | 0.4612 | 0.6422 |      65 |

## KillerWhale, click

Against `calltype_killerwhale_click_context`, which sees the site, the coordinates, the collection
the cut came from and the noise conditions, and no audio.

| model                      |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:---------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_killerwhale_click | 0.6007 |    0.4564 |   0.1444 | -0.0175 | 0.3062 |    0.0793 |         40 |      50 |

| model                              |   estimate |   low |   high |   tapes |
|:-----------------------------------|-----------:|------:|-------:|--------:|
| calltype_killerwhale_click         |     0.5839 | 0.442 | 0.7024 |      65 |
| calltype_killerwhale_click_context |     0.4784 | 0.386 | 0.5988 |      65 |

## KillerWhale, squeal

Against `calltype_killerwhale_squeal_context`, which sees the site, the coordinates, the collection
the cut came from and the noise conditions, and no audio.

| model                       |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:----------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_killerwhale_squeal | 0.7685 |    0.6524 |   0.1161 | -0.0799 | 0.3121 |    0.2398 |         36 |      50 |

| model                               |   estimate |    low |   high |   tapes |
|:------------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_squeal         |     0.7507 | 0.5202 | 0.9214 |      65 |
| calltype_killerwhale_squeal_context |     0.5909 | 0.3932 | 0.7941 |      65 |

## KillerWhale, whistle

Against `calltype_killerwhale_whistle_context`, which sees the site, the coordinates, the collection
the cut came from and the noise conditions, and no audio.

| model                        |   mean |   control |   margin |    low |   high |   p_value |   agreeing |   folds |
|:-----------------------------|-------:|----------:|---------:|-------:|-------:|----------:|-----------:|--------:|
| calltype_killerwhale_whistle | 0.5642 |    0.4632 |   0.1011 | 0.0149 | 0.1872 |    0.0225 |         41 |      50 |

| model                                |   estimate |    low |   high |   tapes |
|:-------------------------------------|-----------:|-------:|-------:|--------:|
| calltype_killerwhale_whistle         |     0.5949 | 0.5076 | 0.632  |      65 |
| calltype_killerwhale_whistle_context |     0.4711 | 0.3729 | 0.5448 |      65 |

## SpermWhale, click

Against `calltype_spermwhale_click_context`, which sees the site, the coordinates, the collection
the cut came from and the noise conditions, and no audio.

| model                     |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:--------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_spermwhale_click | 0.7053 |    0.5801 |   0.1252 | -0.0172 | 0.2677 |    0.0835 |         43 |      50 |

| model                             |   estimate |    low |   high |   tapes |
|:----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_spermwhale_click         |     0.7346 | 0.6019 | 0.8258 |      58 |
| calltype_spermwhale_click_context |     0.6054 | 0.4492 | 0.7331 |      58 |

## SpermWhale, coda

Against `calltype_spermwhale_coda_context`, which sees the site, the coordinates, the collection
the cut came from and the noise conditions, and no audio.

| model                              |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:-----------------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_spermwhale_coda_cnn_small | 0.5655 |    0.4338 |   0.1318 | -0.2158 | 0.4793 |    0.3519 |          4 |       5 |
| calltype_spermwhale_coda           | 0.5702 |    0.5445 |   0.0257 | -0.1823 | 0.2338 |    0.8047 |         23 |      50 |

| model                              |   estimate |    low |   high |   tapes |
|:-----------------------------------|-----------:|-------:|-------:|--------:|
| calltype_spermwhale_coda           |     0.5324 | 0.4345 | 0.6134 |      38 |
| calltype_spermwhale_coda_cnn_small |     0.5485 | 0.4051 | 0.6768 |      38 |
| calltype_spermwhale_coda_context   |     0.4479 | 0.2851 | 0.6338 |      38 |

## SpermWhale, whistle

Against `calltype_spermwhale_whistle_context`, which sees the site, the coordinates, the collection
the cut came from and the noise conditions, and no audio.

| model                       |   mean |   control |   margin |     low |   high |   p_value |   agreeing |   folds |
|:----------------------------|-------:|----------:|---------:|--------:|-------:|----------:|-----------:|--------:|
| calltype_spermwhale_whistle | 0.7117 |    0.5265 |   0.1852 | -0.1035 |  0.474 |    0.2034 |         29 |      50 |

| model                               |   estimate |    low |   high |   tapes |
|:------------------------------------|-----------:|-------:|-------:|--------:|
| calltype_spermwhale_whistle         |      0.782 | 0.4868 | 0.8965 |      58 |
| calltype_spermwhale_whistle_context |      0.522 | 0.3992 | 0.7491 |      58 |
