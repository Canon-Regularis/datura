# What the corpus gives away

The species is written down beside the audio in several places. These are the
measurements of how much each one gives away, and the two questions the audio is
asked once the species is off the table.

## Three fields that name the species

Every result above is explained by what the corpus records alongside the audio.

| field | clips it identifies | detail |
| --- | --- | --- |
| collection code | 91.4% | 7 codes, each in exactly one species |
| recording site | 98.2% | 46 of 47 sites visited for one species |
| native sample rate | see below | 10.2k to 45.4k Hz for killer, 10k to 166.6k for sperm |

The collection code is the field that had gone unmeasured. Every Watkins note opens with it, as in
`BE7A  Squeal.  Reverberation present.` A tape identifier would have made tape grouped folds
sufficient, and this is not one: `BE7A` spans 61 killer whale tapes, `BA2A` 51 sperm whale tapes,
`AC2A` 12 humpback tapes. A held out tape almost always carries a code the training tapes carried
too, so the fold boundary does not hide it.

Splitting held out clips on what a field does to the species:

| model | rate unique | rate shared | code unique | code absent |
| --- | --- | --- | --- | --- |
| logbook | 0.933 | 0.923 | 0.997 | 0.693 |
| metadata | 0.878 | 0.683 | 0.867 | 0.698 |
| acoustic, XGBoost | 0.705 | 0.661 | 0.750 | 0.616 |
| wav2vec2 probe | 0.679 | 0.708 | 0.746 | 0.514 |
| log mel CNN | 0.671 | 0.671 | 0.708 | 0.609 |

There is no shared code column here, because across these three species no code is shared. All seven
belong to one species each. The fourth column is the 359 clips carrying no code at all, and they
span two species over eight tapes, so they are scored over the two they hold. `classes_scored` in
`ambiguity_breakdown.csv` carries that denominator, and averaging a class that cannot appear into it
caps the column at two thirds and reads as a collapse the predictions do not contain.

XGBoost scores 0.705 where the sample rate is unique to a species and 0.661 where several species
share it, a gap of 0.044. The metadata control loses 19 points across the same split and the CNN
loses one.

The logbook falls from 0.997 to 0.693 on the clips with no code, and the metadata control lands on
0.698 while never seeing a code at all. Two models with very different inputs finishing within half
a point of each other says the fall is a property of those 359 clips rather than of the field they
are missing.

Eleven species is where a genuinely shared code exists. On the 2,581 clips whose code is used by more
than one of them the logbook still reaches 0.666 against XGBoost's 0.304, so what the paperwork
carries is more than any one field.

Two checks separate this from a join artefact. `audit_codes_by_species_*.csv` records how many tapes
each code spans, computed without a model. A test builds a corpus where every code sits on exactly
one tape and asserts the logbook's advantage disappears, since tape grouped folds already cover that
case.


## Where the place cost comes from

XGBoost alone, stepping from the tape folds to the real places. The middle row is the
same predictions scored on only the clips a place split can reach, which separates the
72 clips carrying no site from everything else.

| what is being scored | macro-F1 | cost |
| --- | --- | --- |
| tape held out, all 4,160 clips | 0.752 | |
| the same predictions on the 4,088 clips a place split can score | 0.744 | 0.008 |
| pseudo places, same structure and no real place | 0.556 | 0.189 |
| real places | 0.321 | 0.234 |

## Eleven species

`configs/wide.yaml` differs from `configs/base.yaml` in the species list alone: 11 species carrying
at least 10 surviving tapes, 7,723 clips over 228 recordings against 4,160 over 134. Same band,
window, seed and fold rule.

Two tape counts appear in the wide artifacts and they answer different questions. Summing the
surviving tapes of each species gives 238, which is the right figure for a per class statement. Ten
of those are counted twice, because eight tapes carry two of the eleven species, so the number of
independent recordings is 228. Sample size claims use the second.

| model | 3 species | 11 species |
| --- | --- | --- |
| logbook | 0.997 | 0.973 |
| metadata | 0.868 | 0.575 |
| acoustic, XGBoost | 0.752 | 0.425 |
| wav2vec2 probe | 0.739 | 0.437 |

Every score falls, and the gap widens rather than narrowing. XGBoost lands 0.548 below the logbook
at p = 5.5e-15 with 50 of 50 agreeing, against -0.245 on three species, and the probe 0.536 below at
p = 5.0e-20. The pretrained encoder is fractionally ahead of the hand engineered features here,
0.437 against 0.425, and both are less than half of what the paperwork reaches.

The metadata control loses most of its standing: 0.575 against the logbook's 0.973, a gap of 0.398
at p = 1.5e-08 with 50 of 50 agreeing, where on three species that gap was 0.129 and did not
resolve.

Which field carries the logbook changes with breadth. On three species the collection code alone
reaches 0.995 and the model splits mostly on sample rate and latitude. On eleven, most gain goes to
`cond_water_noise` and `cond_reverberation`, the noise conditions the recordist wrote down, with the
collection code at 8%. The confound is the written description as a whole rather than any one field.

Two things temper the per class numbers here. Eight of the 228 recordings carry more than one of the
eleven classes, and long finned pilot whale is the worst affected: 6 of its 18 tapes and 639 of its
1,067 clips sit on tapes it shares with sperm whale. Grouping keeps each tape whole so nothing
crosses a fold boundary, but those two classes are not scored on independent evidence. Separately,
eleven classes over 228 recordings leaves one test tape behind some classes in some folds, so a per
fold score for northern right whale or walrus rests on a single recording.

The networks were not run on the wide set. `python -m src.train.cnn --config configs/wide.yaml
--name cnn_small` adds them, and `configs/wide.yaml` declares trees only so a pipeline run does not
train them by surprise.


## Call types

Species is answered by the paperwork, so the audio is asked a question the paperwork cannot answer:
given that this is a sperm whale, does the clip contain a click? Each call type is posed inside one
species as a binary task against a context control seeing site, coordinates, collection code and
noise conditions. Eight tasks clear 60 clips over 10 tapes. Trees on ten repeats, plus one network on
a single split.

| task | audio | control | margin | p | agreeing |
| --- | --- | --- | --- | --- | --- |
| killer whale, click | 0.597 | 0.481 | +0.116 | 0.198 | 37 of 50 |
| killer whale, whistle | 0.567 | 0.524 | +0.043 | 0.202 | 36 of 50 |
| killer whale, squeal | 0.769 | 0.733 | +0.036 | 0.733 | 32 of 50 |
| sperm whale, click | 0.710 | 0.741 | -0.032 | 0.582 | 29 of 50 |
| sperm whale, whistle | 0.713 | 0.746 | -0.034 | 0.841 | 31 of 50 |
| killer whale, chirp | 0.556 | 0.619 | -0.063 | 0.135 | 34 of 50 |
| killer whale, call | 0.695 | 0.772 | -0.077 | 0.406 | 30 of 50 |
| sperm whale, coda | 0.577 | 0.671 | -0.094 | 0.144 | 39 of 50 |
| sperm whale, coda, CNN | 0.453 | 0.631 | -0.178 | 0.127 | 5 of 5 |

Three of nine margins are positive and not one comparison resolves in either direction. Sperm whale
coda used to be the exception, at -0.162 with p = 0.023, and it stopped being one when the control
learned to read a site name as a name rather than as a position in the alphabet. The audio models
here are unchanged; only the bar they are measured against moved.

This table used to read the opposite way, and the reason is worth stating. The control was once
denied the four header fields, clip duration among them. A Watkins note is written against a whole
cut, so a longer cut is more likely to contain any given call whatever the animal was doing, and a
control without that field was clearing a lower bar than the model beside it. Handing it over moved
killer whale whistle from +0.101 at p = 0.022 to +0.045 at p = 0.19, and moved sperm whale coda from
+0.026 to -0.162. Nothing about any audio model changed.

Sperm whale click resolved at +0.141 and p = 0.044 while the control could not see the collection
code. Giving the control that field moved it to +0.125 and p = 0.084 with no change to the audio
model.

Coda records a labelling failure. Its clips run to a median of 64 seconds while a coda lasts a few,
so most windows inherit a label they do not deserve. Restricting to clips under eight seconds lifted
the margin from +0.012 to +0.105 on one split and reads as +0.026 across fifty. The guard is kept
because the discarded labels are known to be wrong, and it is declared beside the call type in
`configs/call_types.yaml`.


## Explainability

Band occlusion on held out tapes costs humpback recall 0.12 below 330 Hz and nothing above 650 Hz.
Killer whale loses most above 3.7 kHz. Sperm whale loses a little in every band and much in none,
consistent with broadband clicks. Grad-CAM peaks land at 248 and 50 Hz for humpback, and at 1.0 and
4.7 kHz for killer whale, straddling the whistle band rather than sitting on it.

