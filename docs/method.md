# Method

How every number here was produced. The README carries the results.

**Resampling.** Everything is resampled to one rate and anything recorded below it is dropped rather
than upsampled, since upsampling leaves an empty band above the old Nyquist that a classifier will
use as a label. What survives shares an identical 0 to 5 kHz band. It does not survive as an
equipment neutral band, and the section above measures by how much.

**Folds.** `StratifiedGroupKFold` over a group named by the config, applied across species so a
group carrying two labels is never split. `base_10k` groups by tape, which is the first five
characters of the clip id: `5401800A` and `54018001` are both cuts of tape `54018`. Humpback has 604
clips over 14 tapes, so clip counts overstate the sample size by roughly an order of magnitude.
`tests/test_splits.py` fails if a group lands on both sides of a fold.

**Repeats.** The whole split reruns under shifted seeds, giving fifty estimates instead of five. A
model and its control run the identical plan, so repeat three fold two of one pairs with repeat three
fold two of the other. Fold counts are printed beside every number, and a repeat that returns the
same partition is counted once rather than again.

**The paired test.** Any two folds of a five fold split train on sets sharing three quarters of their
data, and each repeat reuses the same clips, so the differences are correlated and the ordinary
standard error is far too small. A plain paired t test over the fifty species differences returns
p = 0.00008; the corrected resampled test, replacing `1/n` with `1/n + 1/(k-1)`, returns p = 0.25 on
the same numbers. Every p value here is the corrected one.

The correction narrows what repeating a split can buy rather than forbidding it. The second term does
not fall as repeats are added, so ten repeats are worth far less than fifty independent folds, and
`tests/test_uncertainty.py` pins both halves of that: the species differences stay unresolved across
ten repeats, and a second set that does cross the threshold on repeats alone is recorded beside them.

**How many comparisons.** Seventy are reported across the seven configurations, so reading each at
0.05 on its own expects more than one to clear the bar carrying nothing.
`data/metadata/report/MULTIPLICITY.md` adjusts across all of them at once. Twenty seven survive, and
every one is either an audio model losing to a model that hears nothing or one no-audio model beating
another. Not a single positive audio comparison survives, in any configuration.

**Intervals.** Bootstrapped over whole tapes rather than clips. Resampling clips counts the same
recording many times and returns an interval several times too narrow; a test builds one interval
each way and asserts the clip version comes out narrower.

**Model selection.** The two network capacities were chosen on validation curves alone, with no test
fold involved. At 2.8 M parameters the CNN peaks at epochs 14, 16, 2, 6 and 20 across the five folds;
at 0.15 M it peaks at 7, 13, 4, 8 and 13.


## The corpus

Watkins Marine Mammal Sound Database, full cuts release: 15,248 mono WAV files across 54 species,
free for personal and academic use. Two sources are involved.

| source | provides | fetched by |
| --- | --- | --- |
| `archive.org/details/watkins-marine-mammal-sound-database-full-cuts` | the audio, 6.7 GB | `src.data.download` |
| `huggingface.co/datasets/ivangtorre/watkins-marine-mammal-full-cuts` | notes, sites, coordinates | `src.data.annotations` |

Everything in this repository that is not a species score rests on the second source. It is read with
HTTP range requests, about a fifth of a megabyte out of each 587 MB shard, and the parsed result is
committed so nothing needs refetching.

The three species under study, before and after the two filters that drop a clip. The sample rate
floor removes 57 humpback clips across 3 tapes. The half second minimum removes 413 more, 366 of them
sperm whale, which is the larger of the two. `audit_dropped_base_10k.csv` carries both.

| species | clips | tapes | kept clips | kept tapes | hours |
| --- | --- | --- | --- | --- | --- |
| KillerWhale | 2647 | 65 | 2601 | 65 | 1.65 |
| SpermWhale | 1379 | 63 | 1013 | 58 | 12.23 |
| HumpbackWhale | 604 | 14 | 546 | 12 | 1.95 |

Fin whale is excluded: 569 of its 580 clips run between 320 and 640 Hz, putting its Nyquist below
320 Hz and leaving no usable band shared with anything else.

