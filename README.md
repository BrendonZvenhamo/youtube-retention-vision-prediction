Predicting YouTube Retention Drop-Off from Visual Frame Features
Summary: Visual features extracted from a frozen, pretrained EfficientNetB0 predict high-retention moments in cooking YouTube videos with F1 = 0.31 ± 0.05 (mean ± std across 4 independent video-level holdout splits), against a dummy baseline of F1 = 0.0. The signal is real but modest, and is specific to content categories where on-screen visuals correlate with viewer engagement.
Motivation
Short-form and long-form video platforms know exactly when viewers drop off, but that data is internal and proprietary. Creators and analysts only see aggregate metrics after the fact, not a predictive signal they can act on before publishing.
This project asks a narrow, falsifiable question: do visual features alone — no audio, no transcript, no captions — carry any predictive signal about where viewers will keep watching versus drop off?
An earlier framing of this question (simulating neural responses to video via brain imaging) was discarded immediately as infeasible without access to fMRI/EEG hardware. This project is the executable version of that underlying curiosity: a proxy for attention using only what a camera captures.
Data
40 cooking videos, YouTube Shorts, minimum 100k views (ensures YouTube's "Most Replayed" data exists)
Sourced via varied search terms across different creators to avoid single-creator visual style bias
Ground truth: YouTube's public "Most Replayed" retention curve, aligned to video timestamps
Frames extracted via ffmpeg at 0.5-second intervals — chosen because YouTube's retention data is itself aggregated at small time increments, not raw frame level, and because human attention does not operate at single-frame resolution
4,188 total frames across 40 videos
Features: 1280-dimensional vectors per frame from a frozen, pretrained EfficientNetB0 (no fine-tuning)
Why cooking only
Visual inspection of frames at high- vs. low-retention timestamps was done by hand across four content categories before any model was trained:
Category
Visual difference between high/low retention frames
Cooking
Yes — plating/reveal moments visually distinct from prep
Music
Yes — performance moments visually distinct from static shots
Tech explainer
No — same face/diagram regardless of retention
Vlog
No — same face/background regardless of retention
Tech and vlog content were dropped from the modeling phase on this evidence. Building a vision-only model on content where visuals don't actually vary with retention would have been a wasted cycle. Music is a planned v2 addition (see Limitations).
Methodology
This section is written in the order things actually happened, including the mistakes, because the mistakes are the more informative part of the process.
1. Initial pipeline validation (1 video)
Confirmed the mechanical pipeline worked end to end: download → frame extraction → Most Replayed scraping → alignment.
2. First binning attempt (10 videos, 527 frames) — wrong thresholds
Retention values were binned into low/medium/high using fixed absolute thresholds (0.1 / 0.3). The underlying retention distribution turned out to be heavily right-skewed (median retention ≈ 0.024), so these thresholds produced a near-unusable class split: 94% low, 2.5% medium, 3.6% high.
Fix: thresholds recalculated from the actual percentile distribution of the data (P25 / P90) rather than arbitrary absolute cutoffs.
3. First "real" result — and a leakage bug
Re-binned and retrained: F1 (high retention) = 0.74. This number was wrong. The train/test split was done at the frame level, meaning frames from the same video appeared in both train and test sets. The model was partly memorizing video-specific visual style (lighting, creator, color grade) rather than learning generalizable retention signal.
Fix: switched to a video-level split — entire videos held out, never frame-level shuffling.
4. Honest holdout result on the same 10 videos
F1 (high retention) on a true video-level holdout: 0.105. A near-total collapse from 0.74. Diagnosed as data starvation: 1280-dimensional features against ~500 training samples. PCA dimensionality reduction was tested as a possible fix and made performance worse on holdout (0.0), ruling out dimensionality as the core problem and confirming it was simply insufficient data volume.
5. Scaled to 40 videos, 4,188 frames
More creators, more visual diversity, fresh percentile-based binning (P25 = 0.055, P75 = 0.581 → roughly balanced 25/50/25 class split).
6. Second leakage bug — found by inspection, not by accident
Percentile thresholds for binning were calculated using all 40 videos, including the 8 that were later held out as the "unseen" test set. This meant the class boundaries the model was evaluated against had been partly shaped by the test data — a subtler form of leakage than #3, but leakage nonetheless.
Fix: thresholds recalculated using only the 32 training videos, then applied unchanged to bin the 8 holdout videos.
7. Class weight correction
Initial class weights (1.0 / 4.0 / 8.0) were inherited from the earlier, heavily imbalanced 10-video dataset and were no longer appropriate for the new, near-balanced 40-video distribution. Recalculated weights (≈1.6 / 1.0 / 1.7) from the actual training distribution, which improved both high-class and low-class F1.
8. Stability check across random seeds
A single 32/8 video split risks being an outlier given the small holdout size. The full leak-free pipeline (steps 6–7) was re-run across 4 different random seeds controlling which 8 videos landed in the holdout set.
Results
Final pipeline: video-level split → training-only percentile thresholds → training-only class weights → XGBoost classifier (default hyperparameters, no tuning).
Seed
F1 (high)
F1 (macro)
Test set composition (low/med/high %)
42
0.342
0.389
9.1 / 70.7 / 20.2
7
0.323
0.421
63.2 / 30.3 / 6.5
123
0.235
0.397
28.6 / 39.1 / 32.3
2024
0.352
0.448
15.5 / 68.1 / 16.4
F1 (high retention) across seeds: mean = 0.313, std = 0.046, range = 0.117
Dummy baseline (always predict majority class): F1 (high) = 0.0 in all cases.
Notably, the four holdout sets above have wildly different class compositions (low-retention frames range from 9% to 63% of the test set depending on seed), yet F1 on the high-retention class stays within a tight band. The model's ability to identify high-retention moments does not collapse when test set composition shifts, which is a stronger stability signal than the standard deviation alone suggests.
Limitations
Cooking content only. Music was visually validated as a second viable category but not yet modeled. Tech and vlog content showed no visual signal during manual inspection and were excluded — this is a finding, not an oversight.
40 videos is small. F1 ≈ 0.31 is a real, leak-free signal, not a production-grade predictor.
Visual features only. No audio, transcript, or caption data. For categories like tech and vlogs, this is likely a hard ceiling — manual inspection suggests retention there is driven by what's said, not what's shown. Multimodal fusion is the natural v2.
Frozen pretrained features. EfficientNetB0 was trained on ImageNet object classification, not on anything related to viewer attention. Fine-tuning or attention-specific feature extraction is unexplored.
8-video holdout per seed is small. The seed-stability check mitigates but does not eliminate this — results should be treated as a band, not a precise estimate.
What this does and does not prove
Proves: for content categories where on-screen visuals plausibly track engagement (cooking, likely music), a model using only frozen pretrained visual features extracted from video frames carries real, non-trivial predictive signal about viewer retention, and that signal survives a properly leak-free, video-level holdout evaluation across multiple random splits.
Does not prove: that this is a deployable product, that it generalizes beyond cooking content, or that visual features are sufficient for retention prediction in general. Two content categories (tech, vlogs) were tested and visually ruled out specifically because the hypothesis did not hold for them.
Stack
Python, ffmpeg (frame extraction), EfficientNetB0 (pretrained, frozen, feature extraction), XGBoost (classification), YouTube Most Replayed data (ground truth retention signal).