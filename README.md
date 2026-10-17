# Remote Photoplethysmography (rPPG) for Deepfake Detection

## 1. Project Overview

This project investigates the use of remote photoplethysmography (rPPG) as a physiological signal-processing approach for detecting facial deepfakes. rPPG estimates pulse-related blood-volume variations from subtle changes in facial skin reflectance captured by a conventional camera. These variations are typically weak, spatially distributed, and sensitive to motion, illumination, compression, and measurement noise.

The central hypothesis is that synthetic facial videos may exhibit inconsistencies in the temporal or spatial characteristics of pulse-related signals. The objective is to investigate whether robust rPPG extraction and physiological signal analysis can provide useful evidence for distinguishing authentic facial videos from manipulated ones.

The initial development phase established a reproducible video-processing environment, implemented a green-channel frequency-domain baseline, and evaluated its heart-rate estimation accuracy against synchronized contact photoplethysmography (PPG) measurements.

**Initial empirical finding:** The discrete-point, green-channel baseline achieved a mean absolute heart-rate error of **17.54 BPM** over 43 temporal windows from subject 1 of UBFC-rPPG Dataset 2. The largest absolute window error was **60.12 BPM**. These results motivate a redesign of the signal-extraction and spectral-estimation stages before the pipeline is used for deepfake classification.

> **Scope of this result:** This is a physiological signal-estimation benchmark on one subject, not a deepfake-detection accuracy result. No deepfake classification performance is claimed at this stage.

---

## 2. System Architecture

### 2.1 Processing pipeline

The initial baseline consists of the following stages:

1. **Video ingestion:** Read uncompressed facial video frames while preserving their temporal order and associating each frame with its sampling time.
2. **Facial landmark detection:** Use MediaPipe Face Mesh to detect and track 478 facial landmarks.
3. **Forehead region selection:** Select nine discrete landmarks as sampling locations for a forehead-based region of interest (ROI).
4. **Green-channel signal extraction:** Read the green-channel intensity at each selected point and average the nine values into one scalar observation per frame.
5. **Temporal preprocessing:** Remove the estimated linear trend and apply a fourth-order, zero-phase Butterworth bandpass filter.
6. **Frequency-domain estimation:** Compute the real FFT and select the maximum-magnitude spectral bin within the physiological frequency band.
7. **Reference comparison:** Apply equivalent detrending, filtering, and FFT processing to the synchronized contact PPG signal from the UBFC-rPPG dataset.
8. **Windowed evaluation:** Calculate heart-rate error over overlapping 10-second windows with a 1-second step.

### 2.2 Architecture diagram

```text
Uncompressed Facial Video
          |
          v
 MediaPipe Face Mesh
      478 landmarks
          |
          v
 Nine Forehead Landmark Points
          |
          v
 Green-Channel Pixel Sampling
          |
          v
 Mean Intensity per Frame
          |
          v
 Linear Detrending
          |
          v
 4th-Order Butterworth Bandpass
       0.7–3.0 Hz
          |
          v
       Real FFT
          |
          v
 Dominant Frequency -> BPM
          |
          v
 Sliding-Window Evaluation
          ^
          |
 Synchronized Contact PPG
          |
 Same Reference Preprocessing
```

### 2.3 Environment and dependencies

The initial implementation used a dedicated Python 3.11.9 environment on Windows.

Python 3.14 was replaced for this phase because of compatibility problems involving native-extension wheels and legacy MediaPipe APIs. The environment was stabilized by pinning NumPy below version 2 to avoid the NumPy 2.x binary ABI incompatibility encountered with the selected MediaPipe build.

Core dependencies:

| Component | Purpose |
|---|---|
| Python 3.11.9 | Runtime compatibility |
| `mediapipe==0.10.14` | Facial landmark detection |
| `numpy==1.26.4` | Numerical computation and FFT |
| `opencv-python` | Video decoding and frame processing |
| `scipy` | Detrending, Butterworth filter design, and filtering |

The exact versions of OpenCV and SciPy should be recorded in the project's dependency lockfile to make the environment reproducible.

Windows App Execution Alias conflicts and PATH resolution issues were also addressed during environment setup.

### 2.4 Datasets

**Physiological validation: UBFC-rPPG Dataset 2**

The initial validation uses an uncompressed AVI video and its corresponding `ground_truth.txt` file containing reference physiological measurements, including contact PPG data.

The contact PPG signal provides an independent reference for assessing whether the video-derived signal produces a credible heart-rate estimate.

**Deepfake evaluation: FaceForensics++**

The project also configured access to original-versus-Deepfakes video pairs from FaceForensics++. The intended evaluation uses the lossless/raw (`c0`) compression setting to minimize compression-related confounding during the initial investigation.

Physiological validation and deepfake classification are separate experiments. The UBFC-rPPG benchmark evaluates signal extraction, while FaceForensics++ is intended to evaluate whether physiological features help discriminate authentic and manipulated videos.

Dataset access, licensing, and usage conditions must be followed.

---

## 3. Initial Baseline Implementation

### 3.1 Landmark-based forehead sampling

MediaPipe Face Mesh provides 478 facial landmarks. The baseline selects the following nine indices:

```python
FOREHEAD_LANDMARKS = [
    10, 107, 66, 69, 109,
    338, 296, 299, 336
]
```

For frame \(t\), let \(G_{t,i}\) denote the green-channel pixel intensity at the image coordinate associated with landmark \(i\). The extracted observation is

\[
g(t)=\frac{1}{9}\sum_{i=1}^{9}G_{t,i}.
\]

This produces a single temporal sequence from the facial video.

The approach is computationally inexpensive and straightforward to implement. However, it assumes that the selected points remain representative of the same skin locations over time. Landmark tracking error, image interpolation, local shadows, and small changes in facial position can violate this assumption.

### 3.2 Temporal detrending and bandpass filtering

The raw sequence is detrended to suppress its estimated linear component:

\[
g_d(t)=\operatorname{detrend}(g(t)).
\]

A fourth-order Butterworth bandpass filter is then applied with lower and upper cutoff frequencies of 0.7 Hz and 3.0 Hz, respectively:

\[
0.7\leq f\leq 3.0\ \text{Hz}.
\]

The corresponding heart-rate interval is

\[
60(0.7)\leq HR\leq60(3.0),
\]

or 42–180 BPM.

The implementation uses second-order sections for numerical stability and `scipy.signal.sosfiltfilt` for forward-backward, zero-phase filtering.

Zero-phase filtering avoids the phase delay introduced by a conventional one-pass filter, which is useful when comparing temporal signals. However, forward-backward filtering is noncausal and can introduce edge artifacts. Each evaluation window must contain enough samples for the filter's padding requirements, and boundary behavior should be checked.

### 3.3 FFT-based heart-rate estimation

For a window containing \(N\) uniformly sampled observations at sampling rate \(f_s\), the real FFT is

\[
G[k]=\operatorname{rFFT}\{g_d[n]\}.
\]

The baseline identifies the largest spectral magnitude within the allowed frequency band:

\[
k^*=\underset{k\in\mathcal{K}}{\arg\max}|G[k]|,
\]

where \(\mathcal{K}\) denotes the frequency bins between 0.7 Hz and 3.0 Hz.

The estimated heart rate is

\[
\widehat{HR}=60\frac{k^*f_s}{N}.
\]

The implementation uses `numpy.fft.rfft` and the corresponding frequency grid. The dominant spectral peak is treated as the estimated pulse frequency.

This estimator assumes that the dominant periodic component in the selected band corresponds to the cardiac pulse. Motion artifacts and other periodic variations can violate this assumption.

### 3.4 Evaluation protocol

The baseline was evaluated on subject 1 of UBFC-rPPG Dataset 2 using:

- Window duration: 10 seconds.
- Window step: 1 second.
- Nominal video sampling rate: 30 FPS.
- Number of evaluated windows: 43.
- Reference: synchronized contact PPG.
- Reference processing: equivalent detrending, bandpass filtering, and FFT-based heart-rate estimation.

For each window \(i\), the absolute error is

\[
e_i=|\widehat{HR}_i-HR_i^{\mathrm{ref}}|.
\]

The mean absolute error is

\[
MAE=\frac{1}{M}\sum_{i=1}^{M}e_i,
\]

where \(M=43\).

Applying comparable preprocessing to the reference and estimated signals reduces methodological differences between the two estimators. It does not eliminate possible timestamp misalignment, differences in sampling rates, sensor noise, or differences between the video-derived pulse and contact PPG measurements.

---

## 4. Benchmark Results

### 4.1 Measured baseline

| Metric | Initial baseline | Target / acceptance criterion |
|---|---:|---|
| Dataset | UBFC-rPPG Dataset 2 | Same held-out benchmark protocol |
| Subject | Subject 1 | Multiple subjects |
| Evaluation windows | 43 | Report the total evaluated windows |
| Window duration | 10 seconds | Preserve for direct comparison |
| Window step | 1 second | Preserve for direct comparison |
| Mean absolute error | **17.54 BPM** | To be established and empirically validated |
| Maximum absolute window error | **60.12 BPM** | Reduce; target to be specified |
| Deepfake classification accuracy | Not measured | Evaluate separately |
| Generalization across subjects | Not established | Subject-independent evaluation |

The current results demonstrate that the initial heart-rate estimator has substantial error on the evaluated sequence. The maximum window error also indicates that performance is not uniformly reliable across time.

The maximum absolute error alone does not reveal how frequently large errors occur. Future evaluations should include the median absolute error, error percentiles, valid-window coverage, and the distribution of errors across subjects.

### 4.2 Target-setting policy

No numerical target MAE is claimed in this initial report because no post-redesign benchmark has yet been measured.

The next experiment should compare the redesigned method against the baseline using identical subjects, windows, timestamps, reference processing, and evaluation metrics. An acceptance threshold should be defined before examining the final test results to avoid selecting a target retrospectively.

A lower heart-rate MAE would support improved physiological estimation. It would not, by itself, demonstrate improved deepfake detection.

---

## 5. Technical Post-Mortem: Why the Baseline Failed

### 5.1 Discrete-point sampling is spatially fragile

The nine-point approach approximates the facial ROI using isolated observations rather than integrating information across a continuous skin surface.

A landmark is a geometric estimate, not a guarantee that the same physical patch of skin is sampled in every frame. Small tracking errors can move a sampling coordinate onto a neighboring pixel, shadow boundary, eyebrow region, or specular highlight.

The resulting temporal signal can contain intensity changes caused by movement rather than blood-volume variation.

Averaging nine points reduces some independent pixel noise, but it does not reliably suppress spatially correlated disturbances such as global illumination changes or rigid head motion.

**Proposed correction:** Construct a spatially continuous forehead mask from valid facial landmarks and aggregate information over a larger skin region. A convex hull around selected forehead landmarks is one practical starting point, provided that the resulting polygon is clipped to valid skin pixels and excludes eyes, eyebrows, hair, and other unsuitable areas.

A convex hull is not inherently a skin segmentation algorithm. It can include non-skin pixels inside its boundary, so mask validation and exclusion regions remain necessary.

### 5.2 Green-only intensity is vulnerable to nuisance variation

The green channel often contains useful pulse-related information because hemoglobin absorption affects visible-light reflectance. Nevertheless, the measured green intensity also responds to:

- Ambient illumination changes.
- Exposure and white-balance adjustments.
- Head movement and changing surface orientation.
- Specular reflections.
- Shadows and occlusion.
- Camera processing and compression.

A simplified observation model is

\[
I_c(t)=L_c(t)R_c(t)+\epsilon_c(t),
\]

where \(I_c(t)\) is the observed intensity in color channel \(c\), \(L_c(t)\) represents illumination, \(R_c(t)\) represents surface reflectance, and \(\epsilon_c(t)\) captures measurement noise and unmodeled effects.

The physiological component is small relative to many of these nuisance variations. A green-only mean therefore cannot reliably distinguish pulse-related changes from unrelated intensity fluctuations.

**Proposed correction:** Use a multi-channel, temporally normalized rPPG method such as Plane-Orthogonal-to-Skin (POS). POS combines normalized color-channel signals to suppress selected common-mode and skin-tone-related variations while emphasizing a pulse-related component.

POS is not a universal solution to illumination changes or motion. Its assumptions can fail under severe movement, clipping, unstable exposure, occlusion, or nonuniform illumination.

### 5.3 Ten-second FFT windows have limited intrinsic resolution

At a sampling rate of 30 FPS, a 10-second window contains approximately 300 samples. The native FFT-bin spacing is

\[
\Delta f=\frac{f_s}{N}
=\frac{30}{300}=0.1\ \text{Hz}.
\]

Expressed in heart-rate units,

\[
\Delta HR=60\Delta f=6\ \text{BPM}.
\]

Consequently, a simple maximum-bin estimator returns heart rates on a discrete frequency grid with approximately 6 BPM spacing. A true pulse frequency between adjacent bins can be quantized to either bin, contributing to estimation error.

This is a property of the baseline estimator, not proof that every estimate has at least 6 BPM error. A true frequency can lie near a bin center, and other errors can dominate.

**Proposed correction:** Zero-pad the filtered signal before taking its FFT to create a more finely sampled representation of the spectrum. For example, increasing the FFT length to 4096 produces a denser frequency grid and enables more precise numerical peak localization.

However, zero-padding does **not** increase the information contained in the original 10-second observation or improve its fundamental resolving power. It interpolates the sampled spectrum. Longer observation windows, when appropriate, provide more genuine frequency resolution but reduce temporal responsiveness.

A robust implementation can combine zero-padding with local peak interpolation or a suitable parametric estimator, while evaluating the result against the same reference protocol.

### 5.4 Filtering and peak selection can compound errors

Bandpass filtering removes frequencies outside the chosen range but cannot distinguish a heartbeat from motion or illumination artifacts that occur within that range.

Likewise, selecting the largest in-band FFT peak can produce an incorrect estimate when the pulse component is weak or when another periodic component dominates.

The reference PPG signal may also contain artifacts, and equivalent filtering does not guarantee identical signal quality across sensors.

**Proposed correction:** Record signal-quality indicators, flag unreliable windows, inspect spectral peaks and harmonics, and report both estimation error and the percentage of windows that yield accepted estimates. Any window rejection rules should be established in advance and applied consistently.

---

## 6. The Mathematical Pivot: Spatial Integration + POS + Spectral Interpolation

The next version replaces the baseline's single-channel, discrete-point signal with a spatially aggregated, multi-channel rPPG pipeline.

### 6.1 Continuous forehead masking

Let \(\Omega\) denote the set of valid pixels in the forehead ROI. For each color channel \(c\), calculate the spatial mean:

\[
\bar I_c(t)=
\frac{1}{|\Omega_t|}
\sum_{p\in\Omega_t}I_c(p,t),
\]

where \(\Omega_t\) is the valid ROI at time \(t\).

This aggregates observations across many skin pixels instead of depending on nine isolated coordinates.

The mask should be derived from facial geometry, constrained to the visible image, and checked for valid skin coverage. Temporal mask stability and the number of valid pixels should be logged so that tracking failures can be distinguished from physiological failures.

### 6.2 Plane-Orthogonal-to-Skin (POS)

The POS algorithm introduced by Wang et al. uses temporally normalized color signals and a projection-based combination designed to emphasize pulse-related color variation.

For a temporal window, define the normalized channel signals as

\[
C_n(t)=\frac{\bar I_c(t)}{\mu_c}-1,
\]

where \(\mu_c\) is the mean intensity of channel \(c\) over that window. The normalized red, green, and blue signals form a color vector.

For the common POS formulation, two projected signals are constructed:

\[
S_1(t)=C_g(t)-C_b(t),
\]

\[
S_2(t)=C_g(t)+C_b(t)-2C_r(t),
\]

where \(C_r,C_g,C_b\) are the normalized channel signals.

The signals are combined using a window-dependent scaling factor:

\[
\alpha=\frac{\sigma(S_1)}{\sigma(S_2)},
\]

\[
h(t)=S_1(t)+\alpha S_2(t).
\]

Here, \(\sigma(\cdot)\) denotes the standard deviation within the processing window. The combined signal \(h(t)\) is the POS pulse estimate, subject to the chosen channel ordering and implementation convention.

The implementation must verify channel order, normalization interval, windowing, scaling, and overlap handling against the original algorithm. Degenerate cases such as a near-zero \(\sigma(S_2)\) require explicit numerical safeguards.

POS provides a principled multi-channel alternative to raw green-channel averaging. It does not guarantee successful pulse recovery in every video and should be evaluated empirically against the original baseline.

**Reference**

Wang, W., den Brinker, A. C., Stuijk, S., and de Haan, G. (2017). *Algorithmic Principles of Remote PPG*. IEEE Transactions on Biomedical Engineering, 64(7), 1479–1491. https://doi.org/10.1109/TBME.2016.2609282

### 6.3 Zero-padded spectral peak estimation

After signal extraction, detrending, and bandpass filtering, compute a zero-padded real FFT:

\[
H_L[k]=\operatorname{rFFT}\{h[n],N_{\mathrm{FFT}}\},
\]

where \(N_{\mathrm{FFT}}\) is larger than the original window length \(N\).

The associated frequency grid is

\[
f_k=\frac{k f_s}{N_{\mathrm{FFT}}}.
\]

Estimate the dominant in-band frequency from the magnitude spectrum:

\[
k^*=\underset{k\in\mathcal K}{\arg\max}|H_L[k]|,
\]

and convert it to beats per minute:

\[
\widehat{HR}=60f_{k^*}.
\]

A local interpolation method can further refine the peak estimate. The implementation should avoid implying that zero-padding adds independent signal information; it provides finer numerical sampling of the existing spectrum.

### 6.4 Proposed next-generation architecture

```text
Uncompressed Facial Video
          |
          v
 MediaPipe Face Mesh
          |
          v
 Continuous Forehead Mask
 + Valid Skin Pixel Selection
          |
          v
 RGB Spatial Aggregation
          |
          v
 POS Temporal Normalization
 + Projection-Based Combination
          |
          v
 Detrending and Bandpass Filtering
          |
          v
 Zero-Padded FFT
 + Optional Peak Interpolation
          |
          v
 Heart-Rate Estimate
 + Signal-Quality Assessment
          |
          v
 Reference Validation on UBFC-rPPG
          |
          v
 Physiological Feature Evaluation
 on FaceForensics++ Pairs
```

This redesign separates the task into two measurable stages: first, recover a more reliable pulse-related signal; second, determine whether physiological features provide discriminative evidence for deepfake detection.

---

## 7. Experimental Plan and Reproducibility

The next experimental phase should evaluate each proposed change independently before testing the full redesigned pipeline.

### Stage A — Reproduce the baseline

- Re-run the original implementation with fixed dependencies and deterministic window indexing.
- Preserve the 10-second window and 1-second step protocol.
- Record timestamps, reference alignment, estimated BPM, reference BPM, and absolute error for every window.
- Verify that the recorded metrics reproduce the baseline results within a stated tolerance.

### Stage B — Ablation study

Compare the following configurations:

1. Nine-point green-channel mean with native FFT.
2. Continuous forehead ROI with green-channel aggregation.
3. Continuous ROI with POS.
4. POS with zero-padded FFT.
5. POS with zero-padded FFT and optional peak interpolation.

Hold the evaluation protocol constant across configurations. This makes it possible to attribute observed improvements to specific changes rather than to an altered test set or evaluation procedure.

### Stage C — Physiological validation

Expand validation beyond a single subject. Report per-subject metrics as well as aggregate performance, and separate subjects or recordings into development and held-out evaluation groups.

Recommended metrics include:

- Mean and median absolute error in BPM.
- Root mean squared error in BPM.
- Error percentiles and maximum absolute error.
- Valid-window coverage.
- Performance across subjects and recording conditions.
- Sensitivity to motion, illumination, and tracking failures.

The reference signal's sampling rate, timestamp alignment, preprocessing, and heart-rate estimation method should be documented. Reference and estimated heart rates must represent equivalent time intervals.

### Stage D — Deepfake evaluation

Use paired original and manipulated videos from FaceForensics++ under a documented compression setting.

Evaluate physiological feature extraction separately from the final classifier. Potential features include pulse-signal quality, temporal consistency, spectral characteristics, and differences in physiological estimates across facial regions.

Use subject- or source-video-independent splits where feasible, and ensure that corresponding original and manipulated versions do not leak across training and test sets. Evaluate with appropriate classification metrics such as AUROC, precision, recall, F1-score, and confusion matrices.

A successful heart-rate estimator is not necessarily a successful deepfake detector. Classification performance must be measured directly on held-out manipulated and authentic videos.

---

## 8. Current Status

**Completed**

- Windows video-processing environment established with Python 3.11.9.
- Native-extension compatibility stabilized with NumPy 1.26.4 and MediaPipe 0.10.14.
- UBFC-rPPG Dataset 2 reference-validation workflow configured.
- FaceForensics++ original-versus-Deepfakes data acquisition configured.
- MediaPipe-based landmark tracking and nine-point forehead sampling implemented.
- Green-channel extraction, detrending, Butterworth bandpass filtering, and FFT heart-rate estimation implemented.
- Baseline evaluated over 43 windows from UBFC-rPPG subject 1.
- Baseline error quantified and major limitations identified.

**Proposed, not yet validated**

- Continuous forehead masking and skin-pixel aggregation.
- POS-based multi-channel signal extraction.
- Zero-padded FFT and optional peak interpolation.
- Signal-quality assessment and robust window handling.
- Multi-subject validation and deepfake classification experiments.

## 9. Conclusion

The initial experiment established an operational rPPG baseline but exposed substantial heart-rate estimation error: 17.54 BPM MAE and 60.12 BPM maximum absolute window error over 43 windows from a single subject.

The results motivate replacing sparse point sampling with spatial aggregation, replacing raw green-channel averaging with a multi-channel rPPG method such as POS, and improving spectral peak localization. These changes are technically motivated hypotheses, not demonstrated improvements.

The next milestone is a controlled ablation study that quantifies the contribution of each modification. Only after physiological extraction is validated should the project claim or evaluate deepfake-detection performance.

The objective is not simply to produce a pulse estimate, but to determine whether reliable, reproducible physiological evidence can improve the discrimination of authentic and manipulated facial videos.
