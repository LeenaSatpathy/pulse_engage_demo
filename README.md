# affective-signals

An **affective-computing prototype** that detects user *engagement* from
wearable-style biosignals — electrodermal activity (EDA) and heart rate (HR) —
using classic signal processing plus a from-scratch logistic-regression
classifier.

Everything runs on a built-in synthetic signal generator, so there's no
hardware dependency and no biometric data involved. The pipeline mirrors how a
real affective-sensing prototype would be structured.

> Built as an applied-R&D prototype connecting affective computing and applied
> neuroscience for human-AI experiences.

## Pipeline

```
generate ─▶ low-pass filter ─▶ tonic/phasic split ─▶ SCR peak detection
                                       │
                                       ▼
                          per-window feature extraction
                  (SCR count/amplitude, EDA slope/var, HR mean/var/slope)
                                       │
                                       ▼
                    NumPy logistic regression ─▶ engaged / not-engaged
```

* `signals.py` — synthetic EDA+HR generator with arousal events and per-window labels.
* `preprocessing.py` — Butterworth filtering, tonic/phasic decomposition, SCR detection (SciPy).
* `features.py` — per-window affective feature extraction.
* `model.py` — standardization + L2 logistic regression in NumPy, with metrics.

## Install

```bash
pip install -e .
```

## Use

```bash
# Train on synthetic recordings and report metrics + feature weights
python -m affective_signals train --recordings 40

# Visualize one recording (EDA, phasic + detected SCRs, HR with event markers)
python -m affective_signals plot --out recording.png
```

Sample run:

```
windows: 720  (engaged: 354, idle: 366)
accuracy=0.92  precision=0.92  recall=0.93  f1=0.92

standardized feature weights:
        hr_mean: +2.114
        eda_std: +1.201
   scr_amp_mean: +0.859
   ...
```

## Library use

```python
from affective_signals import build_dataset, LogisticRegression, train_test_split, evaluate

X, y = build_dataset(n_recordings=40)
Xtr, Xte, ytr, yte = train_test_split(X, y)
model = LogisticRegression().fit(Xtr, ytr)
print(evaluate(yte, model.predict(Xte)))
```

## Notes

The synthetic generator and the engagement label are simplifications meant for
prototyping the *pipeline*, not validated physiological models. The interfaces
(`Recording`, `extract_features`, `LogisticRegression`) are designed so real
recordings can be dropped in by replacing the generator.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT
