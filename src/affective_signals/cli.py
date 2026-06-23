"""CLI for the affective-signals prototype.

    python -m affective_signals train --recordings 40
    python -m affective_signals plot --out recording.png
"""

from __future__ import annotations

from pathlib import Path

import click

from .features import FEATURE_NAMES, build_dataset, extract_features
from .model import LogisticRegression, evaluate, train_test_split
from .preprocessing import decompose_eda, detect_scrs
from .signals import generate_recording


@click.group()
@click.version_option(version="0.1.0", prog_name="affective-signals")
def cli() -> None:
    """Engagement detection from synthetic EDA/HR biosignals."""


@cli.command()
@click.option("--recordings", default=40, show_default=True)
@click.option("--window-s", default=10.0, show_default=True)
@click.option("--seed", default=0, show_default=True)
def train(recordings: int, window_s: float, seed: int) -> None:
    """Build a dataset, train the classifier, and report metrics."""
    X, y = build_dataset(n_recordings=recordings, window_s=window_s)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.25, seed=seed)
    model = LogisticRegression(seed=seed).fit(Xtr, ytr)
    metrics = evaluate(yte, model.predict(Xte))

    click.echo(f"windows: {len(X)}  (engaged: {int(y.sum())}, idle: {int((y==0).sum())})")
    click.echo(
        f"accuracy={metrics['accuracy']:.2f}  precision={metrics['precision']:.2f}  "
        f"recall={metrics['recall']:.2f}  f1={metrics['f1']:.2f}"
    )
    click.echo(f"confusion: {metrics['confusion']}")
    click.echo("\nstandardized feature weights:")
    for name, w in sorted(
        zip(FEATURE_NAMES, model.w), key=lambda kv: abs(kv[1]), reverse=True
    ):
        click.echo(f"  {name:>13}: {w:+.3f}")


@cli.command()
@click.option("--out", type=click.Path(path_type=Path), default="recording.png", show_default=True)
@click.option("--seed", default=1, show_default=True)
def plot(out: Path, seed: int) -> None:
    """Plot one synthetic recording with detected SCRs and arousal events."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = generate_recording(seed=seed)
    tonic, phasic = decompose_eda(rec.eda, rec.fs)
    peaks = detect_scrs(phasic, rec.fs)
    t = rec.t

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t, rec.eda, color="#2563eb", lw=0.8)
    axes[0].plot(t, tonic, color="#f59e0b", lw=1.2, label="tonic")
    axes[0].set_ylabel("EDA (µS)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Synthetic affective recording")

    axes[1].plot(t, phasic, color="#7c3aed", lw=0.8)
    axes[1].plot(peaks / rec.fs, phasic[peaks], "rv", ms=6, label="SCR")
    axes[1].set_ylabel("phasic EDA")
    axes[1].legend(loc="upper right")

    axes[2].plot(t, rec.hr, color="#dc2626", lw=0.8)
    for et in rec.events:
        axes[2].axvline(et, color="gray", alpha=0.25, lw=0.8)
    axes[2].set_ylabel("HR (bpm)")
    axes[2].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    click.echo(f"Detected {len(peaks)} SCRs across {len(rec.events)} events. Wrote {out}")


if __name__ == "__main__":
    cli()
