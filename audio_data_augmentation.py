#!/usr/bin/env python3
"""Training-only waveform augmentation used by train_cnn_from_wav_logmel.py.

Each train segment gets --augmentation-copies extra copies (augment_waveform):
optional DC removal -> mic EQ -> time shift -> gain -> Gaussian noise
-> recorded-background mix -> clip. Steps disabled by the CLI are no-ops.
apply_background_mix and apply_smooth_mic_eq are reused by device_domain_augmentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from audio_dataset import LABEL_TO_ID, Dataset, read_wav_mono


FeatureExtractor = Callable[[np.ndarray, int, int, int, int], np.ndarray]


def add_augmentation_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_enabled: bool = True,
) -> None:
    """Add conservative audio augmentation controls to a training CLI."""
    group = parser.add_argument_group("training WAV augmentation")
    enable_group = group.add_mutually_exclusive_group()
    enable_group.add_argument(
        "--train-augmentation",
        dest="no_train_augmentation",
        action="store_false",
        help="Enable training-only waveform augmentation.",
    )
    enable_group.add_argument(
        "--no-train-augmentation",
        dest="no_train_augmentation",
        action="store_true",
        help="Disable training-only waveform augmentation.",
    )
    parser.set_defaults(no_train_augmentation=not default_enabled)
    group.add_argument(
        "--augmentation-copies",
        type=int,
        default=1,
        help="Composite augmented copies per original training WAV (default: 1).",
    )
    group.add_argument(
        "--augmentation-eq-db",
        type=float,
        default=6.0,
        help="Maximum smooth microphone-response perturbation in dB.",
    )
    group.add_argument(
        "--augmentation-noise-snr-min-db",
        type=float,
        default=15.0,
        help="Minimum random additive-noise SNR in dB.",
    )
    group.add_argument(
        "--augmentation-noise-snr-max-db",
        type=float,
        default=30.0,
        help="Maximum random additive-noise SNR in dB.",
    )
    group.add_argument(
        "--augmentation-time-shift-ms",
        type=float,
        default=100.0,
        help="Maximum zero-filled time shift in either direction.",
    )
    group.add_argument(
        "--augmentation-gain-db",
        type=float,
        default=6.0,
        help="Maximum random gain in either direction in dB.",
    )
    group.add_argument(
        "--augmentation-gain-min-db",
        type=float,
        default=None,
        help="Optional asymmetric minimum gain. Requires --augmentation-gain-max-db.",
    )
    group.add_argument(
        "--augmentation-gain-max-db",
        type=float,
        default=None,
        help="Optional asymmetric maximum gain. Requires --augmentation-gain-min-db.",
    )
    gaussian_group = group.add_mutually_exclusive_group()
    gaussian_group.add_argument(
        "--augmentation-gaussian-noise",
        dest="augmentation_gaussian_noise",
        action="store_true",
        help="Enable the existing synthetic Gaussian-noise augmentation.",
    )
    gaussian_group.add_argument(
        "--no-augmentation-gaussian-noise",
        dest="augmentation_gaussian_noise",
        action="store_false",
        help="Disable synthetic Gaussian noise, for example when mixing recorded backgrounds.",
    )
    parser.set_defaults(augmentation_gaussian_noise=True)
    group.add_argument(
        "--augmentation-background-mix",
        action="store_true",
        help="Mix a random background-class segment from the training split only.",
    )
    group.add_argument(
        "--augmentation-background-snr-min-db",
        type=float,
        default=5.0,
        help="Minimum signal-to-background ratio for recorded background mixing.",
    )
    group.add_argument(
        "--augmentation-background-snr-max-db",
        type=float,
        default=20.0,
        help="Maximum signal-to-background ratio for recorded background mixing.",
    )
    group.add_argument(
        "--augmentation-pre-dc-removal",
        action="store_true",
        help="Remove waveform DC before applying gain and noise augmentation.",
    )


def augmentation_gain_range(args: argparse.Namespace) -> tuple[float, float]:
    """Resolve the new asymmetric range while preserving the legacy symmetric option."""
    minimum = getattr(args, "augmentation_gain_min_db", None)
    maximum = getattr(args, "augmentation_gain_max_db", None)
    if minimum is None and maximum is None:
        limit = float(args.augmentation_gain_db)
        return -limit, limit
    if minimum is None or maximum is None:
        raise ValueError(
            "--augmentation-gain-min-db and --augmentation-gain-max-db must be used together"
        )
    return float(minimum), float(maximum)


def augmentation_config_from_args(args: argparse.Namespace) -> Dict:
    enabled = not args.no_train_augmentation and args.augmentation_copies > 0
    gain_min_db, gain_max_db = augmentation_gain_range(args)
    gaussian_noise = bool(getattr(args, "augmentation_gaussian_noise", True))
    background_mix = bool(getattr(args, "augmentation_background_mix", False))
    pre_dc_removal = bool(getattr(args, "augmentation_pre_dc_removal", False))
    order = []
    if pre_dc_removal:
        order.append("dc_removal")
    if args.augmentation_eq_db > 0:
        order.append("smooth_mic_eq")
    if args.augmentation_time_shift_ms > 0:
        order.append("time_shift")
    order.append("gain")
    if gaussian_noise:
        order.append("additive_gaussian_noise")
    if background_mix:
        order.append("recorded_background_mix")
    return {
        "training_only": True,
        "enabled": enabled,
        "copies_per_original": int(args.augmentation_copies),
        "smooth_mic_eq_max_db": float(args.augmentation_eq_db),
        "noise_snr_db": [
            float(args.augmentation_noise_snr_min_db),
            float(args.augmentation_noise_snr_max_db),
        ],
        "time_shift_max_ms": float(args.augmentation_time_shift_ms),
        "gain_max_db": float(args.augmentation_gain_db),
        "gain_range_db": [gain_min_db, gain_max_db],
        "pre_dc_removal": pre_dc_removal,
        "gaussian_noise_enabled": gaussian_noise,
        "recorded_background_mix_enabled": background_mix,
        "recorded_background_snr_db": [
            float(getattr(args, "augmentation_background_snr_min_db", 5.0)),
            float(getattr(args, "augmentation_background_snr_max_db", 20.0)),
        ],
        "recorded_background_source": "training_split_background_class_only",
        "random_state": int(args.random_state),
        "order": order,
    }


def validate_augmentation_args(args: argparse.Namespace) -> None:
    if args.augmentation_copies < 0:
        raise ValueError("--augmentation-copies must be >= 0")
    if args.augmentation_eq_db < 0:
        raise ValueError("--augmentation-eq-db must be >= 0")
    if args.augmentation_noise_snr_min_db > args.augmentation_noise_snr_max_db:
        raise ValueError("augmentation minimum SNR must not exceed maximum SNR")
    if args.augmentation_time_shift_ms < 0:
        raise ValueError("--augmentation-time-shift-ms must be >= 0")
    if args.augmentation_gain_db < 0:
        raise ValueError("--augmentation-gain-db must be >= 0")
    gain_min_db, gain_max_db = augmentation_gain_range(args)
    if gain_min_db > gain_max_db:
        raise ValueError("augmentation gain minimum must not exceed maximum")
    background_min = float(getattr(args, "augmentation_background_snr_min_db", 5.0))
    background_max = float(getattr(args, "augmentation_background_snr_max_db", 20.0))
    if background_min > background_max:
        raise ValueError("background SNR minimum must not exceed maximum")


def apply_smooth_mic_eq(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    max_db: float,
) -> np.ndarray:
    """Apply a smooth random frequency response to simulate another microphone."""
    if max_db <= 0 or audio.size == 0:
        return audio.copy()

    spectrum = np.fft.rfft(audio)
    frequencies = np.fft.rfftfreq(audio.size, d=1.0 / sr)
    nyquist = sr / 2.0
    anchors_hz = np.asarray(
        [0.0, 150.0, 400.0, 1000.0, 2000.0, 4000.0, 6000.0, nyquist],
        dtype=np.float32,
    )
    anchors_hz = np.unique(np.clip(anchors_hz, 0.0, nyquist))
    anchor_gain_db = rng.uniform(-max_db, max_db, size=len(anchors_hz))
    gain_db = np.interp(frequencies, anchors_hz, anchor_gain_db)
    gain_linear = np.power(10.0, gain_db / 20.0)
    return np.fft.irfft(spectrum * gain_linear, n=audio.size).astype(np.float32)


def apply_time_shift(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    max_shift_ms: float,
) -> np.ndarray:
    """Shift without wraparound so no artificial boundary is moved into the signal."""
    max_samples = min(int(round(max_shift_ms * sr / 1000.0)), max(audio.size - 1, 0))
    if max_samples <= 0:
        return audio.copy()

    shift = int(rng.integers(-max_samples, max_samples + 1))
    shifted = np.zeros_like(audio)
    if shift > 0:
        shifted[shift:] = audio[:-shift]
    elif shift < 0:
        shifted[:shift] = audio[-shift:]
    else:
        shifted[:] = audio
    return shifted


def apply_gain(audio: np.ndarray, rng: np.random.Generator, max_db: float) -> np.ndarray:
    if max_db <= 0:
        return audio.copy()
    gain_db = float(rng.uniform(-max_db, max_db))
    return (audio * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def apply_gain_range(
    audio: np.ndarray,
    rng: np.random.Generator,
    min_db: float,
    max_db: float,
) -> np.ndarray:
    gain_db = float(rng.uniform(min_db, max_db))
    return (audio * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def apply_additive_noise(
    audio: np.ndarray,
    rng: np.random.Generator,
    snr_min_db: float,
    snr_max_db: float,
) -> np.ndarray:
    signal_rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    if signal_rms <= 1e-8:
        return audio.copy()

    noise = rng.normal(0.0, 1.0, size=audio.size).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-12))
    snr_db = float(rng.uniform(snr_min_db, snr_max_db))
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    return (audio + noise * (target_noise_rms / noise_rms)).astype(np.float32)


def apply_background_mix(
    audio: np.ndarray,
    background: np.ndarray,
    rng: np.random.Generator,
    snr_min_db: float,
    snr_max_db: float,
) -> np.ndarray:
    """Mix a recorded training background at a random signal-to-background ratio."""
    if audio.size == 0 or background.size == 0:
        return audio.copy()
    if background.size > audio.size:
        start = int(rng.integers(0, background.size - audio.size + 1))
        background = background[start : start + audio.size]
    elif background.size < audio.size:
        background = np.pad(background, (0, audio.size - background.size))

    signal_rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    background = (background - np.mean(background, dtype=np.float64)).astype(np.float32)
    background_rms = float(np.sqrt(np.mean(background * background) + 1e-12))
    if signal_rms <= 1e-8 or background_rms <= 1e-8:
        return audio.copy()
    snr_db = float(rng.uniform(snr_min_db, snr_max_db))
    target_background_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    return (audio + background * (target_background_rms / background_rms)).astype(np.float32)


def augment_waveform(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
    background_audio: np.ndarray | None = None,
) -> np.ndarray:
    """Apply all configured perturbations and clamp to normalized PCM range."""
    augmented = np.asarray(audio, dtype=np.float32)
    if getattr(args, "augmentation_pre_dc_removal", False):
        augmented = (augmented - np.mean(augmented, dtype=np.float64)).astype(np.float32)
    augmented = apply_smooth_mic_eq(augmented, sr, rng, args.augmentation_eq_db)
    augmented = apply_time_shift(augmented, sr, rng, args.augmentation_time_shift_ms)
    gain_min_db, gain_max_db = augmentation_gain_range(args)
    augmented = apply_gain_range(augmented, rng, gain_min_db, gain_max_db)
    if getattr(args, "augmentation_gaussian_noise", True):
        augmented = apply_additive_noise(
            augmented,
            rng,
            args.augmentation_noise_snr_min_db,
            args.augmentation_noise_snr_max_db,
        )
    if getattr(args, "augmentation_background_mix", False) and background_audio is not None:
        augmented = apply_background_mix(
            augmented,
            background_audio,
            rng,
            args.augmentation_background_snr_min_db,
            args.augmentation_background_snr_max_db,
        )
    return np.clip(augmented, -1.0, 1.0).astype(np.float32)


def augmented_relative_path(relative_path: str, copy_index: int) -> str:
    path = Path(relative_path)
    return (path.parent / f"{path.stem}__aug{copy_index:02d}{path.suffix}").as_posix()


def augment_training_dataset(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    feature_extractor: FeatureExtractor,
    expected_dim: int,
    feature_contract: str,
) -> Dataset:
    """Append augmented features for train sources after dataset splitting."""
    validate_augmentation_args(args)
    config = augmentation_config_from_args(args)
    if not config["enabled"]:
        print("Training WAV augmentation: disabled")
        return dataset

    rng = np.random.default_rng(args.random_state)
    augmented_features = []
    augmented_labels = []
    augmented_relative_paths = []
    augmented_source_paths = []
    skipped = 0
    background_paths = [
        path
        for label, path in zip(dataset.y, dataset.source_paths)
        if int(label) == LABEL_TO_ID["background"] and path is not None and path.exists()
    ]
    if getattr(args, "augmentation_background_mix", False) and not background_paths:
        raise ValueError("Recorded background mixing requires background samples in training split")
    background_cache: dict[Path, np.ndarray] = {}

    original_count = len(dataset.y)
    for sample_index in range(original_count):
        source_path = dataset.source_paths[sample_index]
        if source_path is None or not source_path.exists():
            skipped += 1
            continue

        try:
            sr, audio = read_wav_mono(source_path, args.target_sr)
            if args.max_duration > 0:
                audio = audio[: int(args.max_duration * sr)]

            for copy_index in range(1, args.augmentation_copies + 1):
                background_audio = None
                if getattr(args, "augmentation_background_mix", False):
                    candidates = [path for path in background_paths if path != source_path]
                    background_path = candidates[int(rng.integers(len(candidates)))] if candidates else source_path
                    if background_path not in background_cache:
                        background_sr, background = read_wav_mono(background_path, args.target_sr)
                        if background_sr != sr:
                            raise ValueError(
                                f"Background sample rate {background_sr} does not match source {sr}"
                            )
                        background_cache[background_path] = background[: audio.size]
                    background_audio = background_cache[background_path]
                augmented_audio = augment_waveform(
                    audio,
                    sr,
                    rng,
                    args,
                    background_audio=background_audio,
                )
                feature = feature_extractor(
                    augmented_audio,
                    sr,
                    args.n_fft,
                    args.hop_length,
                    args.n_mels,
                )
                if feature.shape != (expected_dim,) or not np.all(np.isfinite(feature)):
                    raise ValueError(f"expected finite ({expected_dim},) feature, got {feature.shape}")

                augmented_features.append(feature.astype(np.float32))
                augmented_labels.append(int(dataset.y[sample_index]))
                augmented_relative_paths.append(
                    augmented_relative_path(dataset.relative_paths[sample_index], copy_index)
                )
                # Keep the original path for traceability; augmented audio is not persisted.
                augmented_source_paths.append(source_path)
        except Exception as exc:
            skipped += 1
            print(f"Augmentation skipped {source_path}: {exc}")

    if not augmented_features:
        print(
            "Training WAV augmentation produced no samples; source_paths.json "
            "may be missing or point to unavailable WAV files"
        )
        return dataset

    updated_config = dict(dataset.config)
    updated_config["training_augmentation"] = {
        **config,
        "feature_contract": feature_contract,
        "original_train_samples": original_count,
        "augmented_samples": len(augmented_features),
        "skipped_source_samples": skipped,
    }
    updated_config["num_samples"] = original_count + len(augmented_features)

    augmented_X = np.vstack(augmented_features).astype(np.float32)
    print(
        f"Training WAV augmentation: {original_count} original + "
        f"{len(augmented_features)} augmented = {updated_config['num_samples']} samples"
    )
    return Dataset(
        X=np.vstack([dataset.X, augmented_X]).astype(np.float32),
        y=np.concatenate([dataset.y, np.asarray(augmented_labels, dtype=np.int64)]),
        relative_paths=[*dataset.relative_paths, *augmented_relative_paths],
        source_paths=[*dataset.source_paths, *augmented_source_paths],
        config=updated_config,
    )
