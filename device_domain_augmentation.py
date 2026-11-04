#!/usr/bin/env python3
"""Training-only device and outdoor-domain augmentation for audio classifiers.

The chain deliberately operates on waveform segments before Log-Mel extraction:
device impulse response, smooth microphone EQ, spectral tilt, gain, sensor
noise, recorded outdoor background, synthetic reverberation, then a non-clipping
peak guard.  Validation and test datasets never call this module.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence

import numpy as np
from scipy import signal
from scipy.io import wavfile

from audio_data_augmentation import apply_background_mix, apply_smooth_mic_eq
from audio_dataset import LABEL_TO_ID, Dataset, collect_wavs, read_wav_mono


FeatureExtractor = Callable[[np.ndarray, int, int, int, int], np.ndarray]


def add_device_domain_augmentation_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training-only device/outdoor augmentation")
    enabled = group.add_mutually_exclusive_group()
    enabled.add_argument(
        "--domain-augmentation",
        dest="domain_augmentation",
        action="store_true",
        help="Enable the full device/outdoor augmentation chain (default).",
    )
    enabled.add_argument(
        "--no-domain-augmentation",
        dest="domain_augmentation",
        action="store_false",
        help="Train only on clean sliding windows.",
    )
    parser.set_defaults(domain_augmentation=True)
    group.add_argument("--augmentation-copies", type=int, default=2)
    group.add_argument(
        "--device-ir-dir",
        nargs="*",
        default=None,
        help=(
            "Optional folders of measured device/room impulse WAVs. If omitted, "
            "a synthetic short device impulse response is generated."
        ),
    )
    group.add_argument("--augmentation-device-ir-probability", type=float, default=0.8)
    group.add_argument("--augmentation-device-ir-max-ms", type=float, default=80.0)
    group.add_argument("--augmentation-eq-probability", type=float, default=0.8)
    group.add_argument("--augmentation-eq-db", type=float, default=6.0)
    group.add_argument("--augmentation-tilt-probability", type=float, default=0.8)
    group.add_argument(
        "--augmentation-tilt-db-per-octave",
        type=float,
        default=3.0,
        help="Maximum absolute random spectral slope in dB per octave.",
    )
    group.add_argument("--augmentation-gain-min-db", type=float, default=-20.0)
    group.add_argument("--augmentation-gain-max-db", type=float, default=6.0)
    group.add_argument("--augmentation-noise-probability", type=float, default=0.5)
    group.add_argument("--augmentation-noise-snr-min-db", type=float, default=10.0)
    group.add_argument("--augmentation-noise-snr-max-db", type=float, default=30.0)
    group.add_argument(
        "--outdoor-background-dir",
        nargs="*",
        default=None,
        help=(
            "Optional unlabeled outdoor-background WAV roots. Background-labeled "
            "windows from the training split are always available as a fallback."
        ),
    )
    group.add_argument("--augmentation-background-probability", type=float, default=0.8)
    group.add_argument("--augmentation-background-snr-min-db", type=float, default=-5.0)
    group.add_argument("--augmentation-background-snr-max-db", type=float, default=20.0)
    group.add_argument("--augmentation-reverb-probability", type=float, default=0.5)
    group.add_argument("--augmentation-reverb-rt60-min-s", type=float, default=0.10)
    group.add_argument("--augmentation-reverb-rt60-max-s", type=float, default=0.80)
    group.add_argument("--augmentation-reverb-wet-min", type=float, default=0.10)
    group.add_argument("--augmentation-reverb-wet-max", type=float, default=0.40)
    group.add_argument("--augmentation-peak-limit", type=float, default=0.98)
    group.add_argument(
        "--save-augmented-wavs",
        action="store_true",
        help="Persist generated PCM16 train WAVs for audit; costs additional disk space.",
    )


def probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def validate_device_domain_augmentation_args(args: argparse.Namespace) -> None:
    if args.augmentation_copies < 0:
        raise ValueError("--augmentation-copies must be non-negative")
    for name in (
        "augmentation_device_ir_probability",
        "augmentation_eq_probability",
        "augmentation_tilt_probability",
        "augmentation_noise_probability",
        "augmentation_background_probability",
        "augmentation_reverb_probability",
    ):
        probability(float(getattr(args, name)), f"--{name.replace('_', '-')}")
    if args.augmentation_device_ir_max_ms <= 0.0:
        raise ValueError("--augmentation-device-ir-max-ms must be positive")
    if args.augmentation_eq_db < 0.0 or args.augmentation_tilt_db_per_octave < 0.0:
        raise ValueError("EQ and spectral-tilt limits must be non-negative")
    if args.augmentation_gain_min_db > args.augmentation_gain_max_db:
        raise ValueError("augmentation gain minimum exceeds maximum")
    if args.augmentation_noise_snr_min_db > args.augmentation_noise_snr_max_db:
        raise ValueError("augmentation noise SNR minimum exceeds maximum")
    if args.augmentation_background_snr_min_db > args.augmentation_background_snr_max_db:
        raise ValueError("augmentation background SNR minimum exceeds maximum")
    if not 0.0 < args.augmentation_reverb_rt60_min_s <= args.augmentation_reverb_rt60_max_s:
        raise ValueError("reverb RT60 range must be positive and ordered")
    if not 0.0 <= args.augmentation_reverb_wet_min <= args.augmentation_reverb_wet_max <= 1.0:
        raise ValueError("reverb wet range must satisfy 0 <= min <= max <= 1")
    if not 0.0 < args.augmentation_peak_limit <= 1.0:
        raise ValueError("--augmentation-peak-limit must be in (0, 1]")


def collect_wavs_from_roots(folder_texts: Sequence[str] | None) -> list[Path]:
    paths: list[Path] = []
    for text in folder_texts or []:
        root = Path(text).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        paths.extend(path.resolve() for path in collect_wavs(root))
    return sorted(set(paths))


def preserve_rms(reference: np.ndarray, transformed: np.ndarray) -> np.ndarray:
    reference_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2) + 1.0e-12))
    transformed_rms = float(np.sqrt(np.mean(transformed.astype(np.float64) ** 2) + 1.0e-12))
    if reference_rms <= 1.0e-8 or transformed_rms <= 1.0e-8:
        return transformed.astype(np.float32)
    return (transformed * (reference_rms / transformed_rms)).astype(np.float32)


def fft_convolve_fixed(audio: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    convolved = signal.fftconvolve(audio, impulse, mode="full")[: audio.size]
    return preserve_rms(audio, convolved.astype(np.float32))


def synthetic_device_impulse(
    sr: int, rng: np.random.Generator, max_ms: float
) -> np.ndarray:
    """Generate a causal packaging/device response when measured DIRs are unavailable."""
    maximum = max(16, int(round(max_ms * sr / 1000.0)))
    length = int(rng.integers(max(16, maximum // 4), maximum + 1))
    impulse = np.zeros(length, dtype=np.float32)
    impulse[0] = 1.0
    reflection_count = int(rng.integers(3, 9))
    for _ in range(reflection_count):
        delay = int(rng.integers(1, length))
        envelope = math.exp(-4.0 * delay / max(length - 1, 1))
        impulse[delay] += np.float32(rng.uniform(-0.35, 0.35) * envelope)
    # A short random low-pass pole models package/decimator smoothing without
    # fixing the response to any test microphone.
    pole = float(rng.uniform(0.05, 0.65))
    impulse = signal.lfilter([1.0 - pole], [1.0, -pole], impulse).astype(np.float32)
    impulse /= np.sqrt(np.sum(impulse.astype(np.float64) ** 2) + 1.0e-12)
    return impulse


def load_impulse(path: Path, sr: int, max_ms: float) -> np.ndarray:
    _, impulse = read_wav_mono(path, sr)
    maximum = max(1, int(round(max_ms * sr / 1000.0)))
    impulse = np.asarray(impulse[:maximum], dtype=np.float32)
    if not impulse.size or float(np.max(np.abs(impulse))) <= 1.0e-10:
        raise ValueError(f"Impulse response is empty or silent: {path}")
    impulse /= np.sqrt(np.sum(impulse.astype(np.float64) ** 2) + 1.0e-12)
    return impulse


def apply_spectral_tilt(
    audio: np.ndarray,
    sr: int,
    slope_db_per_octave: float,
) -> np.ndarray:
    if abs(slope_db_per_octave) <= 1.0e-12:
        return audio.copy()
    spectrum = np.fft.rfft(audio)
    frequencies = np.fft.rfftfreq(audio.size, 1.0 / sr)
    reference_hz = 1000.0
    octave = np.log2(np.maximum(frequencies, 50.0) / reference_hz)
    gain_db = np.clip(slope_db_per_octave * octave, -18.0, 18.0)
    gain = np.power(10.0, gain_db / 20.0)
    transformed = np.fft.irfft(spectrum * gain, n=audio.size).astype(np.float32)
    return preserve_rms(audio, transformed)


def colored_sensor_noise(
    sample_count: int, rng: np.random.Generator
) -> tuple[np.ndarray, str]:
    color = "pink" if bool(rng.integers(0, 2)) else "white"
    if color == "white":
        noise = rng.normal(0.0, 1.0, sample_count).astype(np.float32)
    else:
        bins = sample_count // 2 + 1
        real = rng.normal(0.0, 1.0, bins)
        imag = rng.normal(0.0, 1.0, bins)
        frequencies = np.arange(bins, dtype=np.float64)
        shaping = 1.0 / np.sqrt(np.maximum(frequencies, 1.0))
        noise = np.fft.irfft((real + 1j * imag) * shaping, n=sample_count).astype(np.float32)
    noise -= np.mean(noise, dtype=np.float64)
    noise /= np.sqrt(np.mean(noise.astype(np.float64) ** 2) + 1.0e-12)
    return noise.astype(np.float32), color


def add_noise_at_snr(
    audio: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    signal_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1.0e-12))
    noise_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2) + 1.0e-12))
    if signal_rms <= 1.0e-8 or noise_rms <= 1.0e-8:
        return audio.copy()
    target = signal_rms / (10.0 ** (snr_db / 20.0))
    return (audio + noise * (target / noise_rms)).astype(np.float32)


def synthetic_reverb(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    rt60_s: float,
    wet: float,
) -> np.ndarray:
    length = max(16, int(round(min(rt60_s, 1.5) * sr)))
    time = np.arange(length, dtype=np.float64) / sr
    decay = np.power(10.0, -3.0 * time / max(rt60_s, 1.0e-3))
    impulse = rng.normal(0.0, 1.0, length) * decay
    impulse[0] += 8.0
    impulse = impulse.astype(np.float32)
    impulse /= np.sqrt(np.sum(impulse.astype(np.float64) ** 2) + 1.0e-12)
    reverberant = signal.fftconvolve(audio, impulse, mode="full")[: audio.size]
    reverberant = preserve_rms(audio, reverberant.astype(np.float32))
    mixed = (1.0 - wet) * audio + wet * reverberant
    return preserve_rms(audio, mixed.astype(np.float32))


def random_audio_excerpt(
    path: Path,
    target_sr: int,
    sample_count: int,
    rng: np.random.Generator,
    cache: dict[Path, np.ndarray],
) -> np.ndarray:
    if path not in cache:
        _, cache[path] = read_wav_mono(path, target_sr)
    audio = cache[path]
    if audio.size > sample_count:
        start = int(rng.integers(0, audio.size - sample_count + 1))
        return np.asarray(audio[start : start + sample_count], dtype=np.float32)
    if audio.size < sample_count:
        return np.pad(audio, (0, sample_count - audio.size)).astype(np.float32)
    return np.asarray(audio, dtype=np.float32).copy()


def peak_guard(audio: np.ndarray, limit: float) -> tuple[np.ndarray, float, float]:
    absolute = np.abs(audio)
    over_fraction = float(np.mean(absolute > 1.0))
    peak = float(np.max(absolute)) if absolute.size else 0.0
    scale = 1.0 if peak <= limit or peak <= 1.0e-12 else limit / peak
    return (audio * scale).astype(np.float32), float(scale), over_fraction


def apply_domain_chain(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
    *,
    real_ir_path: Path | None,
    background_audio: np.ndarray | None,
    background_path: Path | None,
) -> tuple[np.ndarray, Dict[str, object]]:
    augmented = np.asarray(audio, dtype=np.float32)
    augmented = (augmented - np.mean(augmented, dtype=np.float64)).astype(np.float32)
    details: Dict[str, object] = {
        "device_ir": "none",
        "smooth_eq": False,
        "spectral_tilt_db_per_octave": 0.0,
        "gain_db": 0.0,
        "sensor_noise": "none",
        "sensor_noise_snr_db": "",
        "background_path": str(background_path) if background_path else "",
        "background_snr_db": "",
        "reverb_rt60_s": "",
        "reverb_wet": "",
    }

    if rng.random() < args.augmentation_device_ir_probability:
        if real_ir_path is not None:
            impulse = load_impulse(real_ir_path, sr, args.augmentation_device_ir_max_ms)
            details["device_ir"] = str(real_ir_path)
        else:
            impulse = synthetic_device_impulse(sr, rng, args.augmentation_device_ir_max_ms)
            details["device_ir"] = "synthetic"
        augmented = fft_convolve_fixed(augmented, impulse)

    if rng.random() < args.augmentation_eq_probability and args.augmentation_eq_db > 0.0:
        augmented = apply_smooth_mic_eq(
            augmented, sr, rng, args.augmentation_eq_db
        )
        augmented = preserve_rms(audio, augmented)
        details["smooth_eq"] = True

    if (
        rng.random() < args.augmentation_tilt_probability
        and args.augmentation_tilt_db_per_octave > 0.0
    ):
        slope = float(
            rng.uniform(
                -args.augmentation_tilt_db_per_octave,
                args.augmentation_tilt_db_per_octave,
            )
        )
        augmented = apply_spectral_tilt(augmented, sr, slope)
        details["spectral_tilt_db_per_octave"] = slope

    gain_db = float(
        rng.uniform(args.augmentation_gain_min_db, args.augmentation_gain_max_db)
    )
    augmented = (augmented * (10.0 ** (gain_db / 20.0))).astype(np.float32)
    details["gain_db"] = gain_db

    if rng.random() < args.augmentation_noise_probability:
        noise, color = colored_sensor_noise(augmented.size, rng)
        snr_db = float(
            rng.uniform(
                args.augmentation_noise_snr_min_db,
                args.augmentation_noise_snr_max_db,
            )
        )
        augmented = add_noise_at_snr(augmented, noise, snr_db)
        details["sensor_noise"] = color
        details["sensor_noise_snr_db"] = snr_db

    if background_audio is not None:
        background_snr = float(
            rng.uniform(
                args.augmentation_background_snr_min_db,
                args.augmentation_background_snr_max_db,
            )
        )
        # Pass a fixed SNR range so the shared implementation applies the
        # exact value stored in the augmentation manifest.
        augmented = apply_background_mix(
            augmented,
            background_audio,
            rng,
            background_snr,
            background_snr,
        )
        details["background_snr_db"] = background_snr

    if rng.random() < args.augmentation_reverb_probability:
        rt60 = float(
            rng.uniform(
                args.augmentation_reverb_rt60_min_s,
                args.augmentation_reverb_rt60_max_s,
            )
        )
        wet = float(
            rng.uniform(
                args.augmentation_reverb_wet_min,
                args.augmentation_reverb_wet_max,
            )
        )
        augmented = synthetic_reverb(augmented, sr, rng, rt60, wet)
        details["reverb_rt60_s"] = rt60
        details["reverb_wet"] = wet

    augmented, scale, over_fraction = peak_guard(
        augmented, args.augmentation_peak_limit
    )
    details["peak_guard_scale"] = scale
    details["pre_guard_over_full_scale_fraction"] = over_fraction
    return augmented, details


def augmentation_config_from_args(
    args: argparse.Namespace,
    measured_ir_count: int,
    external_background_count: int,
) -> Dict[str, object]:
    return {
        "training_only": True,
        "enabled": bool(args.domain_augmentation and args.augmentation_copies > 0),
        "copies_per_clean_window": int(args.augmentation_copies),
        "order": [
            "dc_removal",
            "device_impulse_response",
            "smooth_mic_eq",
            "spectral_tilt",
            "gain",
            "sensor_noise_floor",
            "recorded_outdoor_background",
            "synthetic_reverb",
            "non_clipping_peak_guard",
            "pcm16_round_trip",
        ],
        "device_impulse_response": {
            "probability": float(args.augmentation_device_ir_probability),
            "measured_ir_count": measured_ir_count,
            "fallback": "synthetic_short_device_ir",
            "max_ms": float(args.augmentation_device_ir_max_ms),
        },
        "smooth_eq": {
            "probability": float(args.augmentation_eq_probability),
            "max_db": float(args.augmentation_eq_db),
        },
        "spectral_tilt": {
            "probability": float(args.augmentation_tilt_probability),
            "max_abs_db_per_octave": float(args.augmentation_tilt_db_per_octave),
        },
        "gain_range_db": [
            float(args.augmentation_gain_min_db),
            float(args.augmentation_gain_max_db),
        ],
        "sensor_noise": {
            "probability": float(args.augmentation_noise_probability),
            "types": ["white", "pink"],
            "snr_db": [
                float(args.augmentation_noise_snr_min_db),
                float(args.augmentation_noise_snr_max_db),
            ],
        },
        "recorded_background": {
            "probability": float(args.augmentation_background_probability),
            "external_file_count": external_background_count,
            "also_uses_training_background_class": True,
            "snr_db": [
                float(args.augmentation_background_snr_min_db),
                float(args.augmentation_background_snr_max_db),
            ],
        },
        "reverb": {
            "probability": float(args.augmentation_reverb_probability),
            "rt60_s": [
                float(args.augmentation_reverb_rt60_min_s),
                float(args.augmentation_reverb_rt60_max_s),
            ],
            "wet": [
                float(args.augmentation_reverb_wet_min),
                float(args.augmentation_reverb_wet_max),
            ],
        },
        "peak_limit": float(args.augmentation_peak_limit),
        "random_state": int(args.random_state),
        "augmented_wavs_persisted": bool(args.save_augmented_wavs),
    }


def write_manifest(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def augment_training_dataset_device_domain(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    output_dir: Path,
    feature_extractor: FeatureExtractor,
    expected_dim: int,
    feature_contract: str,
) -> Dataset:
    """Append reproducible synthetic-device views to the clean training windows."""
    validate_device_domain_augmentation_args(args)
    measured_ir_paths = collect_wavs_from_roots(args.device_ir_dir)
    external_background_paths = collect_wavs_from_roots(args.outdoor_background_dir)
    config = augmentation_config_from_args(
        args, len(measured_ir_paths), len(external_background_paths)
    )
    if not config["enabled"]:
        updated = dict(dataset.config)
        updated["training_augmentation"] = config
        return Dataset(
            dataset.X,
            dataset.y,
            list(dataset.relative_paths),
            list(dataset.source_paths),
            updated,
        )

    training_background_paths = [
        path
        for label, path in zip(dataset.y, dataset.source_paths)
        if int(label) == LABEL_TO_ID["background"]
        and path is not None
        and path.exists()
    ]
    background_paths = sorted(set([*external_background_paths, *training_background_paths]))
    if args.augmentation_background_probability > 0.0 and not background_paths:
        raise ValueError(
            "Background augmentation requires --outdoor-background-dir or "
            "background-labeled windows in the training split"
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    relative_paths: list[str] = []
    source_paths: list[Path | None] = []
    manifest_rows: list[Dict[str, object]] = []
    audio_cache: dict[Path, np.ndarray] = {}
    skipped = 0
    augmented_root = output_dir / "wav_segments" / "train_augmented"

    for sample_index, source_path in enumerate(dataset.source_paths):
        if source_path is None or not source_path.exists():
            skipped += args.augmentation_copies
            continue
        try:
            sr, clean_audio = read_wav_mono(source_path, args.target_sr)
            clean_audio = clean_audio[: int(round(args.segment_seconds * sr))]
            for copy_index in range(1, args.augmentation_copies + 1):
                # Independent seeds make every row replayable even if dataset
                # ordering changes in a future run.
                seed_sequence = np.random.SeedSequence(
                    [args.random_state, sample_index, copy_index]
                )
                sample_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
                rng = np.random.default_rng(sample_seed)
                real_ir_path = None
                if measured_ir_paths:
                    real_ir_path = measured_ir_paths[int(rng.integers(len(measured_ir_paths)))]

                background_path = None
                background_audio = None
                if (
                    background_paths
                    and rng.random() < args.augmentation_background_probability
                ):
                    background_path = background_paths[int(rng.integers(len(background_paths)))]
                    background_audio = random_audio_excerpt(
                        background_path,
                        args.target_sr,
                        clean_audio.size,
                        rng,
                        audio_cache,
                    )

                augmented, details = apply_domain_chain(
                    clean_audio,
                    sr,
                    rng,
                    args,
                    real_ir_path=real_ir_path,
                    background_audio=background_audio,
                    background_path=background_path,
                )
                pcm16 = np.rint(np.clip(augmented, -1.0, 1.0) * 32767.0).astype(np.int16)
                feature_audio = pcm16.astype(np.float32) / 32767.0
                feature = feature_extractor(
                    feature_audio,
                    sr,
                    args.n_fft,
                    args.hop_length,
                    args.n_mels,
                )
                if feature.shape != (expected_dim,) or not np.all(np.isfinite(feature)):
                    raise ValueError(
                        f"expected finite ({expected_dim},) feature, got {feature.shape}"
                    )

                original_relative = Path(dataset.relative_paths[sample_index])
                augmented_relative = (
                    original_relative.parent
                    / f"{original_relative.stem}__aug{copy_index:02d}.wav"
                )
                persisted_path = None
                if args.save_augmented_wavs:
                    persisted_path = augmented_root / augmented_relative
                    persisted_path.parent.mkdir(parents=True, exist_ok=True)
                    wavfile.write(persisted_path, sr, pcm16)

                features.append(feature.astype(np.float32))
                labels.append(int(dataset.y[sample_index]))
                relative_paths.append(augmented_relative.as_posix())
                source_paths.append(persisted_path)
                manifest_rows.append(
                    {
                        "source_index": sample_index,
                        "source_path": str(source_path),
                        "relative_path": augmented_relative.as_posix(),
                        "label_id": int(dataset.y[sample_index]),
                        "copy_index": copy_index,
                        "seed": sample_seed,
                        **details,
                    }
                )
        except Exception as exc:
            skipped += args.augmentation_copies
            print(f"Device augmentation skipped {source_path}: {exc}")

    if not features:
        raise ValueError("Device-domain augmentation produced no usable training samples")

    manifest_path = output_dir / "augmentation_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    updated_config = dict(dataset.config)
    updated_config["training_augmentation"] = {
        **config,
        "feature_contract": feature_contract,
        "clean_train_windows": int(len(dataset.y)),
        "augmented_windows": int(len(features)),
        "skipped_augmented_windows": int(skipped),
        "manifest_path": str(manifest_path),
    }
    print(
        f"Device-domain augmentation: {len(dataset.y)} clean + {len(features)} "
        f"augmented = {len(dataset.y) + len(features)} train windows"
    )
    return Dataset(
        X=np.vstack([dataset.X, np.vstack(features)]).astype(np.float32),
        y=np.concatenate([dataset.y, np.asarray(labels, dtype=np.int64)]),
        relative_paths=[*dataset.relative_paths, *relative_paths],
        source_paths=[*dataset.source_paths, *source_paths],
        config=updated_config,
    )
