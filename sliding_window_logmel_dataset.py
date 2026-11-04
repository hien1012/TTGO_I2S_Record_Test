#!/usr/bin/env python3
"""Leakage-safe sliding-window WAV dataset builder for Log-Mel training.

Original recordings (or same-content groups across microphone roots) are
assigned to train/validation before any overlapping child windows are made.
External test roots are never used for split selection or augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.io import wavfile

import audio_dataset
from audio_dataset import read_wav_mono
from long_wav_segment_dataset import (
    OriginalWav,
    collect_original_wavs,
    content_group_key,
    split_content_group_train_val,
)


FeatureExtractor = Callable[[np.ndarray, int, int, int, int], np.ndarray]


@dataclass
class TestVotingIndex:
    """Hierarchy linking model windows to decision segments and original WAVs."""

    decisions: List[Dict]
    wavs: List[Dict]
    config: Dict


def float_audio_to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Persist the exact finite-resolution waveform used for feature extraction."""
    return np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def sliding_starts(sample_count: int, window_samples: int, hop_samples: int) -> List[int]:
    """Return starts for complete overlapping windows; incomplete tails are dropped."""
    if min(window_samples, hop_samples) <= 0:
        raise ValueError("window_samples and hop_samples must be positive")
    if hop_samples > window_samples:
        raise ValueError("Sliding-window hop cannot exceed the window duration")
    if sample_count < window_samples:
        return []
    return list(range(0, sample_count - window_samples + 1, hop_samples))


def decision_spans(
    sample_count: int,
    decision_samples: int,
    tail_policy: str,
) -> List[Tuple[int, int]]:
    """Return non-overlapping decision spans before child-window voting.

    The span end is the number of valid source samples.  A short final span is
    returned only for ``pad``; the caller pads it to ``decision_samples``.
    """
    if sample_count <= 0 or decision_samples <= 0:
        return []
    if tail_policy not in {"drop", "pad"}:
        raise ValueError(f"Unsupported test tail policy: {tail_policy}")

    spans: List[Tuple[int, int]] = []
    for start in range(0, sample_count, decision_samples):
        end = min(start + decision_samples, sample_count)
        if end - start < decision_samples and tail_policy == "drop":
            break
        spans.append((start, end))
    return spans


def window_relative_path(
    item: OriginalWav,
    window_index: int,
    start_sample: int,
    window_samples: int,
    sample_rate: int,
) -> Path:
    """Keep the established ``__seg`` convention while recording actual times."""
    start_ms = int(round(start_sample * 1000.0 / sample_rate))
    end_ms = int(round((start_sample + window_samples) * 1000.0 / sample_rate))
    filename = (
        f"{item.relative_path.stem}__seg{window_index:04d}_"
        f"{start_ms:08d}-{end_ms:08d}ms.wav"
    )
    return Path(item.source_tag) / item.relative_path.parent / filename


def build_sliding_dataset(
    originals: Sequence[OriginalWav],
    split_name: str,
    output_dir: Path,
    args,
    *,
    feature_extractor: FeatureExtractor,
    expected_dim: int,
    feature_type: str,
    feature_source: str,
    feature_layout: Dict,
    window_seconds: float,
    hop_seconds: float,
) -> audio_dataset.Dataset:
    """Create fixed-size, overlapping windows independently inside each original."""
    window_seconds = float(window_seconds)
    hop_seconds = float(hop_seconds)
    window_samples = int(round(window_seconds * args.target_sr))
    window_hop_samples = int(round(hop_seconds * args.target_sr))
    segment_root = output_dir / "wav_segments" / split_name

    features: List[np.ndarray] = []
    labels: List[int] = []
    relative_paths: List[str] = []
    source_paths: List[Path] = []
    windows_per_original: Dict[str, int] = {}
    uncovered_tail_samples: Dict[str, int] = {}
    skipped_originals = 0

    for file_index, item in enumerate(originals, start=1):
        try:
            sr, audio = read_wav_mono(item.path, args.target_sr)
            starts = sliding_starts(len(audio), window_samples, window_hop_samples)
            if not starts:
                skipped_originals += 1
                print(f"Skipped shorter-than-{window_seconds:g}s original: {item.path}")
                continue

            windows_per_original[str(item.path)] = len(starts)
            uncovered_tail_samples[str(item.path)] = int(
                len(audio) - (starts[-1] + window_samples)
            )
            for window_index, start in enumerate(starts):
                window = audio[start : start + window_samples]
                pcm16 = float_audio_to_pcm16(window)
                relative_path = window_relative_path(
                    item,
                    window_index,
                    start,
                    window_samples,
                    sr,
                )
                saved_path = segment_root / relative_path
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                wavfile.write(saved_path, sr, pcm16)

                # Re-read-equivalent float values make Python evaluation and
                # firmware comparison reproducible down to the PCM16 boundary.
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
                        f"expected finite feature ({expected_dim},), got {feature.shape}"
                    )
                features.append(feature.astype(np.float32))
                labels.append(item.label_id)
                relative_paths.append(relative_path.as_posix())
                source_paths.append(saved_path)

            print(
                f"  [{file_index}/{len(originals)}] {item.path.name}: "
                f"{len(starts)} windows ({window_seconds:g}s, hop={hop_seconds:g}s) "
                f"-> {split_name}"
            )
        except Exception as exc:
            skipped_originals += 1
            print(f"Error processing original WAV {item.path}: {exc}")

    if not features:
        raise ValueError(f"No complete sliding windows produced for {split_name}")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    config = {
        "split": split_name,
        "split_unit": "original_wav_or_content_group_before_sliding_window",
        "segment_method": "sliding_window",
        "segment_seconds": window_seconds,
        "segment_hop_seconds": hop_seconds,
        "segment_samples": window_samples,
        "segment_hop_samples": window_hop_samples,
        "segment_overlap_samples": window_samples - window_hop_samples,
        "segment_overlap_ratio": 1.0 - window_hop_samples / window_samples,
        "incomplete_tail_policy": "drop",
        "target_sr": args.target_sr,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "n_features": expected_dim,
        "feature_type": feature_type,
        "feature_source": feature_source,
        "feature_layout": feature_layout,
        "class_names": audio_dataset.DEFAULT_CLASS_NAMES,
        "original_file_count": len(originals),
        "processed_original_file_count": len(windows_per_original),
        "skipped_original_file_count": skipped_originals,
        "segment_count": len(y),
        "segments_per_original": windows_per_original,
        "uncovered_tail_samples_per_original": uncovered_tail_samples,
    }
    dataset = audio_dataset.Dataset(X, y, relative_paths, source_paths, config)
    audio_dataset.save_npy_dataset(output_dir / "npy_original_splits" / split_name, dataset)
    print(f"Built {split_name}: {len(originals)} originals -> {len(y)} windows {X.shape}")
    return dataset


def test_window_relative_path(
    item: OriginalWav,
    decision_index: int,
    window_index: int,
    absolute_start_sample: int,
    window_samples: int,
    sample_rate: int,
) -> Path:
    """Build a stable path that preserves both decision and child indices."""
    start_ms = int(round(absolute_start_sample * 1000.0 / sample_rate))
    end_ms = int(round((absolute_start_sample + window_samples) * 1000.0 / sample_rate))
    filename = (
        f"{item.relative_path.stem}__decision{decision_index:04d}_"
        f"window{window_index:04d}_{start_ms:08d}-{end_ms:08d}ms.wav"
    )
    return Path(item.source_tag) / item.relative_path.parent / filename


def decision_relative_path(
    item: OriginalWav,
    decision_index: int,
    start_sample: int,
    decision_samples: int,
    sample_rate: int,
) -> Path:
    """Build the path of the fixed-duration clip represented by one vote."""
    start_ms = int(round(start_sample * 1000.0 / sample_rate))
    end_ms = int(round((start_sample + decision_samples) * 1000.0 / sample_rate))
    filename = (
        f"{item.relative_path.stem}__decision{decision_index:04d}_"
        f"{start_ms:08d}-{end_ms:08d}ms.wav"
    )
    return Path(item.source_tag) / item.relative_path.parent / filename


def build_hierarchical_test_dataset(
    originals: Sequence[OriginalWav],
    output_dir: Path,
    args,
    *,
    feature_extractor: FeatureExtractor,
    expected_dim: int,
    feature_type: str,
    feature_source: str,
    feature_layout: Dict,
) -> Tuple[audio_dataset.Dataset, TestVotingIndex]:
    """Build test windows while retaining decision-segment and WAV ownership."""
    sample_rate = int(args.target_sr)
    window_seconds = float(args.model_window_seconds)
    decision_seconds = float(args.test_decision_seconds)
    vote_hop_seconds = float(args.test_vote_hop_seconds)
    window_samples = int(round(window_seconds * sample_rate))
    decision_samples = int(round(decision_seconds * sample_rate))
    vote_hop_samples = int(round(vote_hop_seconds * sample_rate))
    segment_root = output_dir / "wav_segments" / "test"
    decision_root = output_dir / "wav_decisions" / "test"

    features: List[np.ndarray] = []
    labels: List[int] = []
    relative_paths: List[str] = []
    source_paths: List[Path] = []
    decisions: List[Dict] = []
    wavs: List[Dict] = []
    skipped_originals = 0
    dropped_tail_samples: Dict[str, int] = {}

    for file_index, item in enumerate(originals, start=1):
        feature_base = len(features)
        decision_base = len(decisions)
        try:
            sr, audio = read_wav_mono(item.path, sample_rate)
            spans = decision_spans(len(audio), decision_samples, args.test_tail_policy)
            if not spans:
                skipped_originals += 1
                print(
                    f"Skipped test WAV without a complete {decision_seconds:g}s "
                    f"decision segment: {item.path}"
                )
                continue

            wav_decision_indices: List[int] = []
            covered_samples = 0
            for local_decision_index, (start, valid_end) in enumerate(spans):
                valid_samples = valid_end - start
                decision_audio = np.zeros(decision_samples, dtype=np.float32)
                decision_audio[:valid_samples] = audio[start:valid_end]
                decision_pcm16 = float_audio_to_pcm16(decision_audio)
                decision_rel = decision_relative_path(
                    item,
                    local_decision_index,
                    start,
                    decision_samples,
                    sr,
                )
                decision_path = decision_root / decision_rel
                decision_path.parent.mkdir(parents=True, exist_ok=True)
                wavfile.write(decision_path, sr, decision_pcm16)

                child_starts = sliding_starts(
                    decision_samples,
                    window_samples,
                    vote_hop_samples,
                )
                child_indices: List[int] = []
                child_ranges: List[Dict] = []
                for local_window_index, child_start in enumerate(child_starts):
                    child_pcm16 = decision_pcm16[
                        child_start : child_start + window_samples
                    ]
                    absolute_start = start + child_start
                    relative_path = test_window_relative_path(
                        item,
                        local_decision_index,
                        local_window_index,
                        absolute_start,
                        window_samples,
                        sr,
                    )
                    saved_path = segment_root / relative_path
                    saved_path.parent.mkdir(parents=True, exist_ok=True)
                    wavfile.write(saved_path, sr, child_pcm16)

                    feature_audio = child_pcm16.astype(np.float32) / 32767.0
                    feature = feature_extractor(
                        feature_audio,
                        sr,
                        args.n_fft,
                        args.hop_length,
                        args.n_mels,
                    )
                    if feature.shape != (expected_dim,) or not np.all(np.isfinite(feature)):
                        raise ValueError(
                            f"expected finite feature ({expected_dim},), got {feature.shape}"
                        )

                    global_window_index = len(features)
                    features.append(feature.astype(np.float32))
                    labels.append(item.label_id)
                    relative_paths.append(relative_path.as_posix())
                    source_paths.append(saved_path)
                    child_indices.append(global_window_index)
                    child_ranges.append(
                        {
                            "window_index": global_window_index,
                            "start_sample": int(absolute_start),
                            "end_sample": int(absolute_start + window_samples),
                            "start_seconds": float(absolute_start / sr),
                            "end_seconds": float((absolute_start + window_samples) / sr),
                            "path": str(saved_path),
                        }
                    )

                decision_global_index = len(decisions)
                decisions.append(
                    {
                        "decision_index": decision_global_index,
                        "wav_index": len(wavs),
                        "source_tag": item.source_tag,
                        "original_path": str(item.path),
                        "original_relative_path": item.relative_path.as_posix(),
                        "label_name": item.label_name,
                        "label_id": int(item.label_id),
                        "decision_index_within_wav": int(local_decision_index),
                        "start_sample": int(start),
                        "valid_end_sample": int(valid_end),
                        "target_end_sample": int(start + decision_samples),
                        "start_seconds": float(start / sr),
                        "valid_end_seconds": float(valid_end / sr),
                        "target_end_seconds": float((start + decision_samples) / sr),
                        "valid_samples": int(valid_samples),
                        "padded_samples": int(decision_samples - valid_samples),
                        "decision_audio_path": str(decision_path),
                        "window_indices": child_indices,
                        "windows": child_ranges,
                    }
                )
                wav_decision_indices.append(decision_global_index)
                covered_samples = max(covered_samples, valid_end)

            dropped_tail_samples[str(item.path)] = max(0, len(audio) - covered_samples)
            wavs.append(
                {
                    "wav_index": len(wavs),
                    "source_tag": item.source_tag,
                    "original_path": str(item.path),
                    "original_relative_path": item.relative_path.as_posix(),
                    "label_name": item.label_name,
                    "label_id": int(item.label_id),
                    "sample_count": int(len(audio)),
                    "duration_seconds": float(len(audio) / sr),
                    "decision_indices": wav_decision_indices,
                }
            )
            print(
                f"  [{file_index}/{len(originals)}] {item.path.name}: "
                f"{len(spans)} decisions, "
                f"{sum(len(decisions[index]['window_indices']) for index in wav_decision_indices)} "
                "vote windows -> test"
            )
        except Exception as exc:
            # Do not leave orphaned indices if extraction fails partway through
            # an original WAV. Saved audit WAVs may remain, but no dataset row
            # references them and the next successful file starts cleanly.
            del features[feature_base:]
            del labels[feature_base:]
            del relative_paths[feature_base:]
            del source_paths[feature_base:]
            del decisions[decision_base:]
            skipped_originals += 1
            print(f"Error processing test WAV {item.path}: {exc}")

    if not features or not decisions:
        raise ValueError("No hierarchical test decision windows were produced")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    config = {
        "split": "test",
        "split_unit": "external_original_wav_then_decision_then_sliding_window",
        "segment_method": "hierarchical_sliding_vote",
        "model_window_seconds": window_seconds,
        "model_window_samples": window_samples,
        "test_decision_seconds": decision_seconds,
        "test_decision_samples": decision_samples,
        "test_vote_hop_seconds": vote_hop_seconds,
        "test_vote_hop_samples": vote_hop_samples,
        "test_tail_policy": args.test_tail_policy,
        "target_sr": sample_rate,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "n_features": expected_dim,
        "feature_type": feature_type,
        "feature_source": feature_source,
        "feature_layout": feature_layout,
        "class_names": audio_dataset.DEFAULT_CLASS_NAMES,
        "original_file_count": len(originals),
        "processed_original_file_count": len(wavs),
        "skipped_original_file_count": skipped_originals,
        "decision_count": len(decisions),
        "window_count": len(y),
        "dropped_tail_samples_per_original": dropped_tail_samples,
    }
    dataset = audio_dataset.Dataset(X, y, relative_paths, source_paths, config)
    voting_index = TestVotingIndex(decisions=decisions, wavs=wavs, config=config.copy())
    audio_dataset.save_npy_dataset(output_dir / "npy_original_splits" / "test", dataset)
    audio_dataset.write_json(
        output_dir / "test_voting_manifest.json",
        {"config": voting_index.config, "decisions": decisions, "wavs": wavs},
    )
    print(
        f"Built test hierarchy: {len(wavs)} WAVs -> {len(decisions)} decisions "
        f"-> {len(y)} model windows {X.shape}"
    )
    return dataset, voting_index


def validate_input_output_roots(args, output_dir: Path) -> None:
    roots = [Path(text).expanduser().resolve() for text in [*args.train_dir, *args.test_dir]]
    resolved_output = output_dir.expanduser().resolve()
    for root in roots:
        if resolved_output == root or root in resolved_output.parents:
            raise ValueError(f"Output directory must not be inside input WAV root: {root}")


def load_sliding_train_val_test(
    args,
    output_dir: Path,
    *,
    feature_extractor: FeatureExtractor,
    expected_dim: int,
    feature_type: str,
    feature_source: str,
    feature_layout: Dict,
    training_augmenter=None,
) -> Tuple[
    audio_dataset.Dataset,
    audio_dataset.Dataset,
    audio_dataset.Dataset,
    List[Dict],
    TestVotingIndex,
]:
    """Load multiple train/test roots with leakage-safe train/validation grouping."""
    if not args.train_dir or not args.test_dir:
        raise ValueError("Sliding-window training requires --train-dir and --test-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_input_output_roots(args, output_dir)

    train_pool = collect_original_wavs(args.train_dir, "train pool")
    test_originals = collect_original_wavs(args.test_dir, "external test")
    train_paths = {item.path.resolve() for item in train_pool}
    test_paths = {item.path.resolve() for item in test_originals}
    overlap = sorted(str(path) for path in train_paths & test_paths)
    if overlap:
        raise ValueError(f"Original WAVs overlap between train and test: {overlap[:5]}")

    # Always split by recording/content ID before windowing.  This also handles
    # one training root and guarantees that future duplicate IDs cannot leak.
    train_originals, val_originals, content_keys = split_content_group_train_val(
        train_pool,
        val_split=args.val_split,
        random_state=args.random_state,
    )
    train_ids = {content_group_key(item) for item in train_originals}
    val_ids = {content_group_key(item) for item in val_originals}
    overlap_ids = sorted(train_ids & val_ids)
    if overlap_ids:
        raise AssertionError(f"Train/validation content-ID overlap: {overlap_ids[:5]}")

    manifest_path = output_dir / "original_file_split.json"
    audio_dataset.write_json(
        manifest_path,
        {
            "split_unit": "content_id_before_sliding_window",
            "segment_method": "sliding_window",
            "model_window_seconds": int(args.model_window_seconds),
            "train_window_hop_seconds": float(args.train_window_hop_seconds),
            "test_decision_seconds": int(args.test_decision_seconds),
            "test_vote_hop_seconds": float(args.test_vote_hop_seconds),
            "random_state": int(args.random_state),
            "train_ratio": float(1.0 - args.val_split),
            "val_ratio": float(args.val_split),
            "test_policy": "all_wavs_from_external_test_roots",
            "content_id_split_enabled": True,
            "train_val_id_overlap_count": 0,
            "content_group_keys": content_keys,
            "train_original_wavs": [str(item.path) for item in train_originals],
            "val_original_wavs": [str(item.path) for item in val_originals],
            "test_original_wavs": [str(item.path) for item in test_originals],
        },
    )

    common = {
        "feature_extractor": feature_extractor,
        "expected_dim": expected_dim,
        "feature_type": feature_type,
        "feature_source": feature_source,
        "feature_layout": feature_layout,
    }
    train_data = build_sliding_dataset(
        train_originals,
        "train",
        output_dir,
        args,
        window_seconds=args.model_window_seconds,
        hop_seconds=args.train_window_hop_seconds,
        **common,
    )
    val_data = build_sliding_dataset(
        val_originals,
        "val",
        output_dir,
        args,
        window_seconds=args.model_window_seconds,
        hop_seconds=args.train_window_hop_seconds,
        **common,
    )
    test_data, test_voting_index = build_hierarchical_test_dataset(
        test_originals,
        output_dir,
        args,
        **common,
    )

    for dataset in (train_data, val_data, test_data):
        dataset.config["split_manifest_path"] = str(manifest_path)
        dataset.config["content_group_split"] = True
        dataset.config["validation_source"] = "train_roots_grouped_before_windowing"
    train_data.config["content_group_keys"] = content_keys["train"]
    val_data.config["content_group_keys"] = content_keys["val"]
    train_data.config["content_group_overlap_count"] = 0
    val_data.config["content_group_overlap_count"] = 0
    test_data.config["test_source"] = "external_test_roots_untouched"

    # Save clean splits before generating training-only synthetic device views.
    clean_root = output_dir / "npy_original_splits"
    audio_dataset.save_npy_dataset(clean_root / "train", train_data)
    audio_dataset.save_npy_dataset(clean_root / "val", val_data)
    audio_dataset.save_npy_dataset(clean_root / "test", test_data)
    if training_augmenter is not None:
        train_data = training_augmenter(train_data, args)

    consumed_root = output_dir / "npy_splits"
    audio_dataset.save_npy_dataset(consumed_root / "train", train_data)
    audio_dataset.save_npy_dataset(consumed_root / "val", val_data)
    audio_dataset.save_npy_dataset(consumed_root / "test", test_data)
    return (
        train_data,
        val_data,
        test_data,
        [train_data.config, val_data.config, test_data.config],
        test_voting_index,
    )
