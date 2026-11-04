#!/usr/bin/env python3
"""Split original WAV files by source before creating fixed-length segments.

Entry points used by train_cnn_from_wav_logmel.py:
  - load_segmented_train_val_test: --train-dir pool -> train/val; --test-dir -> test.
  - load_grouped_cross_mic_train_val_test: --grouped-cross-mic-split mode.
Splitting happens on whole recordings first, so segments of one WAV never leak
across splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

import audio_dataset
from audio_dataset import infer_label, read_wav_mono


FeatureExtractor = Callable[[np.ndarray, int, int, int, int], np.ndarray]


@dataclass(frozen=True)
class OriginalWav:
    path: Path
    root: Path
    source_tag: str
    relative_path: Path
    label_name: str
    label_id: int


def source_tags(roots: Sequence[Path]) -> List[str]:
    names = [root.name or "source" for root in roots]
    return [name if names.count(name) == 1 else f"{index:02d}_{name}" for index, name in enumerate(names)]


def collect_original_wavs(folder_texts: Sequence[str], split_name: str) -> List[OriginalWav]:
    """Collect labeled original files without slicing them yet."""
    roots = [Path(text).expanduser().resolve() for text in folder_texts]
    missing = [str(root) for root in roots if not root.exists()]
    if missing:
        raise FileNotFoundError(f"{split_name} input folders not found: {missing}")

    originals: List[OriginalWav] = []
    for root, source_tag in zip(roots, source_tags(roots)):
        for path in audio_dataset.collect_wavs(root):
            label_name = infer_label(path)
            if label_name is None:
                print(f"Skipped unlabeled original WAV: {path}")
                continue
            originals.append(
                OriginalWav(
                    path=path,
                    root=root,
                    source_tag=source_tag,
                    relative_path=path.relative_to(root),
                    label_name=label_name,
                    label_id=audio_dataset.LABEL_TO_ID[label_name],
                )
            )

    if not originals:
        raise ValueError(f"No labeled original WAV files found for {split_name}")
    print(f"Original {split_name} WAV files: {len(originals)}")
    return originals


def split_original_train_val(
    originals: Sequence[OriginalWav],
    val_split: float,
    random_state: int,
) -> Tuple[List[OriginalWav], List[OriginalWav]]:
    """Stratify at original-file level so child segments cannot cross splits."""
    if val_split <= 0 or val_split >= 1:
        raise ValueError("--val-split must be greater than 0 and less than 1")

    indices = np.arange(len(originals))
    labels = np.asarray([item.label_id for item in originals], dtype=np.int64)
    try:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_split,
            stratify=labels,
            random_state=random_state,
        )
    except ValueError as exc:
        counts = {name: int(np.sum(labels == label_id)) for name, label_id in audio_dataset.LABEL_TO_ID.items()}
        raise ValueError(
            "Cannot create a stratified validation split from original WAV files. "
            f"Original-file class counts: {counts}; val_split={val_split}."
        ) from exc

    train_files = [originals[int(index)] for index in train_indices]
    val_files = [originals[int(index)] for index in val_indices]
    print(
        f"Original-file split: {len(train_files)} train / {len(val_files)} val "
        f"(val ratio={val_split:.3f})"
    )
    return train_files, val_files


def group_originals_by_content(originals: Sequence[OriginalWav]) -> Dict[str, List[OriginalWav]]:
    """Group same recording IDs across microphone folders before splitting.

    Multiple training roots often contain the same played content recorded by
    different microphones, for example ``ics/drone_1.wav`` and
    ``pdm/drone_1.wav``.  Splitting those files independently leaks content
    into validation.  The content key deliberately excludes the source/mic tag
    and keeps label plus filename ID so every mic version stays in one split.
    """
    grouped: Dict[str, List[OriginalWav]] = {}
    for item in originals:
        grouped.setdefault(content_group_key(item), []).append(item)
    return grouped


def split_content_group_train_val(
    originals: Sequence[OriginalWav],
    val_split: float,
    random_state: int,
) -> Tuple[List[OriginalWav], List[OriginalWav], Dict[str, List[str]]]:
    """Stratify train/val by recording ID group instead of individual files."""
    if val_split <= 0 or val_split >= 1:
        raise ValueError("--val-split must be greater than 0 and less than 1")

    grouped = group_originals_by_content(originals)
    keys = sorted(grouped)
    labels = np.asarray([grouped[key][0].label_id for key in keys], dtype=np.int64)
    key_indices = np.arange(len(keys))
    try:
        train_indices, val_indices = train_test_split(
            key_indices,
            test_size=val_split,
            stratify=labels,
            random_state=random_state,
        )
    except ValueError as exc:
        counts = {
            name: int(np.sum(labels == label_id))
            for name, label_id in audio_dataset.LABEL_TO_ID.items()
        }
        raise ValueError(
            "Cannot create a stratified validation split from content groups. "
            f"Content-group class counts: {counts}; val_split={val_split}."
        ) from exc

    train_keys = {keys[int(index)] for index in train_indices}
    val_keys = {keys[int(index)] for index in val_indices}
    if train_keys & val_keys:
        raise AssertionError("Content-group train/val overlap detected")

    train_files = [item for key in sorted(train_keys) for item in grouped[key]]
    val_files = [item for key in sorted(val_keys) for item in grouped[key]]
    split_keys = {"train": sorted(train_keys), "val": sorted(val_keys)}
    print(
        "Content-group train/val split: "
        f"{len(train_keys)} train / {len(val_keys)} val groups "
        f"(val ratio={val_split:.3f})"
    )
    return train_files, val_files, split_keys


def content_group_key(item: OriginalWav) -> str:
    """Identify the same played source across PDM/ICS/INMP folder trees."""
    parent = item.relative_path.parts[0].lower() if len(item.relative_path.parts) > 1 else "default"
    batch = parent.split("_", 1)[1] if "_" in parent else parent
    stem = item.relative_path.stem.lower()
    suffix = stem[len(item.label_name) :] if stem.startswith(item.label_name) else stem
    suffix = re.sub(r"[^a-z0-9]+", "", suffix)
    if suffix.isdigit():
        suffix = str(int(suffix))
    return f"{batch}/{item.label_name}/{suffix or '0'}"


def split_pdm_content_groups(
    originals: Sequence[OriginalWav],
    val_split: float,
    test_split: float,
    random_state: int,
) -> Tuple[List[OriginalWav], List[OriginalWav], set[str], Dict[str, List[str]]]:
    """Create disjoint PDM train/val and target-microphone test content IDs."""
    if val_split <= 0 or test_split <= 0 or val_split + test_split >= 1:
        raise ValueError("Grouped split requires positive val/test ratios with val + test < 1")

    grouped: Dict[str, List[OriginalWav]] = {}
    for item in originals:
        grouped.setdefault(content_group_key(item), []).append(item)
    keys = sorted(grouped)
    labels = np.asarray([grouped[key][0].label_id for key in keys], dtype=np.int64)
    key_indices = np.arange(len(keys))

    train_val_idx, test_idx = train_test_split(
        key_indices,
        test_size=test_split,
        stratify=labels,
        random_state=random_state,
    )
    val_ratio_within_train_val = val_split / (1.0 - test_split)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_within_train_val,
        stratify=labels[train_val_idx],
        random_state=random_state,
    )

    train_keys = {keys[int(index)] for index in train_idx}
    val_keys = {keys[int(index)] for index in val_idx}
    test_keys = {keys[int(index)] for index in test_idx}
    if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
        raise AssertionError("Content-group split overlap detected")

    train_items = [item for key in sorted(train_keys) for item in grouped[key]]
    val_items = [item for key in sorted(val_keys) for item in grouped[key]]
    split_keys = {
        "train": sorted(train_keys),
        "val": sorted(val_keys),
        "test": sorted(test_keys),
    }
    print(
        "Content-group split: "
        f"{len(train_keys)} train / {len(val_keys)} val / {len(test_keys)} test groups"
    )
    return train_items, val_items, test_keys, split_keys


def segment_relative_path(item: OriginalWav, segment_index: int, segment_seconds: float) -> Path:
    start_ms = int(round(segment_index * segment_seconds * 1000.0))
    end_ms = int(round((segment_index + 1) * segment_seconds * 1000.0))
    filename = f"{item.relative_path.stem}__seg{segment_index:04d}_{start_ms:08d}-{end_ms:08d}ms.wav"
    return Path(item.source_tag) / item.relative_path.parent / filename


def float_audio_to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Persist the exact fixed-length segment used by later augmentation/error review."""
    return np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def build_segment_dataset(
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
) -> audio_dataset.Dataset:
    """Slice each original independently into non-overlapping full segments."""
    segment_seconds = float(args.max_duration)
    if segment_seconds <= 0:
        raise ValueError("--max-duration must be greater than 0 for long-WAV segmentation")

    segment_samples = int(round(segment_seconds * args.target_sr))
    segment_root = output_dir / "wav_segments" / split_name
    features: List[np.ndarray] = []
    labels: List[int] = []
    relative_paths: List[str] = []
    source_paths: List[Path] = []
    original_segment_counts: Dict[str, int] = {}
    dropped_tail_samples = 0
    skipped_originals = 0

    for file_index, item in enumerate(originals, start=1):
        try:
            sr, audio = read_wav_mono(item.path, args.target_sr)
            full_segment_count = len(audio) // segment_samples
            dropped_tail_samples += len(audio) - full_segment_count * segment_samples
            if full_segment_count == 0:
                skipped_originals += 1
                print(f"Skipped shorter-than-{segment_seconds:g}s original: {item.path}")
                continue

            original_segment_counts[str(item.path)] = int(full_segment_count)
            for segment_index in range(full_segment_count):
                start = segment_index * segment_samples
                segment = audio[start : start + segment_samples]
                pcm16 = float_audio_to_pcm16(segment)

                rel_path = segment_relative_path(item, segment_index, segment_seconds)
                saved_path = segment_root / rel_path
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                wavfile.write(saved_path, sr, pcm16)

                # Extract from the persisted PCM representation so desktop
                # analysis and exported error WAVs reproduce the same samples.
                feature_audio = pcm16.astype(np.float32) / 32767.0
                feature = feature_extractor(
                    feature_audio,
                    sr,
                    args.n_fft,
                    args.hop_length,
                    args.n_mels,
                )
                if feature.shape != (expected_dim,) or not np.all(np.isfinite(feature)):
                    raise ValueError(f"expected finite ({expected_dim},), got {feature.shape}")

                features.append(feature.astype(np.float32))
                labels.append(item.label_id)
                relative_paths.append(rel_path.as_posix())
                source_paths.append(saved_path)
            print(
                f"  [{file_index}/{len(originals)}] {item.path.name}: "
                f"{full_segment_count} x {segment_seconds:g}s -> {split_name}"
            )
        except Exception as exc:
            skipped_originals += 1
            print(f"Error processing original WAV {item.path}: {exc}")

    if not features:
        raise ValueError(f"No complete {segment_seconds:g}-second segments produced for {split_name}")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    config = {
        "split": split_name,
        "split_unit": "original_wav_before_segmentation",
        "segment_seconds": segment_seconds,
        "segment_samples": segment_samples,
        "segment_overlap_samples": 0,
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
        "processed_original_file_count": len(original_segment_counts),
        "skipped_original_file_count": skipped_originals,
        "segment_count": len(y),
        "dropped_tail_samples_total": int(dropped_tail_samples),
        "segments_per_original": original_segment_counts,
    }
    dataset = audio_dataset.Dataset(X, y, relative_paths, source_paths, config)
    audio_dataset.save_npy_dataset(output_dir / "npy_original_splits" / split_name, dataset)
    print(f"Built {split_name}: {len(originals)} originals -> {len(y)} segments {X.shape}")
    return dataset


def load_segmented_train_val_test(
    args,
    output_dir: Path,
    *,
    feature_extractor: FeatureExtractor,
    expected_dim: int,
    feature_type: str,
    feature_source: str,
    feature_layout: Dict,
    training_augmenter=None,
) -> Tuple[audio_dataset.Dataset, audio_dataset.Dataset, audio_dataset.Dataset, List[Dict]]:
    """Split the train WAV pool first, then segment every split independently."""
    if not args.train_dir or not args.test_dir:
        raise ValueError("Long-WAV flow requires both --train-dir and --test-dir")

    input_roots = [
        Path(text).expanduser().resolve()
        for text in [*args.train_dir, *args.test_dir]
    ]
    resolved_output = output_dir.expanduser().resolve()
    for input_root in input_roots:
        if resolved_output == input_root or input_root in resolved_output.parents:
            raise ValueError(
                f"Output directory must not be inside an input WAV directory: {resolved_output}"
            )

    train_pool = collect_original_wavs(args.train_dir, "train pool")
    test_originals = collect_original_wavs(args.test_dir, "test")

    train_paths = {item.path.resolve() for item in train_pool}
    test_paths = {item.path.resolve() for item in test_originals}
    overlap = sorted(str(path) for path in train_paths & test_paths)
    if overlap:
        raise ValueError(f"Original WAV files overlap between train and test: {overlap[:5]}")

    # Split original recordings before slicing.  With multiple train roots,
    # split by content ID so same-named multi-mic recordings never cross
    # train/val, e.g. ics/drone_1.wav and pdm/drone_1.wav stay together.
    grouped_train_split = len(args.train_dir) > 1
    if grouped_train_split:
        train_originals, val_originals, split_keys = split_content_group_train_val(
            train_pool,
            val_split=args.val_split,
            random_state=args.random_state,
        )
    else:
        train_originals, val_originals = split_original_train_val(
            train_pool,
            val_split=args.val_split,
            random_state=args.random_state,
        )
        split_keys = None

    split_manifest_path = output_dir / "original_file_split.json"
    audio_dataset.write_json(
        split_manifest_path,
        {
            "split_unit": (
                "content_group_before_segmentation"
                if grouped_train_split
                else "original_wav_before_segmentation"
            ),
            "random_state": int(args.random_state),
            "train_ratio": float(1.0 - args.val_split),
            "val_ratio": float(args.val_split),
            "test_policy": "all_wavs_from_test_dir",
            "multi_train_root_content_group_split": bool(grouped_train_split),
            "content_group_keys": split_keys,
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
    train_data = build_segment_dataset(train_originals, "train", output_dir, args, **common)
    val_data = build_segment_dataset(val_originals, "val", output_dir, args, **common)
    test_data = build_segment_dataset(test_originals, "test", output_dir, args, **common)
    for dataset in (train_data, val_data, test_data):
        dataset.config["split_manifest_path"] = str(split_manifest_path)
        dataset.config["validation_source"] = "random_stratified_originals_from_train_dir"
        dataset.config["content_group_split"] = False
    if grouped_train_split:
        train_data.config["content_group_split"] = True
        val_data.config["content_group_split"] = True
        train_data.config["content_group_keys"] = split_keys["train"]
        val_data.config["content_group_keys"] = split_keys["val"]
        test_data.config["test_source"] = "all_wavs_from_test_dir_not_group_split"

    # Preserve split metadata in the non-augmented cache used for later model
    # export and ablation runs.
    original_root = output_dir / "npy_original_splits"
    audio_dataset.save_npy_dataset(original_root / "train", train_data)
    audio_dataset.save_npy_dataset(original_root / "val", val_data)
    audio_dataset.save_npy_dataset(original_root / "test", test_data)

    if training_augmenter is not None:
        train_data = training_augmenter(train_data, args)

    split_root = output_dir / "npy_splits"
    audio_dataset.save_npy_dataset(split_root / "train", train_data)
    audio_dataset.save_npy_dataset(split_root / "val", val_data)
    audio_dataset.save_npy_dataset(split_root / "test", test_data)
    return train_data, val_data, test_data, [train_data.config, val_data.config, test_data.config]


def load_grouped_cross_mic_train_val_test(
    args,
    output_dir: Path,
    *,
    feature_extractor: FeatureExtractor,
    expected_dim: int,
    feature_type: str,
    feature_source: str,
    feature_layout: Dict,
    training_augmenter=None,
) -> Tuple[audio_dataset.Dataset, audio_dataset.Dataset, audio_dataset.Dataset, List[Dict]]:
    """Train on PDM while testing unseen content IDs on target microphones."""
    if not args.train_dir or len(args.train_dir) != 1 or not args.test_dir:
        raise ValueError("Grouped cross-mic split requires one PDM train root and target test roots")

    pdm_originals = collect_original_wavs(args.train_dir, "PDM content pool")
    target_originals = collect_original_wavs(args.test_dir, "target microphone pool")
    train_originals, val_originals, test_keys, split_keys = split_pdm_content_groups(
        pdm_originals,
        val_split=args.val_split,
        test_split=args.test_split,
        random_state=args.random_state,
    )
    if getattr(args, "include_target_mics_in_train", False):
        train_keys = set(split_keys["train"])
        val_keys = set(split_keys["val"])
        train_originals.extend(
            item for item in target_originals if content_group_key(item) in train_keys
        )
        val_originals.extend(
            item for item in target_originals if content_group_key(item) in val_keys
        )
        print(
            "Multi-mic grouped training enabled: target-mic recordings are included "
            "only for train/val content groups"
        )
    test_originals = [item for item in target_originals if content_group_key(item) in test_keys]
    represented_test_keys = {content_group_key(item) for item in test_originals}
    missing_keys = sorted(test_keys - represented_test_keys)
    if missing_keys:
        print(f"Warning: {len(missing_keys)} held-out content groups have no target-mic WAV: {missing_keys}")
    if not test_originals:
        raise ValueError("No ICS/INMP recordings match the held-out PDM content groups")

    common = {
        "feature_extractor": feature_extractor,
        "expected_dim": expected_dim,
        "feature_type": feature_type,
        "feature_source": feature_source,
        "feature_layout": feature_layout,
    }
    train_data = build_segment_dataset(train_originals, "train", output_dir, args, **common)
    val_data = build_segment_dataset(val_originals, "val", output_dir, args, **common)
    test_data = build_segment_dataset(test_originals, "test", output_dir, args, **common)
    for dataset, split_name in (
        (train_data, "train"),
        (val_data, "val"),
        (test_data, "test"),
    ):
        dataset.config["content_group_split"] = True
        dataset.config["content_group_keys"] = split_keys[split_name]
        dataset.config["target_mics_in_train"] = bool(
            getattr(args, "include_target_mics_in_train", False)
        )

    # build_segment_dataset persists before grouped metadata is known. Save
    # again here so later NPY-only export runs retain the leakage contract.
    original_root = output_dir / "npy_original_splits"
    audio_dataset.save_npy_dataset(original_root / "train", train_data)
    audio_dataset.save_npy_dataset(original_root / "val", val_data)
    audio_dataset.save_npy_dataset(original_root / "test", test_data)

    if training_augmenter is not None:
        train_data = training_augmenter(train_data, args)
    split_root = output_dir / "npy_splits"
    audio_dataset.save_npy_dataset(split_root / "train", train_data)
    audio_dataset.save_npy_dataset(split_root / "val", val_data)
    audio_dataset.save_npy_dataset(split_root / "test", test_data)
    audio_dataset.write_json(
        output_dir / "content_group_split.json",
        {
            "split_unit": "batch_class_recording_id_before_segmentation",
            "random_state": int(args.random_state),
            "train_ratio": float(1.0 - args.val_split - args.test_split),
            "val_ratio": float(args.val_split),
            "test_ratio": float(args.test_split),
            "target_mics_in_train": bool(getattr(args, "include_target_mics_in_train", False)),
            "keys": split_keys,
        },
    )
    return train_data, val_data, test_data, [train_data.config, val_data.config, test_data.config]
