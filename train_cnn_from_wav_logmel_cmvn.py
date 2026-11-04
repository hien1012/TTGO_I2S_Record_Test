#!/usr/bin/env python3
"""
Train the gain/background-augmented single-channel log-mel CMVN CNN variant.
"""
#python train_cnn_from_wav_logmel_cmvn.py `
#>>   --train-dir raw_audio/new_16kHz/ics raw_audio/new_16kHz/pdm raw_audio/new_16kHz/test/ICS raw_audio/new_16kHz/test/PDM `
#>>   --test-dir C:\Users\Potato\OneDrive\Pictures\Desktop\碩論資料\data\microphone\0812_real_test_dji\ICS C:\Users\Potato\OneDrive\Pictures\Desktop\碩論資料\data\microphone\0812_real_test_dji\INMP C:\Users\Potato\OneDrive\Pictures\Desktop\碩論資料\data\microphone\0812_real_test_dji\PDM `
#>>   --val-split 0.2 `
#>>   --max-duration 2 `
#>>   --augmentation-copies 3 `
#>>   --augmentation-gain-min-db -35 `
#>>   --augmentation-gain-max-db 3 `
#>>   --augmentation-background-snr-min-db -10 `
#>>   --augmentation-background-snr-max-db 10 ` --f-min 250 `
#>>   --f-max 2500 ` --epochs 50 `
#>>   --output-dir cnn_output/logmel_cmvn/augment/icspdm_v3_include_pc_test


from __future__ import annotations

import sys

from train_cnn_from_wav_logmel import main


def option_present(*names: str) -> bool:
    return any(
        argument == name or argument.startswith(f"{name}=")
        for argument in sys.argv[1:]
        for name in names
    )


def add_value_default(flag: str, value: str, *aliases: str) -> None:
    if not option_present(flag, *aliases):
        sys.argv.extend([flag, value])


def add_flag_default(flag: str, *conflicting_flags: str) -> None:
    if not option_present(flag, *conflicting_flags):
        sys.argv.append(flag)


def apply_cmvn_variant_defaults() -> None:
    """Set this variant's defaults without preventing explicit CLI overrides."""
    add_value_default("--input-channels", "cmvn")
    add_value_default("--output-dir", "cnn_from_wav_logmel_cmvn_output", "-o")

    # Required training waveform order:
    # PCM float32 -> DC removal -> gain [-20, +6] dB -> recorded background mix.
    add_flag_default("--train-augmentation", "--no-train-augmentation")
    add_flag_default("--augmentation-pre-dc-removal")
    add_value_default("--augmentation-gain-min-db", "-20")
    add_value_default("--augmentation-gain-max-db", "6")
    add_flag_default("--augmentation-background-mix")
    add_value_default("--augmentation-background-snr-min-db", "5")
    add_value_default("--augmentation-background-snr-max-db", "20")

    # These belong to the original augmentation recipe, not this controlled variant.
    add_value_default("--augmentation-eq-db", "0")
    add_value_default("--augmentation-time-shift-ms", "0")
    add_flag_default(
        "--no-augmentation-gaussian-noise",
        "--augmentation-gaussian-noise",
    )


if __name__ == "__main__":
    apply_cmvn_variant_defaults()
    main()
