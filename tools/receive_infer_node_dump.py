#!/usr/bin/env python3
"""Receive infer_node audio debug dumps from the serial console (or a saved log).

The firmware (INFER_AUDIO_DEBUG_DUMP=1) prints each file as
    @@DUMP BEGIN <stem>
    @@FILE <name> <bytes> <crc32>
    @@D <seq> <base64>
    @@DUMP END <stem> <files> <errors>
Other log lines may be interleaved; they are echoed to the terminal and ignored.
Files are written to <output-dir>/<stem>/ after their size and zlib CRC32 match.

Live from the board (close idf.py monitor first, it holds the port):
  python tools/receive_infer_node_dump.py --port COM14 --reset
From a saved monitor log:
  python tools/receive_infer_node_dump.py --log monitor.log
Then:
  python compare_infer_node_debug_dump.py --dump-dir infer_node_dump
"""

from __future__ import annotations

import argparse
import base64
import binascii
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Iterator

ANSI = re.compile(r"\x1b\[[0-9;]*m")


class DumpReceiver:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.stem: str | None = None
        self.file: dict | None = None
        self.completed: list[dict] = []
        self.problems: list[str] = []
        self.files_seen: list[dict] = []

    def _finish_file(self) -> None:
        if self.file is None or self.stem is None:
            return
        info, data = self.file, bytes(self.file["data"])
        self.file = None
        if info["gaps"]:
            self.problems.append(f"{info['name']}: missing/out-of-order lines near seq {info['gaps'][:5]}")
        if len(data) != info["size"]:
            self.problems.append(f"{info['name']}: got {len(data)} bytes, expected {info['size']}")
            return
        if zlib.crc32(data) != info["crc"]:
            self.problems.append(f"{info['name']}: CRC mismatch")
            return
        target = self.output_dir / self.stem / info["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        info["ok"] = True

    def feed(self, raw_line: str) -> dict | None:
        """Process one console line; returns a summary when a dump ends."""
        line = ANSI.sub("", raw_line).rstrip("\r\n")
        position = line.find("@@")
        if position < 0:
            print(line)
            return None
        parts = line[position:].split()
        tag = parts[0]
        if tag == "@@D" and self.file is not None and len(parts) == 3:
            seq = int(parts[1])
            if seq != self.file["next_seq"]:
                self.file["gaps"].append(self.file["next_seq"])
            self.file["next_seq"] = seq + 1
            try:
                self.file["data"] += base64.b64decode(parts[2], validate=True)
            except (binascii.Error, ValueError):
                self.file["gaps"].append(seq)
        elif tag == "@@FILE" and len(parts) == 4:
            self._finish_file()
            self.file = {"name": Path(parts[1]).name, "size": int(parts[2]), "crc": int(parts[3], 16),
                         "data": bytearray(), "next_seq": 0, "gaps": [], "ok": False}
            self.files_seen.append(self.file)
            print(f"  <- {parts[1]} ({int(parts[2])} bytes)")
        elif tag == "@@DUMP" and len(parts) >= 3 and parts[1] == "BEGIN":
            self.stem, self.file, self.problems, self.files_seen = Path(parts[2]).name, None, [], []
            print(f"=== receiving dump {self.stem}")
        elif tag == "@@DUMP" and len(parts) >= 3 and parts[1] == "END" and self.stem is not None:
            self._finish_file()
            device_files = int(parts[3]) if len(parts) > 3 else None
            device_errors = int(parts[4]) if len(parts) > 4 else None
            ok_files = sum(1 for f in self.files_seen if f["ok"])
            summary = {"stem": self.stem, "dir": str(self.output_dir / self.stem), "files_ok": ok_files,
                       "files_announced": device_files, "device_errors": device_errors,
                       "problems": list(self.problems)}
            self.completed.append(summary)
            self.stem = None
            status = "OK" if not summary["problems"] and device_errors == 0 and ok_files == device_files else "INCOMPLETE"
            print(f"=== dump {summary['stem']}: {ok_files}/{device_files} files written, device errors={device_errors} -> {status}")
            for problem in summary["problems"]:
                print("    !", problem)
            return summary
        elif tag in ("@@ERROR",):
            self.problems.append(line[position:])
            print(line)
        return None


def serial_lines(port: str, baud: int, reset: bool, timeout_s: float) -> Iterator[str]:
    import serial  # pyserial

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = port, baud, 0.5
    ser.dtr = False   # keep IO0 high and EN released on auto-reset boards
    ser.rts = False
    ser.open()
    try:
        if reset:
            ser.rts = True    # EN low
            time.sleep(0.1)
            ser.rts = False
        deadline = time.time() + timeout_s
        buffer = b""
        while time.time() < deadline:
            chunk = ser.read(4096)
            if not chunk:
                continue
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                yield line.decode("utf-8", errors="replace")
        print(f"[timeout after {timeout_s:g} s]")
    finally:
        ser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--port", help="Serial port of the node, e.g. COM14")
    source.add_argument("--log", type=Path, help="Saved console log containing @@DUMP blocks")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--reset", action="store_true", help="Pulse EN via RTS after opening the port")
    parser.add_argument("--count", type=int, default=1, help="Stop after this many dumps (serial mode)")
    parser.add_argument("--timeout", type=float, default=900.0, help="Give up after this many seconds")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("infer_node_dump"))
    args = parser.parse_args()

    receiver = DumpReceiver(args.output_dir.resolve())
    if args.log:
        lines: Iterator[str] = iter(args.log.read_text(encoding="utf-8", errors="replace").splitlines())
    else:
        lines = serial_lines(args.port, args.baud, args.reset, args.timeout)
    try:
        for line in lines:
            if receiver.feed(line) is not None and args.port and len(receiver.completed) >= args.count:
                break
    except KeyboardInterrupt:
        print("[interrupted]")

    if not receiver.completed:
        sys.exit("no complete @@DUMP block received")
    print(f"\nNext: python compare_infer_node_debug_dump.py --dump-dir \"{receiver.output_dir}\"")


if __name__ == "__main__":
    main()
