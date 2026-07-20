from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from typing import Sequence

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
SYNCHRONIZE = 0x00100000
WAIT_TIMEOUT = 0x00000102


def keep_awake_until_process_exits(process_id: int) -> None:
    """Prevent system sleep only while the specified Windows process is alive."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    kernel32.SetThreadExecutionState.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    handle = kernel32.OpenProcess(SYNCHRONIZE, False, process_id)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"cannot monitor process {process_id}")
    try:
        if not kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
            raise OSError(ctypes.get_last_error(), "failed to prevent system sleep")
        while kernel32.WaitForSingleObject(handle, 60_000) == WAIT_TIMEOUT:
            pass
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        kernel32.CloseHandle(handle)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prevent sleep while a training process is running")
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args(argv)
    keep_awake_until_process_exits(args.pid)


if __name__ == "__main__":
    main()
