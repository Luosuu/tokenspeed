# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Proton profiles of a scheduler that replays CUDA graphs.

Proton attributes a replayed kernel to its graph node only when a session was
active while the graph was captured; otherwise the kernel hangs off the graph
launch and finalizing a trace-mode session fails with "Cannot find CPU scope
event for kernel launch". A session started by ``/start_profile`` comes after
the capture, so with ``TOKENSPEED_PROTON_SESSION_DIR`` set each scheduler keeps
one session for its whole life instead:

- :meth:`PersistentProtonSession.start` runs before the ``EventLoop`` is built,
  so the session sees every capture; :meth:`~PersistentProtonSession.park`
  deactivates it once the loop is built;
- ``/start_profile`` moves the session to a new data phase and activates it;
- ``/stop_profile`` ends that phase. Proton's periodic flushing writes a
  completed phase to ``<session>.part_<phase>.chrome_trace``, which is then
  moved to the name a per-profile session would have written.

Proton completes a phase only once a later phase receives a kernel, so each
completion launches one tiny kernel. The same path serves eager servers.
"""

from __future__ import annotations

import functools
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from tokenspeed_kernel.profiling import ProfilingConfig, ProfilingState, proton

FORMAT = "chrome_trace"
# Proton marks a phase complete before it writes the file.
_FLUSH_TIMEOUT_S = 300.0


class PersistentProtonSession:
    """This scheduler process's long-lived Proton session."""

    _instance: PersistentProtonSession | None = None

    def __init__(self, path: Path) -> None:
        self.path = path
        self.id: int | None = None
        self.device: int | None = None
        # The phase being profiled and where its trace goes; None between profiles.
        self.phase: int | None = None
        self.config: ProfilingConfig | None = None
        self._stream = None

    @classmethod
    def start(cls, directory: str) -> PersistentProtonSession:
        """Start this process's session; call before any CUDA graph is captured."""
        session = cls(Path(directory) / f"proton-{os.getpid()}")
        session.path.parent.mkdir(parents=True, exist_ok=True)
        session.id = proton.start(
            str(session.path),
            data="trace",
            hook="triton",
            mode=f"periodic_flushing:format={FORMAT}",
        )
        cls._instance = session
        return session

    @classmethod
    def get(cls) -> PersistentProtonSession | None:
        return cls._instance

    def park(self, device: int) -> None:
        """Stop recording after startup; what capture recorded stays with the session."""
        self.device = device
        self._complete(proton.data.advance_phase(self.id) - 1).unlink()

    def close(self) -> None:
        """Finalize the session without writing a profile (startup failed)."""
        proton.finalize(self.id)
        PersistentProtonSession._instance = None

    def begin(self, config: ProfilingConfig) -> None:
        """Record a profile into a new phase; it is written to ``<config.output>.chrome_trace``."""
        self.phase = proton.data.advance_phase(self.id)
        self.config = config
        proton.activate(self.id)
        # kernel_scope() records CPU scopes only while this state is active.
        state = ProfilingState.get()
        state._config, state._session, state.enabled = config, self.id, True

    def stop(self) -> tuple[int, str]:
        """End the profiled phase; returns it and its output for :meth:`write`.

        Kernels keep landing in the next phase, which is discarded, until
        :meth:`write` deactivates the session.
        """
        phase, output = self.phase, self.config.output
        self.phase, self.config = None, None
        proton.data.advance_phase(self.id)
        state = ProfilingState.get()
        state._config, state._session, state.enabled = None, None, False
        return phase, output

    def write(self, phase: int, output: str) -> Path:
        """Write a stopped ``phase`` to ``<output>.chrome_trace``.

        May run on another thread; the next :meth:`begin` must wait for it.
        """
        part = self._complete(phase)
        target = Path(f"{output}.{FORMAT}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(part, target)
        # Phases between profiles hold nothing anyone asked for.
        for stale in self.path.parent.glob(f"{self.path.name}.part_*.{FORMAT}"):
            stale.unlink(missing_ok=True)
        return target

    def _complete(self, phase: int) -> Path:
        """Flush ``phase`` (older than the current one) to its file and deactivate."""
        self._deactivate_after_kernel()
        part = self.path.with_name(f"{self.path.name}.part_{phase}.{FORMAT}")
        _wait_written(part, lambda: proton.data.is_phase_complete(self.id, phase))
        return part

    def _deactivate_after_kernel(self) -> None:
        import torch

        # Also runs on the control-plane thread, whose dispatch guard rejects
        # tensor ops; ``_sleep`` launches its kernel without dispatch.
        with torch.cuda.device(self.device):
            if self._stream is None:
                self._stream = torch.cuda.Stream()
            with torch.cuda.stream(self._stream):
                torch.cuda._sleep(1)
            proton.deactivate(self.id, flushing=True)


def _wait_written(path: Path, complete) -> None:
    """Wait until Proton has written ``path``: a whole JSON object of stable size."""
    deadline = time.monotonic() + _FLUSH_TIMEOUT_S
    size = -1
    while time.monotonic() < deadline:
        if complete() and path.is_file():
            current = path.stat().st_size
            if current == size and current > 0 and _ends_json_object(path):
                return
            size = current
        time.sleep(0.2)
    raise TimeoutError(f"Proton did not write {path}")


def _ends_json_object(path: Path) -> bool:
    with path.open("rb") as file:
        file.seek(max(0, path.stat().st_size - 16))
        return file.read().rstrip().endswith(b"}")


def detach_proton_session() -> Callable[[], None] | None:
    """Stop a per-profile session from recording; returns the call that writes it.

    Unlike ``stop_profiling``, this leaves writing the profile to the caller,
    which may do it on another thread. Returns None without an active session.
    """
    state = ProfilingState.get()
    session = state._session if state.active else None
    output_format = state._config.output_format if state._config is not None else ""
    state._config, state._session, state.enabled = None, None, False
    if session is None:
        return None
    proton.deactivate(session)
    return functools.partial(proton.finalize, session, output_format)
