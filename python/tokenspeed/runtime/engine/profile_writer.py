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

"""Write profiles off the scheduler's critical path.

``/stop_profile`` only stops recording: Torch traces, VizTracer reports and
Proton traces are written on threads of their own, and the reply comes before
the files exist. Writing a profiled step can take tens of seconds, during which
the scheduler, its TP peers and every caller waiting for the reply would
otherwise stall. The next ``/start_profile`` waits for the previous profile's
files: the process has one VizTracer and one Proton profiler.

So that a reader polling the output directory never takes a half-written file
for a whole one, a file is written under ``.writing/`` next to its final path
and renamed into place once complete. A write that fails leaves
``<output>.failed`` with the error instead, so the reader can stop waiting.

Scheduler processes are stopped with SIGKILL, so a caller that shuts the
server down must first wait for the files it expects.
"""

from __future__ import annotations

import os
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

WRITING_DIR = ".writing"
FAILED_SUFFIX = ".failed"


class ProfileWriter:
    """Runs profile writes on threads, and waits for them."""

    def __init__(self) -> None:
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    def write_atomically(self, output: str, save: Callable[[str], object]) -> None:
        """On a thread, ``save(path)`` to a temporary path, then rename it to ``output``."""

        def write() -> None:
            target = Path(output)
            temporary = target.parent / WRITING_DIR / target.name
            temporary.parent.mkdir(parents=True, exist_ok=True)
            try:
                save(str(temporary))
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

        self._start(output, write)

    def write_in_place(self, output: str, write: Callable[[], object]) -> None:
        """On a thread, ``write()``, which writes ``output`` itself."""
        self._start(output, write)

    def _start(self, output: str, write: Callable[[], object]) -> None:
        def run() -> None:
            try:
                write()
            except Exception:
                logger.exception(f"Failed to write the profile {output}")
                try:
                    Path(f"{output}{FAILED_SUFFIX}").write_text(traceback.format_exc())
                except OSError:
                    logger.exception(f"Failed to mark the profile {output} as failed")

        # Not a daemon: a normal interpreter exit still finishes the file.
        thread = threading.Thread(target=run, name="profile-writer")
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def wait(self) -> None:
        """Block until every write started so far has finished."""
        while True:
            with self._lock:
                pending = [thread for thread in self._threads if thread.is_alive()]
                self._threads = pending
            if not pending:
                return
            for thread in pending:
                thread.join()
