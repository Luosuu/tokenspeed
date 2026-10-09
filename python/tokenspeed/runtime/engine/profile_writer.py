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
the files exist. Writing a
profiled step can take tens of seconds, during which the scheduler, its TP
peers and every caller waiting for the reply would otherwise stall. The next
``/start_profile`` waits for the previous profile's files: the process has
one VizTracer and one Proton profiler.

Scheduler processes are stopped with SIGKILL, so a caller that shuts the
server down must first wait for the files it expects.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class ProfileWriter:
    """Runs profile writes on threads, and waits for them."""

    def __init__(self) -> None:
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    def write(self, name: str, function: Callable[..., object], *args) -> None:
        """Run ``function(*args)`` on a thread; a failure is logged."""

        def run() -> None:
            try:
                function(*args)
            except Exception:
                logger.exception(f"Failed to write the {name} profile")

        # Not a daemon: a normal interpreter exit still finishes the file.
        thread = threading.Thread(target=run, name=f"profile-writer-{name}")
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
