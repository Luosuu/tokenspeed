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

"""Record which requests each model forward serves, while VizTracer profiles.

To follow a request from its client to the GPU, a trace must say which
forwards served it. While a VizTracer profile runs, each forward is wrapped in
a ``forward_batch`` slice on the forward thread, with the ``rid`` of every
request in the batch. The kernel scopes the forward launches, and their flows
to Proton's kernels, nest inside that slice. Nothing is recorded otherwise.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

FORWARD_BATCH_NAME = "forward_batch"
FORWARD_BATCH_CATEGORY = "tokenspeed.requests"


def _active_viztracer():
    try:
        from viztracer import get_tracer
    except ImportError:
        return None
    tracer = get_tracer()
    return tracer if tracer is not None and tracer.enable else None


@contextlib.contextmanager
def forward_batch_scope(forward_op: Any) -> Iterator[None]:
    """Record ``forward_op``'s requests around the forward, if VizTracer is profiling."""
    tracer = _active_viztracer() if forward_op is not None else None
    if tracer is None:
        yield
        return
    start = tracer.getts()
    try:
        yield
    finally:
        tracer.add_raw(
            {
                "ph": "X",
                "name": FORWARD_BATCH_NAME,
                "cat": FORWARD_BATCH_CATEGORY,
                "ts": start,
                "dur": tracer.getts() - start,
                "args": {
                    "request_ids": [str(rid) for rid in forward_op.request_ids],
                    "num_extends": int(forward_op.num_extends()),
                },
            }
        )
