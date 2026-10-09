import unittest
from types import SimpleNamespace
from unittest import mock

from tokenspeed.runtime.execution import forward_trace


class _Tracer:
    def __init__(self):
        self.now = 10.0
        self.events = []

    def getts(self):
        self.now += 5.0
        return self.now

    def add_raw(self, event):
        self.events.append(event)


def _forward_op(request_ids, num_extends):
    return SimpleNamespace(request_ids=request_ids, num_extends=lambda: num_extends)


class TestForwardBatchScope(unittest.TestCase):
    def test_records_the_requests_of_a_profiled_forward(self):
        tracer = _Tracer()
        with mock.patch.object(forward_trace, "_active_viztracer", return_value=tracer):
            with forward_trace.forward_batch_scope(_forward_op(["a", "b"], 1)):
                tracer.add_raw({"name": "kernel scope"})

        kernel, batch = tracer.events
        self.assertEqual(kernel["name"], "kernel scope")
        self.assertEqual(
            (batch["name"], batch["cat"], batch["ph"]),
            ("forward_batch", "tokenspeed.requests", "X"),
        )
        self.assertEqual(batch["args"], {"request_ids": ["a", "b"], "num_extends": 1})
        # The slice spans the forward, so the scopes it launches nest inside it.
        self.assertEqual((batch["ts"], batch["dur"]), (15.0, 5.0))

    def test_records_nothing_without_a_profile_or_a_forward(self):
        with mock.patch.object(forward_trace, "_active_viztracer", return_value=None):
            with forward_trace.forward_batch_scope(_forward_op(["a"], 0)):
                pass
        tracer = _Tracer()
        with mock.patch.object(forward_trace, "_active_viztracer", return_value=tracer):
            with forward_trace.forward_batch_scope(None):
                pass
        self.assertEqual(tracer.events, [])


if __name__ == "__main__":
    unittest.main()
