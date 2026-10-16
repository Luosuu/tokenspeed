import unittest
from types import SimpleNamespace
from unittest import mock

from tokenspeed.runtime.execution import forward_trace


class _Tracer:
    def __init__(self):
        self.now = 10.0
        self.events = []
        self.enable = True

    def getts(self):
        self.now += 5.0
        return self.now

    def add_raw(self, event):
        self.events.append(event)


def _forward_op(request_ids, num_extends):
    return SimpleNamespace(request_ids=request_ids, num_extends=lambda: num_extends)


class TestForwardBatch(unittest.TestCase):
    def test_records_the_requests_of_a_profiled_forward(self):
        tracer = _Tracer()
        forward_op = _forward_op(["a", "b"], 1)
        with mock.patch("viztracer.get_tracer", return_value=tracer, create=True):
            self.assertIs(forward_trace.profiling_tracer(forward_op), tracer)
        start = tracer.getts()
        tracer.add_raw({"name": "kernel scope"})
        forward_trace.record_forward_batch(tracer, forward_op, start, tracer.getts())

        kernel, batch = tracer.events
        self.assertEqual(kernel["name"], "kernel scope")
        self.assertEqual(
            (batch["name"], batch["cat"], batch["ph"]),
            ("forward_batch", "tokenspeed.requests", "X"),
        )
        self.assertEqual(
            batch["args"], {"request_ids": ["a", "b"], "num_extends": 1}
        )
        # The slice spans the forward, so the scopes it launches nest inside it.
        self.assertEqual((batch["ts"], batch["dur"]), (15.0, 5.0))

    def test_records_nothing_without_a_profile_or_a_forward(self):
        stopped = _Tracer()
        stopped.enable = False
        for tracer in (None, stopped):
            with mock.patch("viztracer.get_tracer", return_value=tracer, create=True):
                self.assertIsNone(
                    forward_trace.profiling_tracer(_forward_op(["a"], 0))
                )
        with mock.patch("viztracer.get_tracer", return_value=_Tracer(), create=True):
            self.assertIsNone(forward_trace.profiling_tracer(None))


class TestForwardBatchWithVizTracer(unittest.TestCase):
    def test_the_slice_nests_with_every_frame_viztracer_records(self):
        try:
            from viztracer import VizTracer
        except ImportError:
            self.skipTest("viztracer is not installed")
        import json
        import os
        import tempfile

        def forward():
            sum(range(1000))

        def run(forward_op):
            # The shape of ModelExecutorDevice's _forward.
            tracer = forward_trace.profiling_tracer(forward_op)
            start = tracer.getts() if tracer is not None else None
            try:
                return forward()
            finally:
                if tracer is not None:
                    forward_trace.record_forward_batch(
                        tracer, forward_op, start, tracer.getts()
                    )

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "trace.json")
            tracer = VizTracer(output_file=output, min_duration=0, verbose=0)
            tracer.start()
            run(_forward_op(["a"], 0))
            tracer.stop()
            tracer.save()
            with open(output) as file:
                events = json.load(file)["traceEvents"]
        events = [e for e in events if e.get("ph") == "X"]

        [batch] = [e for e in events if e["name"] == "forward_batch"]
        start, end = batch["ts"], batch["ts"] + batch["dur"]
        for event in events:
            if event is batch or event["tid"] != batch["tid"]:
                continue
            other_start, other_end = event["ts"], event["ts"] + event["dur"]
            inside = start <= other_start and other_end <= end
            outside = other_end <= start or end <= other_start
            around = other_start <= start and end <= other_end
            self.assertTrue(inside or outside or around, event["name"])
        self.assertTrue(any("forward" in e["name"] and e is not batch for e in events))


if __name__ == "__main__":
    unittest.main()
