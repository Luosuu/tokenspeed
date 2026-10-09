import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from test.runtime.test_request_handler_profile import _make_handler, _start_req
from types import SimpleNamespace
from unittest import mock

from tokenspeed_kernel import profiling

from tokenspeed.runtime.engine import proton_session as proton_session_mod
from tokenspeed.runtime.engine import request_handler as request_handler_mod
from tokenspeed.runtime.engine.io_struct import ProfileReq, ProfileReqType
from tokenspeed.runtime.engine.proton_session import PersistentProtonSession
from tokenspeed.runtime.utils.env import envs


class FakeProton:
    """Proton's sessions, data phases and periodic flushing, without a GPU."""

    def __init__(self):
        self.path = None
        self.phase = 0
        self.active = False
        self.written = -1
        self.records = {}  # phase -> names recorded in it
        self.finalized = []
        self.data = SimpleNamespace(
            advance_phase=self.advance_phase,
            is_phase_complete=lambda session, phase: phase <= self.written,
        )

    def start(self, path, *, data, hook, mode):
        assert (data, hook, mode) == (
            "trace",
            "triton",
            "periodic_flushing:format=chrome_trace",
        )
        self.path, self.active = path, True
        return 7

    def record(self, name):
        if self.active:
            self.records.setdefault(self.phase, []).append(name)

    def advance_phase(self, session):
        self.phase += 1
        return self.phase

    def activate(self, session):
        self.active = True

    def deactivate(self, session, flushing=False):
        # Periodic flushing writes the phases before the latest one with a kernel.
        if flushing and self.phase in self.records:
            for phase in range(self.written + 1, self.phase):
                names = self.records.pop(phase, [])
                Path(f"{self.path}.part_{phase}.chrome_trace").write_text(
                    json.dumps({"traceEvents": names})
                )
            self.written = self.phase - 1
        self.active = False

    def finalize(self, session, output_format=""):
        self.finalized.append(session)


class TestPersistentProtonSession(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.proton = FakeProton()
        for patcher in (
            mock.patch.object(proton_session_mod, "proton", self.proton),
            mock.patch.object(PersistentProtonSession, "_instance", None),
            # The marker kernel that lets periodic flushing complete a phase.
            mock.patch.object(
                PersistentProtonSession,
                "_deactivate_after_kernel",
                lambda session: (
                    self.proton.record("marker"),
                    self.proton.deactivate(session.id, flushing=True),
                ),
            ),
            mock.patch.object(
                request_handler_mod, "proton_available", return_value=True
            ),
            mock.patch.object(request_handler_mod.torch.distributed, "all_reduce"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        profiling.ProfilingState.reset()
        self.addCleanup(profiling.ProfilingState.reset)

        # Startup: the session records the CUDA-graph capture, then is parked.
        self.session = PersistentProtonSession.start(str(self.tmp / "session"))
        self.proton.record("graph capture")
        self.session.park(device=0)

    def _profile(self, handler, profile_id, kernel):
        req = _start_req(str(self.tmp / profile_id))
        req.profile_id = profile_id
        self.assertTrue(handler.profile(req).success)
        state = profiling.ProfilingState.get()
        self.assertTrue(state.active and state._session == self.session.id)
        self.proton.record(kernel)
        result = handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))
        self.assertTrue(result.success)
        self.assertFalse(profiling.ProfilingState.get().active)
        return self.tmp / profile_id / f"{profile_id}-TP0.proton.chrome_trace"

    def test_startup_records_capture_and_keeps_no_file(self):
        self.assertIs(PersistentProtonSession.get(), self.session)
        self.assertEqual(
            self.proton.path, str(self.tmp / "session" / f"proton-{os.getpid()}")
        )
        self.assertFalse(self.proton.active)
        self.assertEqual(list((self.tmp / "session").iterdir()), [])

    def test_each_profile_is_one_phase_written_where_tokenspeed_writes_it(self):
        handler = _make_handler()
        for step in (3, 4):
            output = self._profile(handler, f"step-{step}", f"decode {step}")
            self.assertEqual(
                json.loads(output.read_text()), {"traceEvents": [f"decode {step}"]}
            )
            self.assertFalse(self.proton.active)
            self.assertEqual(list((self.tmp / "session").iterdir()), [])
        self.assertEqual(self.proton.finalized, [])

    def test_background_write_comes_after_the_reply(self):
        release = threading.Event()
        write = PersistentProtonSession.write
        with mock.patch.object(
            PersistentProtonSession,
            "write",
            lambda session, *args: (release.wait(5), write(session, *args))[1],
        ), envs.TOKENSPEED_PROFILE_SAVE_IN_BACKGROUND.override(True):
            handler = _make_handler()
            output = self._profile(handler, "step-3", "decode")
            self.assertFalse(output.exists())
            release.set()
            # The next profile starts only once the previous one is written.
            handler.profile(_start_req(str(self.tmp / "step-4")))
            self.assertTrue(output.exists())
            handler.profile(ProfileReq(type=ProfileReqType.STOP_PROFILE))
            handler.profile_writer.wait()


if __name__ == "__main__":
    unittest.main()
