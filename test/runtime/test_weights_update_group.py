import unittest
from types import SimpleNamespace
from unittest import mock

from torch.distributed import distributed_c10d as c10d

from tokenspeed.runtime.execution import model_runner as model_runner_mod


class TestInitWeightsUpdateGroup(unittest.TestCase):
    def test_group_is_not_split_from_the_bound_default_group(self):
        # torch >= 2.14 splits from a device-bound default group, which would
        # leave the trainer out of the weight-update group.
        default = SimpleNamespace(bound_device_id="cuda:0")
        seen = []

        def helper(world_size, rank, *args, group_name, **kwargs):
            seen.append((world_size, rank, group_name, default.bound_device_id))
            return object(), None

        runner = model_runner_mod.ModelRunner.__new__(model_runner_mod.ModelRunner)
        runner.global_rank = 1
        runner.gpu_id = 0
        obj = SimpleNamespace(
            rank_offset=1,
            world_size=3,
            group_name="weights",
            backend="nccl",
            master_address="127.0.0.1",
            master_port=29500,
        )
        store = mock.Mock()
        with mock.patch.object(
            c10d, "is_initialized", return_value=True
        ), mock.patch.object(
            c10d, "_get_default_group", return_value=default
        ), mock.patch.object(
            c10d, "_new_process_group_helper", helper
        ), mock.patch.object(
            c10d, "rendezvous", return_value=iter([(store, 2, 3)])
        ), mock.patch.object(
            c10d, "PrefixStore"
        ), mock.patch.object(
            model_runner_mod.torch.cuda, "set_device"
        ):
            ok, message = runner.init_weights_update_group(obj)

        self.assertTrue(ok, message)
        self.assertEqual(seen, [(3, 2, "weights", None)])
        self.assertEqual(default.bound_device_id, "cuda:0")


if __name__ == "__main__":
    unittest.main()
