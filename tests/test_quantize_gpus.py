import dataclasses
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from textual.app import App
from textual.widgets import SelectionList

from launcher import launch, screens
from launcher.data import QUANTIZATION_TOOLS, SERVERS
from launcher.environments import NativeEnvironment
from launcher.gpu import GpuRow
from launcher.models import AIModel, AIType, CacheMount, ModelFormat


def model_for(fmt: ModelFormat) -> AIModel:
    return AIModel(
        id="test-model", name="Test model", path="/models/test.Q8_0.gguf",
        format=fmt, ai_types=frozenset({AIType.TEXT_GEN}),
    )


def gpu_row(index: int, error: str = "") -> GpuRow:
    return GpuRow(
        index=index, bus_id="03", name="Test GPU", power_w=0, power_limit_w=350,
        mem_used_mib=1024, mem_total_mib=24576, util_pct=0, temp_c=30,
        pcie_gen="4/4", pcie_width_cur=16, pcie_width_cap=16, error=error,
    )


class PickerApp(App):
    CSS_PATH = "../launcher/model_launcher.tcss"

    def __init__(self, screen) -> None:
        super().__init__()
        self.initial_screen = screen

    def on_mount(self) -> None:
        self.push_screen(self.initial_screen)


class QuantizeCommandTests(unittest.TestCase):
    def test_device_and_cuda_environment_for_both_tools(self) -> None:
        env = NativeEnvironment(server_id="test", name="test", bin_dir="/tools")
        dest = CacheMount("test", Path("/cache"), 0)
        with patch.object(launch, "_tool_env", return_value=env):
            for registered in QUANTIZATION_TOOLS:
                tool = dataclasses.replace(registered, cuda_home="/cuda toolkit")
                model = model_for(next(iter(tool.input_formats)))
                with self.subTest(tool=tool.id):
                    cmd, _ = launch.build_quantize_cmd(
                        tool, model, tool.quant_options[0], dest, "2,7",
                    )
                    self.assertEqual(cmd[:3], [
                        "env", "CUDA_VISIBLE_DEVICES=2,7", "CUDA_HOME=/cuda toolkit",
                    ])
                    cmd, _ = launch.build_quantize_cmd(
                        tool, model, tool.quant_options[0], dest,
                    )
                    self.assertEqual(
                        cmd[1], f"CUDA_VISIBLE_DEVICES={launch.CONFIG.resolved_cuda_devices}",
                    )

    def test_spawn_and_log_use_preview_command(self) -> None:
        tool = QUANTIZATION_TOOLS[0]
        model = model_for(ModelFormat.GGUF)
        env = NativeEnvironment(server_id="test", name="test", bin_dir="/tools")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = CacheMount("test", root / "output", 0)
            with (
                patch.object(launch, "_tool_env", return_value=env),
                patch.object(launch, "_LOGS_DIR", root / "logs"),
                patch.object(launch, "_spawn_with_pty") as spawn,
            ):
                spawn.return_value = (None, -1)
                cmd, output = launch.build_quantize_cmd(tool, model, "Q4_K_M", dest, "2,7")
                _, log, _ = launch.quantize(tool, model, "Q4_K_M", dest, "2,7")
                self.assertTrue(output.is_dir())
                command = launch.format_cmd(cmd)
                self.assertEqual(spawn.call_args.args[0], command)
                self.assertIn(command, log.read_text())

    def test_explicit_devices_override_inherited_visibility(self) -> None:
        tool = QUANTIZATION_TOOLS[0]
        model = model_for(ModelFormat.GGUF)
        env = NativeEnvironment(server_id="test", name="test", bin_dir="/tools")
        dest = CacheMount("test", Path("/cache"), 0)
        with (
            patch.object(launch, "_tool_env", return_value=env),
            patch.dict(launch._QUANT_BUILDERS, {
                tool.id: lambda *_: ["/bin/sh", "-c", 'printf %s "$CUDA_VISIBLE_DEVICES"'],
            }),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}),
        ):
            cmd, _ = launch.build_quantize_cmd(tool, model, "Q4_K_M", dest, "2,7")
            result = launch.subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout, "2,7")


class QuantizePickerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.model = model_for(ModelFormat.GGUF)
        self.tool = QUANTIZATION_TOOLS[0]
        self.dest = CacheMount("test", Path("/cache"), 0)
        self.env = NativeEnvironment(server_id="test", name="test", bin_dir="/tools")

    async def test_destination_picker_confirmation_and_back(self) -> None:
        picker = screens.QuantizeDestScreen(self.model, self.tool, "Q4_K_M")
        app = PickerApp(picker)
        with (
            patch.object(screens, "CACHE_MOUNTS", [self.dest]),
            patch.object(launch, "cache_disk_info", return_value=(100, 80)),
            patch.object(screens, "poll_gpus", return_value=[
                gpu_row(2), gpu_row(4, "unavailable"), gpu_row(7),
            ]),
            patch.object(launch, "_tool_env", return_value=self.env),
            patch.object(launch, "quantize") as quantize,
        ):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.press("enter")
                self.assertIsInstance(app.screen, screens.QuantizeGpuScreen)
                sl = app.screen.query_one(SelectionList)
                self.assertEqual(sorted(sl.selected), [2, 7])
                sl.deselect_all()
                await pilot.pause()
                await pilot.press("enter")
                self.assertIsInstance(app.screen, screens.QuantizeGpuScreen)
                await pilot.press("space")
                self.assertEqual(sl.selected, [2])
                await pilot.press("a")
                self.assertEqual(sorted(sl.selected), [2, 7])
                sl.deselect(2)
                await pilot.press("right")
                confirm = app.screen
                self.assertIsInstance(confirm, screens.ConfirmQuantize)
                self.assertIn("CUDA_VISIBLE_DEVICES=7", confirm._description)
                confirm._proc_factory()
                quantize.assert_called_once_with(self.tool, self.model, "Q4_K_M", self.dest, "7")
                await pilot.press("escape")
                self.assertIsInstance(app.screen, screens.QuantizeGpuScreen)
                self.assertEqual(sl.selected, [7])
                await pilot.press("escape")
                self.assertIs(app.screen, picker)

    async def test_missing_gpus_cannot_launch_quantization(self) -> None:
        picker = screens.QuantizeGpuScreen(self.model, self.tool, "Q4_K_M", self.dest)
        app = PickerApp(picker)
        with patch.object(screens, "poll_gpus", return_value=[gpu_row(4, "unavailable")]):
            async with app.run_test() as pilot:
                await pilot.press("enter")
                self.assertIs(app.screen, picker)
                self.assertTrue(app._notifications)

    async def test_server_single_gpu_selection_is_preserved(self) -> None:
        server = dataclasses.replace(
            SERVERS[0], max_gpus=1, gpu_count_param=None, concurrency_param=None,
        )
        picker = screens.GpuSelectScreen(server, self.env, None)
        app = PickerApp(picker)
        with patch.object(screens, "poll_gpus", return_value=[gpu_row(2), gpu_row(7)]):
            async with app.run_test() as pilot:
                sl = picker.query_one(SelectionList)
                self.assertEqual(sl.selected, [2])
                await pilot.press("down", "space")
                self.assertEqual(sl.selected, [7])
                self.assertFalse(picker.check_action("select_all", ()))
                self.assertEqual(picker._launch_settings(), ("7", {}))

    async def test_server_missing_gpu_fallback_is_preserved(self) -> None:
        picker = screens.GpuSelectScreen(SERVERS[0], self.env, None)
        app = PickerApp(picker)
        with (
            patch.object(screens, "poll_gpus", return_value=[]),
            patch.object(launch, "build_launch_command", return_value="test-command"),
        ):
            async with app.run_test() as pilot:
                await pilot.pause()
                self.assertIsInstance(app.screen, screens.ConfirmLaunch)

    async def test_server_parallelism_preselection_is_preserved(self) -> None:
        server = next(s for s in SERVERS if s.id == "vllm")
        picker = screens.GpuSelectScreen(server, self.env, self.model)
        app = PickerApp(picker)
        with (
            patch.object(screens, "poll_gpus", return_value=[gpu_row(2), gpu_row(7)]),
            patch.object(screens, "fit_probe_for", return_value=None),
            patch.object(launch, "compute_parallelism", return_value=launch.ParallelismPlan(1, 1, 1)),
        ):
            async with app.run_test() as pilot:
                self.assertEqual(picker.query_one(SelectionList).selected, [2])
                await pilot.press("a")
                self.assertEqual(sorted(picker.query_one(SelectionList).selected), [2, 7])
                self.assertEqual(
                    picker._launch_settings(), ("2", {"tensor_parallel": 1, "pipeline_parallel": 1}),
                )

    async def test_server_slots_are_preserved(self) -> None:
        server = next(s for s in SERVERS if s.id == "llamacpp")
        with patch.object(screens, "fit_probe_for", return_value=None):
            picker = screens.GpuSelectScreen(server, self.env, self.model)
        initial_slots = launch.launch_params(server, self.model)["parallel"]
        app = PickerApp(picker)
        with patch.object(screens, "poll_gpus", return_value=[gpu_row(2), gpu_row(7)]):
            async with app.run_test() as pilot:
                self.assertEqual(picker._launch_settings(), ("2,7", {"parallel": initial_slots}))
                await pilot.press("plus")
                self.assertEqual(picker._launch_settings(), ("2,7", {"parallel": initial_slots + 1}))


if __name__ == "__main__":
    unittest.main()
