import dataclasses
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from textual.app import App
from textual.widgets import SelectionList

from launcher import launch, screens
from launcher.data import QUANTIZATION_TOOLS, SERVERS
from launcher.environments import NativeEnvironment
from launcher.gpu import GpuRow
from launcher.models import AIModel, AIType, ModelFormat, tools_for_model
from launcher.storage import Location


def model_for(fmt: ModelFormat) -> AIModel:
    return AIModel(
        id="test-model", name="Test model",
        path="/models/test.Q8_0.gguf" if fmt == ModelFormat.GGUF else "/models/test-model",
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
    def test_cuda_environment_matches_tool_requirements(self) -> None:
        env = NativeEnvironment(server_id="test", name="test", bin_dir="/tools")
        dest = Location("test", Path("/cache"))
        builders = {tool.id: lambda *_: ["quant-tool"] for tool in QUANTIZATION_TOOLS}
        with (
            patch.object(launch, "_tool_env", return_value=env),
            patch.dict(launch._QUANT_BUILDERS, builders),
        ):
            for registered in QUANTIZATION_TOOLS:
                tool = dataclasses.replace(registered, cuda_home="/cuda toolkit")
                model = model_for(next(iter(tool.input_formats)))
                with self.subTest(tool=tool.id):
                    cmd, _ = launch.build_quantize_cmd(
                        tool, model, tool.quant_options[0], dest, "2,7",
                    )
                    if tool.uses_cuda:
                        self.assertEqual(cmd[:3], [
                            "env", "CUDA_VISIBLE_DEVICES=2,7", "CUDA_HOME=/cuda toolkit",
                        ])
                    else:
                        self.assertEqual(cmd, ["quant-tool"])
                    cmd, _ = launch.build_quantize_cmd(
                        tool, model, tool.quant_options[0], dest,
                    )
                    if tool.uses_cuda:
                        self.assertEqual(
                            cmd[1], f"CUDA_VISIBLE_DEVICES={launch.CONFIG.resolved_cuda_devices}",
                        )
                    else:
                        self.assertEqual(cmd, ["quant-tool"])

    def test_hf_to_gguf_builds_cpu_converter_command(self) -> None:
        tool = next(tool for tool in QUANTIZATION_TOOLS if tool.id == "llama-hf-to-gguf")
        dest = Location("test", Path("/cache"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "llama.cpp"
            (root / "build" / "bin").mkdir(parents=True)
            source = root / "hf-model"
            source.mkdir()
            script = root / "convert_hf_to_gguf.py"
            script.touch()
            model = AIModel(
                id="hf-model", name="HF model", path=str(source),
                format=ModelFormat.SAFETENSORS_BF16,
                ai_types=frozenset({AIType.TEXT_GEN}),
            )
            env = NativeEnvironment(
                server_id="llamacpp", name="test", bin_dir=str(root / "build" / "bin"),
            )
            with patch.object(launch, "_tool_env", return_value=env):
                cmd, output_dir = launch.build_quantize_cmd(tool, model, "BF16", dest)

            output_file = output_dir / "hf-model.BF16.gguf"
            self.assertEqual(output_dir, Path("/cache/text-generation/hf-model-GGUF-BF16"))
            self.assertEqual(cmd, [
                tool.python_executable, str(script),
                "--outfile", str(output_file), "--outtype", "bf16", str(source),
            ])

    def test_hf_to_gguf_is_not_offered_for_vision_models(self) -> None:
        tool = next(tool for tool in QUANTIZATION_TOOLS if tool.id == "llama-hf-to-gguf")
        model = dataclasses.replace(model_for(ModelFormat.SAFETENSORS_BF16), vision=True)
        self.assertNotIn(tool, tools_for_model(QUANTIZATION_TOOLS, model))

    def test_finalize_gguf_registers_the_output_file(self) -> None:
        from launcher import ingest

        tool = next(tool for tool in QUANTIZATION_TOOLS if tool.id == "llama-hf-to-gguf")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            output_dir = root / "converted-GGUF-F16"
            output_dir.mkdir()
            output_file = output_dir / "converted.F16.gguf"
            output_file.touch()
            model = AIModel(
                id="source", name="Source model", path=str(source),
                format=ModelFormat.SAFETENSORS_BF16,
                ai_types=frozenset({AIType.TEXT_GEN}),
            )
            entry = {
                "id": "converted-gguf-f16",
                "name": "Source model",
                "path": str(output_file),
                "format": ModelFormat.GGUF.value,
            }

            def unused_factory() -> tuple[subprocess.Popen[bytes], Path, int]:
                raise AssertionError("finalize_quantization must not start the process")

            job = launch.QuantizeJob(
                unused_factory, model, tool, "F16", output_dir,
            )
            with (
                patch.object(ingest, "copy_model_metadata") as copy_metadata,
                patch.object(ingest, "ingest_model", return_value=entry) as ingest_model,
                patch.object(launch, "AI_MODELS", []),
            ):
                result = launch.finalize_quantization(job)

            copy_metadata.assert_called_once_with(source, output_dir)
            ingest_model.assert_called_once()
            self.assertEqual(ingest_model.call_args.args[0], output_file)
            self.assertEqual(ingest_model.call_args.kwargs["fmt"], ModelFormat.GGUF.value)
            self.assertEqual(ingest_model.call_args.kwargs["entry_overrides"]["quant"], "F16")
            self.assertEqual(result.path, str(output_file))

    def test_spawn_and_log_use_preview_command(self) -> None:
        tool = QUANTIZATION_TOOLS[0]
        model = model_for(ModelFormat.GGUF)
        env = NativeEnvironment(server_id="test", name="test", bin_dir="/tools")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = Location("test", root / "output")
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
        dest = Location("test", Path("/cache"))
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
        self.dest = Location("test", Path("/cache"))
        self.env = NativeEnvironment(server_id="test", name="test", bin_dir="/tools")

    async def test_destination_picker_confirmation_and_back(self) -> None:
        picker = screens.QuantizeDestScreen(self.model, self.tool, "Q4_K_M")
        app = PickerApp(picker)
        with (
            patch.object(screens, "LOCATIONS", [self.dest]),
            patch.object(Location, "online", property(lambda _: True)),
            patch.object(Location, "disk_info", return_value=(100, 80)),
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

    async def test_cpu_conversion_skips_gpu_picker(self) -> None:
        tool = next(tool for tool in QUANTIZATION_TOOLS if tool.id == "llama-hf-to-gguf")
        model = model_for(ModelFormat.SAFETENSORS_BF16)
        picker = screens.QuantizeDestScreen(model, tool, "F16")
        app = PickerApp(picker)
        with (
            patch.object(screens, "LOCATIONS", [self.dest]),
            patch.object(Location, "online", property(lambda _: True)),
            patch.object(Location, "disk_info", return_value=(100, 80)),
            patch.object(launch, "build_quantize_cmd", return_value=(["python", "convert.py"], Path("/cache/out"))),
        ):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.press("enter")
                self.assertIsInstance(app.screen, screens.ConfirmQuantize)

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
