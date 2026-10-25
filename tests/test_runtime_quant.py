import dataclasses
import tempfile
import unittest
from pathlib import Path

from launcher import ingest, launch
from launcher.data import QUANTIZATION_TOOLS, SERVERS
from launcher.environments import CondaEnvironment
from launcher.hf_config import read_parallelism_fields
from launcher.models import AIModel, AIType, ModelFormat, RuntimeQuant, tools_for_model
from launcher.storage import Location


def _server(server_id: str):
    return next(s for s in SERVERS if s.id == server_id)


def _tool(tool_id: str):
    return next(t for t in QUANTIZATION_TOOLS if t.id == tool_id)


def _hf_model(fmt: ModelFormat = ModelFormat.SAFETENSORS_BF16) -> AIModel:
    return AIModel(
        id="hf-model", name="HF model", path="/models/HF-Model", format=fmt,
        ai_types=frozenset({AIType.TEXT_GEN}),
    )


class RuntimeQuantTests(unittest.TestCase):
    @staticmethod
    def _offered(server_id: str, model: AIModel) -> tuple[RuntimeQuant, ...]:
        return tuple(o.quant for o in _server(server_id).runtime_quant_options(model))

    def test_server_offerings_ignore_local_hardware(self) -> None:
        dense, moe = _hf_model(), dataclasses.replace(_hf_model(), moe_experts=64)
        self.assertEqual(self._offered("vllm", dense), (
            RuntimeQuant.FP8_BLOCK, RuntimeQuant.FP8_CHANNEL, RuntimeQuant.FP8,
            RuntimeQuant.MXFP8, RuntimeQuant.BNB_NF4))
        self.assertIn(RuntimeQuant.INT8_EXPERTS, self._offered("vllm", moe))
        self.assertIn(RuntimeQuant.NVFP4_EXPERTS, self._offered("vllm", moe))
        self.assertEqual(self._offered("sglang", dense), (RuntimeQuant.FP8, RuntimeQuant.BNB_NF4))
        self.assertEqual(self._offered("ooba", dense), (RuntimeQuant.BNB_INT8, RuntimeQuant.BNB_NF4))
        self.assertEqual(self._offered("vllm", _hf_model(ModelFormat.SAFETENSORS_FP8)), ())

    def test_hardware_requirements(self) -> None:
        vllm = _server("vllm")
        block = vllm.runtime_quant_option(RuntimeQuant.FP8_BLOCK).requires
        self.assertFalse(block.met_by(86))
        self.assertTrue(block.met_by(90))
        self.assertTrue(block.met_by(0))  # unknown GPUs are never flagged
        self.assertTrue(vllm.runtime_quant_option(RuntimeQuant.FP8).requires.met_by(86))
        self.assertIn("sm_89+", block.summary)

    def test_fit_hints_respect_hardware_and_full_precision_loads(self) -> None:
        model = dataclasses.replace(_hf_model(), size_gb=100)
        on_ampere = dict(model.vram_fit_hints(60, SERVERS, system_sm=86))
        self.assertNotIn(RuntimeQuant.FP8_BLOCK, on_ampere)
        self.assertEqual(on_ampere[RuntimeQuant.FP8], 50)  # vLLM's layer-wise load, not SGLang's
        self.assertIn(RuntimeQuant.FP8_BLOCK, dict(model.vram_fit_hints(60, SERVERS, system_sm=90)))
        sglang_fp8 = _server("sglang").runtime_quant_option(RuntimeQuant.FP8)
        self.assertEqual(sglang_fp8.load_gb(100), 100)

    def test_builders_emit_the_server_flags(self) -> None:
        env = CondaEnvironment(server_id="vllm", name="vllm")
        model = _hf_model()
        cases = {
            ("vllm", RuntimeQuant.FP8): ["--quantization", "fp8"],
            ("vllm", RuntimeQuant.BNB_NF4): ["--quantization", "bitsandbytes"],
            ("sglang", RuntimeQuant.BNB_NF4): ["--quantization", "bitsandbytes"],
            ("ooba", RuntimeQuant.BNB_INT8): ["--load-in-8bit"],
            ("ooba", RuntimeQuant.BNB_NF4): ["--load-in-4bit"],
        }
        for (server_id, quant), expected in cases.items():
            with self.subTest(server=server_id, quant=quant):
                cmd = launch.build_cmd(_server(server_id), env, model, quant)
                joined = " ".join(cmd)
                self.assertIn(" ".join(expected), joined)
        self.assertNotIn("--quantization", launch.build_cmd(_server("vllm"), env, model, None))

    def test_unsupported_quant_is_rejected(self) -> None:
        env = CondaEnvironment(server_id="vllm", name="vllm")
        with self.assertRaises(ValueError):
            launch.build_cmd(_server("vllm"), env, _hf_model(), RuntimeQuant.BNB_INT8)

    def test_every_quant_has_display_info(self) -> None:
        for quant in RuntimeQuant:
            self.assertTrue(quant.label and quant.short and 0 < quant.vram_ratio < 1)


class LlmCompressorToolTests(unittest.TestCase):
    def test_offered_for_safetensors_only(self) -> None:
        offered = {t.id for t in tools_for_model(QUANTIZATION_TOOLS, _hf_model())}
        self.assertTrue({"llmc-fp8", "llmc-int"} <= offered)
        offered_bin = {t.id for t in tools_for_model(QUANTIZATION_TOOLS, _hf_model(ModelFormat.PYTORCH_BIN))}
        self.assertFalse({"llmc-fp8", "llmc-int"} & offered_bin)

    def test_command_and_output(self) -> None:
        tool = _tool("llmc-fp8")
        env = CondaEnvironment(server_id="llm-compressor", name="llm-compressor")
        cmd, output = launch.build_quantize_cmd(
            tool, env, _hf_model(), "FP8_BLOCK", Location("test", Path("/cache")), "0,1",
        )
        self.assertEqual(output, Path("/cache/text-generation/HF-Model-FP8_BLOCK"))
        self.assertEqual(cmd, [
            "env", "CUDA_VISIBLE_DEVICES=0,1",
            str(env.python), str(launch._LLMC_SCRIPT),
            "--scheme", "FP8_BLOCK", "/models/HF-Model", str(output),
        ])
        self.assertTrue(launch._LLMC_SCRIPT.is_file())

    def test_output_naming_templates(self) -> None:
        dest = Location("test", Path("/cache"))
        gguf = AIModel(id="m", name="M", path="/m/Model.Q8_0.gguf", format=ModelFormat.GGUF,
                       ai_types=frozenset({AIType.TEXT_GEN}))
        self.assertEqual(
            launch.compute_quant_output_path(_tool("llama-quantize"), gguf, "Q4_K_M", dest).name,
            "Model-GGUF-Q4_K_M")
        self.assertEqual(
            launch.compute_quant_output_path(_tool("exl2-convert"), _hf_model(), "4.0", dest).name,
            "HF-Model-exl2-4.0bpw")
        self.assertEqual(_tool("exl2-convert").quant_label_template.format(option="4.0"), "4.0 bpw")

    def test_options_cover_newer_hardware(self) -> None:
        mx = _tool("llmc-int").option("MXFP4")
        self.assertEqual(mx.requires.min_sm, 100)
        self.assertFalse(mx.requires.met_by(86))
        self.assertEqual(_tool("llmc-int").option("W4A16").requires.min_sm, 0)

    def test_finalize_records_serving_requirement(self) -> None:
        from unittest.mock import patch
        tool = _tool("llmc-int")
        captured = {}

        def fake_ingest(path, fmt, entry_overrides):
            captured.update(entry_overrides)
            return {"path": "/nowhere", "format": fmt, "id": "x", "name": "x", "ai_types": ["text_gen"]}

        job = launch.QuantizeJob(lambda: None, _hf_model(), tool, "MXFP4", Path("/tmp/out-MXFP4"))  # type: ignore[arg-type]
        with patch("launcher.ingest.copy_model_metadata"), patch("launcher.ingest.ingest_model", fake_ingest), \
                patch.object(launch, "AI_MODELS", []):
            launch.finalize_quantization(job)
        self.assertEqual(captured["min_sm"], 100)
        self.assertIn("MXFP4 needs sm_100+", captured["hw_note"])


class CompressedTensorsDetectionTests(unittest.TestCase):
    @staticmethod
    def _cfg(group: str, weights: dict) -> dict:
        return {"quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {group: {"targets": ["Linear"], "weights": weights}},
        }}

    def test_format_and_scheme(self) -> None:
        fp8 = self._cfg("FP8_BLOCK", {"type": "float", "num_bits": 8, "block_structure": [128, 128]})
        int4 = self._cfg("group_0", {"type": "int", "num_bits": 4, "group_size": 128})
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ingest._detect_hf_format(Path(tmp), fp8), "safetensors_fp8")
            self.assertEqual(ingest._detect_hf_format(Path(tmp), int4), "compressed_tensors")
        self.assertEqual(ingest._detect_compressed_tensors_scheme(fp8), "FP8_BLOCK")
        self.assertEqual(ingest._detect_compressed_tensors_scheme(int4), "")

    def test_block_structure_feeds_parallelism_fields(self) -> None:
        import json
        cfg = {"num_attention_heads": 8, **self._cfg(
            "FP8_BLOCK", {"type": "float", "num_bits": 8, "block_structure": [128, 128]})}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps(cfg))
            fields = read_parallelism_fields(tmp)
        assert fields is not None
        self.assertEqual(fields["weight_block_size"], (128, 128))


class LlmcQuantizeScriptTests(unittest.TestCase):
    """The wrapper's stdlib-only helpers (torch / llmcompressor are imported lazily)."""

    @classmethod
    def setUpClass(cls) -> None:
        import importlib.machinery
        import importlib.util
        loader = importlib.machinery.SourceFileLoader("llmc_quantize", str(launch._LLMC_SCRIPT))
        spec = importlib.util.spec_from_loader("llmc_quantize", loader)
        assert spec is not None
        cls.script = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.script)

    def test_staging_keeps_indexed_shards_and_metadata_only(self) -> None:
        import json
        with tempfile.TemporaryDirectory() as tmp:
            src, staging = Path(tmp) / "src", Path(tmp) / "staging"
            src.mkdir(); staging.mkdir()
            for name in ("model-1.safetensors", "model-2.safetensors", "consolidated.safetensors",
                         "pytorch_model.bin", "pytorch_model.bin.index.json", "config.json",
                         "tokenizer.json", ".gitattributes"):
                (src / name).touch()
            (src / "original").mkdir()
            (src / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}))
            shards = self.script._weight_shards(src)
            skipped = self.script._stage_source(src, shards, staging)
            self.assertEqual([s.name for s in shards], ["model-1.safetensors", "model-2.safetensors"])
            self.assertEqual(sorted(p.name for p in staging.iterdir()), [
                "config.json", "model-1.safetensors", "model-2.safetensors",
                "model.safetensors.index.json", "tokenizer.json",
            ])
            self.assertIn("original/", skipped)
            self.assertIn("consolidated.safetensors", skipped)

    def test_fused_siblings_are_ignored_together(self) -> None:
        shapes = {f"m.l.{p}.weight": [8, 8] for p in ("q_proj", "k_proj", "v_proj", "o_proj",
                                                      "in_proj_a", "in_proj_b")}
        result = self.script._with_fused_siblings({"m.l.k_proj", "m.l.in_proj_a"}, shapes)
        self.assertEqual(result, {"m.l.q_proj", "m.l.k_proj", "m.l.v_proj",
                                  "m.l.in_proj_a", "m.l.in_proj_b"})


class Fp8SourceTests(unittest.TestCase):
    """Converting FP8-only releases (e.g. DeepSeek V3/V4)."""

    @staticmethod
    def _checkpoint(tmp: str, scale_suffix: str | None) -> AIModel:
        import json
        root = Path(tmp)
        (root / "config.json").write_text(json.dumps({"quantization_config": {
            "quant_method": "fp8", "weight_block_size": [128, 128]}}))
        weights = {"model.layers.0.mlp.down_proj.weight": "model-1.safetensors"}
        if scale_suffix:
            weights[f"model.layers.0.mlp.down_proj.{scale_suffix}"] = "model-1.safetensors"
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
        return AIModel(
            id="fp8", name="FP8", path=tmp, format=ModelFormat.SAFETENSORS_FP8,
            ai_types=frozenset({AIType.TEXT_GEN}),
        )

    def test_block_fp8_gets_every_tool_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._checkpoint(tmp, "weight_scale_inv")
            tools = {t.id for t in tools_for_model(QUANTIZATION_TOOLS, model)}
        self.assertEqual(tools, {"llama-hf-to-gguf", "llmc-fp8", "llmc-int", "llmc-dequant"})

    def test_native_layouts_get_llama_cpp_only(self) -> None:
        """DeepSeek-V4-style checkpoints (".scale" tensors, FP4 experts): llama.cpp converts them."""
        with tempfile.TemporaryDirectory() as tmp:
            model = self._checkpoint(tmp, "scale")
            tools = {t.id for t in tools_for_model(QUANTIZATION_TOOLS, model)}
        self.assertEqual(tools, {"llama-hf-to-gguf"})

    def test_fp8_only_options(self) -> None:
        gguf = _tool("llama-hf-to-gguf")
        self.assertIn("FP8-Q8_0", [o.id for o in gguf.options_for(_hf_model(ModelFormat.SAFETENSORS_FP8))])
        self.assertNotIn("FP8-Q8_0", [o.id for o in gguf.options_for(_hf_model())])
        self.assertEqual(gguf.option("FP8-Q8_0").args, ("--outtype", "bf16", "--fp8-as-q8"))
        self.assertFalse(_tool("llmc-dequant").converts(_hf_model()))

    def test_requantize_flag_follows_source_precision(self) -> None:
        from launcher.environments import NativeEnvironment
        env = NativeEnvironment(server_id="llamacpp", name="t", bin_dir="/bin")
        tool, out = _tool("llama-quantize"), Path("/out")

        def cmd(quant: str) -> list[str]:
            model = AIModel(id="g", name="g", path=f"/m/M.{quant}.gguf", format=ModelFormat.GGUF,
                            ai_types=frozenset({AIType.TEXT_GEN}), quant=quant)
            return launch._QUANT_BUILDERS[tool.id](tool, env, model, "Q4_K_M", out)

        self.assertNotIn("--allow-requantize", cmd("BF16"))
        self.assertIn("--allow-requantize", cmd("FP8-Q8_0"))
        self.assertEqual(Path(cmd("FP8-Q8_0")[-2]).name, "M.Q4_K_M.gguf")  # source quant stripped

    def test_large_conversions_spool_to_the_destination(self) -> None:
        from unittest.mock import patch
        from launcher.environments import NativeEnvironment
        tool, out = _tool("llama-hf-to-gguf"), Path("/dest/text-generation/X-GGUF-BF16")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "build" / "bin").mkdir(parents=True)
            (root / "convert_hf_to_gguf.py").touch()
            (root / "src").mkdir()
            env = NativeEnvironment(server_id="llamacpp", name="t", bin_dir=str(root / "build" / "bin"))

            def cmd(size_gb: float) -> list[str]:
                model = AIModel(id="m", name="m", path=str(root / "src"), format=ModelFormat.SAFETENSORS_FP8,
                                ai_types=frozenset({AIType.TEXT_GEN}), size_gb=size_gb)
                return launch._build_llama_hf_to_gguf(tool, env, model, "BF16", out)

            with patch.object(launch.CONFIG, "convert_ram_mib", 100 * 1024):
                self.assertNotIn("--use-temp-file", cmd(50))
                large = cmd(865)
        self.assertIn("--use-temp-file", large)
        self.assertEqual(large[:2], ["env", f"TMPDIR={out}"])


if __name__ == "__main__":
    unittest.main()
