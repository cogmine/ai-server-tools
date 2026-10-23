import functools
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from launcher import ingest, launch
from launcher.data import QUANTIZATION_TOOLS
from launcher.models import AIModel, AIType, ModelFormat, RopeScaling
from launcher import storage
from launcher.storage import Location, model_metadata_dir


def _tool(tool_id: str):
    return next(t for t in QUANTIZATION_TOOLS if t.id == tool_id)


def _gguf_model(path: Path, **fields) -> AIModel:
    return AIModel(
        id="source", name="Source model", path=str(path),
        format=ModelFormat.GGUF, ai_types=frozenset({AIType.TEXT_GEN}), **fields,
    )


class DerivedModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.catalog = self.root / "models_local.json"
        self.catalog.write_text(json.dumps({"models": []}))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _finalize(self, model: AIModel, quant: str, output_dir: Path) -> dict:
        def unused_factory() -> tuple[subprocess.Popen[bytes], Path, int]:
            raise AssertionError("finalize_quantization must not start the process")

        job = launch.QuantizeJob(unused_factory, model, _tool("llama-quantize"), quant, output_dir)
        with (
            patch.object(ingest, "ingest_model",
                         functools.partial(ingest.ingest_model, catalog_path=self.catalog)),
            patch.object(launch, "AI_MODELS", []),
        ):
            launch.finalize_quantization(job)
        return json.loads(self.catalog.read_text())["models"][0]

    def test_requantized_gguf_copies_metadata_and_inherits_model_fields(self) -> None:
        source_dir = self.root / "Model-GGUF-F16"
        source_dir.mkdir()
        (source_dir / "config.json").write_text(json.dumps({
            "architectures": ["MistralForCausalLM"], "max_position_embeddings": 131072,
        }))
        (source_dir / "tokenizer.json").write_text("{}")
        source_file = source_dir / "Model.F16.gguf"
        source_file.touch()
        output_dir = self.root / "Model-GGUF-Q8_0"
        output_dir.mkdir()
        (output_dir / "Model.Q8_0.gguf").touch()
        model = _gguf_model(
            source_file, ctx=131072, arch="MistralForCausalLM", params_m=123000,
            tool_call_parser="hermes", rope_scaling=RopeScaling(factor=2.0),
            disabled_template_patches=frozenset({"system_anywhere"}),
            mmproj_path="/models/mmproj.gguf", vision=True, family="model-family",
            note="too big for VRAM", favorite=True, ctx_override=8192,
            server_params={"llamacpp": {"parallel": 2}},
        )

        entry = self._finalize(model, "Q8_0", output_dir)

        self.assertTrue((output_dir / "config.json").is_file())
        self.assertTrue((output_dir / "tokenizer.json").is_file())
        self.assertEqual(entry["arch"], "MistralForCausalLM")
        self.assertEqual(entry["ctx"], 131072)
        self.assertEqual(entry["params_m"], 123000)
        self.assertEqual(entry["tool_call_parser"], "hermes")
        self.assertEqual(entry["rope_scaling"], {"factor": 2.0})
        self.assertEqual(entry["disabled_template_patches"], ["system_anywhere"])
        self.assertEqual(entry["mmproj_path"], "/models/mmproj.gguf")
        self.assertTrue(entry["vision"])
        self.assertEqual(entry["quant"], "Q8_0")
        self.assertEqual(entry["family"], "model-family")
        for name in ("note", "favorite", "ctx_override", "server_params"):
            self.assertNotIn(name, entry)
        AIModel.from_dict(entry)  # inherited values round-trip through the schema

    def test_catalog_fields_are_inherited_when_source_has_no_config(self) -> None:
        source_file = self.root / "Model.F16.gguf"
        source_file.touch()
        output_dir = self.root / "Model-GGUF-Q4_K_M"
        output_dir.mkdir()
        (output_dir / "Model.Q4_K_M.gguf").touch()
        model = _gguf_model(source_file, ctx=65536, arch="LlamaForCausalLM", moe_experts=8, moe_active=2)

        entry = self._finalize(model, "Q4_K_M", output_dir)

        self.assertEqual(entry["ctx"], 65536)
        self.assertEqual(entry["arch"], "LlamaForCausalLM")
        self.assertEqual((entry["moe_experts"], entry["moe_active"]), (8, 2))

    def test_metadata_dir_for_per_quant_subdir_layout(self) -> None:
        bundle = self.root / "Model-GGUF"
        (bundle / "Q8_0").mkdir(parents=True)
        (bundle / "config.json").write_text("{}")
        gguf = bundle / "Q8_0" / "Model.Q8_0.gguf"
        gguf.touch()
        self.assertEqual(model_metadata_dir(gguf), bundle)
        self.assertEqual(model_metadata_dir(bundle), bundle)


class StorageTests(unittest.TestCase):
    """Copies are separate entries at mirrored paths; deletion removes exactly what it shows."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.nas = Location("NAS", root / "nas")
        self.fast = Location("NVMe", root / "fast")
        for loc in (self.nas, self.fast):
            loc.root.mkdir()
        self.catalog = root / "models_local.json"
        self.models: list[AIModel] = []
        self._patches = [
            patch.object(storage, "LOCATIONS", [self.nas, self.fast]),
            patch.object(launch, "LOCATIONS", [self.nas, self.fast]),
            patch.object(launch, "_CATALOG_PATH", self.catalog),
            patch.object(launch, "AI_MODELS", self.models),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _bundle(self, *quants: str) -> Path:
        bundle = self.nas.root / "text-generation" / "Model-GGUF"
        bundle.mkdir(parents=True)
        (bundle / "config.json").write_text("{}")
        (bundle / "README.md").write_text("readme")
        (bundle / "mmproj-F16.gguf").write_bytes(b"proj")
        for quant in quants:
            (bundle / f"Model.{quant}.gguf").write_bytes(quant.encode())
        return bundle

    def _add(self, path: Path, **fields) -> AIModel:
        entry = {
            "id": fields.pop("id", path.stem.lower()), "name": "Model", "path": str(path),
            "format": fields.pop("format", "gguf"), "ai_types": ["text_gen"], "family": "model",
            **fields,
        }
        catalog = json.loads(self.catalog.read_text()) if self.catalog.exists() else {"models": []}
        catalog["models"].append(entry)
        self.catalog.write_text(json.dumps(catalog))
        model = AIModel.from_dict(entry)
        self.models.append(model)
        return model

    def test_mirrored_path_and_location(self) -> None:
        path = self.nas.root / "text-generation" / "M" / "m.gguf"
        self.assertIs(storage.location_of(path), self.nas)
        self.assertEqual(storage.mirrored_path(path, self.fast), self.fast.root / "text-generation" / "M" / "m.gguf")
        self.assertIsNone(storage.location_of("/elsewhere/m.gguf"))

    def test_copy_mirrors_files_and_registers_snapshot_entry(self) -> None:
        bundle = self._bundle("Q8_0", "Q4_K_M")
        model = self._add(
            bundle / "Model.Q8_0.gguf", mmproj_path=str(bundle / "mmproj-F16.gguf"),
            server_status={"llamacpp": {"llama.cpp": {"none": "good"}}}, favorite=True,
        )

        proc = storage.copy_model(model, self.fast)
        _, stderr = proc.communicate()
        self.assertEqual(proc.returncode, 0, stderr)
        dest = self.fast.root / "text-generation" / "Model-GGUF"
        self.assertEqual(
            sorted(f.name for f in dest.iterdir()),
            ["Model.Q8_0.gguf", "config.json", "mmproj-F16.gguf"],  # not other quants / README
        )

        copies = launch.register_copy(model, self.fast)
        self.assertEqual(len(copies), 1)
        copy = copies[0]
        self.assertEqual((copy.id, copy.family), (model.id, model.family))
        self.assertEqual(copy.path, str(dest / "Model.Q8_0.gguf"))
        self.assertEqual(copy.mmproj_path, str(dest / "mmproj-F16.gguf"))
        self.assertEqual(copy.server_status, model.server_status)
        self.assertTrue(copy.favorite)
        self.assertEqual(launch.copies_of(model), [copy])
        launch.register_copy(model, self.fast)  # idempotent
        self.assertEqual(len(json.loads(self.catalog.read_text())["models"]), 2)

    def test_ingest_recognises_a_copy_at_a_mirrored_path(self) -> None:
        bundle = self._bundle("Q8_0")
        original = self._add(bundle / "Model.Q8_0.gguf", id="model-q8", note="tested")
        copy_path = self.fast.root / "text-generation" / "Model-GGUF" / "Model.Q8_0.gguf"
        copy_path.parent.mkdir(parents=True)
        copy_path.write_bytes(b"Q8_0")
        catalog = json.loads(self.catalog.read_text())
        entry = ingest.build_entry(copy_path, catalog, fmt="gguf")
        self.assertEqual((entry["id"], entry["note"], entry["path"]), (original.id, "tested", str(copy_path)))

    def test_delete_keeps_bundle_while_other_quants_remain(self) -> None:
        bundle = self._bundle("Q8_0", "Q4_K_M")
        mmproj = str(bundle / "mmproj-F16.gguf")
        q8 = self._add(bundle / "Model.Q8_0.gguf", mmproj_path=mmproj)
        self._add(bundle / "Model.Q4_K_M.gguf", mmproj_path=mmproj)

        plan = storage.plan_deletion(q8, self.models)
        self.assertEqual(plan.dirs, ())
        self.assertEqual(plan.files, (bundle / "Model.Q8_0.gguf",))  # shared mmproj kept
        launch.delete_model(plan)
        self.assertFalse((bundle / "Model.Q8_0.gguf").exists())
        self.assertTrue((bundle / "mmproj-F16.gguf").exists())
        self.assertEqual([m.path for m in self.models], [str(bundle / "Model.Q4_K_M.gguf")])
        self.assertEqual(len(json.loads(self.catalog.read_text())["models"]), 1)

    def test_delete_last_model_removes_bundle_and_empty_parents(self) -> None:
        bundle = self._bundle("Q8_0")
        model = self._add(bundle / "Model.Q8_0.gguf", mmproj_path=str(bundle / "mmproj-F16.gguf"))
        plan = storage.plan_deletion(model, self.models)
        self.assertEqual(plan.dirs, (bundle,))
        launch.delete_model(plan)
        self.assertFalse((self.nas.root / "text-generation").exists())
        self.assertTrue(self.nas.root.is_dir())  # never the location root
        self.assertEqual(self.models, [])

    def test_delete_directory_model_takes_entries_sharing_it(self) -> None:
        model_dir = self.nas.root / "text-generation" / "Model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"w")
        st = self._add(model_dir, id="model-bf16", format="safetensors_bf16")
        self._add(model_dir, id="model-bin", format="pytorch_bin")
        plan = storage.plan_deletion(st, self.models)
        self.assertEqual((plan.dirs, len(plan.entries)), ((model_dir,), 2))

    def test_staging_uses_one_copy_per_model(self) -> None:
        bundle = self._bundle("Q8_0")
        nas_copy = self._add(bundle / "Model.Q8_0.gguf", id="m")
        fast_copy = self._add(self.fast.root / "text-generation" / "Model-GGUF" / "Model.Q8_0.gguf", id="m")
        self.assertEqual(launch._one_copy_each([fast_copy, nas_copy]), [nas_copy])  # on disk + first location


if __name__ == "__main__":
    unittest.main()
