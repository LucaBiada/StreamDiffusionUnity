"""Protocol + transport tests. No GPU needed (server runs with --fake-pipeline).

    <venv>\\python.exe -m unittest discover -s tests -v
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import websockets  # noqa: E402
from sd_client import SdClient  # noqa: E402
from server.transports.shmem import ShmChannel  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ShmemTests(unittest.TestCase):
    def test_roundtrip_and_no_old_frames(self):
        name = f"sdtest_{os.getpid()}_a"
        a = ShmChannel.create(name, 64, 48, 4)
        b = ShmChannel.open(name)
        try:
            self.assertIsNone(b.read_latest(0))
            img = np.random.randint(0, 255, (48, 64, 4), dtype=np.uint8)
            a.publish(img, frame_id=7, timestamp_ns=123)
            fr = b.read_latest(0)
            self.assertIsNotNone(fr)
            self.assertEqual(fr.frame_id, 7)
            self.assertEqual(fr.timestamp_ns, 123)
            np.testing.assert_array_equal(fr.pixels, img)
            self.assertIsNone(b.read_latest(fr.index))  # nothing newer
        finally:
            b.close()
            a.close()

    def test_latest_wins(self):
        name = f"sdtest_{os.getpid()}_b"
        a = ShmChannel.create(name, 32, 32, 3)
        b = ShmChannel.open(name)
        try:
            for i in range(1, 6):
                a.publish(np.full((32, 32, 3), i, np.uint8), frame_id=i)
            fr = b.read_latest(0)
            self.assertEqual(fr.frame_id, 5)
            self.assertEqual(int(fr.pixels[0, 0, 0]), 5)
        finally:
            b.close()
            a.close()

    def test_wrong_shape_rejected(self):
        name = f"sdtest_{os.getpid()}_c"
        a = ShmChannel.create(name, 16, 16, 4)
        try:
            with self.assertRaises(ValueError):
                a.publish(np.zeros((8, 8, 4), np.uint8), 1)
        finally:
            a.close()


class EngineCacheTests(unittest.TestCase):
    def _make(self, root, res_hw=(576, 1024), batch=1, with_log=True):
        import json, pathlib
        base = pathlib.Path(root) / "stabilityai"
        vae = base / f"sdxl-turbo--tiny_vae-True--min_batch-1--max_batch-1--optlvl3--mode-img2img--trt10.16--cc120--res-{res_hw[0]}x{res_hw[1]}"
        unet = base / f"sdxl-turbo--habc123--res-{res_hw[0]}x{res_hw[1]}"
        vae.mkdir(parents=True); unet.mkdir(parents=True)
        (vae / "vae_decoder.engine").write_bytes(b"x"); (vae / "vae_encoder.engine").write_bytes(b"x")
        (unet / "unet.engine").write_bytes(b"x")
        if with_log:
            (pathlib.Path(root) / "build_log.jsonl").write_text(
                json.dumps({"engine_dir": unet.name, "engine_file": "unet.engine", "batch_size": batch}) + chr(10))

    def test_cached_missing_and_batch_mismatch(self):
        import tempfile
        from server.engines import engine_status
        with tempfile.TemporaryDirectory() as d:
            # empty cache: everything must be built, with an estimate
            r = engine_status(d, "stabilityai/sdxl-turbo", 1024, 576, 1, True)
            self.assertFalse(r["cached"])
            self.assertEqual(set(r["missing"]), {"vae_decoder", "vae_encoder", "unet"})
            self.assertGreater(r["estimate_minutes"], 0)
            # populated for 1024x576 (library names it HxW), batch 1
            self._make(d)
            r = engine_status(d, "stabilityai/sdxl-turbo", 1024, 576, 1, True)
            self.assertTrue(r["cached"], r)
            # a different resolution is not cached
            self.assertFalse(engine_status(d, "stabilityai/sdxl-turbo", 1024, 512, 1, True)["cached"])
            # same size but a different number of steps needs another UNet engine
            r = engine_status(d, "stabilityai/sdxl-turbo", 1024, 576, 2, True)
            self.assertEqual(r["missing"], ["unet"])
            # tiny VAE off uses different VAE engines
            self.assertIn("vae_decoder", engine_status(d, "stabilityai/sdxl-turbo", 1024, 576, 1, False)["missing"])

    def test_no_engine_dir_never_raises(self):
        from server.engines import engine_status
        self.assertTrue(engine_status(None, "x/y", 512, 512)["cached"])
        self.assertTrue(engine_status("Z:/no/such/engine/dir", "x/y", 512, 512)["missing"])


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.port = free_port()
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "run_server.py"), "--fake-pipeline", "--port", str(self.port),
             "--allow-shutdown", "--token", "sekret"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    break
            except OSError:
                await asyncio.sleep(0.1)
        else:
            self.fail("server did not start")

    async def asyncTearDown(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    async def test_auth_and_bad_messages(self):
        # wrong token is rejected
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="nope")
        r = await c.connect()
        self.assertEqual(r["type"], "error")
        self.assertEqual(r["code"], "bad_token")
        # no hello -> not_authenticated; malformed JSON -> error, server survives
        ws = await websockets.connect(f"ws://127.0.0.1:{self.port}")
        await ws.send(json.dumps({"type": "start"}))
        self.assertEqual(json.loads(await ws.recv())["code"], "not_authenticated")
        await ws.send("this is not json")
        self.assertEqual(json.loads(await ws.recv())["code"], "bad_message")
        await ws.close()
        ok = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        ack = await ok.connect()
        self.assertEqual(ack["type"], "hello_ack")
        u = await ok.request("definitely_not_a_message")
        self.assertEqual(u["code"], "unknown_type")
        await ok.close()

    async def test_full_cycle_and_reconnect(self):
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        await c.connect()
        v = await c.configure_video(64, 48)
        self.assertEqual((v["width"], v["height"]), (64, 48))
        started = await c.request("start")  # model not loaded yet
        self.assertEqual(started["code"], "not_ready")
        r = await c.request("load_model", model_id="fake", width=64, height=48, bogus_key=1)
        self.assertEqual(r["type"], "ack")
        await c.request("set_params", params={"flip_y": False, "guidance_scale": 1.5, "nonsense": 1})
        await c.request("set_prompt", prompt="hello")
        self.assertEqual((await c.request("start"))["type"], "ack")
        img = np.random.randint(0, 255, (48, 64, 4), dtype=np.uint8)
        img[..., 3] = 255
        c.send_frame(img, 42)
        fr = await c.wait_result(42, timeout=5)
        self.assertIsNotNone(fr, "no output frame")
        self.assertEqual(fr.frame_id, 42)
        np.testing.assert_array_equal(fr.pixels[..., :3], 255 - img[..., :3])  # fake pipeline inverts
        self.assertTrue(np.all(fr.pixels[..., 3] == 255))
        await asyncio.sleep(0.7)
        self.assertIsNotNone(c.latest_stats)
        self.assertEqual(c.latest_stats["phase"], "running")
        warns = [e for e in c.events if e["type"] == "warning"]
        self.assertTrue(any("bogus_key" in w["message"] for w in warns))
        self.assertTrue(any("nonsense" in w["message"] for w in warns))
        # client drops; server must keep state and keep running
        await c.close()
        c2 = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        ack = await c2.connect()
        self.assertEqual(ack["state"]["phase"], "running")
        self.assertEqual(ack["state"]["prompt"], "hello")
        self.assertEqual(ack["state"]["params"]["guidance_scale"], 1.5)
        await c2.request("stop")
        await c2.close()

    async def test_flip_y_is_identity_end_to_end(self):
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        await c.connect()
        await c.configure_video(32, 24)
        await c.request("load_model", model_id="fake", width=32, height=24)
        await c.request("set_params", params={"flip_y": True})
        await c.request("start")
        img = np.zeros((24, 32, 4), np.uint8)
        img[:4, :, :3] = (255, 0, 0)  # marker in the first rows of memory
        img[..., 3] = 255
        c.send_frame(img, 1)
        fr = await c.wait_result(1, timeout=5)
        self.assertIsNotNone(fr)
        # flipped in and flipped back out: same row order as sent, colours inverted by the fake model
        np.testing.assert_array_equal(fr.pixels[..., :3], 255 - img[..., :3])
        await c.close()

    async def test_separate_control_image(self):
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        await c.connect()
        v = await c.configure_video(32, 24)
        self.assertTrue(v.get("ctl_name"))
        await c.request("load_model", model_id="fake", width=32, height=24)
        await c.request("start")
        img = np.full((24, 32, 4), 10, np.uint8); img[..., 3] = 255
        ctl = np.full((24, 32, 4), 200, np.uint8); ctl[..., 3] = 255
        # default: the input is the control image
        c.send_control(ctl, 1)
        c.send_frame(img, 1)
        fr = await c.wait_result(1, timeout=5)
        self.assertEqual(int(fr.pixels[0, 0, 0]), 245)            # 255 - input
        # separate: the control ring drives the (fake) model
        await c.request("set_params", params={"control_source": "separate"})
        c.send_control(ctl, 2)
        await asyncio.sleep(0.05)
        c.send_frame(img, 2)
        fr = await c.wait_result(2, timeout=5)
        self.assertEqual(int(fr.pixels[0, 0, 0]), 55)             # 255 - control
        await c.close()

    async def test_processed_control_preview(self):
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        await c.connect()
        v = await c.configure_video(32, 24)
        self.assertTrue(v.get("cpv_name"))
        await c.request("load_model", model_id="fake", width=32, height=24)
        await c.request("start")
        img = np.full((24, 32, 4), 40, np.uint8); img[..., 3] = 255
        # preview off: nothing is published
        for i in range(1, 7):
            c.send_frame(img, i)
            await c.wait_result(i, timeout=5)
        self.assertEqual(c.chan_cpv.write_index(), 0)
        r = await c.request("set_params", params={"control_preview": True})
        self.assertEqual(r["type"], "ack")
        for i in range(7, 20):
            c.send_frame(img, i)
            await c.wait_result(i, timeout=5)
        fr = c.chan_cpv.read_latest(0)
        self.assertIsNotNone(fr)
        self.assertEqual(int(fr.pixels[0, 0, 0]), 215)           # the fake model's "processed" control = 255 - 40
        await c.close()

    async def test_ipadapter_reference_image(self):
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        await c.connect()
        v = await c.configure_video(32, 24)
        self.assertTrue(v.get("ref_name"))
        r = await c.request("load_model", model_id="fake", width=32, height=24, use_ipadapter=True,
                            ipadapters=[{"ipadapter_model_path": "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin",
                                         "image_encoder_path": "h94/IP-Adapter/sdxl_models/image_encoder", "scale": 0.7}])
        self.assertNotEqual(r["type"], "error")
        r = await c.request("set_params", params={"ipadapter": {"scale": 0.4, "enabled": True}})
        self.assertEqual(r["type"], "ack")
        await c.request("start")
        img = np.full((24, 32, 4), 10, np.uint8); img[..., 3] = 255
        ref = np.full((24, 32, 4), 123, np.uint8); ref[..., 3] = 255
        c.send_frame(img, 1)
        fr = await c.wait_result(1, timeout=5)
        self.assertEqual(int(fr.pixels[0, 0, 0]), 245)            # no reference yet: 255 - input
        c.send_reference(ref, 1)
        await asyncio.sleep(0.05)
        c.send_frame(img, 2)
        fr = await c.wait_result(2, timeout=5)
        self.assertEqual(int(fr.pixels[0, 0, 0]), 132)            # reference wins in the test model: 255 - 123
        await c.close()

    def test_ip_adapter_feature_key(self):
        from server import engines

        k = engines.feature_key({"ipadapter": "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"})
        self.assertNotEqual(k, "default")
        self.assertEqual(k, engines.feature_key({"ipadapter": {"ipadapter_model_path": "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"}}))
        self.assertEqual(engines.feature_key({"ipadapter": None}), "default")

    async def test_blending_and_filters_are_accepted(self):
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        await c.connect()
        await c.configure_video(32, 24)
        await c.request("load_model", model_id="fake", width=32, height=24)
        r = await c.request("set_prompt", prompt="a", prompts=[{"text": "cat", "weight": 0.7}, {"text": "dog", "weight": 0.3}],
                            interpolation="slerp", normalize=True)
        self.assertEqual(r["type"], "ack")
        r = await c.request("set_prompt", prompt="a", prompts="nonsense")
        self.assertEqual(r["code"], "bad_message")
        r = await c.request("set_params", params={
            "seed_list": [{"seed": 1, "weight": 0.5}, {"seed": 2, "weight": 0.5}], "seed_interpolation": "average",
            "similar_filter": {"enabled": True, "threshold": 0.95, "max_skip": 5}, "add_noise": False})
        self.assertEqual(r["type"], "ack")
        await c.close()
        c2 = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        ack = await c2.connect()
        st = ack["state"]
        self.assertEqual([p["text"] for p in st["prompts"]], ["cat", "dog"])
        self.assertTrue(st["params"]["similar_filter"]["enabled"])
        self.assertFalse(st["params"]["add_noise"])
        self.assertEqual(len(st["params"]["seed_list"]), 2)
        warns = [e for e in c.events if e["type"] == "warning"]
        self.assertEqual(warns, [], "new params must not be reported as unknown")
        await c2.close()

    async def test_ndi_output_is_published_and_discoverable(self):
        from server.transports.ndi import ndi_capability
        if not ndi_capability()["available"]:
            self.skipTest("NDI runtime / cyndilib not available on this machine")
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        ack = await c.connect()
        self.assertTrue(ack["capabilities"]["ndi_output"]["available"])
        await c.configure_video(64, 48)
        await c.request("load_model", model_id="fake", width=64, height=48)
        name = f"SDU unit test {os.getpid()}"
        r = await c.request("set_ndi_output", enabled=True, name=name)
        self.assertEqual(r["type"], "ack", r)
        await c.request("start")
        img = np.zeros((48, 64, 4), np.uint8); img[..., 3] = 255
        for i in range(1, 25):
            c.send_frame(img, i)
            await asyncio.sleep(0.04)
        await asyncio.sleep(1.0)
        # the sender's own counters
        st = c.latest_stats
        self.assertIsNotNone(st)
        self.assertTrue(st["ndi_enabled"])
        self.assertGreater(st["ndi_frames"], 0)
        # and NDI discovery finds it, as Resolume would
        from cyndilib.finder import Finder
        finder = Finder()
        finder.open()
        found = []
        for _ in range(30):
            found = [n for n in finder.get_source_names() if name in n]
            if found:
                break
            await asyncio.sleep(0.2)
        finder.close()
        self.assertTrue(found, "NDI source was not discovered")
        # switching it off reports the new state
        r = await c.request("set_ndi_output", enabled=False)
        self.assertEqual(r["type"], "ack")
        await asyncio.sleep(0.7)
        self.assertFalse(c.latest_stats["ndi_enabled"])
        r = await c.request("set_ndi_output", enabled="yes")
        self.assertEqual(r["code"], "bad_message")
        await c.close()

    async def test_engine_status_message(self):
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        await c.connect()
        r = await c.request("engine_status", width=1024, height=576, steps=1, model_id="stabilityai/sdxl-turbo")
        self.assertEqual(r["type"], "engine_status")
        self.assertTrue(r["cached"])          # fake pipeline: no engine dir configured
        bad = await c.request("engine_status", width="wide")
        self.assertEqual(bad["code"], "bad_message")
        await c.close()

    async def test_v2v_params_and_load_keys_are_accepted(self):
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        await c.connect()
        r = await c.request("load_model", model_id="stabilityai/sdxl-turbo", width=64, height=64,
                            use_cached_attn=True, use_feature_injection=True, max_cache_maxframes=4)
        self.assertEqual(r["type"], "load_ack" if r["type"] == "load_ack" else r["type"])
        self.assertNotIn("use_cached_attn", r.get("unknown_keys", []))
        r = await c.request("set_params", params={"fi_strength": 0.5, "fi_threshold": 0.9,
                                                  "cache_interval": 2, "cache_maxframes": 3})
        self.assertEqual(r["type"], "ack")
        st = await c.request("get_state")
        self.assertAlmostEqual(st["params"]["fi_strength"], 0.5)
        await c.close()

    def test_engine_feature_key_and_manifest(self):
        import tempfile
        from server import engines

        self.assertEqual(engines.feature_key({}), "default")
        self.assertEqual(engines.feature_key({"mode": "img2img", "cfg_type": "self", "use_cached_attn": False}), "default")
        k = engines.feature_key({"use_cached_attn": True, "use_feature_injection": True, "max_cache_maxframes": 4})
        self.assertEqual(k, "img2img-self-v2v-fi-c4")
        with tempfile.TemporaryDirectory() as d:
            feats = {"use_cached_attn": True, "use_feature_injection": True, "max_cache_maxframes": 4}
            before = engines.engine_status(d, "stabilityai/sdxl-turbo", 512, 512, 1, True, features=feats)
            self.assertIn("unet", before["missing"])
            engines.record_build(d, "stabilityai/sdxl-turbo", 512, 512, 1, feats)
            self.assertIn("stabilityai/sdxl-turbo|512x512|1|" + k, engines._read_manifest(d))

    def test_lora_scan_and_feature_key(self):
        import tempfile
        from pathlib import Path
        from server import engines

        with tempfile.TemporaryDirectory() as d:
            lora = Path(d) / "LoRA" / "styles"
            lora.mkdir(parents=True)
            (lora / "pixel.safetensors").write_bytes(b"x" * 2048)
            (Path(d) / "LoRA" / "readme.txt").write_text("not a lora")
            res = engines.scan_loras(d)
            self.assertEqual([l["name"] for l in res["loras"]], ["styles/pixel.safetensors"])
            self.assertTrue(res["loras"][0]["path"].endswith("styles/pixel.safetensors"))
        self.assertEqual(engines.scan_loras(None)["loras"], [])
        k1 = engines.feature_key({"lora_dict": {"a/b.safetensors": 1.0}})
        k2 = engines.feature_key({"lora_dict": {"a/b.safetensors": 0.3}})      # weight is live: same engine
        k3 = engines.feature_key({"lora_dict": {"c.safetensors": 1.0}})
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertNotEqual(k1, "default")

    def test_model_scan_and_local_model_status(self):
        import tempfile
        from pathlib import Path
        from server import engines

        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "Model"
            (m / "diff" / "unet").mkdir(parents=True)
            (m / "diff" / "model_index.json").write_text("{}")
            (m / "diff" / "unet" / "w.safetensors").write_bytes(b"x" * 100)
            (m / "dream.safetensors").write_bytes(b"x" * 4096)
            (m / "notes.txt").write_text("x")
            names = sorted((x["name"], x["kind"]) for x in engines.scan_models(d)["models"])
            self.assertEqual(names, [("diff", "folder"), ("dream.safetensors", "file")])
        self.assertTrue(engines.is_local_model("E:/AI/models/Model/x.safetensors"))
        self.assertTrue(engines.is_local_model("E:\a\b"))
        self.assertFalse(engines.is_local_model("stabilityai/sdxl-turbo"))
        with tempfile.TemporaryDirectory() as e:
            st = engines.engine_status(e, "E:/m/x.safetensors", 512, 512, 1, True)
            self.assertFalse(st["cached"])
            engines.record_build(e, "E:/m/x.safetensors", 512, 512, 1, {})
            st = engines.engine_status(e, "E:/m/x.safetensors", 512, 512, 1, True)
            self.assertTrue(st["cached"])

    async def test_list_models(self):
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        await c.connect()
        r = await c.request("list_models")
        self.assertEqual(r["type"], "models")
        self.assertIn("models", r)
        await c.close()

    async def test_list_loras_and_lora_weights(self):
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        await c.connect()
        r = await c.request("list_loras")
        self.assertEqual(r["type"], "loras")
        self.assertIn("loras", r)
        r = await c.request("set_params", params={"lora_weights": {"a.safetensors": 0.5}})
        self.assertEqual(r["type"], "ack")
        await c.close()

    async def test_resize_and_shutdown(self):
        c = SdClient(f"ws://127.0.0.1:{self.port}", token="sekret")
        await c.connect()
        v1 = await c.configure_video(32, 32)
        v2 = await c.configure_video(48, 40)
        self.assertNotEqual(v1["in_name"], v2["in_name"])
        self.assertEqual((v2["width"], v2["height"]), (48, 40))
        bad = await c.request("configure_video", transport="shmem", width=3, height=3)
        self.assertEqual(bad["type"], "error")
        r = await c.request("shutdown")
        self.assertEqual(r["type"], "ack")
        for _ in range(50):
            if self.proc.poll() is not None:
                break
            await asyncio.sleep(0.1)
        self.assertIsNotNone(self.proc.poll(), "server did not exit after shutdown")


class IpAdapterKindTests(unittest.TestCase):
    def test_kind_from_path(self):
        from server import ipadapter as ipa

        self.assertEqual(ipa.kind_from_path("h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"), "regular")
        self.assertEqual(ipa.kind_from_path("h94/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"), "plus")
        self.assertEqual(ipa.kind_from_path("h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin"), "faceid")
        self.assertEqual(ipa.kind_from_path("E:\models\IP-Adapter-FaceID-Plus.bin"), "faceid")

    def test_normalize_rejects_incomplete(self):
        from server import ipadapter as ipa

        self.assertIsNone(ipa.normalize({"ipadapter_model_path": "x.bin"}))


if __name__ == "__main__":
    unittest.main()
