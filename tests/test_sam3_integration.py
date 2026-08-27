import json
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "PSC-AVDN" / "script"
sys.path.insert(0, str(SCRIPT_DIR))

from visual_grounding.backends import RawProposal
from visual_grounding.contracts import parse_concepts, parse_views
from visual_grounding.grounding_agent import GroundingAgent
from visual_grounding.service import VisionService


class FakeVision:
    def __init__(self, candidates):
        self.candidates_value = candidates
        self.payloads = []

    def health(self):
        return {"ok": True, "prewarmed": True, "backend": "sam3"}

    def candidates(self, payload):
        self.payloads.append(payload)
        return {"backend": "sam3", "candidates": self.candidates_value}


class FakeResponse:
    def __init__(self, message):
        self.message = message

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": self.message}]}


class Sam3IntegrationTests(unittest.TestCase):
    def test_contracts_reject_invalid_requests(self):
        with self.assertRaises(ValueError):
            parse_views([{"view_id": "main", "role": "main"}])
        with self.assertRaises(ValueError):
            parse_concepts([{"concept_id": "x", "text": "roof"}, {"concept_id": "x", "text": "road"}])

    def test_qwen_selects_registered_main_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "main.jpg"
            Image.new("RGB", (768, 768), "white").save(image_path)
            candidates = [{"candidate_id": "c1:main:001", "concept_role": "target", "view_id": "main", "bbox_2d": [10, 20, 100, 120], "mask_path": ""}]
            agent = GroundingAgent(api_key="k", qwen_url="http://qwen", model="qwen", vision_tool_url="http://vision")
            agent.vision = FakeVision(candidates)
            messages = [
                {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "find_visual_candidates", "arguments": json.dumps({"concepts": [{"concept_id": "c1", "text": "roof", "role": "target"}]})}}]},
                {"role": "assistant", "tool_calls": [{"id": "2", "function": {"name": "finish_target_selection", "arguments": json.dumps({"dest_present": True, "candidate_id": "c1:main:001", "confidence": 0.9, "reason": "visible roof"})}}]},
            ]
            with patch("visual_grounding.grounding_agent.requests.post", side_effect=[FakeResponse(item) for item in messages]):
                result = agent.locate(destination="a roof", image_path=str(image_path), artifact_dir=temp)
            self.assertEqual(result["bbox_2d"], [10.0, 20.0, 100.0, 120.0])
            self.assertEqual(agent.vision.payloads[0]["views"][0]["scale"], 5)

    def test_service_materializes_sam3_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "main.jpg"
            Image.new("RGB", (32, 32), "white").save(image_path)
            args = SimpleNamespace(sam3_checkpoint="unused", device="cpu", score_threshold=0.08)
            service = VisionService(args)
            mask = __import__("numpy").zeros((32, 32), dtype=bool)
            mask[4:12, 5:15] = True
            service.backend = SimpleNamespace(loaded=True, propose=lambda image, text: [RawProposal(mask, [5, 4, 15, 12], 0.8)])
            result = service.candidates({"backend": "sam3", "views": [{"view_id": "main", "role": "main", "image_path": str(image_path), "scale": 5}], "concepts": [{"concept_id": "c1", "text": "roof", "role": "target"}], "artifact_dir": temp})
            self.assertEqual(len(result["candidates"]), 1)
            self.assertTrue(Path(result["candidates"][0]["mask_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
