import json
import logging
import os
from pathlib import Path
from typing import Any
import numpy as np
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class MotionXDataset(Dataset):
    def __init__(
        self, hdf5_file: str | None = None, root_dir: str = "data/MotionX"
    ) -> None:
        super().__init__()

        self.motions = []

        if not hdf5_file:
            self.root_dir = Path(root_dir)
            self.motions = self._find_motions()
            log.debug(f"Found {len(self.motions)} motion files")

    def _find_motions(self):
        """
        Find all motion files in `self.root_dir`.

        :return: List of (category, subset, motion_name, filename).
        """

        motions = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(".npy"):
                    motion_file = Path(os.path.join(dirpath, filename))
                    motion_name = motion_file.name.replace(".npy", "")
                    subset = motion_file.parent.name
                    category = motion_file.parent.parent.name
                    motions.append((category, subset, motion_name, str(motion_file)))
        return motions

    def _parse_text_annotation(
        self, txt_root_dir: str, category: str, subset: str, motion_name: str
    ):
        filename = Path(txt_root_dir) / category / subset / f"{motion_name}.txt"
        with open(filename, "r") as f:
            result = f.read()
            return result.encode("utf-8")

    def _parse_json_annotation(
        self, json_root_dir: str, category: str, subset: str, motion_name: str
    ):
        filename = Path(json_root_dir) / category / subset / f"{motion_name}.json"
        with open(filename, "r") as f:
            result = json.load(f)
            frames = sorted(result.keys(), key=lambda x: int(x))
            return [result[k].encode("utf-8") for k in frames]

    def __len__(self) -> int:
        return len(self.motions)

    def __getitem__(self, index) -> Any:
        category, subset, motion_name, filename = self.motions[index]

        data = np.load(filename)

        face_text = self._parse_text_annotation(
            self.root_dir / "face_texts", category, subset, motion_name
        )
        semantic_label = self._parse_text_annotation(
            self.root_dir / "motionx_seq_text_v1.1", category, subset, motion_name
        )
        body_texts = self._parse_json_annotation(
            self.root_dir / "texts/body_texts", category, subset, motion_name
        )
        hand_texts = self._parse_json_annotation(
            self.root_dir / "texts/hand_texts", category, subset, motion_name
        )

        return {
            "motion_name": motion_name,
            "category": category,
            "subset": subset,
            "num_frames": data.shape[0],
            "semantic_label": semantic_label,
            "face_text": face_text,
            "body_texts": len(body_texts),
            "hand_texts": len(hand_texts),
            "smplx_322": data.shape,
        }
