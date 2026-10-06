"""Penalty for a binder that stays far from the target hotspots.

The search passes ``target_hotspot_mask`` in target-chain order. Residue 1
on that chain is mask index 0. The generated PDB renumbers each chain from
1, so the mask does not need a PDB accession number.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from proteinfoundation.rewards.base_reward import BaseRewardModel, standardize_reward


class HotspotProximityRewardModel(BaseRewardModel):
    """Subtract the angstroms by which a binder misses the nearest hotspot.

    ``d`` is the closest heavy-atom distance from any binder atom to any
    hotspot atom. The model's total is ``-max(0, d - cutoff)``, capped at
    ``max_excess``. A binder already within ``cutoff`` of some hotspot adds
    zero. ``CompositeRewardModel`` adds that total, times this model's
    weight, onto the overall reward.
    """

    IS_FOLDING_MODEL = False
    SUPPORTS_GRAD = False
    SUPPORTS_SAVE_PDB = False

    def __init__(
        self,
        contact_cutoff: float = 4.0,
        max_excess: float = 20.0,
        target_chain: str = "A",
        binder_chain: str = "B",
    ) -> None:
        super().__init__()
        self.contact_cutoff = float(contact_cutoff)
        self.max_excess = float(max_excess)
        self.target_chain = target_chain
        self.binder_chain = binder_chain

    def score(self, pdb_path: str, requires_grad: bool = False, **kwargs: Any) -> dict[str, Any]:
        self._check_capabilities(requires_grad=requires_grad)
        target_chain = kwargs.get("target_chain") or self.target_chain
        binder_chain = kwargs.get("binder_chain") or self.binder_chain
        hotspot_residues = _hotspot_residues(kwargs.get("target_hotspot_mask"))
        distance = _nearest_distance(pdb_path, target_chain, binder_chain, hotspot_residues)
        excess = _excess(distance, self.contact_cutoff, self.max_excess)
        reported = distance if math.isfinite(distance) else self.contact_cutoff + self.max_excess
        return standardize_reward(
            reward={
                "distance": torch.tensor(reported, dtype=torch.float32),
                "excess": torch.tensor(excess, dtype=torch.float32),
                "n_hotspots": torch.tensor(float(len(hotspot_residues)), dtype=torch.float32),
            },
            total_reward=-excess,
        )


def _hotspot_residues(mask: Any) -> list[int]:
    """1-based target-chain residue numbers marked by the mask."""
    if mask is None:
        return []
    if isinstance(mask, torch.Tensor):
        flags = mask.detach().cpu().reshape(-1).tolist()
    else:
        flags = list(mask)
    return [index + 1 for index, flag in enumerate(flags) if flag]


def _excess(distance: float, cutoff: float, cap: float) -> float:
    if not math.isfinite(distance):
        return cap
    return min(cap, max(0.0, distance - cutoff))


def _nearest_distance(
    pdb_path: str,
    target_chain: str,
    binder_chain: str,
    hotspot_residues: list[int],
) -> float:
    if not hotspot_residues:
        return math.inf
    wanted = set(hotspot_residues)
    hotspots: list[tuple[float, float, float]] = []
    binder: list[tuple[float, float, float]] = []
    for line in open(pdb_path):
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        element = (line[76:78].strip() or name[:1]).upper()
        if element == "H" or name.startswith("H"):
            continue
        chain = line[21]
        residue = int(line[22:26])
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if chain == target_chain and residue in wanted:
            hotspots.append(xyz)
        elif chain == binder_chain:
            binder.append(xyz)
    if not hotspots or not binder:
        return math.inf
    return min(math.dist(atom, hotspot) for atom in binder for hotspot in hotspots)
