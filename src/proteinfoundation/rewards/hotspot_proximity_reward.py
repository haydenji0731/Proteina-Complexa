"""Penalty for a binder that stays far from the target hotspots.

``target_hotspot_mask`` covers the target residues only, in the same order as
the target prefix of ``chain_index``. The generated PDB restarts residue
numbers at 1 on every chain, so a hotspot is the chain letter plus its
position on that chain. One target chain is still residue ``index + 1`` on
chain A.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from proteinfoundation.rewards.base_reward import BaseRewardModel, standardize_reward
from proteinfoundation.utils.pdb_utils import PDB_CHAIN_IDS


class HotspotProximityRewardModel(BaseRewardModel):
    """Subtract the mean, over hotspots, of how far the binder misses each one.

    For each hotspot, ``d`` is the closest heavy-atom distance from the binder
    to that residue. That hotspot contributes ``-min(max_excess, max(0, d -
    cutoff))``. The model's total is the mean of those terms, so a contact
    with one hotspot does not excuse a miss of another. ``CompositeRewardModel``
    adds that total, times this model's weight, onto the overall reward.
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
        hotspot_sites = _hotspot_sites(
            kwargs.get("target_hotspot_mask"),
            kwargs.get("chain_index"),
            target_chain,
        )
        distances = _hotspot_distances(pdb_path, binder_chain, hotspot_sites)
        excess, reported = _mean_excess(distances, self.contact_cutoff, self.max_excess)
        return standardize_reward(
            reward={
                "distance": torch.tensor(reported, dtype=torch.float32),
                "excess": torch.tensor(excess, dtype=torch.float32),
                "n_hotspots": torch.tensor(float(len(hotspot_sites)), dtype=torch.float32),
            },
            total_reward=-excess,
        )


def _hotspot_sites(mask: Any, chain_index: Any, fallback_chain: str) -> list[tuple[str, int]]:
    """Hotspots as ``(chain letter, 1-based residue on that chain)``.

    The mask is the target prefix. ``chain_index`` may also contain the binder,
    which is ignored. Numbering restarts at 1 whenever the chain id changes,
    matching ``to_pdb``. Without ``chain_index``, every hotspot is
    ``index + 1`` on ``fallback_chain``.
    """
    if mask is None:
        return []
    if isinstance(mask, torch.Tensor):
        flags = mask.detach().cpu().reshape(-1).tolist()
    else:
        flags = list(mask)
    if not flags:
        return []

    chains = _target_chain_ids(chain_index, len(flags))
    if chains is None:
        return [(fallback_chain, index + 1) for index, flag in enumerate(flags) if flag]

    sites: list[tuple[str, int]] = []
    last_chain: int | None = None
    position = 0
    for flag, chain_id in zip(flags, chains, strict=False):
        if chain_id != last_chain:
            position = 1
            last_chain = chain_id
        else:
            position += 1
        if flag:
            sites.append((_chain_letter(chain_id), position))
    return sites


def _target_chain_ids(chain_index: Any, n_target: int) -> list[int] | None:
    """Chain ids for the target prefix, or None when the index is absent."""
    if chain_index is None:
        return None
    if isinstance(chain_index, torch.Tensor):
        values = chain_index.detach().cpu().reshape(-1).tolist()
    else:
        values = list(chain_index)
    if not values:
        return None
    return [int(chain_id) for chain_id in values[:n_target]]


def _chain_letter(chain_id: int) -> str:
    if 0 <= chain_id < len(PDB_CHAIN_IDS):
        return PDB_CHAIN_IDS[chain_id]
    return "?"


def _excess(distance: float, cutoff: float, cap: float) -> float:
    if not math.isfinite(distance):
        return cap
    return min(cap, max(0.0, distance - cutoff))


def _mean_excess(distances: list[float], cutoff: float, cap: float) -> tuple[float, float]:
    """Mean per-hotspot penalty, and the mean of the distances that produced it.

    No hotspots, or a hotspot with no atoms, pays the full cap. The reported
    distance stays finite: a missing hotspot is recorded as ``cutoff + cap``.
    """
    if not distances:
        return cap, cutoff + cap
    excesses = [_excess(distance, cutoff, cap) for distance in distances]
    reported = [distance if math.isfinite(distance) else cutoff + cap for distance in distances]
    return sum(excesses) / len(excesses), sum(reported) / len(reported)


def _hotspot_distances(
    pdb_path: str,
    binder_chain: str,
    hotspot_sites: list[tuple[str, int]],
) -> list[float]:
    """Closest heavy-atom distance from the binder to each hotspot, in site order."""
    if not hotspot_sites:
        return []
    atoms: dict[tuple[str, int], list[tuple[float, float, float]]] = {site: [] for site in hotspot_sites}
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
        site = (chain, residue)
        if site in atoms:
            atoms[site].append(xyz)
        elif chain == binder_chain:
            binder.append(xyz)
    distances: list[float] = []
    for site in hotspot_sites:
        hotspot = atoms[site]
        if not hotspot or not binder:
            distances.append(math.inf)
        else:
            distances.append(min(math.dist(atom, point) for atom in binder for point in hotspot))
    return distances
