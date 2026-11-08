"""Topology specifications shared by native and ncps-backed recurrent cells.

The NCP comparison uses a degree-preserving directed edge-swap null graph.  It
keeps recurrent/sensory nodes, edge counts, motor outputs, and every node's
in/out degree fixed while allowing swaps to break NCP pathway constraints.  Both
graphs are evaluated by one common synchronous masked CfC update so that the
graph, rather than a different layered execution engine, is the intervention. The
auditable hashes returned by :func:`make_degree_matched_wiring_pair` are stored
with every checkpoint so a topology request cannot be silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TopologySpec:
    kind: str = "dense"
    sparsity: float = 0.50
    seed: int = 0

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "TopologySpec":
        spec = cls(**dict(mapping or {}))
        if spec.kind not in {"dense", "random_sparse", "autoncp"}:
            raise ValueError(f"Unsupported topology: {spec.kind}")
        if not 0.0 <= spec.sparsity < 1.0:
            raise ValueError("Topology sparsity must be in [0, 1)")
        return spec


def recurrent_mask(hidden_dim: int, spec: TopologySpec) -> torch.Tensor:
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if spec.kind == "dense":
        return torch.ones(hidden_dim, hidden_dim, dtype=torch.float32)
    if spec.kind == "autoncp":
        raise ValueError("AutoNCP is available through ncps liquid models, not native cells")
    rng = np.random.default_rng(spec.seed)
    mask = rng.random((hidden_dim, hidden_dim)) >= spec.sparsity
    np.fill_diagonal(mask, True)
    for index in range(hidden_dim):
        if not mask[index].any():
            mask[index, int(rng.integers(0, hidden_dim))] = True
        if not mask[:, index].any():
            mask[int(rng.integers(0, hidden_dim)), index] = True
    return torch.as_tensor(mask, dtype=torch.float32)


def make_ncps_wiring(hidden_dim: int, output_dim: int, spec: TopologySpec):
    try:
        from ncps import wirings
    except ImportError as exc:
        raise RuntimeError("NCP topology requires the ncps package") from exc
    if spec.kind == "autoncp":
        return wirings.AutoNCP(
            hidden_dim,
            output_dim,
            sparsity_level=spec.sparsity,
            seed=spec.seed,
        )
    if spec.kind == "random_sparse":
        return wirings.Random(
            hidden_dim,
            output_dim=output_dim,
            sparsity_level=spec.sparsity,
            random_seed=spec.seed,
        )
    raise ValueError("Dense ncps models use an integer unit count, not a wiring object")


def _matrix_sha256(matrix: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(matrix, dtype=np.int32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _degree_preserving_swaps(
    matrix: np.ndarray, *, seed: int, requested_swaps: int
) -> tuple[np.ndarray, int]:
    """Randomize a directed/bipartite signed graph without changing degrees."""
    source = np.asarray(matrix, dtype=np.int32)
    randomized = source.copy()
    rng = np.random.default_rng(int(seed))
    edges = [tuple(map(int, edge)) for edge in np.argwhere(randomized != 0)]
    successful = 0
    attempts = 0
    maximum_attempts = max(100, requested_swaps * 50)
    while successful < requested_swaps and attempts < maximum_attempts:
        if len(edges) < 2:
            break
        first, second = rng.choice(len(edges), size=2, replace=False)
        first = int(first)
        second = int(second)
        a, b = edges[first]
        c, d = edges[second]
        attempts += 1
        if a == c or b == d or randomized[a, d] != 0 or randomized[c, b] != 0:
            continue
        polarity_ab = int(randomized[a, b])
        polarity_cd = int(randomized[c, d])
        randomized[a, b] = 0
        randomized[c, d] = 0
        randomized[a, d] = polarity_ab
        randomized[c, b] = polarity_cd
        edges[first] = (a, d)
        edges[second] = (c, b)
        successful += 1
    return randomized, successful


def make_degree_matched_wiring_pair(
    *,
    hidden_dim: int,
    input_dim: int,
    output_dim: int,
    sparsity: float,
    ncp_seed: int,
    random_seed: int,
):
    """Return flattened NCP and exact-degree global-random wiring graphs.

    The random graph preserves each recurrent node's in/out degree and each
    sensory feature/node degree exactly.  Global swaps may break every NCP
    pathway constraint.  Both returned objects are one-layer ``Wiring`` graphs;
    :class:`NCPSPolicy` replaces the upstream dense recurrent mask with these
    exact adjacency matrices and slices the first ``output_dim`` motor states.
    """
    try:
        from ncps import wirings
    except ImportError as exc:
        raise RuntimeError("Degree-matched topology requires ncps") from exc
    if output_dim >= hidden_dim - 2:
        raise ValueError("AutoNCP requires output_dim < hidden_dim - 2")
    source_ncp = wirings.AutoNCP(
        int(hidden_dim), int(output_dim), sparsity_level=float(sparsity), seed=int(ncp_seed)
    )
    source_ncp.build(int(input_dim))
    recurrent_edges = int(np.count_nonzero(source_ncp.adjacency_matrix))
    sensory_edges = int(np.count_nonzero(source_ncp.sensory_adjacency_matrix))
    recurrent, recurrent_swaps = _degree_preserving_swaps(
        source_ncp.adjacency_matrix,
        seed=int(random_seed),
        requested_swaps=max(1, 10 * recurrent_edges),
    )
    sensory, sensory_swaps = _degree_preserving_swaps(
        source_ncp.sensory_adjacency_matrix,
        seed=int(random_seed) + 1,
        requested_swaps=max(1, 10 * sensory_edges),
    )
    if recurrent_swaps == 0 or sensory_swaps == 0:
        raise RuntimeError("Global degree-preserving swaps did not change both graphs")

    def flat_wiring(recurrent_matrix: np.ndarray, sensory_matrix: np.ndarray):
        wiring = wirings.Wiring(int(hidden_dim))
        wiring.set_output_dim(int(output_dim))
        wiring.set_input_dim(int(input_dim))
        wiring.adjacency_matrix = np.asarray(recurrent_matrix, dtype=np.int32).copy()
        wiring.sensory_adjacency_matrix = np.asarray(sensory_matrix, dtype=np.int32).copy()
        return wiring

    ncp_wiring = flat_wiring(
        source_ncp.adjacency_matrix, source_ncp.sensory_adjacency_matrix
    )
    random_wiring = flat_wiring(recurrent, sensory)

    ncp_in = np.count_nonzero(source_ncp.adjacency_matrix, axis=0)
    ncp_out = np.count_nonzero(source_ncp.adjacency_matrix, axis=1)
    random_in = np.count_nonzero(recurrent, axis=0)
    random_out = np.count_nonzero(recurrent, axis=1)
    ncp_sensor_out = np.count_nonzero(source_ncp.sensory_adjacency_matrix, axis=1)
    random_sensor_out = np.count_nonzero(sensory, axis=1)
    ncp_sensor_in = np.count_nonzero(source_ncp.sensory_adjacency_matrix, axis=0)
    random_sensor_in = np.count_nonzero(sensory, axis=0)
    exact_degree_match = bool(
        np.array_equal(ncp_in, random_in)
        and np.array_equal(ncp_out, random_out)
        and np.array_equal(ncp_sensor_out, random_sensor_out)
        and np.array_equal(ncp_sensor_in, random_sensor_in)
    )
    node_roles = {}
    for index in source_ncp._motor_neurons:
        node_roles[int(index)] = "motor"
    for index in source_ncp._command_neurons:
        node_roles[int(index)] = "command"
    for index in source_ncp._inter_neurons:
        node_roles[int(index)] = "inter"

    def recurrent_block_counts(matrix: np.ndarray) -> dict[str, int]:
        counts: dict[str, int] = {}
        for source_index, target_index in np.argwhere(matrix != 0):
            key = f"{node_roles[int(source_index)]}_to_{node_roles[int(target_index)]}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def sensory_destination_counts(matrix: np.ndarray) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, target_index in np.argwhere(matrix != 0):
            key = f"sensory_to_{node_roles[int(target_index)]}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    ncp_recurrent_blocks = recurrent_block_counts(source_ncp.adjacency_matrix)
    random_recurrent_blocks = recurrent_block_counts(recurrent)
    ncp_sensory_blocks = sensory_destination_counts(
        source_ncp.sensory_adjacency_matrix
    )
    random_sensory_blocks = sensory_destination_counts(sensory)
    new_pathway_blocks = sorted(
        key
        for key, count in {
            **random_recurrent_blocks,
            **random_sensory_blocks,
        }.items()
        if count > 0
        and {
            **ncp_recurrent_blocks,
            **ncp_sensory_blocks,
        }.get(key, 0)
        == 0
    )
    pathway_constraints_broken = bool(new_pathway_blocks)
    audit = {
        "method": "global_directed_double_edge_swap",
        "hidden_nodes": int(hidden_dim),
        "input_nodes": int(input_dim),
        "motor_nodes": int(output_dim),
        "recurrent_edges": recurrent_edges,
        "sensory_edges": sensory_edges,
        "recurrent_swaps": int(recurrent_swaps),
        "sensory_swaps": int(sensory_swaps),
        "exact_per_node_degree_match": exact_degree_match,
        "ncp_recurrent_sha256": _matrix_sha256(source_ncp.adjacency_matrix),
        "random_recurrent_sha256": _matrix_sha256(recurrent),
        "ncp_sensory_sha256": _matrix_sha256(source_ncp.sensory_adjacency_matrix),
        "random_sensory_sha256": _matrix_sha256(sensory),
        "graphs_differ": bool(
            not np.array_equal(source_ncp.adjacency_matrix, recurrent)
            or not np.array_equal(source_ncp.sensory_adjacency_matrix, sensory)
        ),
        "ncp_seed": int(ncp_seed),
        "random_seed": int(random_seed),
        "global_synchronous_graph_update": True,
        "ncp_pathway_constraints_preserved": not pathway_constraints_broken,
        "ncp_pathway_constraints_broken": pathway_constraints_broken,
        "new_non_ncp_pathway_blocks": new_pathway_blocks,
        "ncp_recurrent_block_counts": ncp_recurrent_blocks,
        "random_recurrent_block_counts": random_recurrent_blocks,
        "ncp_sensory_block_counts": ncp_sensory_blocks,
        "random_sensory_block_counts": random_sensory_blocks,
        "ncp_source_pathway_blocks": [
            "sensory_to_inter",
            "inter_to_command",
            "command_recurrent",
            "command_to_motor",
        ],
    }
    if (
        not exact_degree_match
        or not audit["graphs_differ"]
        or not pathway_constraints_broken
    ):
        raise RuntimeError(f"Invalid degree-matched topology pair: {audit}")
    return ncp_wiring, random_wiring, audit


def make_flat_ncp_wiring(
    *, hidden_dim: int, input_dim: int, output_dim: int, sparsity: float, seed: int
):
    """Build the flattened NCP graph used for parameter-count search."""
    try:
        from ncps import wirings
    except ImportError as exc:
        raise RuntimeError("NCP topology requires ncps") from exc
    source = wirings.AutoNCP(
        int(hidden_dim), int(output_dim), sparsity_level=float(sparsity), seed=int(seed)
    )
    source.build(int(input_dim))
    flattened = wirings.Wiring(int(hidden_dim))
    flattened.set_output_dim(int(output_dim))
    flattened.set_input_dim(int(input_dim))
    flattened.adjacency_matrix = source.adjacency_matrix.copy()
    flattened.sensory_adjacency_matrix = source.sensory_adjacency_matrix.copy()
    audit = {
        "global_synchronous_graph_update": True,
        "recurrent_edges": int(np.count_nonzero(source.adjacency_matrix)),
        "sensory_edges": int(np.count_nonzero(source.sensory_adjacency_matrix)),
    }
    return flattened, audit

