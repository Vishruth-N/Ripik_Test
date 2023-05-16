"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from typing import Tuple, List, Dict, Any

MAIN_PREFIX = "MAIN"
UNKNOWN_TOKEN = "UNK"


def combine_pids(*args):
    return "__".join([str(x) for x in args])


def split_pids(pids: str, maxsplit: int = -1):
    return pids.rsplit("__", maxsplit=maxsplit)


def find_combinations_less_than_k(nums: List[float], k: float):
    def backtrack(nums, k, curr_index, curr_sum, combination, results):
        if curr_sum > k or curr_index >= len(nums):
            return

        # Consider
        combination.append((nums[curr_index], curr_index))
        backtrack(
            nums, k, curr_index, curr_sum + nums[curr_index], combination, results
        )
        combination.pop()

        # dont consider
        backtrack(nums, k, curr_index + 1, curr_sum, combination, results)

        if combination:
            results.add(frozenset((tuple(combination.copy()),)))

    results = set()
    backtrack(nums, k, 0, 0, [], results)
    return results


class FeasibleNode:
    def __init__(
        self, node_id: int, material_id: str, batch_size: float, alt_bom: str
    ) -> None:
        self.node_id = node_id
        self.material_id = material_id
        self.batch_size = batch_size
        self.alt_bom = alt_bom
        self.in_degree = 0
        self.out_degree = 0
        self.value = 0


class FeasibleGraph:
    def __init__(self) -> None:
        self.nodes = {}
        self.adj_list = {}

    def add_node(self, node_key: Tuple):
        assert len(node_key) == 3
        if node_key not in self.nodes:
            node_id = len(self.nodes)
            self.nodes[node_key] = FeasibleNode(
                node_id=node_id,
                material_id=node_key[0],
                batch_size=node_key[1],
                alt_bom=node_key[2],
            )

    def get_node(self, node_key: Tuple):
        if node_key not in self.nodes:
            return None

        return self.nodes[node_key]

    def reset_node_values(self, val: float = 0) -> None:
        for node in self.nodes.values():
            node.value = val

    def add_directed_edge(self, u: Tuple, v: Tuple):
        # Make nodes
        self.add_node(u)
        self.add_node(v)

        # Add connection
        node_uid = self.nodes[u].node_id
        node_vid = self.nodes[v].node_id
        if node_uid not in self.adj_list:
            self.adj_list[node_uid] = []
        if node_vid not in self.adj_list:
            self.adj_list[node_vid] = []

        self.adj_list[node_uid].append(self.nodes[v])

        # Update degree
        self.nodes[u].out_degree += 1
        self.nodes[v].in_degree += 1

    def bfs_sum(self):
        q = []
        visited = set()
        for node in self.nodes.values():
            if node.in_degree == 0:
                q.append(node)

        i = 0
        while i < len(q):
            node = q[i]
            if node.node_id in visited:
                i += 1
                continue

            visited.add(node.node_id)
            for neighbor_node in self.adj_list[node.node_id]:
                neighbor_node.value += node.value
                q.append(neighbor_node)

            i += 1


class PerfectBatch:
    def __init__(self, component_id: str, batch_size: float) -> None:
        self.component_id = component_id
        self.batch_size = batch_size
        self.value = 0
        self.space_consumed = 0
        self.items = []

    def get_space_left(self) -> float:
        return self.batch_size - self.space_consumed

    def get_value(self) -> float:
        return self.value

    def get_items(self) -> List[Dict[str, Any]]:
        return self.items

    def add(self, raw_requirement) -> None:
        self.items.append(raw_requirement)
        self.value += raw_requirement["priority"]
        self.space_consumed += raw_requirement["quantity"]
