"""
TEDS: Tree-Edit-Distance-based Similarity for tables (protocol §6.4).

For biology researchers: a table is turned into a tree (table -> rows ->
cells). TEDS counts the fewest edits (insert, delete, rename a node) that
turn the tool's tree into the answer key's tree, and scales it to 0-1:

    TEDS = 1 - edit_distance / max(nodes in tool tree, nodes in key tree)

Costs follow PubTabNet (Zhong et al., 2020): inserting or deleting a node
costs 1; renaming costs 1 if the node types or cell spans differ, otherwise
(two cells) the normalised edit distance between their texts. TEDS-S is the
same with cell text ignored, so it scores structure only.

The edit distance is computed with the Zhang-Shasha algorithm (1989), which
gives the same optimal distance as the APTED library used by PubTabNet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lxml import html as lxml_html
from rapidfuzz.distance import Levenshtein

from .text_utils import normalise_for_scoring


@dataclass
class Node:
    tag: str
    colspan: int = 1
    rowspan: int = 1
    text: str = ""
    children: list["Node"] = field(default_factory=list)


def teds(tool_html: str | None, key_html: str | None, structure_only: bool = False) -> float:
    """TEDS (or TEDS-S) between two HTML tables; 0 when the tool gave none."""
    if not key_html:
        raise ValueError("the key table has no cells")
    if not tool_html:
        return 0.0
    tool_tree, key_tree = parse_table(tool_html), parse_table(key_html)
    if tool_tree is None:
        return 0.0
    distance = zhang_shasha(tool_tree, key_tree, structure_only)
    return max(0.0, 1.0 - distance / max(count(tool_tree), count(key_tree)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_table(fragment: str) -> Node | None:
    """table -> tr -> td (th counts as td; thead/tbody are flattened)."""
    try:
        root = lxml_html.fragment_fromstring(fragment, create_parent="div")
    except Exception:
        return None
    tables = root.xpath(".//table")
    if not tables:
        return None
    tree = Node("table")
    for tr in tables[0].xpath(".//tr"):
        row = Node("tr")
        for cell in tr:
            if cell.tag not in ("td", "th"):
                continue
            row.children.append(Node(
                "td", int_attr(cell, "colspan"), int_attr(cell, "rowspan"),
                normalise_for_scoring(cell.text_content())))
        tree.children.append(row)
    return tree


def int_attr(element, name: str) -> int:
    try:
        return max(1, int(element.get(name) or 1))
    except ValueError:
        return 1


def count(node: Node) -> int:
    return 1 + sum(count(c) for c in node.children)


def rename_cost(a: Node, b: Node, structure_only: bool) -> float:
    if a.tag != b.tag:
        return 1.0
    if a.tag != "td":
        return 0.0
    if (a.colspan, a.rowspan) != (b.colspan, b.rowspan):
        return 1.0
    if structure_only:
        return 0.0
    if not a.text and not b.text:
        return 0.0
    return Levenshtein.normalized_distance(a.text, b.text)


def postorder(root: Node) -> tuple[list[Node], list[int]]:
    """Nodes in post-order and, for each, the index of its leftmost leaf."""
    nodes, leftmost = [], []

    def visit(node: Node) -> int:
        first_leaf = None
        for child in node.children:
            leaf = visit(child)
            if first_leaf is None:
                first_leaf = leaf
        nodes.append(node)
        index = len(nodes) - 1
        leftmost.append(first_leaf if first_leaf is not None else index)
        return leftmost[index]

    visit(root)
    return nodes, leftmost


def keyroots(leftmost: list[int]) -> list[int]:
    """Nodes with no later node sharing their leftmost leaf."""
    seen, roots = set(), []
    for index in range(len(leftmost) - 1, -1, -1):
        if leftmost[index] not in seen:
            roots.append(index)
            seen.add(leftmost[index])
    return sorted(roots)


def zhang_shasha(a: Node, b: Node, structure_only: bool) -> float:
    """Minimum edit distance between two ordered trees (unit insert/delete)."""
    nodes_a, left_a = postorder(a)
    nodes_b, left_b = postorder(b)
    tree_distance = [[0.0] * len(nodes_b) for _ in nodes_a]
    for i in keyroots(left_a):
        for j in keyroots(left_b):
            forest_distance(i, j, nodes_a, left_a, nodes_b, left_b,
                            tree_distance, structure_only)
    return tree_distance[-1][-1]


def forest_distance(i, j, nodes_a, left_a, nodes_b, left_b, tree_distance,
                    structure_only) -> None:
    """The inner Zhang-Shasha loop for one pair of keyroots."""
    li, lj = left_a[i], left_b[j]
    rows, cols = i - li + 2, j - lj + 2
    forest = [[0.0] * cols for _ in range(rows)]
    for x in range(1, rows):
        forest[x][0] = forest[x - 1][0] + 1
    for y in range(1, cols):
        forest[0][y] = forest[0][y - 1] + 1
    for x in range(1, rows):
        for y in range(1, cols):
            ia, jb = li + x - 1, lj + y - 1
            if left_a[ia] == li and left_b[jb] == lj:
                forest[x][y] = min(
                    forest[x - 1][y] + 1, forest[x][y - 1] + 1,
                    forest[x - 1][y - 1] + rename_cost(nodes_a[ia], nodes_b[jb],
                                                       structure_only))
                tree_distance[ia][jb] = forest[x][y]
            else:
                px, py = left_a[ia] - li, left_b[jb] - lj
                forest[x][y] = min(forest[x - 1][y] + 1, forest[x][y - 1] + 1,
                                   forest[px][py] + tree_distance[ia][jb])
