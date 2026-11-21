"""
The benchmark metrics, one function per protocol section (§6.1-§6.7).

For biology researchers: every function takes the answer key and one tool's
output (both in the common format) and returns a small dict of numbers. A
value of None means "not supported" (the tool does not produce that field)
or "not applicable" (the key has nothing to score, e.g. a paper without
tables). None is never averaged as zero.
"""
from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz
from scipy.stats import kendalltau

from .align import Span, Stream, align_paragraphs, build_stream
from .teds import teds
from .text_utils import (clean_doi, match_similarity, normalise_for_matching,
                         similarity, surname_key)

COVERAGE_THRESHOLD = 0.8      # §6.2
TITLE_THRESHOLD = 0.9         # §6.1 / §6.5
RAW_THRESHOLD = 0.8           # §6.5
PLACEMENT_SLACK = 40          # characters either side of a span (§6.6)


# --- §6.1 metadata ------------------------------------------------------------

def score_metadata(key: dict, tool: dict) -> dict:
    k, t = key["metadata"], tool["metadata"]
    result = {}
    if t["title"] is None:
        result.update(title_exact=None, title_similarity=None)
    else:
        result["title_exact"] = float(normalise_for_matching(t["title"]) ==
                                      normalise_for_matching(k["title"]))
        result["title_similarity"] = similarity(t["title"], k["title"])
    if t["authors"] is None:
        result.update(authors_f1=None, authors_order=None)
    else:
        key_names = [surname_key(a["surname"]) for a in k["authors"]]
        tool_names = [surname_key(a["surname"]) for a in t["authors"]]
        result["authors_f1"] = multiset_f1(tool_names, key_names)
        result["authors_order"] = float(tool_names == key_names)
    # A tool "produces metadata" when it returns an author list at all (Prajnya,
    # GROBID); for such a tool a missing DOI, year or journal scores 0.
    produces_metadata = t["authors"] is not None
    result["doi"] = (float(clean_doi(t["doi"]) == k["doi"])
                     if produces_metadata and k["doi"] else None)
    result["year"] = float(t["year"] == k["year"]) if produces_metadata else None
    if not produces_metadata:
        result["journal"] = None
    else:
        names = {normalise_for_matching(j) for j in k.get("journal_alternates", [])}
        names.add(normalise_for_matching(k["journal"]))
        result["journal"] = float(normalise_for_matching(t["journal"]) in names)
    return result


def multiset_f1(predicted: list, expected: list) -> float:
    if not predicted and not expected:
        return 1.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


# --- §6.2 abstract and body -----------------------------------------------------

def score_abstract(key: dict, tool: dict) -> dict:
    if tool["abstract"] is None or not key["abstract"]:
        return {"abstract_similarity": None}
    return {"abstract_similarity": similarity(tool["abstract"], key["abstract"])}


def score_body(key: dict, tool: dict) -> tuple[dict, list[Span] | None, Stream | None]:
    """Coverage, similarity and reading order; also returns the alignment,
    which the link metrics reuse."""
    if tool["body_paragraphs"] is None:
        return {"body_coverage": None, "body_similarity": None,
                "reading_order_tau": None}, None, None
    stream = build_stream(tool["body_paragraphs"])
    spans = align_paragraphs(key["body_paragraphs"], stream)
    matched = [s for s in spans if s.score >= COVERAGE_THRESHOLD]
    coverage = len(matched) / len(spans) if spans else None
    tau = None
    if len(matched) >= 3:
        tau = kendalltau([s.key_index for s in matched], [s.start for s in matched]).statistic
    body_sim = similarity("\n".join(tool["body_paragraphs"]), "\n".join(key["body_paragraphs"]))
    return {"body_coverage": coverage, "body_similarity": body_sim,
            "reading_order_tau": tau}, spans, stream


# --- §6.3 figures ---------------------------------------------------------------

def score_figures(key: dict, tool: dict) -> dict:
    empty = {"figures_recall": None, "figures_precision": None,
             "figure_caption_similarity": None, "figures_kept_together": None,
             "figures_recall_with_caption_match": None,
             "figures_kept_together_with_caption_match": None}
    if tool["figures"] is None or not key["figures"]:
        return empty
    key_figs = {f["label_number"]: f for f in key["figures"]}
    tool_figs = {f["label_number"]: f for f in tool["figures"]}
    found = [n for n in key_figs if n in tool_figs]
    captions = [similarity(strip_label(tool_figs[n]["caption"]),
                           strip_label(key_figs[n]["caption"])) for n in found]
    kept = sum(1 for n in found if tool_figs[n]["object_count"] == 1)
    by_caption = caption_matches([key_figs[n] for n in key_figs if n not in tool_figs],
                                 tool.get("unlabelled_figure_objects") or [])
    return {"figures_recall": len(found) / len(key_figs),
            "figures_precision": (len(found) / len(tool_figs)) if tool_figs else None,
            "figure_caption_similarity": mean(captions),
            "figures_kept_together": kept / len(key_figs),
            # amendment A5: also credit a figure found with its caption but
            # without its printed label (reported beside, never instead)
            "figures_recall_with_caption_match": (len(found) + by_caption) / len(key_figs),
            "figures_kept_together_with_caption_match": (kept + by_caption) / len(key_figs)}


def caption_matches(missing_key_figures: list[dict], unlabelled_captions) -> int:
    """How many key figures not found by label are matched, one to one, by an
    unlabelled tool figure whose caption is >= 0.8 similar (A5)."""
    if isinstance(unlabelled_captions, int) or not missing_key_figures:
        return 0
    candidates = []
    for k, figure in enumerate(missing_key_figures):
        key_caption = strip_label(figure["caption"])
        for t, caption in enumerate(unlabelled_captions):
            if caption:
                score = similarity(strip_label(caption), key_caption)
                if score >= COVERAGE_THRESHOLD:
                    candidates.append((score, k, t))
    used_key, used_tool = set(), set()
    for _score, k, t in sorted(candidates, reverse=True):
        if k not in used_key and t not in used_tool:
            used_key.add(k)
            used_tool.add(t)
    return len(used_key)


# --- §6.4 tables ----------------------------------------------------------------

def score_tables(key: dict, tool: dict) -> dict:
    empty = {"tables_recall": None, "tables_precision": None, "teds": None,
             "teds_structure": None, "tables_image_only_in_key": None,
             "tables_unlabelled": None}
    if not key["tables"]:
        return empty
    if tool["tables"] is None:
        return {**empty, "tables_image_only_in_key": sum(t["image_only"] for t in key["tables"])}
    key_tabs = {t["label_number"]: t for t in key["tables"]}
    tool_tabs = {}
    for table in tool["tables"]:
        if table["label_number"] and table["label_number"] not in tool_tabs:
            tool_tabs[table["label_number"]] = table
    found = [n for n in key_tabs if n in tool_tabs]
    scored = [n for n in found if key_tabs[n]["html"]]
    return {"tables_recall": len(found) / len(key_tabs),
            # over LABELLED tool tables, as for figures; unlabelled ones are
            # counted separately (they may be real appendix tables)
            "tables_precision": len(found) / len(tool_tabs) if tool_tabs else None,
            "tables_unlabelled": sum(1 for t in tool["tables"] if not t["label_number"]),
            "teds": mean([teds(tool_tabs[n]["html"], key_tabs[n]["html"]) for n in scored]),
            "teds_structure": mean([teds(tool_tabs[n]["html"], key_tabs[n]["html"], True)
                                    for n in scored]),
            "tables_image_only_in_key": sum(t["image_only"] for t in key["tables"])}


# --- §6.5 references --------------------------------------------------------------

def match_references(key_refs: list[dict], tool_refs: list[dict],
                     word_order_free: bool = True) -> dict[str, str]:
    """tool reference id -> key reference id; one to one, best match first:
    by DOI, then title >= 0.9 and same year, then raw text >= 0.8.

    Amendment A6: the raw-text rule compares word sets (rapidfuzz
    token_set_ratio), because the key's raw text for structured JATS
    citations is rebuilt from fields in a different order from the print.
    word_order_free=False gives the frozen rule (plain similarity)."""
    pairs, used_key, used_tool = {}, set(), set()
    key_by_doi = {r["doi"]: r for r in key_refs if r["doi"]}
    for t in tool_refs:
        k = key_by_doi.get(clean_doi(t.get("doi")))
        if k and k["id"] not in used_key:
            pairs[t["id"]] = k["id"]; used_key.add(k["id"]); used_tool.add(t["id"])
    candidates = []
    for t in tool_refs:
        if t["id"] in used_tool:
            continue
        for k in key_refs:
            if k["id"] in used_key:
                continue
            if t.get("title") and k.get("title") and t.get("year") == k.get("year"):
                score = match_similarity(t["title"], k["title"])
                if score >= TITLE_THRESHOLD:
                    candidates.append((2 + score, t["id"], k["id"]))
                    continue
            if t.get("raw") and k.get("raw"):
                score = (fuzz.token_set_ratio(normalise_for_matching(t["raw"]),
                                              normalise_for_matching(k["raw"])) / 100
                         if word_order_free else match_similarity(t["raw"], k["raw"]))
                if score >= RAW_THRESHOLD:
                    candidates.append((score, t["id"], k["id"]))
    for _score, tid, kid in sorted(candidates, reverse=True):
        if tid not in used_tool and kid not in used_key:
            pairs[tid] = kid; used_tool.add(tid); used_key.add(kid)
    return pairs


def score_references(key: dict, tool: dict) -> tuple[dict, dict[str, str]]:
    empty = {"references_recall": None, "references_precision": None,
             "references_recall_frozen_rule": None, "ref_title": None, "ref_year": None, "ref_doi": None, "ref_first_author": None}
    if tool["references"] is None or not key["references"]:
        return empty, {}
    pairs = match_references(key["references"], tool["references"])
    frozen = match_references(key["references"], tool["references"], word_order_free=False)
    key_by_id = {r["id"]: r for r in key["references"]}
    result = {"references_recall": len(pairs) / len(key["references"]),
              "references_precision": len(pairs) / len(tool["references"])
              if tool["references"] else None,
              "references_recall_frozen_rule": len(frozen) / len(key["references"])}
    parsed = [t for t in tool["references"] if t.get("parsed", True)]
    if not parsed:
        return {**empty, **result}, pairs
    fields = {"ref_title": [], "ref_year": [], "ref_doi": [], "ref_first_author": []}
    for t in parsed:
        if t["id"] not in pairs:
            continue
        k = key_by_id[pairs[t["id"]]]
        if k.get("title"):
            fields["ref_title"].append(float(match_similarity(t.get("title"), k["title"])
                                             >= TITLE_THRESHOLD))
        if k.get("year"):
            fields["ref_year"].append(float(t.get("year") == k["year"]))
        if k.get("doi"):
            fields["ref_doi"].append(float(clean_doi(t.get("doi")) == k["doi"]))
        if k.get("first_author_surname"):
            fields["ref_first_author"].append(float(
                surname_key(t.get("first_author_surname")) == surname_key(k["first_author_surname"])))
    return {**result, **{name: mean(values) for name, values in fields.items()}}, pairs


# --- §6.6 / §6.7 in-text links ------------------------------------------------------

def score_links(key: dict, tool: dict, spans: list[Span] | None, stream: Stream | None,
                kind: str, id_map: dict[str, str] | None = None) -> dict:
    """Precision and recall of in-text links, placed by position.

    `kind` is "citations" (§6.6), "figures" (§6.7) or "tables" (amendment A7,
    scored exactly like figures)."""
    field, key_field, prefix = LINK_KINDS[kind]
    if tool.get(field) is None or spans is None:
        return {f"{prefix}_precision": None, f"{prefix}_recall": None}
    key_targets = key_link_targets(key, key_field, kind)
    total_key = sum(len(v) for v in key_targets.values())
    if total_key == 0:
        return {f"{prefix}_precision": None, f"{prefix}_recall": None}
    placed: dict[int, list[str]] = {}
    total_tool = 0
    for link in tool[field]:
        targets = [(id_map or {}).get(t, None) if kind == "citations" else t
                   for t in link[key_field]]
        total_tool += len(targets)
        position = stream.position_of(link["paragraph_index"], link["offset"])
        k = place(position, spans)
        if k is not None:
            placed.setdefault(k, []).extend(t for t in targets if t is not None)
    correct = sum(sum((Counter(placed.get(k, [])) & Counter(v)).values())
                  for k, v in key_targets.items())
    return {f"{prefix}_precision": correct / total_tool if total_tool else None,
            f"{prefix}_recall": correct / total_key}


# kind -> (common-format field, target field, metric prefix)
LINK_KINDS = {"citations": ("citation_links", "reference_ids", "citation"),
              "figures": ("figure_links", "figure_ids", "figure_mention"),
              "tables": ("table_links", "table_ids", "table_mention")}


def key_link_targets(key: dict, key_field: str, kind: str) -> dict[int, list[str]]:
    """Key paragraph index -> its link targets (figure and table ids become
    numbers)."""
    objects = key.get("tables") if kind == "tables" else key.get("figures")
    label_of = {f["id"]: f["label_number"] for f in objects or []}
    targets = {}
    for link in key.get(LINK_KINDS[kind][0]) or []:
        ids = link[key_field]
        if kind != "citations":
            ids = [label_of[i] for i in ids if i in label_of]
        if ids:
            targets[link["paragraph_index"]] = ids
    return targets


def place(position: int | None, spans: list[Span]) -> int | None:
    """The key paragraph whose aligned span holds this position."""
    if position is None:
        return None
    best, best_gap = None, PLACEMENT_SLACK + 1
    for span in spans:
        if span.start < 0 or span.score < 0.5:
            continue
        if span.start <= position <= span.end:
            return span.key_index
        gap = min(abs(position - span.start), abs(position - span.end))
        if gap < best_gap:
            best, best_gap = span.key_index, gap
    return best


LEADING_LABEL = re.compile(r"^\s*(?:\*\*)?\s*(?:fig(?:ure)?s?|table)\.?\s*\d+[a-z]?\s*[.:|\-–—]?\s*",
                           re.IGNORECASE)


def strip_label(caption: str | None) -> str:
    """Caption text without a leading 'Figure 3 |' / 'Fig. 3.' (the label is
    scored on its own; the key's caption never contains it)."""
    return LEADING_LABEL.sub("", caption or "", count=1)


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None
