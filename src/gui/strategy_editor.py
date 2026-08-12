#!/usr/bin/env python3
"""
Strategy Lab's backend — an editor over the REAL config files, not a second
source of truth. S12.

For biology researchers: this is where "which AI model reads which kind of
page" lives. `src/` never names a model directly (see CLAUDE.md's
VLM-agnostic rule) — it asks for a ROLE ("who reads the body pages?") and
one of three files answers. A model swap touches all three, or it is
incomplete:

1. **`config/vlm_profiles/model.ini`** — the llama-server PRESET: where the
   model's weight files live on disk, how much of it goes on the GPU
   (`n-gpu-layers`), context size, sampling. One `[section]` per model.
2. **`config/vlm_profiles/<model>.yaml`** — the PIPELINE PROFILE: whether
   the model follows instructions or only answers fixed prompts
   (`prompt_mode`), how much VRAM it needs, what it can and cannot do.
3. **`profile_registry`** inside `config/research_article_strategy.yaml` —
   the ROLE ASSIGNMENT: which profile (file #2) answers for `extraction`,
   `title_detection`, `layout_detector`, `figure_escalation`,
   `understanding`, `default`.

**Why the registry is edited surgically instead of `yaml.safe_dump`-ed
whole.** `research_article_strategy.yaml` carries load-bearing prose
comments elsewhere in the file (the offline-roster explanation, the
figure_escalation cost/benefit note) that a full parse-and-redump would
silently discard. `profile_registry`'s own comments matter just as much.
So `write_profile_registry_roles` finds the `profile_registry:` block by
text and rewrites only the `role: value` lines inside it, leaving every
byte outside that block — and every comment inside it that isn't attached
to a role being CHANGED — untouched. A role being ADDED gets no comment (a
human can add one later); a role's value being changed keeps its existing
inline comment, since the reasoning that inline comment states usually
still applies to the new value too, or should be reviewed by a human either
way, not silently dropped by a diff engine.

**A single VLM profile YAML (`<model>.yaml`) is NOT edited surgically.**
Those files are shorter, and the plan (`docs/gui_plan.md` §4) explicitly
calls for `yaml.safe_load` / `yaml.safe_dump` with a diff preview — the
diff IS the safety net for comment loss on a save, not a reason to avoid
one. `save_vlm_profile` always returns the diff text so a caller can show
it before committing.

**Every write backs up the file it is about to touch first**
(`<name>.bak-<timestamp>`), the same discipline sessions 8-11 used for the
ledger CSV and document JSON before any live test — this config tree is
what the CLI depends on for every run, so a corrupted copy here breaks more
than the GUI.
"""

from __future__ import annotations

import configparser
import difflib
import io
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.processing.paths import CONFIG_ROOT

MODEL_INI_PATH = CONFIG_ROOT / "vlm_profiles" / "model.ini"
VLM_PROFILES_DIR = CONFIG_ROOT / "vlm_profiles"
STRATEGY_YAML_PATH = CONFIG_ROOT / "research_article_strategy.yaml"

#: A frozen, committed-to-git copy of the config tree, mirroring the layout
#: above one level down. Seeded once (2026-08-11, the S12-gate-verified
#: state: glm-ocr/olmocr-2-7b/deepseek-ocr-v2, zero validator warnings) —
#: see the "Factory defaults" section below for why it is a snapshot, not
#: something recomputed. This is what "Restore to Default" reverts to; it
#: is never itself written to by this module.
FACTORY_DEFAULTS_ROOT = CONFIG_ROOT / "factory_defaults"
FACTORY_MODEL_INI_PATH = FACTORY_DEFAULTS_ROOT / "model.ini"
FACTORY_VLM_PROFILES_DIR = FACTORY_DEFAULTS_ROOT / "vlm_profiles"
FACTORY_STRATEGY_YAML_PATH = FACTORY_DEFAULTS_ROOT / "research_article_strategy.yaml"

#: The six roles `src/` resolves via `@profile_registry.<role>` — see
#: CLAUDE.md's "Modularity contracts" and `config/research_article_strategy.yaml`.
ROLES = [
    "default",
    "title_detection",
    "extraction",
    "layout_detector",
    "figure_escalation",
    "understanding",
]

#: What each role is FOR, shown next to its selector in the UI.
ROLE_DESCRIPTIONS = {
    "default": "Title-page two-shot reading (raw OCR, then structured metadata).",
    "title_detection": "Classifies a candidate page as a title page or not.",
    "extraction": "Reads body pages when the offline GLM two-phase path is enabled. "
                  "Removing this role disables that whole path (a configuration choice, "
                  "not a crash — body_pipeline.py checks for its presence).",
    "layout_detector": "Locates figure/table boxes. 'onnx' selects the CPU detector "
                        "(PP-DocLayoutV3, default); any other value names a VLM grounding profile.",
    "figure_escalation": "A second opinion, only on pages where the cheap pass "
                          "contradicts itself. Optional — deleting this role turns the "
                          "feature off cleanly, it is not a required role.",
    "understanding": "Figure description / structuring. Reserved: no caller in src/ "
                      "today (session 11's deliberate design — see CLAUDE.md). Assigning "
                      "it does nothing yet.",
}

#: Roles that MAY be absent from the registry without the pipeline refusing
#: to start. Everything else (default, title_detection, extraction,
#: layout_detector) has at least one real caller that fails without it.
OPTIONAL_ROLES = {"figure_escalation", "understanding"}

#: For each role, the task names (`config/task_definitions/<name>.yaml`) its
#: assigned model is actually called with — derived by reading the real call
#: sites, not guessed:
#:   default            -> vlm_provider.title_page shot1/shot2 (strategy.py:182-185)
#:   title_detection     -> get_title_detection_config (strategy.py:134-142)
#:   extraction          -> body_pipeline.py:159-160 (raw_ocr is load-bearing;
#:                          table_description falls back to raw_ocr's prompt
#:                          if absent, so it is not listed as required here)
#:   layout_detector      -> grounding, not a fixed_prompts task; not checked
#:   figure_escalation    -> grounding, not a fixed_prompts task; not checked
#:   understanding        -> unwired (see ROLE_DESCRIPTIONS); not checked
ROLE_REQUIRED_TASKS: Dict[str, List[str]] = {
    "default": ["raw_ocr_extraction", "structured_analysis_title_page"],
    "title_detection": ["title_page_detection"],
    "extraction": ["raw_ocr_extraction"],
    "layout_detector": [],
    "figure_escalation": [],
    "understanding": [],
}


# =============================================================================
# model.ini — the server preset layer
# =============================================================================

def read_model_ini(path: Path = MODEL_INI_PATH) -> configparser.ConfigParser:
    """Parse model.ini. Keys are preserved case-sensitively (llama-server flags).

    `strict=False`: the real file has at least one duplicate key (a `mlock`
    listed twice in one preset) that llama-server itself tolerates. The
    default strict parser refuses to even READ the file over that — too
    fragile for a viewer that must open the file as it actually is, warts
    included, rather than crash before showing anything. The later value
    wins, same as llama-server's own last-flag-wins behaviour.
    """
    parser = configparser.ConfigParser(strict=False)
    parser.optionxform = str  # default lower-cases keys; flags like n-gpu-layers must survive as-is
    parser.read(path, encoding="utf-8")
    return parser


def model_ini_sections(parser: configparser.ConfigParser) -> List[str]:
    """Every `[section]` name — each one a model llama-server can load."""
    return list(parser.sections())


def model_ini_section_dict(parser: configparser.ConfigParser, section: str) -> Dict[str, str]:
    """One section's key/value pairs, in file order."""
    return dict(parser.items(section))


def preview_text_diff(before: str, after: str, label: str) -> str:
    """A unified diff between two raw file contents — the generic version
    of `preview_vlm_profile_diff` / `preview_profile_registry_diff`, used
    wherever the caller already has both texts in hand (the raw-text
    editors for model.ini and a VLM profile)."""
    return _unified_diff(before, after, label)


def read_model_ini_text(path: Path = MODEL_INI_PATH) -> str:
    """The whole file, raw — the primary read path for the UI's text editor."""
    return path.read_text(encoding="utf-8")


def save_model_ini_text(text: str, path: Path = MODEL_INI_PATH) -> Path:
    """
    Write model.ini from raw TEXT, verbatim — mirrors `save_vlm_profile_text`
    for the same reason: writing what the user typed, unparsed and
    unrebuilt, is the only way an INI comment survives a save (configparser's
    own writer has no comment model to preserve one through a round-trip).

    Returns:
        The backup file's path.

    Raises:
        configparser.Error: `text` does not parse as valid INI — refused
            before anything is written.
    """
    parser = configparser.ConfigParser(strict=False)
    parser.optionxform = str
    parser.read_string(text)  # validates only; raises on malformed INI
    backup_path = _backup(path)
    _write_text(path, text)
    return backup_path


def write_model_ini_section(parser: configparser.ConfigParser, section: str,
                            values: Dict[str, str], path: Path = MODEL_INI_PATH) -> Path:
    """
    Replace one section's values and write the file back.

    Backs up first. `configparser`'s writer does not preserve comments (INI
    has no standard comment-association model to preserve), so — like the
    YAML profile writer below — this trades comment fidelity for a working
    round-trip; model.ini carries far fewer comments than the YAML profiles.

    Returns:
        The backup file's path.
    """
    if not parser.has_section(section):
        parser.add_section(section)
    for key in list(parser[section].keys()):
        if key not in values:
            parser.remove_option(section, key)
    for key, value in values.items():
        parser.set(section, key, str(value))

    backup_path = _backup(path)
    buffer = io.StringIO()
    parser.write(buffer)
    _write_text(path, buffer.getvalue())
    return backup_path


# =============================================================================
# vlm_profiles/<model>.yaml — the pipeline profile layer
# =============================================================================

def list_vlm_profiles(root: Path = VLM_PROFILES_DIR) -> List[str]:
    """Every profile name (no extension), sorted."""
    return sorted(p.stem for p in root.glob("*.yaml"))


def load_vlm_profile_text(name: str, root: Path = VLM_PROFILES_DIR) -> str:
    """The profile's raw text — used to build a diff before a save."""
    return (root / f"{name}.yaml").read_text(encoding="utf-8")


def load_vlm_profile(name: str, root: Path = VLM_PROFILES_DIR) -> Dict[str, Any]:
    """The profile, parsed. Empty dict if the file is empty (not expected, but not fatal)."""
    with open(root / f"{name}.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def preview_vlm_profile_diff(name: str, updated: Dict[str, Any],
                             root: Path = VLM_PROFILES_DIR) -> str:
    """
    A unified diff of what `save_vlm_profile` WOULD write, without writing it.

    This is the safety net the module docstring promises: `yaml.safe_dump`
    strips comments, so showing the diff before committing is how a user
    notices — and can still cancel.
    """
    before = load_vlm_profile_text(name, root=root) if (root / f"{name}.yaml").exists() else ""
    after = yaml.safe_dump(updated, sort_keys=False, allow_unicode=True)
    return _unified_diff(before, after, name)


def save_vlm_profile(name: str, updated: Dict[str, Any],
                     root: Path = VLM_PROFILES_DIR) -> Path:
    """
    Write a profile's full contents back as YAML. Backs up first.

    Returns:
        The backup file's path (empty-named backup if the profile is new —
        `_backup` handles a missing source file by returning the intended
        path with nothing written, so a caller can still log "no prior
        version").
    """
    path = root / f"{name}.yaml"
    backup_path = _backup(path)
    _write_text(path, yaml.safe_dump(updated, sort_keys=False, allow_unicode=True))
    return backup_path


def save_vlm_profile_text(name: str, text: str, root: Path = VLM_PROFILES_DIR) -> Path:
    """
    Write a profile from raw YAML TEXT, verbatim — the primary save path the
    UI uses. Unlike `save_vlm_profile` (which re-serializes a dict and so
    strips comments), this writes exactly what the user typed: any comment
    they did not touch survives untouched, because it was never parsed away
    and rebuilt in the first place.

    Args:
        text: the full proposed file content.

    Returns:
        The backup file's path.

    Raises:
        yaml.YAMLError: `text` does not parse as YAML — refused before
            anything is written, so a typo cannot corrupt the file the CLI
            also depends on.
    """
    yaml.safe_load(text)  # validates only; the parsed value is not used here
    path = root / f"{name}.yaml"
    backup_path = _backup(path)
    _write_text(path, text)
    return backup_path


def render_profile_yaml(data: Dict[str, Any]) -> str:
    """Dict -> starter YAML text, for pre-filling a new profile's editor
    after `duplicate_vlm_profile` (which returns a dict, not text)."""
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def duplicate_vlm_profile(source_name: str, new_name: str, model_name: str,
                          root: Path = VLM_PROFILES_DIR) -> Dict[str, Any]:
    """
    Clone an existing profile as the starting point for a model this
    codebase has never heard of — the plan's explicit answer to "the user
    must be able to point the tool at a model without editing Python."

    Does NOT write the file — returns the new profile dict for the caller
    to review (and edit further) before `save_vlm_profile`.

    Args:
        source_name: an existing profile to clone the shape of.
        new_name: the new profile's name (must not already exist).
        model_name: becomes the new profile's `model_name` field — the value
            that must later match a `[section]` in model.ini.

    Raises:
        FileNotFoundError: source profile does not exist.
        ValueError: new_name already exists.
    """
    if not (root / f"{source_name}.yaml").exists():
        raise FileNotFoundError(f"Source profile not found: {source_name}")
    if (root / f"{new_name}.yaml").exists():
        raise ValueError(f"A profile named '{new_name}' already exists.")

    cloned = load_vlm_profile(source_name, root=root)
    cloned["model_name"] = model_name
    return cloned


# =============================================================================
# profile_registry — the role assignment layer (surgical edit)
# =============================================================================

_ROLE_LINE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(\S+)?(\s*#.*)?\s*$")


def read_profile_registry(strategy_path: Path = STRATEGY_YAML_PATH) -> Dict[str, str]:
    """Every role currently assigned. Roles absent from the file are absent here too."""
    config = yaml.safe_load(strategy_path.read_text(encoding="utf-8")) or {}
    return dict(config.get("profile_registry", {}))


def write_profile_registry_roles(new_roles: Dict[str, Optional[str]],
                                 strategy_path: Path = STRATEGY_YAML_PATH) -> Path:
    """
    Apply role changes to the `profile_registry:` block in place, leaving
    every other line of the file — including every comment — untouched.

    Args:
        new_roles: role -> new model name. A value of None DELETES that
            role's line (the documented way to disable an optional role,
            e.g. `figure_escalation`). A role not mentioned in this dict is
            left exactly as it was.

    Returns:
        The backup file's path.

    Raises:
        ValueError: `profile_registry:` was not found in the file (the
            surgical approach depends on it existing — this should never
            happen against the shipped config, and a caller seeing this
            error should stop rather than guess where to write).
    """
    text = strategy_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    start, end = _find_registry_block(lines)
    block = lines[start:end]
    block = _apply_role_changes(block, new_roles)

    backup_path = _backup(strategy_path)
    new_text = "".join(lines[:start] + block + lines[end:])
    _write_text(strategy_path, new_text)
    return backup_path


def preview_profile_registry_diff(new_roles: Dict[str, Optional[str]],
                                  strategy_path: Path = STRATEGY_YAML_PATH) -> str:
    """A unified diff of what `write_profile_registry_roles` would change, without writing it."""
    text = strategy_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start, end = _find_registry_block(lines)
    block = _apply_role_changes(lines[start:end], new_roles)
    after = "".join(lines[:start] + block + lines[end:])
    return _unified_diff(text, after, strategy_path.name)


# =============================================================================
# Validation — three checks, each already cost a real debugging session
# =============================================================================

def validate_model_name_matches_ini(profile: Dict[str, Any],
                                    ini_sections: List[str]) -> Optional[str]:
    """
    `model_name` in a profile must name a `[section]` in model.ini, or the
    first request to that model fails at runtime with no earlier warning —
    nothing in the pipeline checks this today (`glm-ocr.yaml`'s own comment
    says so). Returns an error string, or None if it matches.

    A cloud profile (`model_type` starting `cloud_api_`, e.g. Gemini) is
    never in model.ini by design — it names a hosted API, not a local
    llama-server preset — so it is exempt from this check rather than a
    permanent false positive (four of the seven shipped profiles are cloud).
    """
    if str(profile.get("model_type", "")).startswith("cloud_api"):
        return None
    model_name = profile.get("model_name")
    if not model_name:
        return "Profile has no model_name — llama-server would not know which preset to load."
    if model_name not in ini_sections:
        return (f"model_name '{model_name}' has no matching [section] in model.ini. "
                f"Available: {', '.join(sorted(ini_sections)) or '(none)'}")
    return None


def validate_fixed_prompts_for_role(role: str, profile: Dict[str, Any]) -> List[str]:
    """
    A `prompt_mode: fixed` profile that is missing a `fixed_prompts` entry
    for a task its assigned role actually calls is a role that is silently
    broken — the provider has nothing to substitute and fails at that call,
    not at assignment time. Returns a list of missing task names (empty if fine).
    """
    if profile.get("prompt_mode") != "fixed":
        return []
    fixed_prompts = profile.get("fixed_prompts", {}) or {}
    required = ROLE_REQUIRED_TASKS.get(role, [])
    return [task for task in required if task not in fixed_prompts]


#: Roles active within the SAME processing phase — VRAM co-residency only
#: matters within a phase. Crossing phases (title page, then body pages) is
#: a deliberate, already-budgeted model SWAP
#: (`ResearchArticleStrategy._reserve_gpu_for` claims the GPU for the title
#: model before the title page runs, precisely so this swap is controlled,
#: not accidental) — not a misconfiguration, so it must not be flagged the
#: same way an unplanned same-phase clash is. Grouping derived from the real
#: call sites: `default`/`title_detection` both serve the title-page shots
#: (strategy.py `_init_title_page`); `extraction`/`figure_escalation`/
#: `understanding` all belong to body-page processing (`body_pipeline.py`).
#: `layout_detector` is excluded — 'onnx' (the default) runs on CPU with no
#: VRAM cost, and even a VLM grounding profile there is checked at whatever
#: phase actually invokes it, which this simple two-group model does not
#: yet distinguish for that role. `understanding` is ALSO excluded, for a
#: different reason: it has no caller anywhere in `src/` today (session 11's
#: deliberate design, ROLE_DESCRIPTIONS says so) — a role that never runs
#: cannot conflict with anything, and warning about it would contradict the
#: "does nothing yet" note shown right next to its selector. Add it back to
#: the body-phase group the day it gets a real caller.
ROLE_PHASE_GROUPS: List[List[str]] = [
    ["default", "title_detection"],
    ["extraction", "figure_escalation"],
]

#: **These groups are not a guess — they were MEASURED.** On 2026-07-10, on the
#: 8188 MiB card, with the llama-server router (build b9936):
#:
#:     baseline (nothing loaded)            0 MiB
#:     + glm-ocr                         2535 MiB   (+2535)
#:     + deepseek-ocr-v2                 7464 MiB   (+4929)   both "loaded"
#:     peak during interleaved inference 7940 MiB   (248 MiB headroom)
#:     olmocr-2-7b ALONE                 7907 MiB   -> nothing co-resides with it
#:
#: So the body phase (`extraction` + `figure_escalation` = glm + deepseek)
#: genuinely co-resides and the escalation path needs ZERO model swaps, while
#: the title phase's olmocr-2-7b must swap — which is exactly what
#: `_reserve_gpu_for` does deliberately every run. Comparing across the two
#: groups would flag that deliberate swap as an error. The measurement lives in
#: SLM's `default` profile (pre-V6/V4 history); do not re-derive it by trial
#: and error, as session 17 did before finding it.

#: What a sum of weights MISSES. Measured above: the two models' weights total
#: 7464 MiB, but peak VRAM during interleaved inference is 7940 MiB — vision
#: encode allocates transient buffers on top of the weights. That is 476 MiB,
#: rounded up here. Without this margin a pair summing to 7.9 GB "fits" on an
#: 8.0 GB card on paper and OOMs in practice.
VISION_TRANSIENT_OVERHEAD_GB = 0.5


def compute_vram_warnings(role_profiles: Dict[str, Dict[str, Any]],
                          card_vram_gb: Optional[float]) -> List[str]:
    """
    Flags model PAIRS, ASSIGNED WITHIN THE SAME PHASE, that will not fit on
    the card together — the trap CLAUDE.md's "VRAM residency" section
    documents: two models that do not list each other under
    `co_resident_with` still "work" (llama-swap loads on demand), just ~3x
    slower with byte-identical output, a failure invisible in results and
    visible only in the clock.

    A first version of this function compared every pair across ALL six
    roles, which flagged the title model against the body models on the
    project's own live config — a swap the pipeline already performs
    deliberately every run, not an error. See `ROLE_PHASE_GROUPS`, whose
    grouping the 2026-07-10 measurements independently confirm.

    **Two things this gets right that a plain sum of weights does not, both
    derived from those measurements rather than from reasoning:**

    1. **Transient buffers are counted** (`VISION_TRANSIENT_OVERHEAD_GB`).
       Weights are not peak: glm + deepseek weigh 7464 MiB and peak at 7940
       MiB under interleaved inference.
    2. **`co_resident_with` does not override arithmetic.** That declaration
       records a pair VERIFIED to co-reside — but it was verified on a
       specific card (8188 MiB). It is a statement about two models, not a
       promise about every GPU. On a smaller card the pair still does not
       fit, and silently trusting the declaration would hide a real problem
       on exactly the hardware most likely to hit it. Strategy Lab exists
       partly so an 8 GB card and a 64 GB card can run different rosters
       (`docs/gui_plan.md` §4), so the card is a variable here, not a
       constant. A declared pair that does not fit gets its own, clearer
       warning rather than the generic swap-thrash one.

    Args:
        role_profiles: role -> that role's loaded profile dict (only roles
            with a real, resolvable profile need be included).
        card_vram_gb: the GPU's total VRAM, or None if it could not be
            detected (in which case no warning fires — nothing to compare
            against, and a false warning is worse than a missing one).

    Returns:
        Human-readable warnings, one per unfit same-phase pair of DISTINCT models.
    """
    if not card_vram_gb:
        return []

    warnings: List[str] = []
    for group in ROLE_PHASE_GROUPS:
        by_model: Dict[str, Dict[str, Any]] = {}
        for role in group:
            profile = role_profiles.get(role)
            if not profile:
                continue
            name = profile.get("model_name")
            if name:
                by_model.setdefault(name, profile)

        names = sorted(by_model)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                left_profile, right_profile = by_model[left], by_model[right]
                left_vram = (left_profile.get("resource_requirements", {}) or {}).get("vram_per_instance_gb", 0)
                right_vram = (right_profile.get("resource_requirements", {}) or {}).get("vram_per_instance_gb", 0)
                # The declaration is only required on ONE side — glm-ocr.yaml lists
                # deepseek-ocr-v2, not the reverse — so check both directions.
                # The declaration is only required on ONE side — glm-ocr.yaml lists
                # deepseek-ocr-v2, not the reverse — so check both directions.
                left_co_resident = (left_profile.get("resource_requirements", {}) or {}).get("co_resident_with", []) or []
                right_co_resident = (right_profile.get("resource_requirements", {}) or {}).get("co_resident_with", []) or []
                declared = right in left_co_resident or left in right_co_resident

                # Weights plus the transient buffers vision encode allocates on
                # top of them — see VISION_TRANSIENT_OVERHEAD_GB for the measurement.
                needed = (left_vram or 0) + (right_vram or 0) + VISION_TRANSIENT_OVERHEAD_GB
                if needed <= card_vram_gb:
                    continue  # they fit, declared or not

                if declared:
                    warnings.append(
                        f"'{left}' ({left_vram} GB) and '{right}' ({right_vram} GB) are "
                        f"declared co-resident, but that needs ~{needed:.1f} GB "
                        f"(weights plus ~{VISION_TRANSIENT_OVERHEAD_GB} GB of vision buffers) "
                        f"and this card has {card_vram_gb:.1f} GB. The declaration was "
                        f"verified on a larger card — it is not a promise about this one. "
                        f"Expect a swap on every alternation, or pick a smaller model."
                    )
                else:
                    warnings.append(
                        f"'{left}' ({left_vram} GB) + '{right}' ({right_vram} GB) needs "
                        f"~{needed:.1f} GB with vision buffers, over the "
                        f"{card_vram_gb:.1f} GB card, and they are not declared "
                        f"co-resident. They will swap on every alternation — "
                        f"~3x slower, output unchanged (see CLAUDE.md, 'VRAM residency')."
                    )
    return warnings


# =============================================================================
# Factory defaults — the idiot-proof reset
# =============================================================================
#
# Every write in this module already backs up the file it touches, which
# undoes the LAST edit. That is not the same guarantee as "get back to a
# known-good state" — a user five edits deep, unsure which one broke
# something, has no single button that gets them out. V5 had exactly this
# (`prajnya_studio/gui/pages/strategy_lab.py`'s "Restore Factory Settings",
# backed by a pristine `research_article_strategy.FACTORY_DEFAULT.yaml`
# sitting next to the live config) — the same shape is rebuilt here rather
# than reinvented, at `config/factory_defaults/`.
#
# The snapshot is frozen, not recomputed: it was copied from the live tree
# on 2026-08-11, the exact state the S12 gate verified (zero validator
# warnings, the glm-ocr/olmocr-2-7b/deepseek-ocr-v2 roster documented
# throughout CLAUDE.md). Nothing in this module ever writes INTO
# `factory_defaults/` — updating the baseline, if it's ever needed, is a
# deliberate developer action (copy the live files over), not something a
# GUI button does, so "restore defaults" can never restore something a
# previous mistake had already baked into the baseline.

def has_factory_default_profile(name: str, factory_root: Path = FACTORY_VLM_PROFILES_DIR) -> bool:
    """Whether `name` ships a frozen baseline — false for a profile a user
    created via Duplicate, which has no "factory" version to revert to."""
    return (factory_root / f"{name}.yaml").exists()


def preview_model_ini_restore_diff(path: Path = MODEL_INI_PATH,
                                   factory_path: Path = FACTORY_MODEL_INI_PATH) -> str:
    """Diff between the live model.ini and the frozen default, without writing."""
    return preview_text_diff(read_model_ini_text(path), read_model_ini_text(factory_path),
                             "model.ini")


def restore_model_ini_default(path: Path = MODEL_INI_PATH,
                              factory_path: Path = FACTORY_MODEL_INI_PATH) -> Path:
    """Overwrites the live model.ini with the frozen default. Backs up first."""
    return save_model_ini_text(read_model_ini_text(factory_path), path=path)


def preview_vlm_profile_restore_diff(name: str, root: Path = VLM_PROFILES_DIR,
                                     factory_root: Path = FACTORY_VLM_PROFILES_DIR) -> str:
    """Diff between a live profile and its frozen default, without writing.

    Raises FileNotFoundError if `name` has no factory default — check
    `has_factory_default_profile` first; this is not meant to be guessed at
    from a caught exception.
    """
    if not has_factory_default_profile(name, factory_root=factory_root):
        raise FileNotFoundError(f"No factory default for profile '{name}' "
                                f"(it was created after the baseline was frozen).")
    return preview_text_diff(load_vlm_profile_text(name, root=root),
                             load_vlm_profile_text(name, root=factory_root),
                             f"{name}.yaml")


def restore_vlm_profile_default(name: str, root: Path = VLM_PROFILES_DIR,
                                factory_root: Path = FACTORY_VLM_PROFILES_DIR) -> Path:
    """Overwrites a live profile with its frozen default. Backs up first."""
    if not has_factory_default_profile(name, factory_root=factory_root):
        raise FileNotFoundError(f"No factory default for profile '{name}' "
                                f"(it was created after the baseline was frozen).")
    return save_vlm_profile_text(name, load_vlm_profile_text(name, root=factory_root), root=root)


def _profile_registry_restore_changes(
        strategy_path: Path = STRATEGY_YAML_PATH,
        factory_strategy_path: Path = FACTORY_STRATEGY_YAML_PATH) -> Dict[str, Optional[str]]:
    """The role changes `write_profile_registry_roles` needs to bring the
    live registry back to the frozen default — every role the default sets
    (to its default value) or removes (a role present live but absent in
    the default becomes None, i.e. deleted), computed by DIFFING the two,
    not by copying the default file whole (that would strip the live
    file's comments, the exact thing this module's surgical-edit design
    exists to avoid)."""
    live = read_profile_registry(strategy_path)
    default = read_profile_registry(factory_strategy_path)
    changes: Dict[str, Optional[str]] = {}
    for role in set(live) | set(default):
        default_value = default.get(role)
        if live.get(role) != default_value:
            changes[role] = default_value  # None deletes a role live-only
    return changes


def preview_profile_registry_restore_diff(
        strategy_path: Path = STRATEGY_YAML_PATH,
        factory_strategy_path: Path = FACTORY_STRATEGY_YAML_PATH) -> str:
    """Diff `write_profile_registry_roles` would produce restoring defaults, without writing."""
    changes = _profile_registry_restore_changes(strategy_path, factory_strategy_path)
    return preview_profile_registry_diff(changes, strategy_path)


def restore_profile_registry_default(
        strategy_path: Path = STRATEGY_YAML_PATH,
        factory_strategy_path: Path = FACTORY_STRATEGY_YAML_PATH) -> Path:
    """Restores `profile_registry` to its frozen default, role by role,
    surgically — see `_profile_registry_restore_changes`. Backs up first."""
    changes = _profile_registry_restore_changes(strategy_path, factory_strategy_path)
    return write_profile_registry_roles(changes, strategy_path)


def restore_all_defaults(
        root: Path = VLM_PROFILES_DIR, factory_root: Path = FACTORY_VLM_PROFILES_DIR,
        model_ini_path: Path = MODEL_INI_PATH, factory_model_ini_path: Path = FACTORY_MODEL_INI_PATH,
        strategy_path: Path = STRATEGY_YAML_PATH,
        factory_strategy_path: Path = FACTORY_STRATEGY_YAML_PATH) -> Dict[str, Path]:
    """Restores model.ini, every profile that HAS a factory default, and
    profile_registry, in that order. Returns {what: backup_path} for
    everything actually changed — a profile already matching its default,
    or one with no default at all, is not touched and not in the result."""
    backups: Dict[str, Path] = {}

    if preview_model_ini_restore_diff(model_ini_path, factory_model_ini_path).strip():
        backups["model.ini"] = restore_model_ini_default(model_ini_path, factory_model_ini_path)

    for name in list_vlm_profiles(root):
        if not has_factory_default_profile(name, factory_root=factory_root):
            continue
        if preview_vlm_profile_restore_diff(name, root=root, factory_root=factory_root).strip():
            backups[f"{name}.yaml"] = restore_vlm_profile_default(name, root=root, factory_root=factory_root)

    if _profile_registry_restore_changes(strategy_path, factory_strategy_path):
        backups["profile_registry"] = restore_profile_registry_default(strategy_path, factory_strategy_path)

    return backups


def detect_gpu_vram_gb() -> Optional[float]:
    """
    Best-effort read of the card's total VRAM via nvidia-smi — the same tool
    CLAUDE.md's operational notes already use for verifying a clean card.
    Returns None (not a guess) if nvidia-smi is unavailable or fails; a
    caller should treat that as "let the user type it in", not as 0.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        first_line = result.stdout.strip().splitlines()[0]
        return round(float(first_line.strip()) / 1024, 1)
    except Exception:
        return None


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _write_text(path: Path, text: str) -> None:
    """Write `text` to `path` as UTF-8 with LF line endings — the ONLY write
    path this module uses.

    For biology researchers: Windows and Linux disagree about how a line
    ends. Windows uses two invisible characters (CR + LF), Linux and Mac use
    one (LF). Python's ordinary `write_text` "helpfully" translates every LF
    into whatever the current operating system prefers — so on Windows,
    saving a config file silently rewrote all ~129 of its line endings even
    when the user changed a single word.

    Nothing broke: YAML and INI parse either ending happily. But git then
    reported the file as modified while `git diff` showed no change at all,
    which is exactly the kind of noise that trains people to ignore
    `git status`. `newline="\\n"` turns the translation off, so the bytes
    written are the bytes we asked for.

    `text` is normalized to LF FIRST, because `newline="\\n"` only stops
    Python adding CRLF — it happily writes a CRLF that was already in the
    string. That case is not hypothetical: HTML specifies that a browser
    `<textarea>` submits its contents with CRLF line breaks, so the raw-text
    editors for `model.ini` and the VLM profiles hand us CRLF whenever the
    user has actually typed in them. Normalizing here means every write path
    lands as LF regardless of where the text came from.

    The other half of this fix is `.gitattributes` (`* text=auto eol=lf`),
    which stops `git checkout` re-introducing CRLF from the opposite
    direction. Neither half works alone — see that file's header.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    path.write_text(normalized, encoding="utf-8", newline="\n")


def _backup(path: Path) -> Path:
    """Copies `path` to `<path>.bak-<unix-timestamp>` before it is overwritten.

    If `path` does not exist yet (a brand-new profile), returns the WOULD-BE
    backup path with nothing written — there is nothing to preserve, and a
    caller can tell the two cases apart by checking `path.exists()` first.
    """
    backup_path = path.with_name(f"{path.name}.bak-{int(time.time())}")
    if path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def _unified_diff(before: str, after: str, label: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"{label} (current)", tofile=f"{label} (proposed)",
    )
    return "".join(diff)


def _find_registry_block(lines: List[str]) -> tuple[int, int]:
    """Returns [start, end) line indices of the `profile_registry:` block,
    where `end` is the first line back at column 0 that isn't blank."""
    start = None
    for index, line in enumerate(lines):
        if line.startswith("profile_registry:"):
            start = index
            break
    if start is None:
        raise ValueError("'profile_registry:' not found in the strategy YAML — "
                         "refusing to guess where to write.")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line[0].isspace():
            end = index
            break
    return start, end


def _apply_role_changes(block: List[str], new_roles: Dict[str, Optional[str]]) -> List[str]:
    """Rewrites/deletes/appends role lines within an already-isolated block."""
    block = list(block)
    remaining = dict(new_roles)

    result: List[str] = []
    for line in block:
        match = _ROLE_LINE.match(line)
        role_name = match.group(2) if match else None
        if role_name in remaining:
            new_value = remaining.pop(role_name)
            if new_value is None:
                continue  # delete this role's line entirely
            indent, _, _, comment = match.groups()
            result.append(f"{indent}{role_name}: {new_value}{comment or ''}\n")
        else:
            result.append(line)

    # Anything left in `remaining` is a role being ADDED that had no
    # existing line — append it just before the block's trailing blank
    # lines (if any), so it reads as part of the registry, not orphaned
    # after a gap.
    trailing_blank_count = 0
    for line in reversed(result):
        if line.strip() == "":
            trailing_blank_count += 1
        else:
            break
    insert_at = len(result) - trailing_blank_count
    for role_name, value in remaining.items():
        if value is None:
            continue  # deleting a role that never existed — nothing to do
        result.insert(insert_at, f"  {role_name}: {value}\n")
        insert_at += 1

    return result
