"""
The output JSON is a published contract, and this file is what holds it.

For biology researchers: another program (Shastrasaara, the search-and-answer
layer) reads the JSON files this pipeline writes. Until v6.4.0 the shape of
those files was written down nowhere, so a change here reached the reader only
when its own tests failed, or did not fail at all. Two such changes are on
record: table cells became LaTeX in v6.3.0, and chapter page ranges turned out
to be counted from zero. `contract/prajnya_document.schema.json` is the written
promise, and these tests make breaking it loud, on THIS side of the boundary.

Four kinds of check:
  1. the schema is sound and its version matches the code's;
  2. what the writer produces satisfies it;
  3. every real document in `run_outputs/` satisfies it (skipped if absent);
  4. the schema has teeth -- deliberately damaged documents are refused --
     and its vocabularies cannot drift from the code that emits them.
"""
import copy
import json
import re
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from src.processing.document_json import CONTRACT_VERSION, build_document_json
from src.processing.models import (ContentPage, DocumentProfile, DocumentType,
                                   ExtractedMetadata, ProcessingResult)
from src.processing.utils.page_classifier import PageType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "contract" / "prajnya_document.schema.json"
RUN_OUTPUTS = REPO_ROOT / "run_outputs"


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


SCHEMA = _load_schema()
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


def _errors(document: dict) -> list:
    """Every way `document` breaks the contract, as readable strings."""
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message[:120]}"
            for e in VALIDATOR.iter_errors(document)]


# --- a document shaped like the pipeline's, built through the real writer -----

def _page_with_everything() -> ContentPage:
    """A model-read page carrying a figure, a table and a heading."""
    return ContentPage(page_number=1, content="# Results\nText.", quality_score=0.9, metadata={
        "processing_method": "glm_fullpage",
        "page_type": "simple_visual",
        "figures": [{"figure_id": "p1_fig1", "page": 1, "bbox": [10, 20, 300, 400],
                     "caption": "Fig. 1. A figure.", "image_path": "figures/doc/p1_fig1.png",
                     "printed_label": "Fig. 1", "label_key": "fig1", "panel_group_size": 1}],
        "tables": [{"table_id": "p1_tab1", "page": 1, "bbox": [10, 500, 300, 700],
                    "caption": "Table 1.", "image_path": "tables/doc/p1_tab1.png",
                    "grid": '<table border="1"><tr><td>$R_{\\mathrm{b}}$</td><td>0.347</td></tr></table>'}],
        "headings": [{"text": "Results", "bbox": [50.9, 94.5, 290.0, 101.4], "size_ratio": 1.4,
                      "weight_ratio": 1.2, "caps_fraction": 0.1, "is_bold": True}],
        "has_reference_heading": False,
        "reference_density": 0.0,
        "page_time": 12.8,
    })


def _profile(tmp_path: Path, pages=None, chapters=None, errors=None) -> DocumentProfile:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    profile = DocumentProfile(pdf_path=pdf, page_count=2, file_size=9,
                              document_type=DocumentType.RESEARCH_ARTICLE)
    profile.processing_result = ProcessingResult(
        metadata=ExtractedMetadata(title="A paper", authors=["A. Author"],
                                   doi="10.1016/j.test.2026.01.001", year=2026,
                                   validation_source="zotero_doi_lookup",
                                   validation_confidence=0.95),
        pages=pages if pages is not None else [_page_with_everything()],
        chapters=chapters if chapters is not None else [],
        errors=errors or [],
    )
    return profile


@pytest.fixture
def document(tmp_path) -> dict:
    """A valid document, as the real writer produces it."""
    chapter = {"number": 1, "title": "Chapter one", "authors": ["B. Writer"],
               "start_page_printed": 1, "page_range": [10, 12], "page_range_confidence": "header"}
    return build_document_json(_profile(tmp_path, chapters=[chapter]), elapsed=3.5)


# --- 1. the schema itself ---------------------------------------------------------

class TestSchemaIsSound:
    def test_it_is_a_valid_json_schema(self):
        jsonschema.Draft202012Validator.check_schema(SCHEMA)

    def test_its_version_matches_the_code(self):
        """Change the format and the schema together, or this fails."""
        assert SCHEMA["x-contract-version"] == CONTRACT_VERSION

    def test_the_version_has_major_dot_minor_shape(self):
        assert re.fullmatch(r"\d+\.\d+", CONTRACT_VERSION)


# --- 2. what the writer produces ---------------------------------------------------

class TestWriterOutputSatisfiesTheContract:
    def test_a_full_document_is_valid(self, document):
        assert _errors(document) == []

    def test_the_contract_version_is_stamped(self, document):
        assert document["document_info"]["contract_version"] == CONTRACT_VERSION

    def test_a_document_that_does_not_say_its_version_is_still_valid(self, document):
        """Every file written by 6.3.0 or earlier lacks the stamp (and the citations block)."""
        del document["document_info"]["contract_version"]
        del document["citations"]
        assert _errors(document) == []

    def test_a_contract_1_0_document_needs_no_citations_block(self, document):
        """Files written by v6.4.0 and v6.5.0 say 1.0 and have no block."""
        document["document_info"]["contract_version"] = "1.0"
        del document["citations"]
        assert _errors(document) == []

    def test_the_citations_block_is_written(self, document):
        """From contract 1.1 every file carries it, at every level."""
        assert document["citations"]["reference_list"] == {"status": "not_requested"}
        assert document["citations"]["source_label"] == "Author, 2026"

    def test_a_document_with_a_found_reference_list_is_valid(self, tmp_path):
        """Level `online`: the finder's list is written into the block."""
        profile = _profile(tmp_path)
        profile.processing_result.citation_level = "online"
        profile.processing_result.citation_references = {
            "reference_list": {"status": "found", "source": "crossref", "style": "numbered",
                               "count": 1, "local_count": 1, "counts_agree": True},
            "references": [{"position": 1, "number": 1, "first_author": "Bella", "year": 2016,
                            "title": "A title", "container": "A journal",
                            "doi": "10.1042/bj20151169", "source": "crossref",
                            "printed_text": None}],
            "report": {"not": "written"},
        }
        doc = build_document_json(profile, elapsed=1.0)
        assert _errors(doc) == []
        assert doc["citations"]["reference_list"]["status"] == "found"
        assert len(doc["citations"]["references"]) == 1

    def test_a_document_with_markers_is_valid_and_offsets_are_true(self, tmp_path):
        profile = _profile(tmp_path)
        profile.processing_result.citation_level = "online"
        profile.processing_result.citation_references = {
            "reference_list": {"status": "found", "source": "crossref", "style": "numbered",
                               "count": 1, "local_count": 1, "counts_agree": True},
            "references": [{"position": 1, "number": 1, "first_author": "Bella", "year": 2016,
                            "title": None, "container": None, "doi": None,
                            "source": "crossref", "printed_text": None}],
            "markers": [{"page_number": 1, "text": "[1]", "kind": "bracket", "numbers": [1],
                         "context_before": "Text.", "content_offset": None}],
        }
        profile.processing_result.pages[0].content = "# Results\nText. [1]"
        doc = build_document_json(profile, elapsed=1.0)
        assert _errors(doc) == []
        assert doc["citations"]["markers"][0]["content_offset"] == len("# Results\nText. ")

    def test_a_failed_page_is_valid_and_says_why(self, tmp_path):
        failed = ContentPage(page_number=2, content="", metadata={
            "processing_method": "glm_failed", "error": "HTTP 500"})
        doc = build_document_json(_profile(tmp_path, pages=[failed]), elapsed=1.0)
        assert _errors(doc) == []

    def test_a_chapter_whose_boundary_was_not_found_is_valid(self, document):
        """The corpus book has two: page_range is null, not absent and not a pair."""
        document["profile"]["processing_result"]["chapters"].append(
            {"number": 2, "title": "Chapter two", "authors": [], "page_range": None,
             "page_range_confidence": None})
        assert _errors(document) == []

    def test_a_document_with_no_pages_is_valid(self, tmp_path):
        assert _errors(build_document_json(_profile(tmp_path, pages=[]), elapsed=1.0)) == []

    def test_unknown_extra_keys_are_allowed(self, document):
        """Additive changes must never break a reader."""
        document["pages"][0]["metadata"]["a_key_added_next_year"] = {"x": 1}
        document["document_info"]["another_new_key"] = True
        assert _errors(document) == []


# --- 3. the real corpus --------------------------------------------------------------

REAL_DOCUMENTS = sorted((RUN_OUTPUTS / "documents").glob("*.v6.json"))


@pytest.mark.skipif(not REAL_DOCUMENTS, reason="no run_outputs/documents here (gitignored)")
class TestRealOutputsSatisfyTheContract:
    """Verified against the real files, not fixtures: the lesson of sessions 13-16."""

    @pytest.mark.parametrize("path", REAL_DOCUMENTS, ids=lambda p: p.name[:30])
    def test_document_is_valid(self, path):
        document = json.loads(path.read_text(encoding="utf-8"))
        assert _errors(document) == []

    @pytest.mark.parametrize("path", REAL_DOCUMENTS, ids=lambda p: p.name[:30])
    def test_every_crop_file_exists(self, path):
        """image_path is relative to run_outputs/, as the schema says."""
        document = json.loads(path.read_text(encoding="utf-8"))
        missing = [rec["image_path"]
                   for page in document["pages"]
                   for kind in ("figures", "tables")
                   for rec in page["metadata"].get(kind, [])
                   if not (RUN_OUTPUTS / rec["image_path"]).is_file()]
        assert missing == []

    @pytest.mark.parametrize("path", REAL_DOCUMENTS, ids=lambda p: p.name[:30])
    def test_crop_ids_are_unique_within_a_document(self, path):
        document = json.loads(path.read_text(encoding="utf-8"))
        for id_key, kind in (("figure_id", "figures"), ("table_id", "tables")):
            ids = [rec[id_key] for page in document["pages"]
                   for rec in page["metadata"].get(kind, [])]
            assert len(ids) == len(set(ids)), kind


# --- 3b. the published examples ------------------------------------------------------

EXAMPLES = sorted((REPO_ROOT / "contract" / "examples").glob("*.json"))


class TestThePublishedExamplesFollowTheContract:
    """
    `contract/examples/` is what another program is built against before real
    files exist. An example that breaks the contract would teach its reader
    the wrong shape.
    """

    def test_there_are_examples(self):
        assert len(EXAMPLES) >= 2

    @pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
    def test_example_is_valid(self, path):
        assert _errors(json.loads(path.read_text(encoding="utf-8"))) == []

    @pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
    def test_example_says_the_current_contract_version(self, path):
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["document_info"]["contract_version"] == CONTRACT_VERSION

    @pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
    def test_every_marker_is_where_it_says_and_cites_a_listed_reference(self, path):
        """The example's offsets and numbers must be true, or it misleads."""
        document = json.loads(path.read_text(encoding="utf-8"))
        content = {page["page_number"]: page["content"] for page in document["pages"]}
        positions = {ref["position"] for ref in document["citations"]["references"]}
        for marker in document["citations"]["markers"]:
            text, offset = marker["text"], marker["content_offset"]
            page_text = content[marker["page_number"]]
            assert page_text[offset:offset + len(text)] == text
            assert page_text[:offset].endswith(marker["context_before"])
            assert set(marker["numbers"]) <= positions

    @pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
    def test_the_label_is_what_the_code_would_derive(self, path):
        """An example whose label the writer would never produce is a wrong promise."""
        from src.processing.citations import source_label

        document = json.loads(path.read_text(encoding="utf-8"))
        label, by_file_name = source_label(document["metadata_final"],
                                           document["document_info"]["filename"])
        assert document["citations"]["source_label"] == label
        assert document["citations"]["cited_by_file_name"] is by_file_name


# --- 3c. the real example: one paper's unedited output ----------------------------------

REAL_EXAMPLES = sorted((REPO_ROOT / "contract" / "examples").glob("*/*.v6.json"))


class TestTheRealExampleFollowsTheContract:
    """
    `contract/examples/<paper>/` is a real run, copied as written. It is checked
    like any other example, except that the contract allows a marker's
    `content_offset` to be null (the mark was not found exactly once), which the
    hand-written examples never use.
    """

    def test_there_is_a_real_example(self):
        assert REAL_EXAMPLES

    @pytest.mark.parametrize("path", REAL_EXAMPLES, ids=lambda p: p.parent.name)
    def test_example_is_valid(self, path):
        assert _errors(json.loads(path.read_text(encoding="utf-8"))) == []

    @pytest.mark.parametrize("path", REAL_EXAMPLES, ids=lambda p: p.parent.name)
    def test_every_placed_marker_is_where_it_says(self, path):
        document = json.loads(path.read_text(encoding="utf-8"))
        content = {page["page_number"]: page["content"] for page in document["pages"]}
        positions = {ref["position"] for ref in document["citations"]["references"]}
        for marker in document["citations"]["markers"]:
            assert set(marker["numbers"]) <= positions
            offset = marker["content_offset"]
            if offset is None:
                continue
            page_text = content[marker["page_number"]]
            assert page_text[offset:offset + len(marker["text"])] == marker["text"]

    @pytest.mark.parametrize("path", REAL_EXAMPLES, ids=lambda p: p.parent.name)
    def test_every_included_crop_is_listed_in_a_manifest(self, path):
        """The folder keeps a few crops; each must be one the manifests name."""
        folder = path.parent
        listed = set()
        for manifest in folder.glob("*/*/manifest.json"):
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for record in data.get("figures", []) + data.get("tables", []):
                listed.add(record["image_path"])
        crops = {png.relative_to(folder).as_posix() for png in folder.rglob("*.png")}
        assert crops and crops <= listed


# --- 4. teeth, and no drift from the code ---------------------------------------------

def _drop(*path):
    def mutate(doc):
        target = doc
        for step in path[:-1]:
            target = target[step]
        del target[path[-1]]
    return mutate


def _set(value, *path):
    def mutate(doc):
        target = doc
        for step in path[:-1]:
            target = target[step]
        target[path[-1]] = value
    return mutate


DAMAGE = {
    "a page without its number": _drop("pages", 0, "page_number"),
    "a page numbered from zero": _set(0, "pages", 0, "page_number"),
    "a page whose text is not a string": _set(["x"], "pages", 0, "content"),
    "a table grid that is not a string": _set([["a", "b"]], "pages", 0, "metadata", "tables", 0, "grid"),
    "an empty table grid": _set("", "pages", 0, "metadata", "tables", 0, "grid"),
    "a table without its image": _drop("pages", 0, "metadata", "tables", 0, "image_path"),
    "a figure box with three numbers": _set([1, 2, 3], "pages", 0, "metadata", "figures", 0, "bbox"),
    "a figure without a caption key": _drop("pages", 0, "metadata", "figures", 0, "caption"),
    "an invented reading route": _set("made_up_route", "pages", 0, "metadata", "processing_method"),
    "an invented page type": _set("scanned", "pages", 0, "metadata", "page_type"),
    "an invented document type": _set("podcast", "document_info", "document_type"),
    "a version without the v": _set("6.3.0", "document_info", "pipeline_version"),
    "authors as one string": _set("Smith", "metadata_final", "authors"),
    "no summary": _drop("summary"),
    "no pages": _drop("pages"),
    "a chapter with a negative page range": _set([-1, 3], "profile", "processing_result", "chapters", 0, "page_range"),
    "a chapter range of three numbers": _set([1, 2, 3], "profile", "processing_result", "chapters", 0, "page_range"),
    "no pdf_path": _drop("profile", "pdf_path"),
    "a 1.1 document without its citations block": _drop("citations"),
    "an invented citation level": _set("cloud", "citations", "level"),
    "a citations block that does not say its level": _drop("citations", "level"),
    "an invented metadata source": _set("google", "citations", "metadata_source"),
    "an empty source label": _set("", "citations", "source_label"),
    "no cited_by_file_name": _drop("citations", "cited_by_file_name"),
    "an invented reference list status": _set({"status": "maybe"}, "citations", "reference_list"),
    "references that are not a list": _set(None, "citations", "references"),
    "a reference without its position": _set([{"number": 1, "first_author": "Doi", "year": 2005,
                                               "doi": None, "source": "crossref"}],
                                             "citations", "references"),
    "a marker on page zero": _set([{"page_number": 0, "text": "[1]", "kind": "bracket", "numbers": [1],
                                    "context_before": "", "content_offset": None}],
                                  "citations", "markers"),
}


class TestTheSchemaHasTeeth:
    @pytest.mark.parametrize("name", sorted(DAMAGE))
    def test_damage_is_refused(self, document, name):
        assert _errors(document) == [], "the starting document must be valid"
        DAMAGE[name](document)
        assert _errors(document) != [], f"the contract accepted {name}"


#: Each field a contract version added, with that version (audit D6). Before
#: v6.8 the schema defined them but required only 1.1's citations block, so a
#: file claiming 1.4 without `author_names` still passed.
ADDED_IN = {
    "mentions": ("1.2", ("mentions",)),
    "a page's body": ("1.3", ("pages", 0, "body")),
    "a page's furniture": ("1.3", ("pages", 0, "metadata", "furniture")),
    "a page's outside_body": ("1.3", ("pages", 0, "metadata", "outside_body")),
    "author_names": ("1.4", ("metadata_final", "author_names")),
    "status": ("1.5", ("document_info", "status")),
    "failed_pages": ("1.5", ("document_info", "failed_pages")),
}


def _version_before(version: str) -> str:
    major, minor = version.split(".")
    return f"{major}.{int(minor) - 1}"


class TestEachVersionRequiresWhatItAdded:
    @pytest.mark.parametrize("name", sorted(ADDED_IN))
    def test_a_current_file_without_it_is_refused(self, document, name):
        assert _errors(document) == []
        _drop(*ADDED_IN[name][1])(document)
        assert _errors(document) != [], f"the contract accepted a file without {name}"

    @pytest.mark.parametrize("name", sorted(ADDED_IN))
    def test_a_file_from_before_that_version_may_lack_it(self, document, name):
        """Older files stay valid: the rule starts at the version that added it."""
        version, path = ADDED_IN[name]
        document["document_info"]["contract_version"] = _version_before(version)
        _drop(*path)(document)
        assert _errors(document) == []

    @pytest.mark.parametrize("name", sorted(ADDED_IN))
    def test_a_later_version_still_requires_it(self, document, name):
        document["document_info"]["contract_version"] = "1.12"
        _drop(*ADDED_IN[name][1])(document)
        assert _errors(document) != []


def _emitted_processing_methods() -> set:
    """Every processing_method value written anywhere in src/, read from the source."""
    found = set()
    for source in (REPO_ROOT / "src").rglob("*.py"):
        for line in source.read_text(encoding="utf-8").splitlines():
            for match in re.finditer(r'processing_method"\s*:\s*"(\w+)"(?:\s+if\b.*?\belse\s+"(\w+)")?', line):
                found.update(g for g in match.groups() if g)
    return found


class TestVocabulariesCannotDriftFromTheCode:
    """A new route or type added in src/ must be added to the contract, on purpose."""

    def test_every_emitted_processing_method_is_in_the_contract(self):
        allowed = set(SCHEMA["$defs"]["page"]["properties"]["metadata"]["properties"]
                      ["processing_method"]["enum"])
        emitted = _emitted_processing_methods()
        assert emitted, "the scan found nothing -- the regex has stopped working"
        assert emitted <= allowed, f"emitted but not in the contract: {emitted - allowed}"

    def test_page_types_are_exactly_the_classifiers(self):
        allowed = set(SCHEMA["$defs"]["page"]["properties"]["metadata"]["properties"]["page_type"]["enum"])
        assert allowed - {None} == {member.value for member in PageType}

    def test_citation_vocabularies_are_exactly_the_codes(self):
        """A level, source or list status added in src/ must be added here, on purpose."""
        from src.processing.citations.block import METADATA_SOURCES, REFERENCE_LIST_STATUSES
        from src.processing.citations.level import CITATION_LEVELS, LEVEL_SET_BY

        block = SCHEMA["$defs"]["citations"]["properties"]
        assert set(block["level"]["enum"]) == set(CITATION_LEVELS)
        assert set(block["level_set_by"]["enum"]) == set(LEVEL_SET_BY)
        assert set(block["metadata_source"]["enum"]) == set(METADATA_SOURCES)
        assert (set(block["reference_list"]["properties"]["status"]["enum"])
                == set(REFERENCE_LIST_STATUSES))

    def test_document_types_are_exactly_the_models(self):
        allowed = set(SCHEMA["properties"]["document_info"]["properties"]["document_type"]["enum"])
        assert allowed == {member.value for member in DocumentType}
