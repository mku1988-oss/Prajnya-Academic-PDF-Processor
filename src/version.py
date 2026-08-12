"""
Prajnya Version Management

Central version definition for the entire Prajnya project.
Update this file when releasing new versions.
"""

__version__ = "5.2.0"
"""Project version (semantic versioning: MAJOR.MINOR.PATCH)"""

PIPELINE_VERSION = "v5.0.0"
"""Pipeline version identifier used in output metadata"""

# Version history
# v5.2.0 - GLM-OCR two-phase reading engine: a VLM-agnostic table-region
#          detector, content-drop coverage gate, model residency swapping,
#          figure crop+tag for multimodal RAG. Also: author order preserved
#          across merges, Zotero DOI-first lookup, text-layer DOI no longer
#          discarded. MINOR rather than PATCH because the reading engine is
#          new; PIPELINE_VERSION stays v5.0.0 because the OUTPUT SCHEMA is
#          unchanged, and that is what the identifier stamps into results.
# v5.1.0 - OpenRouter integration (NVIDIA Nemotron 12B VL and other models)
# v5.0.0 - Automated deployment system, GitFlow workflow, centralized version management
# v4.3.0 - Phase 3.8 completion (legacy)
