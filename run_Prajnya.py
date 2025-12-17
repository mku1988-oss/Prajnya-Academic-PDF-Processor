#!/usr/bin/env python3
"""
run_Prajnya.py — Prajnya unified runner (integrated to your repo)

Modes:
  - --pdf PATH           : Mode A (single PDF). optional --pages "5" or "2-4"
  - --folder PATH        : Mode B (batch folder). optional --pages
  - --noninteractive     : Mode C (no interactive prompts, just flagging)
  - --review             : Mode D (interactive ledger review; runs outside pytest/CI)

This script is adapted to your codebase:
- DocumentOrchestrator.process_document(self, pdf_path: Union[str, Path], pages_to_process: Optional[List[int]])
- CsvLedgerBackend(...) with update_record(LedgerRecord) and commit()
- LedgerRecord model from src/processing/ledger/semantic_ledger.py
- generate_hash(file_path) from semantic_ledger.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from dotenv import load_dotenv
load_dotenv()

# Project imports (actual modules in your repo)
from src.processing.document_orchestrator import DocumentOrchestrator
from src.processing.ledger.csv_backend import CsvLedgerBackend
from src.processing.ledger.semantic_ledger import LedgerRecord, generate_hash

# Logging with UTF-8 support for Unicode characters (arrows, checkmarks, etc.)
# Configure root logger to handle all loggers in the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run_debug.log", mode='w', encoding='utf-8')
    ]
)

# Get logger for this module
logger = logging.getLogger("run_Prajnya")

RUN_OUTPUT_DIR = Path("run_outputs")
LEDGER_CSV_PATH = Path("persistence/semantic_ledger.csv")


def parse_pages_to_list(pages: Optional[str]) -> Optional[List[int]]:
    """Parse '5' or '2-5' into [5] or [2,3,4,5]. Return None if pages is None."""
    if not pages:
        return None
    s = pages.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        try:
            start = int(a)
            end = int(b)
            if start <= 0 or end <= 0 or start > end:
                raise ValueError("Invalid page range")
            return list(range(start, end + 1))
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid pages range: {pages}") from e
    else:
        try:
            p = int(s)
            if p <= 0:
                raise ValueError
            return [p]
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid page number: {pages}") from e


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


class PrajnyaRunner:
    def __init__(self, ledger_path: Union[str, Path] = LEDGER_CSV_PATH):
        # instantiate orchestrator
        self.orch = DocumentOrchestrator()
        # ledger backend (CSV)
        self.ledger = CsvLedgerBackend(str(ledger_path))

    def assess_metadata_completeness(
        self, 
        metadata: Dict[str, Any],
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Assess if metadata meets completeness requirements.
        
        Based on user requirement: title + authors + DOI + journal ALL required
        
        Args:
            metadata: Metadata dict from ProcessingResult
            strict: If True, ALL fields must be present (default)
            
        Returns:
            Assessment dict with structure:
            {
                "is_complete": bool,
                "missing_fields": List[str],
                "suspicious_items": List[str],
                "confidence": float,
                "requires_review": bool
            }
        """
        assessment = {
            "is_complete": True,
            "missing_fields": [],
            "suspicious_items": [],
            "confidence": 1.0,
            "requires_review": False
        }
        
        # Get provenance if available
        provenance = metadata.get("provenance", {})
        
        # Check required fields (USER REQUIREMENT: ALL FOUR)
        if not metadata.get("title"):
            assessment["is_complete"] = False
            assessment["missing_fields"].append("title")
        
        authors = metadata.get("authors")
        if not authors or (isinstance(authors, list) and len(authors) == 0):
            assessment["is_complete"] = False
            assessment["missing_fields"].append("authors")
        
        if not metadata.get("doi"):
            assessment["is_complete"] = False
            assessment["missing_fields"].append("doi")
        
        if not metadata.get("journal"):
            assessment["is_complete"] = False
            assessment["missing_fields"].append("journal")
        
        # Check provenance flags
        if provenance.get("suspicious_authors"):
            assessment["suspicious_items"].append("authors")
            assessment["requires_review"] = True
        
        # Check confidence score
        confidence = provenance.get("validation_confidence", 
                                   metadata.get("validation_confidence", 1.0))
        if confidence is None:
            confidence = 0.0
        assessment["confidence"] = float(confidence)
        
        # Low confidence requires review
        if confidence < 0.75:
            assessment["requires_review"] = True
            assessment["suspicious_items"].append(f"low_confidence({confidence:.2f})")
        
        # Final decision
        if assessment["missing_fields"] or assessment["suspicious_items"]:
            assessment["requires_review"] = True
        
        return assessment

    def process_single(self, pdf_path: Union[str, Path], pages: Optional[List[int]] = None, noninteractive: bool = False, concurrency_enabled: bool = True, force: bool = False) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        start = datetime.utcnow()
        logger.info("Processing %s (pages=%s, noninteractive=%s)", pdf_path, pages, noninteractive)

        try:
            doc_hash = generate_hash(str(pdf_path))
            existing_record = self.ledger.get_record(doc_hash)
            
            # If record exists and is marked as 'reviewed' (or complete), skip processing
            # If record exists and is marked as 'reviewed' (or complete), skip processing
            # Unless force is True
            if existing_record and existing_record.review_status == "reviewed" and not force:
                logger.info(f"⏩ SKIPPING {pdf_path.name} - Already processed and reviewed (ID: {doc_hash[:8]})")
                
                # Return a mock result structure so batch summary works
                return {
                    "pdf": pdf_path.name,
                    "success": True,
                    "processing_time": 0.0,
                    "metadata": existing_record.model_dump(),
                    "pages_processed": 0,
                    "skipped": True
                }
        except Exception as e:
            logger.warning(f"Failed to check ledger for existing record: {e}")

        # Initialize result object
        res: Dict[str, Any] = {
            "pdf": pdf_path.name,
            "success": False,
            "error_details": None,
            "processing_time": None,
            "metadata": {},
            "pages_processed": 0,
        }
        
        assessment = {
            "is_complete": False,
            "missing_fields": [],
            "suspicious_items": [],
            "confidence": 0.0,
            "requires_review": True,
            "review_status": "not_assessed"
        }

        # We must keep orches defined for FIX 3
        orches = None
        processing_result = None

        try:
            # ------------------------------------------------------------------
            # EXECUTE MAIN PIPELINE
            # ------------------------------------------------------------------
            orches = self.orch.process_document(pdf_path, pages_to_process=pages, concurrency_enabled=concurrency_enabled)

            if isinstance(orches, dict):
                res["success"] = bool(orches.get("success", True))
                res["error_details"] = orches.get("error_details")
                res["processing_time"] = float(orches.get("processing_time", 0.0)) or (
                    (datetime.utcnow() - start).total_seconds()
                )

                processing_result = orches.get("processing_result")
                if processing_result:
                    md = processing_result.metadata.model_dump() if processing_result.metadata else {}
                    res["metadata"] = md
                    res["pages_processed"] = len(processing_result.pages)
                else:
                    res["metadata"] = {}
                    res["pages_processed"] = 0

            else:
                # fallback for old architectures
                res["success"] = True
                res["processing_time"] = (datetime.utcnow() - start).total_seconds()
                res["metadata"] = getattr(orches, "metadata", {}) or {}
                res["pages_processed"] = getattr(orches, "pages_processed", 0) or 0

            # ------------------------------------------------------------------
            # METADATA COMPLETENESS ASSESSMENT
            # ------------------------------------------------------------------
            md = res.get("metadata") or {}
            document_id = md.get("source_hash") or generate_hash(str(pdf_path))

            # Use new assessment function
            assessment = self.assess_metadata_completeness(md)

            logger.info(f"Metadata assessment: {assessment}")

            # ------------------------------------------------------------------
            # INTERACTIVE REVIEW (if needed and mode allows)
            # ------------------------------------------------------------------
            if assessment["requires_review"] and not noninteractive:
                print("\n" + "="*60)
                print("⚠️  METADATA REVIEW REQUIRED")
                print("="*60)
                print(f"Missing fields: {assessment['missing_fields']}")
                print(f"Suspicious items: {assessment['suspicious_items']}")
                print(f"Confidence: {assessment['confidence']:.2f}")
                print()
                
                # Call interactive review function
                md = self._interactive_review(md, assessment)
                
                # Update result with edited metadata
                res["metadata"] = md
                
                # Mark as reviewed
                assessment["requires_review"] = False
                assessment["review_status"] = "reviewed"
            else:
                assessment["review_status"] = "provisional" if assessment["requires_review"] else "reviewed"

            # ------------------------------------------------------------------
            # LEDGER UPDATE
            # ------------------------------------------------------------------
            try:
                existing = self.ledger.get_record(document_id)
                
                if existing:
                    # Update record with latest metadata
                    existing.title = md.get("title")
                    existing.doi = md.get("doi")
                    existing.journal = md.get("journal")
                    existing.year = md.get("year")
                    
                    authors = md.get("authors") or []
                    if isinstance(authors, str):
                        authors = [a.strip() for a in authors.split(",") if a.strip()]
                    existing.authors = authors
                    existing.requires_review = assessment.get("requires_review", False)
                    existing.review_needed_fields = assessment.get("missing_fields", [])
                    existing.review_status = "reviewed" if not assessment["requires_review"] else "provisional"
                    existing.last_updated = datetime.utcnow().isoformat()
                    
                    self.ledger.update_record(existing)
                else:
                    # Create new record
                    authors = md.get("authors") or []
                    if isinstance(authors, str):
                        authors = [a.strip() for a in authors.split(",") if a.strip()]
                    
                    prov = md.get("provenance", {})
                    validation_confidence = prov.get("validation_confidence", 0.0)
                    validation_source = prov.get("validation_source", "prajnya_runner")
                    new_rec = LedgerRecord(
                        document_id=document_id,
                        pdf_filename=pdf_path.name,
                        title=md.get("title"),
                        doi=md.get("doi"),
                        authors=authors,
                        journal=md.get("journal"),
                        year=md.get("year"),
                        
                        validation_confidence=validation_confidence,
                        validation_source=validation_source,
                        requires_review=assessment.get("requires_review", False),
                        review_needed_fields=assessment.get("missing_fields", []),
                        review_status="reviewed" if not assessment["requires_review"] else "provisional",
                        last_reviewed_by=None,
                        manual_entry=False,
                        
                        last_updated=datetime.utcnow().isoformat(),
                        retrieval_provenance=md.get("retrieval_provenance", {}),
                        source_hash=document_id
                    )
                    self.ledger.update_record(new_rec)
                
                # CRITICAL: Ledger flush NOW happens in Runner
                self.ledger.commit()
                logger.info("Ledger updated successfully")
                
            except Exception as e:
                logger.error(f"Ledger update failed: {e}", exc_info=True)

        except Exception as e:
            logger.exception("Critical pipeline failure for PDF %s: %s", pdf_path, e)
            res["success"] = False
            res["error_details"] = str(e)
            res["processing_time"] = (datetime.utcnow() - start).total_seconds()
            # DO NOT RETURN — we still need to write JSON

        # ------------------------------------------------------------------
        # STRUCTURED JSON OUTPUT
        # MOVED HERE: After ledger update, so user corrections are captured
        # ------------------------------------------------------------------
        # NOTE: In batch mode, this writes provisional metadata
        # The JSON will be regenerated after interactive review via regenerate_json()
        # ------------------------------------------------------------------
        try:
            out_dir = Path("run_outputs") / "documents"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            pages = []
            if processing_result:
                pages = [p.model_dump() for p in processing_result.pages]
            
            # Create a clean copy without nested provenance
            metadata_clean = res.get("metadata", {}).copy()
            
            # Extract provenance for metadata_sources before removing
            provenance = metadata_clean.get("provenance", {})
            
            # Remove the circular provenance dict from metadata
            metadata_clean.pop("provenance", None)
            structured_output = {
                "document_info": {
                    "filename": pdf_path.name,
                    "document_type": orches.get("document_type") if isinstance(orches, dict) else None,
                    "processing_time": res["processing_time"],
                    "success": res["success"],
                    "pipeline_version": "v4.3.0-phase3.8"  # NEW
                },
                
                "metadata_final": metadata_clean,  # Use cleaned metadata
                
                "metadata_sources": {  # Extract from provenance dict
                    "vlm": provenance.get("validation_source", "unknown"),
                    "zotero": provenance.get("zotero_source", "none"),
                    "confidence": provenance.get("validation_confidence", 0.0)
                },
                
                "review_status": assessment.get("review_status", "not_assessed"),  # NEW
                "review_needed_fields": assessment.get("missing_fields", []),  # NEW
                
                "pages": pages,
                
                "summary": {
                    "pages_processed": len(pages),
                    "errors": processing_result.errors if processing_result else [],
                }
            }
            
            output_path = out_dir / f"{pdf_path.stem}.json"
            write_json(output_path, structured_output)
            
            logger.info(f"Structured JSON saved: {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to write structured JSON: {e}")

        # Final return — only once, after JSON is written
        return res

    def _interactive_review(
        self, 
        metadata: Dict[str, Any],
        assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Interactive metadata correction loop.
        
        Args:
            metadata: Current metadata dict
            assessment: Completeness assessment result
            
        Returns:
            Corrected metadata dict
        """
        while True:
            print("\nCurrent metadata:")
            print(f"1) Title   : {metadata.get('title')}")
            print(f"2) Authors : {metadata.get('authors')}")
            print(f"3) Journal : {metadata.get('journal')}")
            print(f"4) DOI     : {metadata.get('doi')}")
            print(f"5) Year    : {metadata.get('year')}")
            print(f"6) Done (save and continue)")
            
            choice = input("\nChoose field to edit (1-6): ").strip()
            
            if choice == "1":
                new_val = input("New Title: ").strip()
                if new_val:
                    metadata["title"] = new_val
                    
            elif choice == "2":
                new_val = input("New authors (comma-separated): ").strip()
                if new_val:
                    metadata["authors"] = [a.strip() for a in new_val.split(",") if a.strip()]
                    
            elif choice == "3":
                new_val = input("New Journal: ").strip()
                if new_val:
                    metadata["journal"] = new_val
                    
            elif choice == "4":
                new_val = input("New DOI: ").strip()
                metadata["doi"] = new_val if new_val else None
                
            elif choice == "5":
                new_val = input("New Year (YYYY): ").strip()
                if new_val:
                    try:
                        metadata["year"] = int(new_val)
                    except ValueError:
                        print("Invalid year format")
                        
            elif choice == "6":
                # Validate completeness before allowing exit
                check = self.assess_metadata_completeness(metadata)
                if check["missing_fields"]:
                    print(f"\n⚠️  Still missing: {check['missing_fields']}")
                    cont = input("Save anyway? (y/N): ").strip().lower()
                    if cont != "y":
                        continue
                break
            else:
                print("Invalid choice. Pick 1-6.")
        
        return metadata

    def regenerate_json_from_ledger(self, pdf_paths: List[Path]) -> None:
        """
        Regenerate JSON files from ledger records after interactive review.
        
        This ensures JSON files contain user-corrected metadata, not provisional data.
        
        Args:
            pdf_paths: List of PDF paths whose JSON needs regeneration
        """
        logger.info(f"Regenerating JSON for {len(pdf_paths)} PDFs from ledger...")
        
        for pdf_path in pdf_paths:
            try:
                # Get ledger record with corrected metadata
                doc_id = generate_hash(str(pdf_path))
                record = self.ledger.get_record(doc_id)
                
                if not record:
                    logger.warning(f"No ledger record found for {pdf_path.name}")
                    continue
                
                # Build metadata dict from ledger
                metadata_clean = {
                    "title": record.title,
                    "authors": record.authors,
                    "doi": record.doi,
                    "journal": record.journal,
                    "year": record.year,
                    "publisher": None,
                    "validation_confidence": record.validation_confidence,
                    "validation_source": record.validation_source,
                    "source_hash": record.source_hash
                }
                
                # Create Phase 3.8 compliant structure
                structured_output = {
                    "document_info": {
                        "filename": pdf_path.name,
                        "document_type": "research_article",
                        "processing_time": 0.0,  # Not re-processing
                        "success": True,
                        "pipeline_version": "v4.3.0-phase3.8"
                    },
                    
                    "metadata_final": metadata_clean,
                    
                    "metadata_sources": {
                        "vlm": record.validation_source,
                        "zotero": "user_corrected",
                        "confidence": record.validation_confidence
                    },
                    
                    "review_status": record.review_status,
                    "review_needed_fields": [],
                    
                    "pages": [],  # Preserved from original processing
                    
                    "summary": {
                        "pages_processed": 0,
                        "errors": [],
                        "note": "Regenerated from ledger after user review"
                    }
                }
                
                # Write updated JSON
                out_dir = Path("run_outputs") / "documents"
                out_dir.mkdir(parents=True, exist_ok=True)
                output_path = out_dir / f"{pdf_path.stem}.json"
                
                write_json(output_path, structured_output)
                logger.info(f"Regenerated JSON: {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to regenerate JSON for {pdf_path.name}: {e}")

    def run_folder(self, folder: Union[str, Path], recursive: bool = False, noninteractive: bool = False, pages: Optional[List[int]] = None) -> Dict[str, Any]:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        pdfs = sorted(folder.rglob("*.pdf")) if recursive else sorted(folder.glob("*.pdf"))
        processed = []
        failed = []
        needs_review = []  # FIX BUG #9: Track PDFs needing review
        tstart = datetime.utcnow()
        
        # Review prompts will be deferred until the end
        for p in pdfs:
            result = self.process_single(p, pages=pages, noninteractive=True)  # Force noninteractive
            if result.get("success"):
                processed.append(result)
                # Check if this PDF needs review
                md = result.get("metadata", {})
                doc_id = md.get("source_hash") or generate_hash(str(p))
                record = self.ledger.get_record(doc_id)
                if record and record.requires_review:
                    needs_review.append(p)
            else:
                failed.append(result)
        
        total_time = (datetime.utcnow() - tstart).total_seconds()
        
        for result in processed:
            if "metadata" in result and isinstance(result["metadata"], dict):
                result["metadata"].pop("provenance", None)
        
        for result in failed:
            if "metadata" in result and isinstance(result["metadata"], dict):
                result["metadata"].pop("provenance", None)
        
        if needs_review and not noninteractive:
            print("\n" + "="*60)
            print(f"📋 BATCH PROCESSING COMPLETE")
            print("="*60)
            print(f"Processed: {len(processed)} PDFs")
            print(f"Failed: {len(failed)} PDFs")
            print(f"Needs Review: {len(needs_review)} PDFs")
            print()
            print("The following PDFs need metadata review:")
            for pdf in needs_review:
                print(f"  - {pdf.name}")
            print()
            
            choice = input("Start interactive review now? (y/N): ").strip().lower()
            if choice == "y":
                self.review_ledger_interactive()
                
                logger.info("Regenerating JSON files with corrected metadata...")
                self.regenerate_json_from_ledger(needs_review)
                print("\n✅ JSON files updated with corrected metadata!")
            else:
                print("\nReview skipped. Run 'python run_Prajnya.py --review' later.")
                print("Note: JSON files contain provisional metadata until review is complete.")
        
        summary = {"processed": processed, "failed": failed, "needs_review": [str(p) for p in needs_review], "total_time": total_time}
        # write summary
        ensure = Path("run_outputs")
        ensure.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        write_json(ensure / f"batch_summary_results_{ts}.json", summary)
        write_json(ensure / "batch_summary_results.json", summary)
        return summary

    def _record_needs_review(self, rec: LedgerRecord) -> bool:
        """Decide whether a ledger record needs interactive review."""
        # If schema has metadata_flag, use it.
        rdict = rec.model_dump() if hasattr(rec, "model_dump") else {}
        if rdict.get("metadata_flag") == "needs_user_review":
            return True
        # Fallback heuristics: missing critical metadata
        if not rec.title or not rec.authors or len(rec.authors) == 0 or not rec.doi:
            return True
        return False

    def review_ledger_interactive(self):
        """Interactive post-batch ledger review. Must be run from terminal (not pytest)."""
        # collect ledger entries
        # get all records from ledger backend implementation
        # CsvLedgerBackend stores records in _records dict mapping document_id->LedgerRecord
        entries: List[LedgerRecord] = []
        try:
            # prefer public API if exists
            if hasattr(self.ledger, "get_all_records"):
                entries = self.ledger.get_all_records()
            else:
                # try internal _records
                if hasattr(self.ledger, "_records"):
                    entries = list(self.ledger._records.values())
                else:
                    # last resort: try read via commit/load interface by constructing CsvLedgerBackend again
                    logger.error("Ledger backend does not expose records for review.")
                    return
        except Exception as e:
            logger.exception("Failed to enumerate ledger records: %s", e)
            return

        to_review = [r for r in entries if self._record_needs_review(r)]
        if not to_review:
            print("No ledger entries require manual review.")
            return

        print(f"{len(to_review)} ledger entries require review. Launching interactive correction...")

        for rec in to_review:
            print("\n" + "=" * 60)
            print(f"Reviewing: {rec.pdf_filename or rec.document_id}")
            print("=" * 60)
            while True:
                print("\nCurrent values:")
                print("1) Title  :", rec.title)
                print("2) Authors:", rec.authors)
                print("3) Journal:", rec.journal)
                print("4) DOI    :", rec.doi)
                print("5) Skip / Done (save and continue)")
                ch = input("Choose field to edit (1-5): ").strip()
                if ch == "1":
                    newv = input("New Title (leave blank to keep): ").strip()
                    if newv != "":
                        rec.title = newv
                elif ch == "2":
                    newv = input("New authors (comma-separated, leave blank to keep): ").strip()
                    if newv != "":
                        rec.authors = [a.strip() for a in newv.split(",") if a.strip()]
                elif ch == "3":
                    newv = input("New Journal (leave blank to keep): ").strip()
                    if newv != "":
                        rec.journal = newv
                elif ch == "4":
                    newv = input("New DOI (leave blank to clear): ").strip()
                    rec.doi = newv if newv else None
                elif ch == "5":
                    # mark reviewed by clearing metadata_flag if exists
                    try:
                        rdict = rec.model_dump()
                        if "metadata_flag" in rdict:
                            rec_dict = rdict
                            rec_dict["metadata_flag"] = ""
                            # Create updated LedgerRecord and update in ledger
                            updated = LedgerRecord(**rec_dict)
                            self.ledger.update_record(updated)
                        else:
                            self.ledger.update_record(rec)
                        self.ledger.commit()
                    except Exception:
                        try:
                            self.ledger.update_record(rec)
                            self.ledger.commit()
                        except Exception as e:
                            logger.exception("Failed to save corrected record: %s", e)
                    print("Saved; moving on.")
                    break
                else:
                    print("Invalid selection. Pick 1-5.")

        print("Interactive review complete. Ledger updated.")

# ---- CLI ----
def build_parser():
    p = argparse.ArgumentParser(prog="run_Prajnya")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--pdf", type=str, help="Single PDF to process")
    g.add_argument("--folder", type=str, help="Folder of PDFs to process (non-recursive by default)")
    g.add_argument("--review", action="store_true", help="Run interactive ledger review (Mode D)")

    p.add_argument("--pages", type=str, default=None, help="Single page or range: '5' or '2-5'")
    p.add_argument("--noninteractive", action="store_true", help="Do not prompt during run; mark unresolved metadata for later review")
    p.add_argument("--recursive", action="store_true", help="When using --folder, use recursive search")
    p.add_argument("--verbosity", "-v", action="count", default=0, help="Increase logging verbosity")
    p.add_argument("--no-concurrency", action="store_true", help="Disable concurrent processing (Phase 4.6) for diagnostic testing")
    p.add_argument("--force", action="store_true", help="Force re-processing even if already in ledger")
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbosity >= 2:
        logger.setLevel(logging.DEBUG)
    elif args.verbosity == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    # Initialize VLM Server
    from src.llm.providers.server_manager import ServerManager
    import yaml

    # Load configuration from strategy YAML
    server_config = None
    model_to_load = "olmocr-2-7b"
    try:
        config_path = Path("src/processing/strategies/config/research_article_strategy.yaml")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                # Extract local_server configuration
                server_config = cfg.get('vlm_provider', {}).get('local_server', {})
                # Extract default model
                model_to_load = cfg.get('profile_registry', {}).get('default', "olmocr-2-7b")
    except Exception as e:
        logger.warning(f"Could not read strategy config: {e}")

    # Initialize ServerManager from config (or fallback to defaults)
    if server_config and server_config.get('enabled', True):
        server_manager = ServerManager.from_config(server_config)
    else:
        logger.info("Using default ServerManager configuration")
        server_manager = ServerManager(backend="router_mode")

    try:
        if not server_manager.ensure_started():
            logger.error("Failed to start VLM server")
            return 1

        # Load model for Phase 5.5
        if not server_manager.load_model(model_to_load):
            logger.warning(f"Failed to load VLM model '{model_to_load}'. Processing may fail if no model is loaded.")

        pages_list = None
        if args.pages:
            pages_list = parse_pages_to_list(args.pages)

        runner = PrajnyaRunner()
        
        # Initialize review lists for finally block
        needs_review = []
        ledger_candidates = []

        if args.review:
            # interactive review mode
            runner.review_ledger_interactive()
            return 0

        if args.pdf:
            p = Path(args.pdf)
            if not p.exists():
                logger.error("PDF not found: %s", p)
                return 2
            res = runner.process_single(p, pages=pages_list, noninteractive=args.noninteractive, concurrency_enabled=not args.no_concurrency, force=args.force)
            # output
            ensure = Path("run_outputs")
            ensure.mkdir(parents=True, exist_ok=True)
            out = ensure / f"single_{p.stem}_result.json"
            write_json(out, res)
            print(json.dumps(res, indent=2, default=str))
            return 0

        if args.folder:
            folder = Path(args.folder)
            if not folder.exists():
                logger.error("Folder not found: %s", folder)
                return 2
            summary = runner.run_folder(folder, recursive=args.recursive, noninteractive=args.noninteractive, pages=pages_list)
            print(json.dumps(summary, indent=2, default=str))

            # print which processed need review
            needs_review = []
            for r in summary.get("processed", []):
                md = r.get("metadata") or {}
                if md.get("metadata_flag") == "needs_user_review":
                    needs_review.append(r["pdf"])
            # Also check ledger for heuristic missing fields
            ledger_candidates = []

    finally:
        # Stop server only if we started it (ServerManager handles this check)
        server_manager.stop()
        try:
            if hasattr(runner.ledger, "_records"):
                for rec in runner.ledger._records.values():
                    if not rec.title or not rec.authors or len(rec.authors) == 0 or not rec.doi:
                        ledger_candidates.append(rec.pdf_filename or rec.document_id)
        except Exception:
            pass

        merged = sorted(set(needs_review + ledger_candidates))
        if merged:
            print("\nThe following PDFs likely need manual metadata correction:")
            for p in merged:
                print(" -", p)
            if not args.noninteractive:
                print("\nRun `python run_Prajnya.py --review` to interactively fix them now.")
        else:
            print("\nNo immediate metadata review required.")

        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
