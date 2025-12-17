# prajnya_studio/gui/pages/batch_commander.py
from nicegui import ui
import asyncio
import os
import logging
from datetime import datetime
from pathlib import Path

from .. import theme
from ..app_state import AppState
from ...prajnya import orchestrator, logger as prajnya_logger

# Module-level logger for this page
logger = logging.getLogger(__name__)

# --- GLOBAL STATE (Persists across tab switches) ---
_global_batch_processor = None
_global_batch_ledger = None
from ..utils.batch_processor import BatchProcessor
from ..utils.batch_ledger import BatchLedger
from ..utils.confidence_utils import derive_field_confidence
from ...prajnya.orchestrator import get_orchestrator

class LogStream(ui.log):
    """A self-updating log element that connects to the Prajnya logger."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classes('w-full h-full font-mono text-xs')
        self.style('background: transparent; color: #E0E0E0;')
        # Poll for new logs every 100ms
        self._timer = ui.timer(0.1, self._update)
    
    def _update(self):
        # Get new entries from the backend deque handler
        if prajnya_logger.GLOBAL_LOG_HANDLER:
            entries = prajnya_logger.GLOBAL_LOG_HANDLER.get_new_entries()
            for entry in entries:
                # Format: [HH:MM:SS] [LEVEL] Message
                ts = entry['ts'].split('T')[1][:8]
                line = f"[{ts}] [{entry['level']}] {entry['msg']}"
                self.push(line)

def mount():
    # Access globals to persist state across tab switches
    global _global_batch_processor, _global_batch_ledger
    
    # --- CSS: Force Standard Components to match Vedic Palette ---
    ui.add_head_html(f'''
        <style>
            /* Toggles (Switches) -> Vermillion (Agni) */
            .q-toggle__inner--truthy {{ color: {theme.THEME_LIGHT["vermillion"]} !important; }}
            .q-toggle__track {{ color: {theme.THEME_LIGHT["vermillion"]} !important; opacity: 0.4; }}
            
            /* Linear Progress -> Default to Indigo */
            .q-linear-progress__model {{ color: {theme.THEME_LIGHT["indigo"]}; }}
            .q-linear-progress__track {{ opacity: 0.15; }}
        </style>
    ''')
    
    state = AppState.get_instance()
    
    state = AppState.get_instance()
    
    # FIX 1: Safety wrapper for UI notifications (prevents crash on tab switch)
    def safe_notify(*args, **kwargs):
        try:
            ui.notify(*args, **kwargs)
        except (RuntimeError, TypeError):
            # Ignore errors if the page has been unmounted
            pass
    
    # --- State Variables for this page ---
    uploaded_files = []
    
    # Initialize OR Retrieve Singleton Processor
    batch_orchestrator = None
    vlm_offline = False
    api_key_missing = False
    error_message = ""

    if _global_batch_processor is None:
        try:
            batch_orchestrator = get_orchestrator()
            _global_batch_processor = BatchProcessor(batch_orchestrator)
        except ValueError as e:
            # Catch configuration errors (e.g., missing API key for cloud mode)
            error_str = str(e)
            if "GEMINI_API_KEY" in error_str:
                logger.error(f"Cloud mode enabled but API key missing: {e}")
                api_key_missing = True
                error_message = error_str
            else:
                # Other ValueError - re-raise
                raise
        except ConnectionError as e:
            logger.warning(f"VLM backend offline: {e}")
            vlm_offline = True
            error_message = str(e)
            # Don't crash - we'll show a warning UI instead

    # Local references for convenience
    batch_processor = _global_batch_processor
    # Note: batch_ledger is accessed via _global_batch_ledger in functions below

    # Show error banner if VLM is unavailable or API key is missing
    if api_key_missing:
        # Specific banner for missing Gemini API key
        with ui.card().classes('w-full mb-4').style(
            f'background: {theme.THEME_LIGHT["error"]}22; '
            f'border: 2px solid {theme.THEME_LIGHT["error"]}; '
            f'border-radius: 8px; '
            f'padding: 16px;'
        ):
            with ui.row().classes('items-center gap-3'):
                ui.icon('cloud_off', size='32px').style(f'color: {theme.THEME_LIGHT["error"]};')
                with ui.column().classes('gap-1'):
                    ui.label('Gemini API Key Missing').style(
                        f'font-weight: 600; '
                        f'color: {theme.THEME_LIGHT["text"]};'
                    )
                    ui.label('Cloud mode is enabled in Strategy Lab, but GEMINI_API_KEY is not set.').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'font-size: 13px;'
                    )
                    ui.label('Solution: Add GEMINI_API_KEY="your-key-here" to the .env file, or switch to Local Mode in Strategy Lab.').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'font-size: 13px;'
                    )
                    ui.label('Get your API key at: https://aistudio.google.com/app/apikey').style(
                        f'color: {theme.THEME_LIGHT["indigo"]}; '
                        f'font-size: 12px; '
                        f'font-family: var(--prajnya-font-mono);'
                    )
    elif vlm_offline or batch_processor is None:
        # Original banner for VLM backend offline
        with ui.card().classes('w-full mb-4').style(
            f'background: {theme.THEME_LIGHT["warning"]}22; '
            f'border: 2px solid {theme.THEME_LIGHT["warning"]}; '
            f'border-radius: 8px; '
            f'padding: 16px;'
        ):
            with ui.row().classes('items-center gap-3'):
                ui.icon('warning', size='32px').style(f'color: {theme.THEME_LIGHT["warning"]};')
                with ui.column().classes('gap-1'):
                    ui.label('VLM Backend Offline').style(
                        f'font-weight: 600; '
                        f'color: {theme.THEME_LIGHT["text"]};'
                    )
                    ui.label('The llama.cpp VLM server is not running. Start it and refresh the page to enable batch processing.').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'font-size: 13px;'
                    )
                    ui.label('Expected at: http://localhost:8080/health').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'font-size: 12px; '
                        f'font-family: var(--prajnya-font-mono);'
                    )

    # --- Logic Handlers ---
    
    async def handle_upload(e):
        try:
            # 1. Robust Name Extraction
            if hasattr(e, 'file'):
                name = getattr(e.file, 'name', None) or getattr(e.file, 'filename', None)
            else:
                name = getattr(e, 'name', None)
            
            if not name:
                name = f"upload_{datetime.now().strftime('%H%M%S')}.pdf"

            # 2. Define staging path
            import sys
            if sys.platform == "win32":
                base_dir = os.environ.get("APPDATA", os.path.expanduser("~"))
                upload_dir = os.path.join(base_dir, "PrajnyaStudio", "uploads")
            else:
                upload_dir = os.path.join(os.path.expanduser("~"), ".local", "share", "PrajnyaStudio", "uploads")
            
            os.makedirs(upload_dir, exist_ok=True)
            dst_path = os.path.join(upload_dir, name)

            # 3. Save content (FIXED: Await the async read)
            with open(dst_path, 'wb') as f:
                # The error confirmed e.file.read() is a coroutine, so we must await it
                content = await e.file.read()
                f.write(content)
            
            uploaded_files.append(dst_path)
            file_list_label.text = f"{len(uploaded_files)} files ready"
            ui.notify(f"Saved: {name}", type='positive')
            
        except Exception as err:
            ui.notify(f"Upload failed: {str(err)}", type='negative')
            print(f"Upload Error: {err}")

    def clear_queue():
        """Clear all uploaded files and reset UI."""
        uploaded_files.clear()
        if batch_processor:
            batch_processor.clear_queue()
        
        file_list_label.text = "0 files ready"
        
        if 'upload' in locals() or 'upload' in globals():
             # This check is safe because clear_queue is a closure in mount()
             # where 'upload' is defined later. Python closures resolve nicely.
             try:
                 upload.reset()
             except:
                 pass
        
        ui.notify("Queue cleared", type='info', icon='delete')
        logger.info("User cleared the batch queue")
        
    async def start_batch_processing():
        global _global_batch_ledger

        # Validate inputs
        if not uploaded_files:
            ui.notify('No files uploaded! Please upload PDF files first.', type='warning')
            return

        # UI Updates
        btn_start.disable()
        btn_stop.enable()
        btn_stop.visible = True
        btn_kill.enable()
        btn_kill.visible = True
        spinner.visible = True
        progress_bar.value = 0.0
        status_label.text = "Initializing batch..."

        try:
            # Create new ledger for this batch (Global)
            _global_batch_ledger = BatchLedger()

            # Update UI
            batch_id_label.text = f"BatchID: {_global_batch_ledger.batch_id}"

            # Add all files to ledger as pending
            for file_path in uploaded_files:
                _global_batch_ledger.add_entry(file_path, status='pending')

            update_ledger_table()

            # Add files to batch processor queue
            batch_processor.clear_queue()
            batch_processor.add_files(uploaded_files)

            # Register callbacks to CURRENT UI elements
            # This is critical: refreshing the page breaks old links, we must re-link
            batch_processor.register_callback('on_file_start', on_file_start)
            batch_processor.register_callback('on_file_complete', on_file_complete)
            batch_processor.register_callback('on_file_error', on_file_error)
            batch_processor.register_callback('on_progress_update', on_progress_update)
            batch_processor.register_callback('on_batch_complete', on_batch_complete)

            ui.notify(f'Starting batch {_global_batch_ledger.batch_id}: {len(uploaded_files)} files', type='info')
            logger.info(f"🚀 Starting batch {_global_batch_ledger.batch_id}: {len(uploaded_files)} files")

            # Start processing
            summary = await batch_processor.start_processing()

            # Batch complete logic
            success_count = summary['success_files']
            error_count = summary['error_files']
            total_time = summary['processing_time']

            ui.notify(
                f'Batch complete! {success_count}/{len(uploaded_files)} successful ({error_count} errors) in {total_time:.1f}s',
                type='positive' if error_count == 0 else 'warning',
                icon='check_circle'
            )
            status_label.text = "Batch Complete"
            progress_bar.value = 1.0

        except Exception as e:
            ui.notify(f"Batch processing error: {str(e)}", type='negative')
            status_label.text = "Error"
            logger.error(f"Batch processing failed: {e}", exc_info=True)

        finally:
            btn_start.enable()
            btn_stop.disable()
            btn_stop.visible = False
            btn_kill.visible = False
            spinner.visible = False
    
    def stop_processing():
        batch_processor.stop_processing()
        btn_start.enable()
        btn_stop.disable()
        btn_stop.visible = False
        btn_kill.visible = False # Hide kill button too
        spinner.visible = False
        ui.notify('Stopping batch processing...', type='warning')
        logger.info("User requested batch processing stop")

    def kill_batch():
        """Phase 4.5: Hard Kill Switch (The Ejectable Engine)."""
        logger.critical("⚡ USER INITIATED HARD KILL!")
        batch_processor.kill_current_job()
        
        btn_start.enable()
        btn_stop.disable()
        btn_stop.visible = False
        btn_kill.visible = False
        spinner.visible = False
        
        ui.notify('⚡ ENGINE EJECTED - Batch Terminated Immediately', type='negative', icon='bolt')
        status_label.text = "ABORTED (Hard Kill)"

    def update_metadata_confidence(confidence: dict):
        """
        Phase 6.6f: Update Metadata Confidence card with real-time scores.

        Args:
            confidence: Dict with keys 'title', 'authors', 'abstract', 'publication', 'doi'
                       Values are floats 0.0-1.0
        """
        from gui.utils.confidence_utils import get_confidence_level, format_confidence_display

        for field_key, score in confidence.items():
            if field_key in confidence_labels:
                # Get level and formatted display
                level = get_confidence_level(score)
                display_text = format_confidence_display(score)

                # Get color based on level
                color_map = {
                    'High': theme.THEME_LIGHT['success'],      # Forest Green
                    'Medium': theme.THEME_LIGHT['warning'],    # Amber
                    'Low': theme.THEME_LIGHT['error'],         # Error Red
                    'Missing': theme.THEME_LIGHT['text_muted'] # Gray muted
                }
                color = color_map.get(level, theme.THEME_LIGHT['text_muted'])

                # Update UI element
                label = confidence_labels[field_key]
                label.text = display_text
                label.style(f'color: {color};')

    def update_ledger_table():
        """
        Phase 6.6g: Update Ledger Preview table with current batch data.

        Clears and recreates table rows from batch_ledger entries.
        """
        if not _global_batch_ledger:
            return

        ledger_table_container.clear()
        rows = _global_batch_ledger.get_table_rows()

        # Create new rows
        with ledger_table_container:
            for row in rows:
                filename = row['filename']
                status = row['status']
                prov = row['prov']

                # Determine status color
                if '✓ OK' in status:
                    status_color = theme.THEME_LIGHT['success']
                elif '△ ERR' in status:
                    status_color = theme.THEME_LIGHT['error']
                elif '↻ RUN' in status:
                    status_color = theme.THEME_LIGHT['warning']
                else:  # '○ WAIT'
                    status_color = theme.THEME_LIGHT['text_muted']

                with ui.row().classes('w-full items-center gap-2 mb-1'):
                    ui.label(filename).classes('text-xs flex-1').style(
                        f'color: {theme.THEME_LIGHT["text"]}; '
                        f'font-family: var(--prajnya-font-mono);'
                    )
                    ui.label(status).classes('text-xs').style(
                        f'color: {status_color}; '
                        f'font-weight: 500; '
                        f'width: 60px;'
                    )
                    ui.label(prov).classes('text-xs').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'width: 40px; '
                        f'text-align: center;'
                    )

    async def on_file_start(file_path: str):
        """
        Called when file processing starts.

        Updates:
        - Ledger entry status to 'processing'
        - UI status label
        - Ledger table display
        - Output log
        """
        file_name = Path(file_path).name
        status_label.text = f"Processing: {file_name}"
        if _global_batch_ledger:
            _global_batch_ledger.update_entry_status(file_path, 'processing')
            update_ledger_table()

        # Log
        logger.info(f"[START] {file_name}")

    async def on_file_complete(result: dict):
        """
        Called when file completes successfully.

        Updates:
        - Ledger entry with success status and metadata
        - Derives confidence scores
        - Output log with processing time
        """
        file_path = result.get('file_path', '')
        file_name = Path(file_path).name
        processing_time = result.get('processing_time', 0.0)

        # Extract metadata and confidence
        processing_result = result.get('processing_result')
        if processing_result and hasattr(processing_result, 'metadata'):
            metadata = processing_result.metadata

            # Derive per-field confidence scores
            field_confidence = derive_field_confidence(metadata)

            # Build metadata dict for ledger
            metadata_dict = {
                'title': metadata.title,
                'authors': metadata.authors,
                'doi': metadata.doi,
                'publisher': metadata.publisher,
                'year': metadata.year
            }

            # Determine provenance flag
            # Simple heuristic: if validation_confidence < 0.7, mark with △
            provenance = '△' if (metadata.validation_confidence or 0.0) < 0.7 else '-'

            if _global_batch_ledger:
                _global_batch_ledger.update_entry_status(
                    file_path, 'success', metadata=metadata_dict,
                    confidence=field_confidence, provenance=provenance
                )
            update_metadata_confidence(field_confidence)
        else:
            if _global_batch_ledger:
                _global_batch_ledger.update_entry_status(file_path, 'success')

        update_ledger_table()

        # Log
        logger.info(f"[DONE] {file_name} ({processing_time:.2f}s)")

    async def on_file_error(file_path: str, error: str):
        """
        Called when file processing fails.

        Updates:
        - Ledger entry with error status and message
        - Ledger table display
        - Output log
        """
        file_name = Path(file_path).name

        # Update ledger with error
        if _global_batch_ledger:
            _global_batch_ledger.update_entry_status(file_path, 'error', provenance='△', error=str(error))
            update_ledger_table()

        # Log
        logger.error(f"[ERROR] {file_name}: {error}")

    async def on_progress_update(progress: dict):
        """
        Called periodically with progress stats.

        Updates:
        - All 3 progress bars (current, error, total)
        - Percent label
        - Files processed badge
        - Error count label
        """
        # Extract progress data
        current_pct = progress.get('current_progress', 0.0)
        error_pct = progress.get('error_progress', 0.0)
        total_pct = progress.get('total_progress', 0.0)
        processed_files = progress.get('processed_files', 0)
        total_files = progress.get('total_files', 0)
        error_files = progress.get('error_files', 0)

        # Update progress bars (using UI element references)
        progress_bar.value = current_pct
        error_progress.value = error_pct
        total_progress.value = total_pct

        # Update labels
        percent_label.text = f"{int(current_pct * 100)}% Complete"
        files_loaded_badge.text = f"{processed_files}/{total_files} processed"
        error_label.text = f"Errors: {error_files}"

    async def on_batch_complete(summary: dict):
        """
        Called when entire batch finishes.
        """
        success = summary.get('success_files', 0)
        total = summary.get('total_files', 0)
        errors = summary.get('error_files', 0)
        processing_time = summary.get('processing_time', 0.0)

        # Export ledger if available
        if _global_batch_ledger:
            try:
                custom_output = folder_input.value.strip() if folder_input.value else None
                if custom_output:
                    output_dir = Path(custom_output) / "batch_results"
                else:
                    output_dir = Path("run_outputs/batch_results")

                output_dir.mkdir(parents=True, exist_ok=True)
                ledger_path = output_dir / f"batch_ledger_{_global_batch_ledger.batch_id}.json"
                _global_batch_ledger.export_json(str(ledger_path))

                logger.info(f"📊 Ledger exported: {ledger_path}")
                safe_notify(f'Ledger saved to: {output_dir}', type='info', icon='folder')

            except Exception as e:
                logger.error(f"Failed to export ledger: {e}")
                safe_notify(f'Ledger export failed: {str(e)}', type='negative')

        # Log summary
        logger.info(
            f"[COMPLETE] Batch finished: {success}/{total} successful, "
            f"{errors} errors in {processing_time:.1f}s"
        )

    def export_logs():
        """Export all logs to a timestamped file."""
        try:
            import logging
            from pathlib import Path
            import copy

            # Get all log entries from the global handler
            if not prajnya_logger.GLOBAL_LOG_HANDLER:
                ui.notify('No logs available to export', type='warning')
                return

            # Use history deque instead of stream deque
            # The stream deque (.deque) is cleared by terminal polling
            # The history deque (.history) is never cleared and retains full session history
            try:
                all_entries = prajnya_logger.GLOBAL_LOG_HANDLER.get_history()
            except RuntimeError as deque_err:
                # If deque is still mutating, wait a moment and retry once
                import time
                time.sleep(0.1)
                all_entries = prajnya_logger.GLOBAL_LOG_HANDLER.get_history()

            if not all_entries:
                ui.notify('No logs to export', type='info')
                return

            # Create export directory in user-accessible location
            # Using run_outputs/logs (relative to execution directory)
            export_dir = Path("run_outputs/logs")
            export_dir.mkdir(parents=True, exist_ok=True)

            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = export_dir / f"prajnya_studio_log_{timestamp}.txt"

            # Write logs to file
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(f"Prajnya Studio Log Export\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Entries: {len(all_entries)}\n")
                f.write("=" * 80 + "\n\n")

                for entry in all_entries:
                    ts = entry.get('ts', 'N/A')
                    level = entry.get('level', 'INFO')
                    msg = entry.get('msg', '')
                    f.write(f"[{ts}] [{level:8s}] {msg}\n")

            ui.notify(f'Logs exported to {export_path.name}', type='positive', icon='check_circle')
            logging.info(f"Exported {len(all_entries)} log entries to {export_path}")

        except Exception as e:
            ui.notify(f'Export failed: {str(e)}', type='negative', icon='error')
            logging.error(f"Log export failed: {e}", exc_info=True)

    def change_log_level(level_name: str):
        """Change the logging level for the root logger."""
        try:
            import logging

            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR
            }

            if level_name in level_map:
                logging.root.setLevel(level_map[level_name])
                ui.notify(f'Log level set to {level_name}', type='info', icon='info')
                logging.info(f"Log level changed to {level_name}")

        except Exception as e:
            ui.notify(f'Failed to change log level: {str(e)}', type='negative')
            logging.error(f"Log level change failed: {e}")

    # --- UI Layout (v2.0 - Design Bible Compliant) ---

    # Removed absolute positioning block - moved to sticky header below
    pass

    # Header Section with golden underline
    # --- Sticky Action Header (Starts at Top) ---
    with ui.row().classes('w-full items-center justify-between sticky top-0 z-50 py-4 px-6 mb-6').style(
        f'background: {theme.THEME_LIGHT["canvas"]}; '
        f'border-bottom: 2px solid {theme.THEME_LIGHT["gold"]};'
    ):
        # LEFT: Title and Description
        with ui.column().classes('gap-1'):
            ui.label('Batch Commander').classes('text-3xl').style(
                f'font-family: var(--prajnya-font-display); '
                f'font-weight: 600; '
                f'color: {theme.THEME_LIGHT["indigo"]}; '
                f'font-variant: small-caps;'
            )
            ui.label('Process multiple PDF files with custom parameters.').classes('text-sm').style(
                f'color: {theme.THEME_LIGHT["text_muted"]}; '
                f'font-family: var(--prajnya-font-body);'
            )

        # RIGHT: Action Buttons (Moved from absolute position)
        with ui.row().classes('gap-4 items-center'):
            # Spinner (Visible when processing)
            spinner = ui.spinner(size='2em').props('color=red')
            spinner.visible = False

            # Start Button (Primary Action)
            btn_start = ui.button('▶ Start Batch', on_click=start_batch_processing).style(
                f'background: {theme.THEME_LIGHT["vermillion"]} !important; '
                f'color: white !important; '
                f'padding: 10px 24px; '
                f'font-weight: 600; '
                f'text-transform: uppercase; '
                f'font-size: 13px; '
                f'border-radius: 8px; '
                f'box-shadow: 0 4px 12px rgba(184, 71, 42, 0.3);'
            )
            
            # Stop/Kill row (hidden initially)
            with ui.row().classes('gap-2'):
                btn_stop = ui.button('■ STOP', on_click=stop_processing).props('flat').style(
                    f'background: {theme.THEME_LIGHT["warning"]} !important; '
                    f'color: white !important; '
                    f'padding: 8px 20px; '
                    f'font-weight: 600; '
                    f'font-size: 12px; '
                    f'border-radius: 6px;'
                )
                btn_stop.disable()
                btn_stop.visible = False
                
                btn_kill = ui.button('⚡ KILL', on_click=kill_batch).props('flat').style(
                    f'background: {theme.THEME_LIGHT["error"]} !important; '
                    f'color: white !important; '
                    f'padding: 8px 16px; '
                    f'font-weight: 800; '
                    f'font-size: 12px; '
                    f'border-radius: 6px;'
                )
                btn_kill.disable()
                btn_kill.visible = False

    with ui.row().classes('w-full h-full gap-6 items-stretch'):

        # --- LEFT COLUMN: Controls (~55% width) ---
        with ui.column().classes('gap-4 flex-1').style('max-width: 55%;'):

            # Top Row: Upload Files + Set Output Destination (side-by-side)
            with ui.row().classes('w-full gap-4 items-stretch'):

                # Upload Files Card (LEFT - 50%)
                with ui.card().classes('flex-1').style(
                    f'background: {theme.THEME_LIGHT["card"]}; '
                    f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
                    f'border-radius: 8px; '
                    f'padding: 16px;'
                ):
                    # Section Header with golden underline
                    ui.label('UPLOAD FILES').classes('text-sm mb-2').style(
                        f'font-family: var(--prajnya-font-body); '
                        f'font-weight: 600; '
                        f'color: {theme.THEME_LIGHT["indigo"]}; '
                        f'text-transform: uppercase; '
                        f'letter-spacing: 0.5px;'
                    )
                    # Golden underline (Design Bible)
                    ui.element('div').classes('w-full mb-3').style(
                        f'height: 2px; '
                        f'background: {theme.THEME_LIGHT["gold"]};'
                    )

                    # Hidden upload element with unique ID
                    upload = ui.upload(on_upload=handle_upload, multiple=True, auto_upload=True).props('flat').style('display: none;')
                    upload._props['id'] = 'batch-file-upload'

                    # Clean Drop Zone with dashed gold border
                    drop_zone = ui.column().classes('w-full items-center justify-center gap-3').style(
                        f'border: 2px dashed {theme.THEME_LIGHT["border_gold"]}; '
                        f'border-radius: 8px; '
                        f'padding: 32px 16px; '
                        f'background: transparent; '
                        f'min-height: 160px; '
                        f'position: relative; '
                        f'cursor: pointer;'
                    )

                    # Click handler to trigger file picker
                    drop_zone.on('click', lambda: ui.run_javascript(
                        'document.getElementById("batch-file-upload").querySelector("input[type=file]").click()'
                    ))

                    with drop_zone:
                        ui.icon('upload_file', size='48px').style(f'color: {theme.THEME_LIGHT["gold"]};')
                        ui.label('Drag & Drop PDFs').classes('text-sm font-medium').style(
                            f'color: {theme.THEME_LIGHT["text"]}; '
                            f'font-family: var(--prajnya-font-body);'
                        )
                        ui.label('or click to browse').classes('text-xs').style(
                            f'color: {theme.THEME_LIGHT["text_muted"]}; '
                            f'font-family: var(--prajnya-font-body);'
                        )

                        # Browse Files Button (Indigo - Design Bible)
                        ui.button('Browse Files', icon='folder_open').style(
                            f'background: {theme.THEME_LIGHT["indigo"]} !important; '
                            f'color: white !important; '
                            f'margin-top: 8px;'
                        )

                        # Clear Queue Button (Subtle text button)
                        ui.button('Clear Queue', icon='delete', on_click=clear_queue).props('flat dense size=sm').style(
                            f'color: {theme.THEME_LIGHT["text_muted"]} !important; '
                            f'margin-top: 4px;'
                        )

                    file_list_label = ui.label('0 files ready').classes('text-sm').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'margin-top: 8px; '
                        f'font-family: var(--prajnya-font-mono);'
                    )

                # Set Output Destination Card (RIGHT - 50%)
                with ui.card().classes('flex-1').style(
                    f'background: {theme.THEME_LIGHT["card"]}; '
                    f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
                    f'border-radius: 8px; '
                    f'padding: 16px;'
                ):
                    ui.label('SET OUTPUT DESTINATION').classes('text-sm mb-2').style(
                        f'font-family: var(--prajnya-font-body); '
                        f'font-weight: 600; '
                        f'color: {theme.THEME_LIGHT["indigo"]}; '
                        f'text-transform: uppercase; '
                        f'letter-spacing: 0.5px;'
                    )
                    # Golden underline (Design Bible)
                    ui.element('div').classes('w-full mb-3').style(
                        f'height: 2px; '
                        f'background: {theme.THEME_LIGHT["gold"]};'
                    )

                    ui.label('Batch Folder Path').classes('text-xs mb-2').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'font-family: var(--prajnya-font-body);'
                    )
                    folder_input = ui.input(
                        value=state.settings.get('default_batch_folder'),
                        placeholder='e.g., D:/Documents/Research/Output or leave empty for default'
                    ).props('outlined dense').classes('w-full')

                    # Help text
                    ui.label('Enter custom output path or leave empty to use default (run_outputs/batch_results)').classes('text-xs mt-1').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'font-family: var(--prajnya-font-body); '
                        f'font-style: italic;'
                    )

                    # Switches moved below
                    with ui.row().classes('w-full justify-between gap-2 mt-4'):
                        rec_switch = ui.switch('Recursive', value=state.settings.get('recursive_default'))
                        rev_switch = ui.switch('Skip Reviewed', value=state.settings.get('skip_review_default'))

            # Batch Progress Section (3 Progress Bars) - GROWS to fill space
            with ui.card().classes('w-full flex-1').style(
                f'background: {theme.THEME_LIGHT["card"]}; '
                f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
                f'border-radius: 8px; '
                f'padding: 16px;'
            ):
                # Header row with count badge
                with ui.row().classes('w-full items-center justify-between mb-2'):
                    ui.label('BATCH PROGRESS').classes('text-sm').style(
                        f'font-family: var(--prajnya-font-body); '
                        f'font-weight: 600; '
                        f'color: {theme.THEME_LIGHT["indigo"]}; '
                        f'text-transform: uppercase; '
                        f'letter-spacing: 0.5px;'
                    )
                    files_loaded_badge = ui.label('0 files loaded').classes('text-xs').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'font-family: var(--prajnya-font-mono);'
                    )

                # Golden underline (Design Bible)
                ui.element('div').classes('w-full mb-3').style(
                    f'height: 2px; '
                    f'background: {theme.THEME_LIGHT["gold"]};'
                )

                # Processing status line
                with ui.row().classes('w-full items-center gap-2 mb-3'):
                    ui.icon('sync', size='16px').style(f'color: {theme.THEME_LIGHT["indigo"]};')
                    status_label = ui.label('Processing: waiting...').classes('text-sm').style(
                        f'color: {theme.THEME_LIGHT["text"]}; '
                        f'font-family: var(--prajnya-font-body);'
                    )
                    ui.space()
                    percent_label = ui.label('0% Complete').classes('text-sm font-mono').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]};'
                    )

                # Progress Bar 1: Current File (Indigo - Structure)
                progress_bar = ui.linear_progress(value=0.0).props('stripe').style(
                    f'color: {theme.THEME_LIGHT["indigo"]}; '  # Sets the bar color
                    'height: 10px; '
                    'border-radius: 4px; '
                    '--q-linear-progress-stripe-color: rgba(255, 255, 255, 0.2);'
                ).classes('mb-2')

                # Progress Bar 2: Errors (Vermillion)
                with ui.row().classes('w-full items-center gap-2 mb-2'):
                    error_label = ui.label('Errors: 0').classes('text-xs').style(
                        f'color: {theme.THEME_LIGHT["error"]}; '
                        f'width: 100px;'
                    )
                    error_progress = ui.linear_progress(value=0.0).props('color=negative').style(
                        'height: 8px; '
                        'flex: 1;'
                    )

                # Progress Bar 3: Batch Complete (Gold/Amber)
                with ui.row().classes('w-full items-center gap-2'):
                    ui.label('Batch Complete').classes('text-xs').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'width: 100px;'
                    )
                    total_progress = ui.linear_progress(value=0.0).props('color=warning').style(
                        'height: 8px; '
                        'flex: 1;'
                    )

            # (Buttons moved to top-right action panel)

        # --- RIGHT COLUMN: Terminal + Future Panels ---
        with ui.column().classes('flex-1 gap-4'):

            # Output Log (Terminal) - Card-based
            with ui.card().classes('w-full').style(
                f'background: {theme.THEME_LIGHT["card"]}; '
                f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
                f'border-radius: 8px; '
                f'padding: 0; '
                f'overflow: hidden;'
            ):
                # Card Header (consistent with other cards - no shaded background)
                with ui.row().classes('w-full items-center gap-3').style(
                    f'padding: 16px 16px 8px 16px;'  # Match other cards
                ):
                    ui.label('OUTPUT LOG').classes('text-sm').style(
                        f'font-family: var(--prajnya-font-body); '
                        f'font-weight: 600; '
                        f'color: {theme.THEME_LIGHT["indigo"]}; '
                        f'text-transform: uppercase; '
                        f'letter-spacing: 0.5px;'
                    )

                    ui.label('PROV/VAR').classes('text-xs').style(
                        f'background: {theme.THEME_LIGHT["warning"]}; '
                        f'color: white; '
                        f'padding: 2px 8px; '
                        f'border-radius: 4px; '
                        f'font-family: var(--prajnya-font-mono); '
                        f'font-weight: 600;'
                    )

                # Golden underline (Design Bible - aligned with other cards)
                ui.element('div').classes('w-full').style(
                    f'height: 2px; '
                    f'background: {theme.THEME_LIGHT["gold"]};'
                )
                
                # Controls row (below golden line)
                with ui.row().classes('w-full items-center gap-3').style(
                    f'padding: 8px 16px; '
                    f'background: {theme.THEME_LIGHT["card"]};'
                ):
                    # Log Level Control
                    log_level_select = ui.select(
                        {
                            'DEBUG': 'DEBUG',
                            'INFO': 'INFO',
                            'WARNING': 'WARNING',
                            'ERROR': 'ERROR'
                        },
                        value='INFO',
                        label='Level',
                        on_change=lambda e: change_log_level(e.value)
                    ).classes('w-24').props('dense outlined').style(
                        f'color: {theme.THEME_LIGHT["text"]};'
                    )

                    ui.space()

                    # Export Log Button
                    ui.button('Export', icon='download', on_click=lambda: export_logs())\
                        .props('dense flat size=sm').style(f'color: {theme.THEME_LIGHT["indigo"]} !important;')

                # Collapsible Terminal (Accordion style for space management)
                with ui.expansion('Show Terminal', icon='terminal').props('dense default-opened').classes('w-full').style(
                    'background: transparent; border: none;'
                ):
                    # Terminal (Dark, as designed) - REDUCED HEIGHT to fit all 3 panels
                    with ui.card().classes('w-full').style(
                        f'background: {theme.THEME_LIGHT["terminal_bg"]}; '
                        f'border-radius: 0; '
                        f'padding: 0; '
                        f'height: 220px; '
                        f'display: flex; '
                        f'flex-direction: column;'
                    ):
                        # Mac-style Header
                        with ui.row().classes('w-full items-center gap-2').style(
                            'background: #0D0D0D; '
                            'padding: 8px 12px;'
                        ):
                            ui.element('div').style('width:12px; height:12px; border-radius:50%; background:#FF5F56')
                            ui.element('div').style('width:12px; height:12px; border-radius:50%; background:#FFBD2E')
                            ui.element('div').style('width:12px; height:12px; border-radius:50%; background:#27C93F')
                            ui.space()
                            ui.label('process_stream.log').classes('text-xs text-gray-500 font-mono')

                        # Real Log Stream
                        log_stream = LogStream()

            # Metadata Confidence Panel (Color-coded with edit icons)
            with ui.card().classes('w-full').style(
                f'background: {theme.THEME_LIGHT["card"]}; '
                f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
                f'border-radius: 8px; '
                f'padding: 16px;'
            ):
                ui.label('METADATA CONFIDENCE').classes('text-sm mb-2').style(
                    f'font-family: var(--prajnya-font-body); '
                    f'font-weight: 600; '
                    f'color: {theme.THEME_LIGHT["indigo"]}; '
                    f'text-transform: uppercase; '
                    f'letter-spacing: 0.5px;'
                )
                # Golden underline (Design Bible)
                ui.element('div').classes('w-full mb-3').style(
                    f'height: 2px; '
                    f'background: {theme.THEME_LIGHT["gold"]};'
                )
                confidence_labels = {}

                # Define metadata fields
                metadata_fields = [
                    ('✱', 'Title', 'title'),
                    ('‖', 'Author(s)', 'authors'),
                    ('≡', 'Abstract', 'abstract'),
                    ('▢', 'Publication', 'publication'),
                    ('⌀', 'DOI', 'doi'),
                ]

                for icon, field_name, field_key in metadata_fields:
                    with ui.row().classes('w-full items-center gap-2'):
                        ui.label(icon).style(f'color: {theme.THEME_LIGHT["gold"]}; width: 16px; font-size: 14px;')
                        ui.label(f'{field_name}:').classes('text-xs').style(f'color: {theme.THEME_LIGHT["text"]}; width: 70px;')

                        # Store reference to confidence label for updates
                        confidence_labels[field_key] = ui.label('Pending...').classes('text-xs font-medium flex-1').style(
                            f'color: {theme.THEME_LIGHT["text_muted"]};'
                        )

                        # Edit action icon (right side)
                        ui.icon('edit', size='14px').classes('cursor-pointer').style(
                            f'color: {theme.THEME_LIGHT["text_muted"]}; '
                            f'opacity: 0.5;'
                        )

            # Ledger Preview Panel - GROWS to fill space
            with ui.card().classes('w-full flex-1').style(
                f'background: {theme.THEME_LIGHT["card"]}; '
                f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
                f'border-radius: 8px; '
                f'padding: 16px;'
            ):
                # Header with BatchID badge
                with ui.row().classes('w-full items-center justify-between mb-2'):
                    ui.label('LEDGER PREVIEW').classes('text-sm').style(
                        f'font-family: var(--prajnya-font-body); '
                        f'font-weight: 600; '
                        f'color: {theme.THEME_LIGHT["indigo"]}; '
                        f'text-transform: uppercase; '
                        f'letter-spacing: 0.5px;'
                    )
                    batch_id_label = ui.label('BatchID: Pending').classes('text-xs font-mono').style(
                        f'background: {theme.THEME_LIGHT["sidebar"]}; '
                        f'padding: 2px 8px; '
                        f'border-radius: 4px; '
                        f'color: {theme.THEME_LIGHT["text_muted"]};'
                    )

                # Golden underline (Design Bible)
                ui.element('div').classes('w-full mb-3').style(
                    f'height: 2px; '
                    f'background: {theme.THEME_LIGHT["gold"]};'
                )

                # Table header
                with ui.row().classes('w-full items-center gap-2 mb-2 pb-2').style(
                    f'border-bottom: 1px solid {theme.THEME_LIGHT["border"]};'
                ):
                    ui.label('File Name').classes('text-xs flex-1').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'font-weight: 600;'
                    )
                    ui.label('Status').classes('text-xs').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'font-weight: 600; '
                        f'width: 60px;'
                    )
                    ui.label('Prov.').classes('text-xs').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'font-weight: 600; '
                        f'width: 40px;'
                    )
                ledger_table_container = ui.column().classes('w-full')
                
    # --- CRITICAL: State Rehydration (At the very end of mount) ---
    
    # Restore file list from global processor
    if batch_processor.files_queue:
        uploaded_files.extend(batch_processor.files_queue)
        file_list_label.text = f"{len(uploaded_files)} files ready"

    # If processor has activity (running OR finished), restore the UI state
    if batch_processor.status != 'idle':
        
        # 1. Restore Stats & Progress Visuals
        stats = batch_processor.get_progress()
        
        import asyncio
        asyncio.create_task(on_progress_update(stats))

        # 2. Restore Ledger Table
        if _global_batch_ledger:
            batch_id_label.text = f"BatchID: {_global_batch_ledger.batch_id}"
            update_ledger_table()

        # 3. Handle specific states
        if batch_processor.status == 'processing':
            # Active: Disable controls, show spinner
            btn_start.disable()
            btn_stop.enable()
            btn_stop.visible = True
            spinner.visible = True
            status_label.text = f"Processing: {stats.get('current_file_name', 'Resuming...')}"
            
            # RE-REGISTER CALLBACKS (Critical for active batches)
            batch_processor.register_callback('on_file_start', on_file_start)
            batch_processor.register_callback('on_file_complete', on_file_complete)
            batch_processor.register_callback('on_file_error', on_file_error)
            batch_processor.register_callback('on_progress_update', on_progress_update)
            batch_processor.register_callback('on_batch_complete', on_batch_complete)
            
            safe_notify('Batch monitor reconnected', type='positive', icon='wifi')

        elif batch_processor.status == 'completed':
            # Finished: Show 100%, Enable start
            btn_start.enable()
            btn_stop.visible = False
            spinner.visible = False
            status_label.text = "Batch Complete"
            progress_bar.value = 1.0
            percent_label.text = "100% Complete"