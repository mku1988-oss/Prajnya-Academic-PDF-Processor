"""
Ledger Master - Advanced ledger management interface (Sprint 4 MVP)

Features:
- Search & filter panel (search text, review status, confidence)
- Data table with sortable columns and pagination
- CSV export functionality

MVP Scope (Phase 1):
- Real-time search (title, author, DOI)
- Review status filter (All/Not Reviewed/Reviewed)
- Confidence threshold slider
- Result count display
"""

from nicegui import ui
from pathlib import Path
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import csv
import io

# Import ledger backend
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.processing.ledger.csv_backend import CsvLedgerBackend
from src.processing.ledger.semantic_ledger import LedgerRecord
from .. import theme

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

LEDGER_CSV_PATH = Path("persistence/semantic_ledger.csv")
ENTRIES_PER_PAGE = 20

#  ============================================================================
# STATE MANAGEMENT
# ============================================================================

class LedgerMasterState:
    """State management for Ledger Master page."""

    def __init__(self):
        self.ledger: Optional[CsvLedgerBackend] = None
        self.all_entries: List[LedgerRecord] = []
        self.filtered_entries: List[LedgerRecord] = []
        self.displayed_entries: List[LedgerRecord] = []

        # Filter state
        self.search_query: str = ""
        self.review_status_filter: str = "all"  # all, not_reviewed, reviewed
        self.confidence_threshold: float = 0.0  # 0.0 to 1.0

        # Pagination state
        self.current_page: int = 1
        self.total_pages: int = 1

        # Sort state
        self.sort_column: str = "last_updated"  # default sort by date
        self.sort_ascending: bool = False  # newest first

        # Selection state (for checkboxes)
        self.selected_entries: set = set()  # Set of document_ids
        self.select_all_value: bool = False

        # UI elements (for reactive updates)
        self.result_count_label: Optional[ui.label] = None
        self.table_container: Optional[ui.column] = None
        self.pagination_container: Optional[ui.row] = None
        self.batch_actions_container: Optional[ui.column] = None
        self.dialog_container: Optional[ui.element] = None

        # Debounce task for search
        self.search_task: Optional[asyncio.Task] = None

    def load_ledger(self):
        """Load all records from CSV ledger."""
        try:
            self.ledger = CsvLedgerBackend(str(LEDGER_CSV_PATH))
            self.all_entries = self.ledger.get_all_records()
            logger.info(f"Loaded {len(self.all_entries)} entries from ledger")

            # Initial filter (show all)
            self.apply_filters()

        except Exception as e:
            logger.error(f"Failed to load ledger: {e}")
            ui.notify(f"Failed to load ledger: {e}", type='negative')
            self.all_entries = []
            self.filtered_entries = []

    def apply_filters(self):
        """Apply search and filter criteria to all entries."""
        results = self.all_entries.copy()

        # Apply search query (title, authors, DOI)
        if self.search_query.strip():
            query = self.search_query.lower().strip()
            results = [
                entry for entry in results
                if (
                    (entry.title and query in entry.title.lower()) or
                    (entry.doi and query in entry.doi.lower()) or
                    any(query in author.lower() for author in entry.authors)
                )
            ]

        # Apply review status filter
        if self.review_status_filter == "not_reviewed":
            results = [e for e in results if e.review_status in ("not_reviewed", "provisional")]
        elif self.review_status_filter == "reviewed":
            results = [e for e in results if e.review_status == "reviewed"]
        # "all" = no filtering

        # Apply confidence threshold
        results = [
            e for e in results
            if e.validation_confidence >= self.confidence_threshold
        ]

        self.filtered_entries = results
        self.current_page = 1  # Reset to first page
        self.update_pagination()

    def update_pagination(self):
        """Calculate pagination and update displayed entries."""
        total_entries = len(self.filtered_entries)
        self.total_pages = max(1, (total_entries + ENTRIES_PER_PAGE - 1) // ENTRIES_PER_PAGE)

        # Ensure current page is valid
        if self.current_page > self.total_pages:
            self.current_page = self.total_pages

        # Get entries for current page
        start_idx = (self.current_page - 1) * ENTRIES_PER_PAGE
        end_idx = start_idx + ENTRIES_PER_PAGE
        self.displayed_entries = self.filtered_entries[start_idx:end_idx]

    def sort_entries(self, column: str):
        """Sort filtered entries by column."""
        if self.sort_column == column:
            # Toggle sort direction
            self.sort_ascending = not self.sort_ascending
        else:
            # New column, default to ascending (except date)
            self.sort_column = column
            self.sort_ascending = (column != "last_updated")

        # Sort logic
        if column == "title":
            self.filtered_entries.sort(
                key=lambda e: (e.title or "").lower(),
                reverse=not self.sort_ascending
            )
        elif column == "authors":
            self.filtered_entries.sort(
                key=lambda e: (e.authors[0] if e.authors else "").lower(),
                reverse=not self.sort_ascending
            )
        elif column == "doi":
            self.filtered_entries.sort(
                key=lambda e: (e.doi or "").lower(),
                reverse=not self.sort_ascending
            )
        elif column == "review_status":
            # Sort order: not_reviewed -> provisional -> reviewed
            status_order = {"not_reviewed": 0, "provisional": 1, "reviewed": 2}
            self.filtered_entries.sort(
                key=lambda e: status_order.get(e.review_status, 0),
                reverse=not self.sort_ascending
            )
        elif column == "confidence":
            self.filtered_entries.sort(
                key=lambda e: e.validation_confidence,
                reverse=not self.sort_ascending
            )
        elif column == "last_updated":
            self.filtered_entries.sort(
                key=lambda e: e.last_updated,
                reverse=not self.sort_ascending
            )

        self.update_pagination()

# Global state instance
state = LedgerMasterState()

# ============================================================================
# UI COMPONENTS
# ============================================================================

def create_search_bar():
    """Create search bar with debounced input."""

    async def on_search_change(e):
        """Handle search input with 300ms debounce."""
        # Cancel previous search task
        if state.search_task:
            state.search_task.cancel()

        # Update query - get value from the args
        new_value = e.args if e.args is not None else ""
        state.search_query = new_value

        # Create new debounced task
        async def debounced_search():
            await asyncio.sleep(0.3)  # 300ms debounce
            state.apply_filters()
            refresh_ui()

        state.search_task = asyncio.create_task(debounced_search())

    with ui.row().classes('w-full items-center gap-4 mb-4'):
        ui.icon('search', size='md').classes('text-gray-400')
        search_input = ui.input(
            placeholder='Search by title, author, or DOI...'
        ).classes('flex-1').props('outlined dense')
        search_input.on('update:model-value', on_search_change)


def create_filter_panel():
    """Create filter controls (review status + confidence)."""

    # FIXED: Explicit background color and border
    with ui.card().classes('w-full mb-4').style(
        f'background-color: {theme.THEME_LIGHT["card"]} !important; '
        f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
        f'border-radius: 8px; padding: 16px;'
    ):
        ui.label('Filters').classes('text-lg font-bold mb-2').style(f'color: {theme.THEME_LIGHT["indigo"]};')

        with ui.row().classes('w-full items-center gap-8'):
            # Review Status Filter
            with ui.column().classes('gap-1'):
                ui.label('Review Status').classes('text-sm font-semibold text-gray-700')

                def on_status_change(e):
                    state.review_status_filter = e.value
                    state.apply_filters()
                    refresh_ui()

                ui.radio(
                    ['all', 'not_reviewed', 'reviewed'],
                    value='all',
                    on_change=on_status_change
                ).props('inline').bind_value_to(state, 'review_status_filter')

            # Confidence Threshold Slider
            with ui.column().classes('flex-1 gap-1'):
                confidence_label = ui.label(
                    f'Confidence Threshold: ≥ {state.confidence_threshold:.0%}'
                ).classes('text-sm font-semibold text-gray-700')

                def on_confidence_change(e):
                    state.confidence_threshold = e.value
                    confidence_label.text = f'Confidence Threshold: ≥ {e.value:.0%}'
                    state.apply_filters()
                    refresh_ui()

                ui.slider(
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    value=0.0,
                    on_change=on_confidence_change
                ).props('label-always').classes('w-full')

def create_result_count():
    """Create result count display."""
    total = len(state.all_entries)
    filtered = len(state.filtered_entries)

    state.result_count_label = ui.label(
        f'Showing {filtered} of {total} entries'
    ).classes('text-sm text-gray-600 mb-4')


def create_data_table():
    """Create data table with sortable columns."""

    with ui.column().classes('w-full') as container:
        state.table_container = container

        if not state.displayed_entries:
            # Empty state
            with ui.card().classes('w-full p-12 items-center justify-center').style(f'background: {theme.THEME_LIGHT["sidebar"]};'):
                ui.icon('inbox', size='4em').classes('text-gray-300 mb-4')
                if len(state.all_entries) == 0:
                    ui.label('No entries in ledger').classes('text-xl font-bold text-gray-400')
                    ui.label('Process some PDFs to see them here').classes('text-sm text-gray-500 mt-2')
                else:
                    ui.label('No results match filters').classes('text-xl font-bold text-gray-400')
                    ui.label('Try adjusting your search or filters').classes('text-sm text-gray-500 mt-2')
        else:
            # Data table with sortable columns - wrapped in scrollable container
            with ui.element('div').classes('w-full overflow-x-auto').style('max-width: 100%;'):
                with ui.column().classes('min-w-max'):
                    create_table_header()
                    create_table_rows()


def create_table_header():
    """Create sortable table header."""

    def sort_icon(column: str) -> str:
        """Get sort indicator for column."""
        if state.sort_column == column:
            return '↓' if not state.sort_ascending else '↑'
        return ''

    def on_column_click(column: str):
        """Handle column header click for sorting."""
        state.sort_entries(column)
        refresh_ui()

    def on_select_all_change(e):
        """Handle select all checkbox change."""
        state.select_all_value = e.value
        if e.value:
            for entry in state.displayed_entries:
                state.selected_entries.add(entry.document_id)
        else:
            state.selected_entries.clear()
        refresh_ui()

    # FIXED: Sidebar Cream background + Gold Border
    # Header row
    with ui.row().classes('items-center gap-2 p-3 rounded-t-lg border-b-2').style(
        f'background: {theme.THEME_LIGHT["sidebar"]}; '
        f'border-color: {theme.THEME_LIGHT["border_gold"]}; '
        f'min-width: 1260px;'
    ):
        # Checkbox column
        with ui.element('div').style('width: 40px; flex-shrink: 0;'):
            ui.checkbox(value=state.select_all_value, on_change=on_select_all_change)

        # Helper for headers - FIXED to remove Blue
        def header_btn(text, col):
            # We use a clickable div instead of a button to avoid Quasar color override fights
            with ui.element('div').classes('cursor-pointer flex items-center').style(
                f'width: 100%; height: 100%; color: {theme.THEME_LIGHT["indigo"]}; font-weight: 700; font-size: 0.875rem;'
            ).on('click', lambda: on_column_click(col)):
                ui.label(f'{text} {sort_icon(col)}')

        # Columns
        with ui.element('div').style('width: 400px; flex-shrink: 0;'): header_btn('Title', 'title')
        with ui.element('div').style('width: 220px; flex-shrink: 0;'): header_btn('Authors', 'authors')
        with ui.element('div').style('width: 180px; flex-shrink: 0;'): header_btn('DOI', 'doi')
        with ui.element('div').style('width: 120px; flex-shrink: 0;'): header_btn('Status', 'review_status')
        with ui.element('div').style('width: 100px; flex-shrink: 0;'): header_btn('Conf.', 'confidence')
        with ui.element('div').style('width: 140px; flex-shrink: 0;'): header_btn('Date', 'last_updated')

        # Actions column (Not clickable)
        with ui.element('div').style('width: 60px; flex-shrink: 0; text-align: center;'):
            ui.label('Actions').style(f'color: {theme.THEME_LIGHT["indigo"]}; font-weight: 700; font-size: 0.875rem;')


def create_table_rows():
    """Create table rows for displayed entries."""

    for idx, entry in enumerate(state.displayed_entries):
        bg_style = 'background: transparent;' if idx % 2 == 0 else 'background: rgba(44, 62, 80, 0.03);'
        
        with ui.row().classes('items-center gap-2 p-3 border-b').style(
            f'{bg_style} border-color: {theme.THEME_LIGHT["border"]}; min-width: 1260px; min-height: 48px;'
        ):

            # Checkbox - 40px
            with ui.element('div').style('width: 40px; flex-shrink: 0;'):
                # Create closure to capture current entry
                def make_checkbox_handler(entry_id):
                    def handler(e):
                        if e.value:
                            state.selected_entries.add(entry_id)
                        else:
                            state.selected_entries.discard(entry_id)
                            state.select_all_value = False  # Uncheck select all if any row unchecked
                        # CRITICAL: Refresh UI to show/hide batch actions
                        refresh_ui()
                    return handler

                is_selected = entry.document_id in state.selected_entries
                ui.checkbox(value=is_selected, on_change=make_checkbox_handler(entry.document_id))

            # Title - 400px, no wrapping
            with ui.element('div').style('width: 400px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'):
                title_text = entry.title or 'Untitled'
                ui.label(title_text).classes('text-sm text-gray-800').props('title="{}"'.format(title_text))

            # Authors - 220px, no wrapping
            with ui.element('div').style('width: 220px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'):
                if entry.authors and len(entry.authors) > 0:
                    author_text = entry.authors[0]
                    if len(entry.authors) > 1:
                        author_text += f' et al. ({len(entry.authors)})'
                else:
                    author_text = 'Unknown'
                ui.label(author_text).classes('text-sm text-gray-600').props('title="{}"'.format(author_text))

            # DOI - 180px, no wrapping
            with ui.element('div').style('width: 180px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'):
                doi_text = entry.doi or 'N/A'
                ui.label(doi_text).classes('text-xs text-gray-500 font-mono').props('title="{}"'.format(doi_text))

            # Review Status - 120px, no wrapping
            with ui.element('div').style('width: 120px; flex-shrink: 0; text-align: center;'):
                status_text = entry.review_status.replace('_', ' ').title()
                status_color = {
                    'Not Reviewed': theme.THEME_LIGHT['error'],   # Vermillion (Needs Attention)
                    'Provisional': theme.THEME_LIGHT['warning'],  # Amber
                    'Reviewed': theme.THEME_LIGHT['success']      # Green
                }.get(status_text, theme.THEME_LIGHT['text_muted'])
                
                ui.label(status_text).style(f'color: {status_color}; font-weight: 500; font-size: 0.875rem;')

            # Confidence - 100px, centered
            with ui.element('div').style('width: 100px; flex-shrink: 0; text-align: center;'):
                confidence_pct = int(entry.validation_confidence * 100)
                confidence_color = 'text-green-600' if confidence_pct >= 80 else 'text-yellow-600' if confidence_pct >= 60 else 'text-red-600'
                ui.label(f'{confidence_pct}%').classes(f'text-sm {confidence_color} font-semibold')

            # Date - 140px
            with ui.element('div').style('width: 140px; flex-shrink: 0;'):
                try:
                    # Parse ISO format date
                    date_obj = datetime.fromisoformat(entry.last_updated.replace('Z', '+00:00'))
                    date_text = date_obj.strftime('%Y-%m-%d')
                except:
                    date_text = entry.last_updated[:10] if len(entry.last_updated) >= 10 else entry.last_updated
                ui.label(date_text).classes('text-sm text-gray-600')

            # Actions - 60px
            with ui.element('div').style('width: 60px; flex-shrink: 0; text-align: center;'):
                # Create actions menu for this entry
                create_actions_menu(entry)


def create_actions_menu(entry: LedgerRecord):
    """Create actions menu button for a table row."""

    # Create closure to capture current entry
    def make_action_handler(current_entry):
        """Factory to create action handlers with correct entry reference."""

        def on_view_details():
            """Open view details modal (read-only)."""
            logger.info(f'Opening view details for entry: {current_entry.document_id[:8]}...')

            if not state.dialog_container:
                ui.notify('Error: UI state lost. Please refresh.', type='negative')
                return

            try:
                with state.dialog_container:
                    with ui.dialog() as dialog, ui.card().classes('p-6').style('min-width: 600px; max-width: 800px;'):
                        ui.label('Entry Details').classes('text-2xl font-bold mb-6')

                        # Title
                        ui.label('Title:').classes('text-sm font-semibold text-gray-600 mb-1')
                        ui.label(current_entry.title or 'Untitled').classes('text-base mb-4')

                        # Authors
                        ui.label('Authors:').classes('text-sm font-semibold text-gray-600 mb-1')
                        if current_entry.authors:
                            for author in current_entry.authors:
                                ui.label(f'• {author}').classes('text-base ml-2')
                        else:
                            ui.label('No authors').classes('text-base text-gray-400')
                        ui.label('').classes('mb-4')  # Spacing

                        # DOI
                        ui.label('DOI:').classes('text-sm font-semibold text-gray-600 mb-1')
                        if current_entry.doi:
                            with ui.row().classes('items-center gap-2'):
                                ui.label(current_entry.doi).classes('text-base')
                                ui.link('🔗 Open', f'https://doi.org/{current_entry.doi}', new_tab=True).classes('text-blue-600')
                        else:
                            ui.label('Not available').classes('text-base text-gray-400')
                        ui.label('').classes('mb-4')

                        # Journal
                        ui.label('Journal:').classes('text-sm font-semibold text-gray-600 mb-1')
                        ui.label(current_entry.journal or 'Not available').classes('text-base mb-4')

                        # Year
                        ui.label('Year:').classes('text-sm font-semibold text-gray-600 mb-1')
                        ui.label(str(current_entry.year) if current_entry.year else 'Not available').classes('text-base mb-4')

                        # Review Status
                        ui.label('Review Status:').classes('text-sm font-semibold text-gray-600 mb-1')
                        status_text = current_entry.review_status.replace('_', ' ').title()
                        status_color = {
                            'Not Reviewed': 'text-orange-600',
                            'Provisional': 'text-yellow-600',
                            'Reviewed': 'text-green-600'
                        }.get(status_text, 'text-gray-600')
                        ui.label(status_text).classes(f'text-base {status_color} font-medium mb-4')

                        # Confidence Score
                        ui.label('Confidence Score:').classes('text-sm font-semibold text-gray-600 mb-1')
                        confidence_pct = int(current_entry.validation_confidence * 100)
                        confidence_color = 'text-green-600' if confidence_pct >= 80 else 'text-yellow-600' if confidence_pct >= 60 else 'text-red-600'
                        ui.label(f'{confidence_pct}%').classes(f'text-base {confidence_color} font-semibold mb-4')

                        # Validation Source
                        ui.label('Validation Source:').classes('text-sm font-semibold text-gray-600 mb-1')
                        ui.label(current_entry.validation_source or 'Not available').classes('text-base mb-4')

                        # Last Updated
                        ui.label('Last Updated:').classes('text-sm font-semibold text-gray-600 mb-1')
                        ui.label(current_entry.last_updated).classes('text-base mb-4')

                        # PDF Filename
                        ui.label('PDF File:').classes('text-sm font-semibold text-gray-600 mb-1')
                        ui.label(current_entry.pdf_filename or 'Not available').classes('text-base mb-4')

                        # Document ID
                        ui.label('Document ID:').classes('text-sm font-semibold text-gray-600 mb-1')
                        doc_id_short = current_entry.document_id[:50] + '...' if len(current_entry.document_id) > 50 else current_entry.document_id
                        ui.label(doc_id_short).classes('text-xs text-gray-500 font-mono mb-6')

                        # Buttons
                        with ui.row().classes('w-full justify-end gap-4 mt-6'):
                                # Synchronous nested handler
                                def open_edit():
                                    dialog.close()
                                    on_edit_metadata() # Call sync function directly

                                ui.button('Edit', on_click=open_edit).props('outlined')
                                ui.button('Close', on_click=dialog.close).props('color=primary')

                    dialog.open()
                    logger.info(f'View details modal opened successfully')
            except Exception as e:
                logger.error(f'Failed to open view details modal: {e}', exc_info=True)
                ui.notify(f'Failed to open details: {e}', type='negative')

        def on_edit_metadata():
            """Open edit metadata modal with form fields."""
            logger.info(f'Opening edit metadata for entry: {current_entry.document_id[:8]}...')
            
            if not state.dialog_container:
                return

            try:
                import copy
                edited_entry = copy.deepcopy(current_entry)

                # Validation functions
                def validate_title(value):
                    if not value or len(value.strip()) < 3:
                        return "Title must be at least 3 characters"
                    return None

                def validate_doi(value):
                    if value and not value.startswith('10.'):
                        return "DOI must start with '10.'"
                    return None

                def validate_year(value):
                    if value and (value < 1900 or value > 2100):
                        return "Year must be between 1900 and 2100"
                    return None

                def validate_authors(value):
                    """Validate authors text area (one per line)."""
                    if not value or not value.strip():
                        return "At least one author is required"
                    authors = [line.strip() for line in value.split('\n') if line.strip()]
                    if len(authors) == 0:
                        return "At least one author is required"
                    for author in authors:
                        if len(author) < 2:
                            return f"Author name too short: '{author}'"
                    return None

                # Error state
                errors = {
                    'title': None,
                    'authors': None,
                    'doi': None,
                    'year': None
                }

                # Save handler
                def save_changes():
                    nonlocal errors

                    # Get values from inputs
                    title_val = title_input.value.strip() if title_input.value else ""
                    authors_val = authors_input.value.strip() if authors_input.value else ""
                    doi_val = doi_input.value.strip() if doi_input.value else ""
                    journal_val = journal_input.value.strip() if journal_input.value else ""
                    year_val = year_input.value if year_input.value else None

                    # Validate all fields
                    errors['title'] = validate_title(title_val)
                    errors['authors'] = validate_authors(authors_val)
                    errors['doi'] = validate_doi(doi_val) if doi_val else None
                    errors['year'] = validate_year(year_val) if year_val else None

                    # Show errors
                    if errors['title']:
                        title_error.text = errors['title']
                        title_error.set_visibility(True)
                    else:
                        title_error.set_visibility(False)

                    if errors['authors']:
                        authors_error.text = errors['authors']
                        authors_error.set_visibility(True)
                    else:
                        authors_error.set_visibility(False)

                    if errors['doi']:
                        doi_error.text = errors['doi']
                        doi_error.set_visibility(True)
                    else:
                        doi_error.set_visibility(False)

                    if errors['year']:
                        year_error.text = errors['year']
                        year_error.set_visibility(True)
                    else:
                        year_error.set_visibility(False)

                    # If any errors, stop
                    if any(errors.values()):
                        ui.notify('Please fix validation errors', type='warning')
                        return

                    # Update edited entry
                    edited_entry.title = title_val
                    edited_entry.authors = [line.strip() for line in authors_val.split('\n') if line.strip()]
                    edited_entry.doi = doi_val if doi_val else None
                    edited_entry.journal = journal_val if journal_val else None
                    edited_entry.year = year_val
                    edited_entry.review_status = review_status_radio.value

                    try:
                        # Save to ledger
                        state.ledger.update_record(edited_entry)
                        state.ledger.commit()

                        # Refresh UI
                        state.load_ledger()
                        refresh_ui()

                        # Close modal
                        dialog.close()

                        ui.notify(f'✅ Updated metadata for "{edited_entry.title[:50]}..."', type='positive')
                        logger.info(f'Updated metadata for entry {edited_entry.document_id}')

                    except Exception as e:
                        logger.error(f'Failed to save metadata: {e}', exc_info=True)
                        ui.notify(f'Failed to save metadata: {e}', type='negative')

                with state.dialog_container:
                    with ui.dialog() as dialog, ui.card().classes('p-6').style('min-width: 600px; max-width: 800px;'):
                        ui.label('Edit Metadata').classes('text-2xl font-bold mb-6')

                        # Title (required)
                        ui.label('Title *').classes('text-sm font-semibold mb-1')
                        title_input = ui.input(value=edited_entry.title or '').classes('w-full')
                        title_error = ui.label('').classes('text-xs text-red-600 mb-4')
                        title_error.set_visibility(False)

                        # Authors (required, multi-line text area)
                        ui.label('Authors * (one per line)').classes('text-sm font-semibold mb-1 mt-4')
                        authors_text = '\n'.join(edited_entry.authors) if edited_entry.authors else ''
                        authors_input = ui.textarea(value=authors_text).classes('w-full').props('rows=5')
                        authors_error = ui.label('').classes('text-xs text-red-600 mb-4')
                        authors_error.set_visibility(False)

                        # DOI (optional)
                        ui.label('DOI').classes('text-sm font-semibold mb-1 mt-4')
                        doi_input = ui.input(value=edited_entry.doi or '').classes('w-full').props('placeholder="10.xxxx/xxxx"')
                        doi_error = ui.label('').classes('text-xs text-red-600 mb-4')
                        doi_error.set_visibility(False)

                        # Journal (optional)
                        ui.label('Journal').classes('text-sm font-semibold mb-1 mt-4')
                        journal_input = ui.input(value=edited_entry.journal or '').classes('w-full')
                        ui.label('').classes('mb-4')  # Spacing

                        # Year (optional)
                        ui.label('Year').classes('text-sm font-semibold mb-1 mt-4')
                        year_input = ui.number(value=edited_entry.year, min=1900, max=2100).classes('w-full')
                        year_error = ui.label('').classes('text-xs text-red-600 mb-4')
                        year_error.set_visibility(False)

                        # Review Status
                        ui.label('Review Status').classes('text-sm font-semibold mb-1 mt-4')
                        review_status_radio = ui.radio(
                            ['not_reviewed', 'provisional', 'reviewed'],
                            value=edited_entry.review_status
                        ).props('inline').classes('mb-6')

                        # Buttons
                        with ui.row().classes('w-full justify-end gap-4 mt-6'):
                                ui.button('Cancel', on_click=dialog.close).props('flat')
                                ui.button('Save', on_click=save_changes).props('color=primary')

                    dialog.open()
                    logger.info(f'Edit metadata modal opened successfully')
            except Exception as e:
                logger.error(f'Failed to open edit metadata modal: {e}', exc_info=True)
                ui.notify(f'Failed to open edit form: {e}', type='negative')

        def on_mark_reviewed():
            """Mark entry as reviewed."""
            try:
                # Update entry status
                current_entry.review_status = "reviewed"

                # Update ledger backend (only takes record, not document_id)
                state.ledger.update_record(current_entry)
                state.ledger.commit()

                # Refresh UI
                state.load_ledger()
                refresh_ui()

                ui.notify(
                    f'✅ Marked "{current_entry.title[:50]}..." as reviewed',
                    type='positive'
                )
                logger.info(f'Marked entry {current_entry.document_id} as reviewed')

            except Exception as e:
                logger.error(f'Failed to mark as reviewed: {e}', exc_info=True)
                ui.notify(f'Failed to mark as reviewed: {e}', type='negative')

        def on_mark_not_reviewed():
            """Mark entry as not reviewed."""
            try:
                # Update entry status
                current_entry.review_status = "not_reviewed"

                # Update ledger backend (only takes record, not document_id)
                state.ledger.update_record(current_entry)
                state.ledger.commit()

                # Refresh UI
                state.load_ledger()
                refresh_ui()

                ui.notify(
                    f'✅ Marked "{current_entry.title[:50]}..." as not reviewed',
                    type='positive'
                )
                logger.info(f'Marked entry {current_entry.document_id} as not reviewed')

            except Exception as e:
                logger.error(f'Failed to mark as not reviewed: {e}', exc_info=True)
                ui.notify(f'Failed to mark as not reviewed: {e}', type='negative')

        def on_delete_entry():
            """Delete entry with confirmation."""
            title_short = current_entry.title[:60] if current_entry.title else "Untitled"

            def delete_confirmed():
                """Handle confirmed deletion."""
                try:
                    # Remove from ledger
                    if current_entry.document_id in state.ledger._records:
                        del state.ledger._records[current_entry.document_id]
                        state.ledger.commit()

                    # Close dialog
                    dialog.close()

                    # Refresh UI
                    state.load_ledger()
                    refresh_ui()

                    ui.notify(
                        f'🗑️ Deleted "{title_short}"',
                        type='positive'
                    )
                    logger.info(f'Deleted entry {current_entry.document_id}')

                except Exception as e:
                    logger.error(f'Failed to delete entry: {e}', exc_info=True)
                    ui.notify(f'Failed to delete entry: {e}', type='negative')
                    dialog.close()

            # Create confirmation dialog
            with ui.dialog() as dialog, ui.card().classes('p-6'):
                ui.label('Delete Entry?').classes('text-xl font-bold mb-4')
                ui.label(f'"{title_short}"').classes('text-sm text-gray-700 mb-2')
                ui.label('This action cannot be undone.').classes('text-sm text-gray-500 mb-6')

                with ui.row().classes('w-full justify-end gap-4'):
                    ui.button('Cancel', on_click=dialog.close).props('flat')
                    ui.button('Delete', on_click=delete_confirmed).props('color=negative')

            # Open dialog
            dialog.open()

        def on_open_pdf():
            """Open PDF in system default viewer."""
            import subprocess
            import platform

            try:
                # Get PDF path
                pdf_path = None

                # Try entry.pdf_filename first
                if current_entry.pdf_filename:
                    pdf_path = Path(current_entry.pdf_filename)

                    # If relative path, try common locations
                    if not pdf_path.is_absolute():
                        # Try multiple common locations in order
                        search_paths = [
                            pdf_path,  # Current directory
                            Path('sample_pdfs') / pdf_path.name,  # sample_pdfs/
                            Path('pdfs') / pdf_path.name,  # pdfs/
                            Path('sample_pdfs/Extracted') / pdf_path.name,  # sample_pdfs/Extracted/
                        ]

                        pdf_path = None
                        for search_path in search_paths:
                            if search_path.exists():
                                pdf_path = search_path
                                break

                # Check if file exists
                if pdf_path and pdf_path.exists():
                    # Open PDF with system default viewer
                    system = platform.system()

                    if system == 'Windows':
                        import os
                        os.startfile(str(pdf_path))
                    elif system == 'Darwin':  # macOS
                        subprocess.run(['open', str(pdf_path)], check=True)
                    else:  # Linux
                        subprocess.run(['xdg-open', str(pdf_path)], check=True)

                    ui.notify(
                        f'📄 Opening "{pdf_path.name}"',
                        type='info'
                    )
                    logger.info(f'Opened PDF: {pdf_path}')

                else:
                    # PDF not found
                    ui.notify(
                        f'❌ PDF file not found: {current_entry.pdf_filename or "No filename"}',
                        type='warning',
                        timeout=5000
                    )
                    logger.warning(f'PDF not found for entry {current_entry.document_id}: {current_entry.pdf_filename}')

            except Exception as e:
                logger.error(f'Failed to open PDF: {e}', exc_info=True)
                ui.notify(f'Failed to open PDF: {e}', type='negative')

        return on_view_details, on_edit_metadata, on_mark_reviewed, on_mark_not_reviewed, on_delete_entry, on_open_pdf

    # Get action handlers
    view_details, edit_metadata, mark_reviewed, mark_not_reviewed, delete_entry, open_pdf = make_action_handler(entry)

    # Create menu button
    with ui.button(icon='more_vert').props('flat dense round').style(
        f'color: {theme.THEME_LIGHT["indigo"]} !important; background: transparent !important; box-shadow: none !important;'
    ):
        with ui.menu().props('auto-close') as menu:
            ui.menu_item('View Details', on_click=view_details, auto_close=True)
            ui.menu_item('Edit Metadata', on_click=edit_metadata, auto_close=True)
            ui.separator()
            ui.menu_item('Mark as Reviewed', on_click=mark_reviewed, auto_close=True)
            ui.menu_item('Mark as Not Reviewed', on_click=mark_not_reviewed, auto_close=True)
            ui.separator()
            ui.menu_item('Delete Entry', on_click=delete_entry, auto_close=True)
            ui.menu_item('Open PDF', on_click=open_pdf, auto_close=True)


def create_batch_actions():
    """Create batch action buttons (shown when entries selected)."""

    if not state.selected_entries:
        return  # Don't show if nothing selected

    selected_count = len(state.selected_entries)

    with ui.row().classes('w-full items-center justify-between mb-4 p-3 rounded-lg').style(
        f'background: rgba(44, 62, 80, 0.05); border: 1px solid {theme.THEME_LIGHT["indigo"]};'
    ):
        ui.label(f'{selected_count} entries selected').classes('text-sm font-semibold').style(f'color: {theme.THEME_LIGHT["indigo"]};')

        with ui.row().classes('gap-2'):
            ui.button(
                'Mark Selected as Reviewed',
                icon='check_circle',
                on_click=on_batch_mark_reviewed
            ).style(f'background: {theme.THEME_LIGHT["indigo"]} !important; color: white !important;')

            ui.button(
                'Delete Selected',
                icon='delete',
                on_click=on_batch_delete
            ).style(f'background: {theme.THEME_LIGHT["indigo"]} !important; color: white !important;')


def on_batch_mark_reviewed():
    """Mark all selected entries as reviewed."""

    selected_entries = [
        entry for entry in state.all_entries
        if entry.document_id in state.selected_entries
    ]

    if not selected_entries:
        return

    count = len(selected_entries)

    def confirm_mark():
        try:
            # Update each entry
            for entry in selected_entries:
                entry.review_status = "reviewed"
                state.ledger.update_record(entry)

            # Commit once
            state.ledger.commit()

            # Close dialog
            dialog.close()

            # Refresh UI
            state.selected_entries.clear()
            state.select_all_value = False
            state.load_ledger()
            refresh_ui()

            ui.notify(f'✅ Marked {count} entries as reviewed', type='positive')
            logger.info(f'Batch marked {count} entries as reviewed')

        except Exception as e:
            logger.error(f'Batch mark reviewed failed: {e}', exc_info=True)
            ui.notify(f'Failed to mark as reviewed: {e}', type='negative')
            dialog.close()

    # Show confirmation
    with ui.dialog() as dialog, ui.card().classes('p-6'):
        ui.label('Mark Entries as Reviewed?').classes('text-xl font-bold mb-4')
        ui.label(f'Mark {count} selected entries as reviewed?').classes('text-base mb-4')

        ui.label('Entries:').classes('font-semibold mb-2')
        for entry in selected_entries[:5]:  # Show first 5
            title_short = entry.title[:50] if entry.title else "Untitled"
            ui.label(f'• {title_short}...').classes('text-sm ml-4')
        if count > 5:
            ui.label(f'• ... and {count - 5} more').classes('text-sm ml-4 text-gray-600')
        ui.label('').classes('mb-4')

        with ui.row().classes('w-full justify-end gap-4'):
            ui.button('Cancel', on_click=dialog.close).props('flat')
            ui.button('Mark Reviewed', on_click=confirm_mark).props('color=primary')

    dialog.open()


def on_batch_delete():
    """Delete all selected entries with backup."""
    import json

    selected_entries = [
        entry for entry in state.all_entries
        if entry.document_id in state.selected_entries
    ]

    if not selected_entries:
        return

    count = len(selected_entries)

    def confirm_delete():
        try:
            # Backup entries
            backup_dir = Path('run_outputs/backups')
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f'batch_delete_{count}_entries_{timestamp}.json'
            backup_path = backup_dir / backup_filename

            backup_data = []
            for entry in selected_entries:
                backup_data.append({
                    'document_id': entry.document_id,
                    'title': entry.title,
                    'authors': entry.authors,
                    'doi': entry.doi,
                    'journal': entry.journal,
                    'year': entry.year,
                    'review_status': entry.review_status,
                    'pdf_filename': entry.pdf_filename
                })

            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2)

            # Delete from ledger
            for entry in selected_entries:
                if entry.document_id in state.ledger._records:
                    del state.ledger._records[entry.document_id]

            # Commit once
            state.ledger.commit()

            # Close dialog
            dialog.close()

            # Refresh UI
            state.selected_entries.clear()
            state.select_all_value = False
            state.load_ledger()
            refresh_ui()

            ui.notify(
                f'🗑️ Deleted {count} entries\n💾 Backed up to: {backup_path}',
                type='positive',
                timeout=5000
            )
            logger.info(f'Batch deleted {count} entries (backed up to {backup_path})')

        except Exception as e:
            logger.error(f'Batch delete failed: {e}', exc_info=True)
            ui.notify(f'Failed to delete entries: {e}', type='negative')
            dialog.close()

    # Show strong confirmation
    with ui.dialog() as dialog, ui.card().classes('p-6'):
        ui.label('Delete Entries?').classes('text-xl font-bold mb-4')
        ui.label(f'⚠️  Delete {count} selected entries?').classes('text-base font-semibold text-red-600 mb-4')
        ui.label('This action CANNOT be undone!').classes('text-base font-bold mb-4')

        ui.label('Entries to delete:').classes('font-semibold mb-2')
        for entry in selected_entries[:5]:  # Show first 5
            title_short = entry.title[:50] if entry.title else "Untitled"
            ui.label(f'• {title_short}...').classes('text-sm ml-4')
        if count > 5:
            ui.label(f'• ... and {count - 5} more').classes('text-sm ml-4 text-gray-600')
        ui.label('').classes('mb-4')

        ui.label('Entries will be backed up before deletion.').classes('text-sm text-gray-600 mb-6')

        with ui.row().classes('w-full justify-end gap-4'):
            ui.button('Cancel', on_click=dialog.close).props('flat')
            ui.button('Delete', on_click=confirm_delete).props('color=negative')

    dialog.open()


def create_pagination():
    """Create pagination controls."""

    with ui.row().classes('w-full justify-center items-center gap-4 mt-4') as container:
        state.pagination_container = container

        if state.total_pages > 1:
            # Previous button
            prev_btn = ui.button(
                '← Previous',
                on_click=lambda: go_to_page(state.current_page - 1)
            ).props('flat')
            prev_btn.set_enabled(state.current_page > 1)

            # Page indicator
            ui.label(f'Page {state.current_page} of {state.total_pages}').classes('text-sm text-gray-600')

            # Next button
            next_btn = ui.button(
                'Next →',
                on_click=lambda: go_to_page(state.current_page + 1)
            ).props('flat')
            next_btn.set_enabled(state.current_page < state.total_pages)


def go_to_page(page: int):
    """Navigate to a specific page."""
    if 1 <= page <= state.total_pages:
        state.current_page = page
        state.update_pagination()
        refresh_ui()


def refresh_ui():
    """Refresh dynamic UI elements."""
    # Update result count
    if state.result_count_label:
        total = len(state.all_entries)
        filtered = len(state.filtered_entries)
        state.result_count_label.text = f'Showing {filtered} of {total} entries'

    # Refresh batch actions
    if state.batch_actions_container:
        state.batch_actions_container.clear()
        with state.batch_actions_container:
            create_batch_actions()

    # Refresh table
    if state.table_container:
        state.table_container.clear()
        with state.table_container:
            create_data_table()

    # Refresh pagination
    if state.pagination_container:
        state.pagination_container.clear()
        with state.pagination_container:
            create_pagination()


# ============================================================================
# EXPORT HELPER FUNCTIONS
# ============================================================================

def sanitize_cite_key_component(text: str) -> str:
    """
    Remove/replace special characters for BibTeX cite key.

    Args:
        text: Component of cite key (author, journal, etc.)

    Returns:
        Sanitized string safe for cite key use
    """
    import re
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Replace common accents
    replacements = {
        'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
        'ã': 'a', 'õ': 'o', 'ñ': 'n', 'ç': 'c'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove spaces and special characters, keep only alphanumeric and underscores
    text = re.sub(r'[^a-z0-9_]', '', text)

    return text


def generate_cite_key(entry: LedgerRecord) -> str:
    """
    Generate BibTeX cite key in format: lastname_journal_year

    Examples:
        - smith_nature_2023
        - doe_science_nd (no year)
        - anonymous_unknown_2024 (no author/journal)

    Args:
        entry: LedgerRecord to generate cite key for

    Returns:
        Unique cite key string
    """
    # Get first author's last name
    if entry.authors and len(entry.authors) > 0:
        # Format is usually "Last, First" or "Last"
        first_author = entry.authors[0]
        if ',' in first_author:
            lastname = first_author.split(',')[0].strip()
        else:
            # If no comma, take first word as last name
            lastname = first_author.split()[0] if ' ' in first_author else first_author
        lastname = sanitize_cite_key_component(lastname)
    else:
        lastname = "anonymous"

    # Get journal abbreviation (first 10 chars, no spaces)
    if entry.journal:
        journal_abbrev = entry.journal.replace(' ', '')[:10]
        journal_abbrev = sanitize_cite_key_component(journal_abbrev)
    else:
        journal_abbrev = "unknown"

    # Get year
    year = str(entry.year) if entry.year else "nd"

    return f"{lastname}_{journal_abbrev}_{year}"


def escape_bibtex(text: str) -> str:
    """
    Escape special characters for BibTeX format.

    BibTeX requires escaping: { } \ $ % & _ #

    Args:
        text: Text to escape

    Returns:
        Escaped text safe for BibTeX
    """
    if not text:
        return ""

    # Order matters - backslash must be first
    replacements = [
        ('\\', r'\\'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('$', r'\$'),
        ('%', r'\%'),
        ('&', r'\&'),
        ('_', r'\_'),
        ('#', r'\#'),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return text


def format_authors_bibtex(authors: List[str]) -> str:
    """
    Format authors for BibTeX: "Last, First and Last, First"

    Args:
        authors: List of author names (format: "Last, First" or "Last")

    Returns:
        BibTeX-formatted author string

    Example:
        Input: ["Smith, John", "Doe, Jane", "Chen, Wei"]
        Output: "Smith, John and Doe, Jane and Chen, Wei"
    """
    if not authors or len(authors) == 0:
        return "Anonymous"

    # Join with " and " as required by BibTeX
    return " and ".join(authors)


def detect_entry_type(entry: LedgerRecord) -> str:
    """
    Detect BibTeX entry type based on document metadata.

    Future-proof for book processing (currently defaults to article).

    Args:
        entry: LedgerRecord to classify

    Returns:
        BibTeX entry type (article, book, inbook, etc.)
    """
    # Future: Check entry.document_type or other fields
    # For now, default to article for research papers
    return "article"


def format_bibtex_entry(entry: LedgerRecord, cite_key: str) -> str:
    """
    Format a single ledger entry as BibTeX entry.

    Args:
        entry: LedgerRecord to format
        cite_key: Generated cite key

    Returns:
        Formatted BibTeX entry string
    """
    entry_type = detect_entry_type(entry)

    # Start with entry header
    lines = [f"@{entry_type}{{{cite_key},"]

    # Title (required)
    title = escape_bibtex(entry.title) if entry.title else "Untitled"
    lines.append(f"  title={{{title}}},")

    # Authors (required)
    author_str = format_authors_bibtex(entry.authors)
    author_str = escape_bibtex(author_str)
    lines.append(f"  author={{{author_str}}},")

    # Journal (optional - include if available)
    if entry.journal:
        journal = escape_bibtex(entry.journal)
        lines.append(f"  journal={{{journal}}},")

    # Year (optional - use "nd" if missing)
    year = str(entry.year) if entry.year else "nd"
    lines.append(f"  year={{{year}}},")

    # DOI (optional - use "notfound" if missing)
    if entry.doi:
        doi = escape_bibtex(entry.doi)
        lines.append(f"  doi={{{doi}}},")
    else:
        lines.append(f"  doi={{notfound}},")

    # Note (constant)
    lines.append(f"  note={{Exported from Prajnya Studio}}")

    # Close entry (no comma on last field)
    lines.append("}")

    return "\n".join(lines)


# ============================================================================
# CSV EXPORT FUNCTIONALITY
# ============================================================================

def export_to_csv():
    """Export selected or filtered ledger entries to CSV file."""

    # Determine what to export
    if state.selected_entries:
        # Export selected entries only
        entries_to_export = [
            entry for entry in state.filtered_entries
            if entry.document_id in state.selected_entries
        ]
        export_scope = f"{len(entries_to_export)} selected"
    elif state.filtered_entries:
        # No selection - export all filtered entries
        entries_to_export = state.filtered_entries
        export_scope = f"{len(entries_to_export)} filtered"
    else:
        ui.notify('No entries to export', type='warning')
        return

    try:
        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'ledger_export_{timestamp}.csv'

        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'Title',
            'Authors',
            'DOI',
            'Journal',
            'Year',
            'Review Status',
            'Confidence',
            'Validation Source',
            'Date Processed',
            'PDF Filename'
        ])

        # Write data rows
        for entry in entries_to_export:
            # Format authors as comma-separated string
            authors_str = '; '.join(entry.authors) if entry.authors else ''

            # Format confidence as percentage
            confidence_pct = f'{entry.validation_confidence:.0%}'

            # Format date
            try:
                date_obj = datetime.fromisoformat(entry.last_updated.replace('Z', '+00:00'))
                date_str = date_obj.strftime('%Y-%m-%d %H:%M:%S')
            except:
                date_str = entry.last_updated

            # Write row
            writer.writerow([
                entry.title or '',
                authors_str,
                entry.doi or '',
                entry.journal or '',
                entry.year or '',
                entry.review_status.replace('_', ' ').title(),
                confidence_pct,
                entry.validation_source or '',
                date_str,
                entry.pdf_filename or ''
            ])

        # Get CSV content
        csv_content = output.getvalue()
        output.close()

        # Also save to local file as backup (in run_outputs/exports/)
        export_dir = Path('run_outputs/exports')
        export_dir.mkdir(parents=True, exist_ok=True)
        local_path = export_dir / filename

        with open(local_path, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)

        # Trigger browser download
        ui.download(csv_content.encode('utf-8'), filename)

        ui.notify(
            f'✅ Exported {export_scope} entries\n📁 Browser download: {filename}\n💾 Also saved: {local_path}',
            type='positive',
            position='top',
            timeout=5000
        )

        logger.info(f'Exported {export_scope} entries to CSV: {filename} (local: {local_path})')

    except Exception as e:
        logger.error(f'CSV export failed: {e}', exc_info=True)
        ui.notify(f'Export failed: {e}', type='negative')


# ============================================================================
# BIBTEX EXPORT FUNCTIONALITY
# ============================================================================

def export_to_bibtex():
    """Export selected or filtered ledger entries to BibTeX file."""

    # Determine what to export (same logic as CSV)
    if state.selected_entries:
        # Export selected entries only
        entries_to_export = [
            entry for entry in state.filtered_entries
            if entry.document_id in state.selected_entries
        ]
        export_scope = f"{len(entries_to_export)} selected"
    elif state.filtered_entries:
        # No selection - export all filtered entries
        entries_to_export = state.filtered_entries
        export_scope = f"{len(entries_to_export)} filtered"
    else:
        ui.notify('No entries to export', type='warning')
        return

    try:
        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'ledger_export_{timestamp}.bib'

        # Build BibTeX content
        bibtex_entries = []
        cite_keys_used = set()

        for entry in entries_to_export:
            # Generate cite key
            cite_key = generate_cite_key(entry)

            # Handle duplicate cite keys by appending counter
            original_cite_key = cite_key
            counter = 2
            while cite_key in cite_keys_used:
                cite_key = f"{original_cite_key}_{counter}"
                counter += 1

            cite_keys_used.add(cite_key)

            # Format BibTeX entry
            bibtex_entry = format_bibtex_entry(entry, cite_key)
            bibtex_entries.append(bibtex_entry)

        # Join all entries with blank line between
        bibtex_content = "\n\n".join(bibtex_entries)

        # Also save to local file as backup (in run_outputs/exports/)
        export_dir = Path('run_outputs/exports')
        export_dir.mkdir(parents=True, exist_ok=True)
        local_path = export_dir / filename

        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(bibtex_content)

        # Trigger browser download
        ui.download(bibtex_content.encode('utf-8'), filename)

        ui.notify(
            f'✅ Exported {export_scope} entries to BibTeX\n📁 Browser download: {filename}\n💾 Also saved: {local_path}',
            type='positive',
            position='top',
            timeout=5000
        )

        logger.info(f'Exported {export_scope} entries to BibTeX: {filename} (local: {local_path})')

    except Exception as e:
        logger.error(f'BibTeX export failed: {e}', exc_info=True)
        ui.notify(f'Export failed: {e}', type='negative')


# ============================================================================
# MAIN MOUNT FUNCTION
# ============================================================================

def mount():
    """Mount the Ledger Master page (v2.0)."""

    # --- CSS: Force Standard Components to match Vedic Palette ---
    ui.add_head_html(f'''
        <style>
            /* Checkboxes & Radios -> Vermillion */
            .q-checkbox__inner--truthy, .q-radio__inner--truthy {{ color: {theme.THEME_LIGHT["vermillion"]} !important; }}
            .q-checkbox__inner .q-checkbox__check, .q-radio__inner .q-radio__check {{ color: {theme.THEME_LIGHT["vermillion"]} !important; }}
            
            /* Sliders -> Indigo */
            .q-slider {{ color: {theme.THEME_LIGHT["indigo"]} !important; }}
            .q-slider__track-container {{ background: rgba(44, 62, 80, 0.15) !important; }}
            .q-slider__selection {{ background: {theme.THEME_LIGHT["indigo"]} !important; }}
            .q-slider__thumb {{ color: {theme.THEME_LIGHT["indigo"]} !important; }}
        </style>
    ''')

    with ui.column().classes('w-full p-4'):
        # --- Sticky Action Header ---
        with ui.row().classes('w-full items-center justify-between sticky top-0 z-50 py-4 px-6 mb-6').style(
            f'background: {theme.THEME_LIGHT["canvas"]}; '
            f'border-bottom: 2px solid {theme.THEME_LIGHT["gold"]};'
        ):
            # LEFT: Title
            with ui.column().classes('gap-1'):
                ui.label('Ledger Master').classes('text-3xl').style(
                    f'font-family: var(--prajnya-font-display); font-weight: 600; color: {theme.THEME_LIGHT["indigo"]}; font-variant: small-caps;'
                )
                ui.label('Search, filter, and manage processed PDF entries').classes('text-sm').style(
                    f'font-family: var(--prajnya-font-body); color: {theme.THEME_LIGHT["text_muted"]};'
                )

            # RIGHT: Action Buttons
            with ui.row().classes('gap-4 items-center'):
                ui.button('Export CSV', icon='download', on_click=export_to_csv).style(
                    f'background: {theme.THEME_LIGHT["vermillion"]} !important; color: white;'
                )
                ui.button('Export BibTeX', icon='menu_book', on_click=export_to_bibtex).props('outline').style(
                    f'color: {theme.THEME_LIGHT["vermillion"]} !important; border-color: {theme.THEME_LIGHT["vermillion"]} !important;'
                )

        state.dialog_container = ui.element('div')

        # Load ledger data
        state.load_ledger()

        # Search Bar
        create_search_bar()

        # Filter Panel
        create_filter_panel()

        # Result Count
        create_result_count()

        # Batch Actions (only shown when entries selected)
        with ui.column().classes('w-full') as batch_actions_container:
            state.batch_actions_container = batch_actions_container
            create_batch_actions()

        # Data Table
        create_data_table()

        # Pagination
        create_pagination()

        # Export Buttons
        # Buttons moved to sticky header
        pass
