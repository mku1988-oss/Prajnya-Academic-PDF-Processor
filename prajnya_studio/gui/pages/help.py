"""
Help Page - v2.1 (Documentation Implementation)

Provides user documentation via expandable cards:
- Quick Start Guide: 4-step workflow for new users
- Configuration Reference: YAML strategy overview
- FAQ: Common questions and answers
- Support: Contact information and version

Sprint 3 - Feature: Help Documentation
"""
from nicegui import ui
from .. import theme


# =============================================================================
# CONTENT DATA
# =============================================================================

QUICK_START_STEPS = [
    {
        "step": 1,
        "title": "Configure Strategy",
        "location": "Strategy Lab",
        "details": [
            "Select a processing preset: Fast, Balanced, or Accurate",
            "Choose your VLM model (OlmOCR 2.7B recommended)",
            "Click 'Apply Settings' to save your configuration"
        ]
    },
    {
        "step": 2,
        "title": "Upload PDFs",
        "location": "Batch Commander",
        "details": [
            "Drag & drop research paper PDFs into the upload zone",
            "Or click to browse and select files",
            "Files appear in the staging queue"
        ]
    },
    {
        "step": 3,
        "title": "Process & Monitor",
        "location": "Batch Commander",
        "details": [
            "Click 'Start Processing' to begin extraction",
            "Watch real-time progress in the log terminal",
            "View metadata confidence scores as files complete"
        ]
    },
    {
        "step": 4,
        "title": "Review Results",
        "location": "Ledger Master",
        "details": [
            "Search and filter processed documents",
            "Edit extracted metadata if corrections are needed",
            "Export results to CSV for external use"
        ]
    }
]

FAQ_ITEMS = [
    {
        "q": "What file types are supported?",
        "a": "Currently only PDF files are supported. The system is optimized for research articles and academic papers."
    },
    {
        "q": "How long does processing take?",
        "a": "Processing time varies by preset: Fast (~1 min/PDF), Balanced (~2-3 min/PDF), Accurate (~5 min/PDF). Complex papers with many figures may take longer."
    },
    {
        "q": "What VLM models are available?",
        "a": "OlmOCR 2.7B (default, 82% accuracy) and OlmOCR 7B (stable, 76% accuracy). The 2.7B model is recommended for most use cases."
    },
    {
        "q": "How do I change processing speed?",
        "a": "Navigate to Strategy Lab and select a different preset. 'Fast' prioritizes speed, 'Accurate' prioritizes quality."
    },
    {
        "q": "Where are results saved?",
        "a": "Extracted metadata is saved to the Semantic Ledger (CSV) in the 'run_outputs/' folder. Batch logs are also preserved."
    },
    {
        "q": "Can I edit extracted metadata?",
        "a": "Yes! In Ledger Master, click the action menu (⋮) for any entry and select 'Edit Metadata' to make corrections."
    },
    {
        "q": "What if processing fails?",
        "a": "Check the log output for error details. Try re-processing with the 'Accurate' preset. Some scanned PDFs may require OCR preprocessing."
    }
]

CONFIG_SECTIONS = [
    {
        "name": "Processing Mode",
        "key": "processing_mode",
        "desc": "Controls extraction strategy: 'adaptive' (auto-detect), 'vlm_only', or 'text_only'"
    },
    {
        "name": "VLM Profile",
        "key": "vlm.profile",
        "desc": "Specifies which vision-language model to use for extraction"
    },
    {
        "name": "Title Resolution",
        "key": "vlm.title_page_resolution",
        "desc": "Image render resolution for title page (higher = more accurate but slower)"
    },
    {
        "name": "Retry Settings",
        "key": "vlm.max_retries",
        "desc": "Number of retry attempts for failed VLM calls"
    }
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _card_style() -> str:
    """Returns the standard card styling for help sections."""
    return (
        f'background: {theme.THEME_LIGHT["card"]}; '
        f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
        f'border-radius: 8px; '
        f'padding: 24px;'
    )


def _create_expandable_card(icon: str, title: str, subtitle: str, content_builder):
    """
    Creates an expandable card with header and collapsible content.
    
    Args:
        icon: Emoji icon for the card header
        title: Main title text
        subtitle: Description text below title
        content_builder: Function that creates the expanded content UI
    """
    with ui.expansion(
        text='',
        icon=None,
    ).classes('w-full').style(
        f'background: {theme.THEME_LIGHT["card"]}; '
        f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
        f'border-radius: 8px; '
        f'margin-bottom: 16px;'
    ) as expansion:
        # Custom header slot
        with expansion.add_slot('header'):
            with ui.row().classes('w-full items-center gap-3 py-2'):
                ui.label(icon).classes('text-2xl')
                with ui.column().classes('gap-0'):
                    ui.label(title).classes('text-xl font-bold').style(
                        f'color: {theme.THEME_LIGHT["indigo"]};'
                    )
                    ui.label(subtitle).classes('text-sm').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]};'
                    )
        
        # Content area
        with ui.column().classes('w-full gap-4 pt-2'):
            content_builder()


def _build_quick_start_content():
    """Builds the Quick Start Guide content."""
    for step in QUICK_START_STEPS:
        with ui.card().classes('w-full').style(
            f'background: {theme.THEME_LIGHT["canvas"]}; '
            f'border-left: 4px solid {theme.THEME_LIGHT["indigo"]}; '
            f'padding: 16px;'
        ):
            with ui.row().classes('items-center gap-3 mb-2'):
                # Step number badge
                ui.label(str(step["step"])).classes(
                    'w-8 h-8 rounded-full flex items-center justify-center text-white font-bold'
                ).style(f'background: {theme.THEME_LIGHT["indigo"]};')
                
                with ui.column().classes('gap-0'):
                    ui.label(step["title"]).classes('font-bold').style(
                        f'color: {theme.THEME_LIGHT["text"]};'
                    )
                    ui.label(step["location"]).classes('text-xs').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]};'
                    )
            
            # Details list
            with ui.column().classes('ml-11 gap-1'):
                for detail in step["details"]:
                    with ui.row().classes('items-start gap-2'):
                        ui.label('•').style(f'color: {theme.THEME_LIGHT["indigo"]};')
                        ui.label(detail).classes('text-sm').style(
                            f'color: {theme.THEME_LIGHT["text"]};'
                        )


def _build_faq_content():
    """Builds the FAQ accordion content."""
    for i, item in enumerate(FAQ_ITEMS):
        with ui.expansion(item["q"]).classes('w-full').style(
            f'background: {theme.THEME_LIGHT["canvas"]}; '
            f'border-radius: 4px; '
            f'margin-bottom: 8px;'
        ):
            ui.label(item["a"]).classes('text-sm py-2').style(
                f'color: {theme.THEME_LIGHT["text"]};'
            )


def _build_config_reference_content():
    """Builds the Configuration Reference content."""
    # Intro text
    ui.label(
        "The strategy configuration file (YAML) controls how PDFs are processed. "
        "You can edit these settings in Strategy Lab → Advanced Settings."
    ).classes('text-sm mb-4').style(f'color: {theme.THEME_LIGHT["text"]};')
    
    # Config table
    with ui.column().classes('w-full gap-2'):
        # Table header
        with ui.row().classes('w-full gap-2 pb-2').style(
            f'border-bottom: 1px solid {theme.THEME_LIGHT["border"]};'
        ):
            ui.label('Setting').classes('flex-1 font-bold text-sm').style(
                f'color: {theme.THEME_LIGHT["indigo"]};'
            )
            ui.label('YAML Key').classes('flex-1 font-bold text-sm').style(
                f'color: {theme.THEME_LIGHT["indigo"]};'
            )
            ui.label('Description').classes('flex-[2] font-bold text-sm').style(
                f'color: {theme.THEME_LIGHT["indigo"]};'
            )
        
        # Table rows
        for section in CONFIG_SECTIONS:
            with ui.row().classes('w-full gap-2 py-2').style(
                f'border-bottom: 1px solid {theme.THEME_LIGHT["border"]};'
            ):
                ui.label(section["name"]).classes('flex-1 text-sm').style(
                    f'color: {theme.THEME_LIGHT["text"]};'
                )
                ui.label(section["key"]).classes('flex-1 text-sm font-mono').style(
                    f'color: {theme.THEME_LIGHT["warning"]};'
                )
                ui.label(section["desc"]).classes('flex-[2] text-sm').style(
                    f'color: {theme.THEME_LIGHT["text_muted"]};'
                )


def _build_support_content():
    """Builds the Support section content."""
    # Contact cards row
    with ui.row().classes('w-full gap-4'):
        # GitHub Issues
        with ui.card().classes('flex-1').style(
            f'background: {theme.THEME_LIGHT["canvas"]}; '
            f'padding: 16px; text-align: center;'
        ):
            ui.label('🐙').classes('text-3xl mb-2')
            ui.label('GitHub Issues').classes('font-bold mb-1').style(
                f'color: {theme.THEME_LIGHT["text"]};'
            )
            ui.label('Report bugs and request features').classes('text-xs mb-2').style(
                f'color: {theme.THEME_LIGHT["text_muted"]};'
            )
            ui.link(
                'github.com/shastrasaara',
                'https://github.com',
                new_tab=True
            ).classes('text-sm').style(f'color: {theme.THEME_LIGHT["indigo"]};')
        
        # Email Support
        with ui.card().classes('flex-1').style(
            f'background: {theme.THEME_LIGHT["canvas"]}; '
            f'padding: 16px; text-align: center;'
        ):
            ui.label('📧').classes('text-3xl mb-2')
            ui.label('Email Support').classes('font-bold mb-1').style(
                f'color: {theme.THEME_LIGHT["text"]};'
            )
            ui.label('Direct assistance for complex issues').classes('text-xs mb-2').style(
                f'color: {theme.THEME_LIGHT["text_muted"]};'
            )
            ui.label('support@shastrasaara.dev').classes('text-sm').style(
                f'color: {theme.THEME_LIGHT["indigo"]};'
            )
    
    # Version info
    ui.separator().classes('my-4')
    with ui.row().classes('w-full justify-between items-center'):
        ui.label('Prajnya Studio').classes('font-bold').style(
            f'color: {theme.THEME_LIGHT["text"]};'
        )
        ui.label('Version 5.0.0 (Saraswati Release)').classes('text-sm').style(
            f'color: {theme.THEME_LIGHT["text_muted"]};'
        )


# =============================================================================
# MAIN MOUNT
# =============================================================================

def mount():
    """
    Mount the Help & Documentation page.
    
    Renders four expandable sections:
    1. Quick Start Guide - 4-step workflow tutorial
    2. Configuration Reference - YAML settings overview
    3. FAQ - Common questions with answers
    4. Support - Contact info and version
    """
    with ui.column().classes('w-full max-w-4xl mx-auto gap-6'):

        # Header (Design Bible: Cinzel, Small Caps, 28-32px)
        ui.label('Help & Documentation').classes('text-3xl mb-1').style(
            f'font-family: var(--prajnya-font-display); '
            f'font-weight: 600; '
            f'color: {theme.THEME_LIGHT["indigo"]}; '
            f'font-variant: small-caps;'
        )
        ui.label('User Guide & Reference').classes('text-sm').style(
            f'font-family: var(--prajnya-font-body); '
            f'color: {theme.THEME_LIGHT["text_muted"]};'
        )

        ui.separator().style(f'background: {theme.THEME_LIGHT["border"]}; margin: 24px 0;')

        # Section 1: Quick Start Guide
        _create_expandable_card(
            icon='📘',
            title='Quick Start Guide',
            subtitle='Get started with Prajnya Studio in 4 simple steps',
            content_builder=_build_quick_start_content
        )

        # Section 2: Configuration Reference
        _create_expandable_card(
            icon='⚙️',
            title='Configuration Reference',
            subtitle='Understand YAML strategy settings',
            content_builder=_build_config_reference_content
        )

        # Section 3: FAQ
        _create_expandable_card(
            icon='❔',
            title='Frequently Asked Questions',
            subtitle='Answers to common questions',
            content_builder=_build_faq_content
        )

        # Section 4: Support
        _create_expandable_card(
            icon='💬',
            title='Support',
            subtitle='Get help and report issues',
            content_builder=_build_support_content
        )
