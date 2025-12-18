# .\prajnya_studio\gui\pages\strategy_lab.py
"""
Strategy Lab - Level 1 (Quick Settings)
Visual configuration editor for VLM processing strategies

Sprint 3 - Feature 1
Allows users to:
- Select processing presets (Fast/Balanced/Accurate)
- Choose VLM models (olmocr-2-7b, olmocr-7b)
- Save configuration changes with automatic backup
- Hot-reload strategy without restarting application
"""

from nicegui import ui
import asyncio
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import shutil
import logging

from gui.utils.model_registry import discover_vlm_models

from gui import theme, components
from gui.app_state import AppState
from prajnya.orchestrator import reset_orchestrator, get_orchestrator

# Set up logger
logger = logging.getLogger(__name__)

# Configuration paths
STRATEGY_CONFIG_PATH = Path("src/processing/strategies/config/research_article_strategy.yaml")
FACTORY_DEFAULT_PATH = Path("src/processing/strategies/config/research_article_strategy.FACTORY_DEFAULT.yaml")
BACKUP_DIR = Path("src/processing/strategies/config/backups")

# Preset configurations
PRESETS = {
    "fast": {
        "name": "Fast Mode",
        "description": "PyMuPDF + Basic VLM (~1 min/PDF)",
        "processing_mode": "adaptive",
        "profile": "olmocr-7b",
        "title_resolution": 1024,
        "adaptive_enabled": True,
        "two_shot_enabled": False,
    },
    "balanced": {
        "name": "Balanced (Recommended)",
        "description": "Adaptive Routing + Two-Shot Title (~2.5 min/PDF)",
        "processing_mode": "adaptive",
        "profile": "olmocr-2-7b",
        "title_resolution": 1288,
        "adaptive_enabled": True,
        "two_shot_enabled": True,
    },
    "accurate": {
        "name": "Maximum Accuracy",
        "description": "Two-Shot Everything (~4 min/PDF)",
        "processing_mode": "two_shot",
        "profile": "olmocr-2-7b",
        "title_resolution": 1288,
        "adaptive_enabled": False,
        "two_shot_enabled": True,
    },
}

# Discover models dynamically
AVAILABLE_MODELS_LOCAL, AVAILABLE_MODELS_CLOUD = discover_vlm_models()

# Combined dict for validation (backward compatibility)
AVAILABLE_MODELS = {**AVAILABLE_MODELS_LOCAL, **AVAILABLE_MODELS_CLOUD}


def mount():
    """Mount the Strategy Lab Level 1 UI component."""
    state = AppState.get_instance()

    # --- CSS: Force Standard Radios to match Vedic Palette ---
    # FIX: Added specific rules for Sliders (Indigo) alongside Radios (Vermillion)
    ui.add_head_html(f'''
        <style>
            /* Radios & Checkboxes -> Vermillion (Agni) */
            .q-radio__inner--truthy, .q-checkbox__inner--truthy {{ color: {theme.THEME_LIGHT["vermillion"]} !important; }}
            .q-radio__inner .q-radio__check, .q-checkbox__inner .q-checkbox__check {{ color: {theme.THEME_LIGHT["vermillion"]} !important; }}
            
            /* Text Utilities */
            .text-positive {{ color: {theme.THEME_LIGHT["success"]} !important; }}
            
            /* Sliders -> Indigo (Structure/Precision) */
            .q-slider {{ color: {theme.THEME_LIGHT["indigo"]} !important; }}
            .q-slider__selection {{ background: {theme.THEME_LIGHT["indigo"]} !important; }}
            .q-slider__thumb {{ color: {theme.THEME_LIGHT["indigo"]} !important; }}
            .q-slider__track-container {{ background: rgba(44, 62, 80, 0.15) !important; }} /* Faint Indigo Track */
            
            /* Tabs -> Indigo (Active) */
            .q-tab--active .q-tab__indicator {{ height: 3px; background: {theme.THEME_LIGHT["vermillion"]}; }}
            .q-tab--active .q-tab__content {{ color: {theme.THEME_LIGHT["indigo"]}; }}
        </style>
    ''')

    # --- State Variables ---
    current_config = {}
    selected_preset = "balanced"
    config_loaded = False
    use_cloud_mode = False  # Cloud mode disabled by default

    # --- UI Layout ---
    with ui.column().classes('w-full p-4'):
        
        # --- Sticky Action Header ---
        with ui.row().classes('w-full items-center justify-between sticky top-0 z-50 py-4 px-6 mb-6').style(
            f'background: {theme.THEME_LIGHT["canvas"]}; '
            f'border-bottom: 2px solid {theme.THEME_LIGHT["gold"]};'
        ):
            # LEFT: Title
            with ui.column().classes('gap-1'):
                ui.label('Strategy Laboratory').classes('text-3xl').style(
                    f'font-family: var(--prajnya-font-display); '
                    f'font-weight: 600; '
                    f'color: {theme.THEME_LIGHT["indigo"]}; '
                    f'font-variant: small-caps;'
                )
                ui.label('Configure VLM processing settings').classes('text-sm').style(
                    f'color: {theme.THEME_LIGHT["text_muted"]}; '
                    f'font-family: var(--prajnya-font-body);'
                )

            # RIGHT: Action Buttons
            with ui.row().classes('gap-2 items-center'):
                btn_load = ui.button(
                    'Load Current Config', 
                    icon='download',
                    on_click=lambda: asyncio.create_task(load_config_async())
                ).props('outline color=primary')

                btn_preview = ui.button(
                    'Preview Changes',
                    icon='visibility',
                    on_click=lambda: preview_advanced_settings()
                ).props('outline color=indigo')
                btn_preview.disable()

                btn_apply = ui.button(
                    'Apply & Save', 
                    icon='check_circle', 
                    on_click=lambda: asyncio.create_task(apply_config_async())
                ).style(f'background: {theme.THEME_LIGHT["vermillion"]} !important; color: white !important;')
                btn_apply.disable()

        # Main Content Container
        with ui.column().classes('w-full max-w-3xl gap-4'):

            # ============================================================================
            # 1. VLM PROVIDER MODE CARD (Moved to Top)
            # ============================================================================
            # ============================================================================
            # 1. VLM PROVIDER MODE (Flattened & Moved to Top)
            # ============================================================================
            with ui.column().classes('w-full mb-2'): # Flattened container

                # Header with mode badge
                with ui.row().classes('w-full items-center mb-4'):
                    ui.label('🌐 VLM Provider Mode').classes('text-lg font-bold').style(
                        f'color: {theme.THEME_LIGHT["indigo"]}; '
                        f'font-family: var(--prajnya-font-body);'
                    )
                    ui.space()
                    # Badge showing current mode
                    mode_badge = ui.badge('Local', color='green').classes('text-sm px-3 py-1')

                # Cloud mode toggle switch
                cloud_switch = ui.switch(
                    'Use Cloud API (Gemini)',
                    value=False
                ).classes('text-base font-medium')

                # Info panel (dynamic based on mode)
                info_panel = ui.column().classes('w-full mt-4 p-4 rounded').style(
                    f'background: rgba(44, 62, 80, 0.05); '
                    f'border-left: 4px solid {theme.THEME_LIGHT["success"]};'  # Default Green
                )

                with info_panel:
                    with ui.row().classes('items-center mb-2'):
                        mode_icon = ui.icon('computer', size='24px').classes('text-green-700')
                        mode_title = ui.label('Local Mode').classes('font-bold ml-2 text-green-900')
                    mode_desc = ui.label(
                        '• Requires: llama-server running on localhost:8080\n'
                        '• Cost: Free (uses your GPU)\n'
                        '• Speed: Depends on GPU hardware\n'
                        '• VRAM: 7.6GB minimum (olmocr-7b)'
                    ).classes('text-sm whitespace-pre-line leading-relaxed').style(
                        f'color: {theme.THEME_LIGHT["text"]};'
                    )

            # ============================================================================
            # 1.1 QUICK SETTINGS HEADER
            # ============================================================================
            ui.label('Quick Settings').classes('text-sm mb-1').style(
                f'font-family: var(--prajnya-font-body); '
                f'font-weight: 600; '
                f'color: {theme.THEME_LIGHT["indigo"]}; '
                f'text-transform: uppercase; '
                f'letter-spacing: 0.5px;'
            )
            # Golden underline
            ui.element('div').classes('w-full mb-4').style(
                f'height: 2px; '
                f'background: {theme.THEME_LIGHT["gold"]};'
            )

            # ============================================================================
            # 1.5. SERVER CONFIGURATION CARD
            # ============================================================================
            
            # Handler for path changes
            def update_path_logic(new_path):
                nonlocal current_config
                # Ensure structure exists
                if 'vlm_provider' not in current_config:
                    current_config['vlm_provider'] = {}
                if 'local_server' not in current_config['vlm_provider']:
                    current_config['vlm_provider']['local_server'] = {}
                
                current_config['vlm_provider']['local_server']['server_executable'] = new_path
                
                # Enable Apply button
                btn_apply.enable()
                btn_preview.enable()

            def on_server_path_change(e):
                update_path_logic(e.value)

            # Retrieve Server Path Logic (Initial State)
            initial_server_path = ""
            try:
                if state.server_manager:
                     # Access protected _backend for path info
                     backend = getattr(state.server_manager, "_backend", None)
                     if backend:
                         if hasattr(backend, 'batch_file') and backend.batch_file:
                             initial_server_path = str(backend.batch_file)
                         elif hasattr(backend, 'llama_swap_exe') and backend.llama_swap_exe:
                             initial_server_path = str(backend.llama_swap_exe)
            except Exception as e:
                logger.warning(f"Could not retrieve server path for UI: {e}")

            with ui.card().classes('w-full').style(
                f'background: {theme.THEME_LIGHT["card"]}; '
                f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
                f'border-radius: 8px; '
                f'padding: 24px;'
            ):
                 with ui.row().classes('w-full items-center mb-4'):
                    ui.label('🔧 Server Configuration').classes('text-lg font-bold').style(
                        f'color: {theme.THEME_LIGHT["indigo"]}; '
                        f'font-family: var(--prajnya-font-body);'
                    )
                 
                 # Info panel for input area
                 server_panel = ui.column().classes('w-full p-4 rounded').style(
                    f'background: rgba(44, 62, 80, 0.05); '
                    f'border-left: 4px solid {theme.THEME_LIGHT["indigo"]};'
                 )

                 with server_panel:
                    ui.label('Server Executable Path').classes('text-xs font-semibold uppercase tracking-wider mb-1').style(f'color: {theme.THEME_LIGHT["text_muted"]};')
                    
                    # Editable Path Input with Browse Button
                    with ui.input(
                        value=initial_server_path,
                        on_change=on_server_path_change
                    ).props('dense outlined').classes('w-full font-mono text-xs').style('background: white;') as server_path_input:
                        server_path_input.tooltip("Path to llama-server executable")
                        with server_path_input.add_slot('append'):
                            ui.button(
                                icon='folder_open',
                                on_click=lambda: open_file_picker(
                                    callback=lambda p: (
                                        setattr(server_path_input, 'value', p),
                                        update_path_logic(p)
                                    ),
                                    initial_path=server_path_input.value
                                )
                            ).props('flat dense rounded').classes('text-indigo-600')

                    # Model Config Info (Relative Path)
                    ui.label('Model Configuration File').classes('text-xs font-semibold uppercase tracking-wider mt-3 mb-1').style(f'color: {theme.THEME_LIGHT["text_muted"]};')
                    
                    relative_ini_path = "src/llm/providers/config/model_profiles/model.ini"
                    with ui.row().classes('items-center gap-1'):
                         ui.icon('description', size='xs').style(f'color: {theme.THEME_LIGHT["text_muted"]};')
                         ui.label(relative_ini_path).classes('text-xs font-mono break-all').style(f'color: {theme.THEME_LIGHT["text"]};')
                    
                    ui.label(
                        'Modify this file to configure model paths and parameters.'
                    ).classes('text-xs mt-1').style(f'color: {theme.THEME_LIGHT["text_muted"]}; font-style: italic;')

            # ============================================================================
            # 2. VLM MODEL CARD (Moved Up)
            # ============================================================================
            with ui.card().classes('w-full').style(
                f'background: {theme.THEME_LIGHT["card"]}; '
                f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
                f'border-radius: 8px; '
                f'padding: 24px;'
            ):
                # Header with Confidence badge
                with ui.row().classes('w-full items-center mb-3'):
                    ui.label('🤖 VLM Model').classes('text-lg font-semibold').style(
                        f'color: {theme.THEME_LIGHT["text"]}; '
                        f'font-family: var(--prajnya-font-body);'
                    )
                    ui.space()

                    # Confidence badge
                    ui.label('Confidence: 82%').classes('text-sm').style(
                        f'color: {theme.THEME_LIGHT["text_muted"]}; '
                        f'font-family: var(--prajnya-font-mono);'
                    )

                # Dropdown (styled as outlined select)
                model_selector = ui.select(
                    AVAILABLE_MODELS_LOCAL,  # Start with local models
                    value="olmocr-2-7b",
                    label="Select Model"
                ).props('outlined').classes('w-full').style(
                    f'background: {theme.THEME_LIGHT["card"]}; '
                    f'border-radius: 6px;'
                )
                
                # Model Info Text (Validation/Help)
                model_info_text = ui.label('Used for title page and body page processing').classes('text-xs text-gray-500 mt-2 ml-1')

            # ============================================================================
            # 3. CONFIGURATION PREVIEW (Moved Up)
            # ============================================================================
            with ui.card().classes('w-full').style(
                f'background: {theme.THEME_LIGHT["card"]}; '
                f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
                f'border-radius: 8px; '
                f'padding: 0; overflow: hidden;'
            ):
                with ui.expansion('👁️ Configuration Preview', icon='code').classes('w-full').style(
                    f'background: {theme.THEME_LIGHT["sidebar"]};'
                ).props('header-class="text-indigo-900 font-bold"'):
                    with ui.column().classes('p-4 w-full'):
                        config_preview = ui.markdown('# Loading...').classes('text-xs')

            # ============================================================================
            # 4. QUICK SETTINGS / PROCESSING MODE (Moved Down)
            # ============================================================================
            with ui.card().classes('w-full').style(
                f'background: {theme.THEME_LIGHT["card"]}; '
                f'border: 1px solid {theme.THEME_LIGHT["border"]}; '
                f'border-radius: 8px; '
                f'padding: 24px;'
            ) as processing_mode_card:  # NAMED for disabling later
                # Header Row
                with ui.row().classes('w-full items-center justify-between mb-2'):
                    with ui.row().classes('items-center'):
                        ui.icon('tune', size='24px').style(f'color: {theme.THEME_LIGHT["indigo"]};')
                        ui.label('Processing Mode').classes('text-xl font-semibold ml-2').style(
                            f'color: {theme.THEME_LIGHT["text"]}; '
                            f'font-family: var(--prajnya-font-body);'
                        )
                    
                    # Ready Badge
                    with ui.row().classes('items-center gap-2'):
                        ui.html('''
                            <div style="width: 8px; height: 8px; border-radius: 50%; background: #2D5A3D; animation: pulse 2s infinite;"></div>
                            <style>@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } }</style>
                        ''', sanitize=False)
                        
                        ui.label('Ready').classes('text-sm font-medium').style(
                            f'color: {theme.THEME_LIGHT["success"]};'
                        )

                ui.element('div').classes('w-full mb-4').style(
                    f'height: 2px; '
                    f'background: {theme.THEME_LIGHT["gold"]};'
                )
                
                # Cloud Mode Banner (Phase 4.8.2)
                cloud_banner = ui.column().classes('w-full mb-4').style(
                    'background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px;'
                )
                cloud_banner.set_visibility(False) # Hidden by default

                with cloud_banner:
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('info', size='sm').classes('text-blue-600')
                        ui.label('LOCAL VLM ONLY').classes('font-bold text-blue-900')
                    ui.label('Cloud API uses optimized single-pass processing. No preset selection needed.').classes('text-sm text-blue-800 mt-1') 

                # Preset Selector Header
                ui.label('PROCESSING MODE').classes('text-sm mb-3').style(
                    f'font-family: var(--prajnya-font-body); '
                    f'font-weight: 600; '
                    f'color: {theme.THEME_LIGHT["indigo"]}; '
                    f'text-transform: uppercase; '
                    f'letter-spacing: 0.5px;'
                )

                # Radio Cards Container
                radio_container = ui.column().classes('w-full gap-3')

            # --- Logic Functions for Cards ---

            def render_radio_cards():
                """Renders the radio cards based on current selection state."""
                radio_container.clear()

                with radio_container:
                    # Fast Mode Card
                    components.radio_card(
                        key='fast',
                        icon='⚡',
                        title=PRESETS['fast']['name'],
                        description=PRESETS['fast']['description'],
                        selected=(selected_preset == 'fast'),
                        on_select=on_preset_select
                    )

                    # Balanced Mode Card
                    components.radio_card(
                        key='balanced',
                        icon='⚖️',
                        title=PRESETS['balanced']['name'],
                        description=PRESETS['balanced']['description'],
                        selected=(selected_preset == 'balanced'),
                        on_select=on_preset_select
                    )

                    # Accurate Mode Card
                    components.radio_card(
                        key='accurate',
                        icon='🎯',
                        title=PRESETS['accurate']['name'],
                        description=PRESETS['accurate']['description'],
                        selected=(selected_preset == 'accurate'),
                        on_select=on_preset_select
                    )

            def on_preset_select(key: str):
                """Handle preset selection and update UI."""
                nonlocal selected_preset, current_config
                selected_preset = key
                render_radio_cards()

                if current_config:
                    preview_config = apply_preset_to_config(
                        current_config,
                        PRESETS[key],
                        model_selector.value,
                        use_cloud_mode
                    )
                    update_config_preview(preview_config, key)
                else:
                    update_preset_preview(key, model_selector.value)

                ui.notify(f'Switched to {PRESETS[key]["name"]}', type='info', position='top')

            def on_model_change(e):
                """Update configuration preview when model changes."""
                nonlocal current_config
                if current_config:
                    preview_config = apply_preset_to_config(
                        current_config,
                        PRESETS[selected_preset],
                        e.args,
                        use_cloud_mode
                    )
                    update_config_preview(preview_config, selected_preset)
                else:
                    update_preset_preview(selected_preset, e.args)

            def on_cloud_mode_toggle(e):
                """Handle cloud mode toggle and update UI."""
                nonlocal use_cloud_mode
                use_cloud_mode = e.args

                if e.args:  # Cloud mode ON
                    # Update available models to cloud options
                    model_selector.options = AVAILABLE_MODELS_CLOUD
                    model_selector.value = "gemini-3-pro"
                    model_selector.update()
                    model_info_text.text = 'Uses Lite Router: Title→Gemini, Text→PyMuPDF, Visual→Gemini'

                    # Update UI indicators
                    mode_badge.text = 'Cloud'
                    mode_badge.props(f'color=blue')

                    info_panel.style(f'border-left: 4px solid {theme.THEME_LIGHT["indigo"]};') # Blue border
                    mode_icon.name = 'cloud'
                    mode_icon.props('class=text-blue-700')
                    mode_title.text = 'Cloud Mode'
                    mode_title.props('class=text-blue-900')
                    
                    mode_desc.text = (
                        '• Requires: GEMINI_API_KEY in .env file\n'
                        '• Cost: ~$0.011/page (Gemini 3 Pro), ~$0.001/page (2.5 Flash)\n'
                        '• Free tier: 100 pages/day\n'
                        '• Speed: Fast (cloud processing, no GPU needed)'
                    )

                    ui.notify(
                        'Cloud mode enabled. Ensure GEMINI_API_KEY is set in .env file',
                        type='info',
                        position='top',
                        timeout=3000
                    )
                    processing_mode_card.style('opacity: 0.6; pointer-events: none; filter: grayscale(0.5);')
                    cloud_banner.set_visibility(True)
                    try:
                        fallback_model_adv.disable()
                        title_mode_adv.disable()
                        title_resolution_adv.disable()
                        body_mode_adv.disable()
                        temperature_adv.disable()
                        max_tokens_adv.disable()
                        repeat_penalty_adv.disable()
                        local_routing_container.set_visibility(False)
                        cloud_routing_container.set_visibility(True)
                    except:
                        pass # UI elements might not be initialized yet on first load

                else:  # Local mode ON
                    # Update available models to local options
                    model_selector.options = AVAILABLE_MODELS_LOCAL
                    model_selector.value = "olmocr-2-7b"
                    model_selector.update()
                    model_info_text.text = 'Used for title page and body page processing'

                    # Update UI indicators
                    mode_badge.text = 'Local'
                    mode_badge.props(f'color=green')
                    
                    info_panel.style(f'border-left: 4px solid {theme.THEME_LIGHT["success"]};') # Green border
                    mode_icon.name = 'computer'
                    mode_icon.props('class=text-green-700')
                    mode_title.text = 'Local Mode'
                    mode_title.props('class=text-green-900')
                    
                    mode_desc.text = (
                        '• Requires: llama-server running on localhost:8080\n'
                        '• Cost: Free (uses your GPU)\n'
                        '• Speed: Depends on GPU hardware\n'
                        '• VRAM: 7.6GB minimum (olmocr-7b)'
                    )

                    ui.notify('Local mode enabled', type='info', position='top', timeout=2000)
                    processing_mode_card.style('opacity: 1.0; pointer-events: auto; filter: none;')
                    cloud_banner.set_visibility(False)
                    try:
                        fallback_model_adv.enable()
                        title_mode_adv.enable()
                        title_resolution_adv.enable()
                        body_mode_adv.enable()
                        temperature_adv.enable()
                        max_tokens_adv.enable()
                        repeat_penalty_adv.enable()
                        local_routing_container.set_visibility(True)
                        cloud_routing_container.set_visibility(False)
                    except:
                        pass

                # Enable Apply button (configuration changed)
                btn_apply.enable()
                btn_preview.enable()
            
            # Hook up events
            model_selector.on('update:model-value', on_model_change)
            cloud_switch.on('update:model-value', on_cloud_mode_toggle)

            # Initial render
            render_radio_cards()

        # Initialize preview with default preset on page load
        ui.timer(0.1, lambda: update_preset_preview(selected_preset, model_selector.value), once=True)

        # Advanced Settings Panel (Level 2) - Separate Card
        with ui.card().classes('w-full').style(
            f'background: {theme.THEME_LIGHT["card"]}; '  # FIXED: Matches main card cream
            f'border: 1px dashed {theme.THEME_LIGHT["border_gold"]}; '  # FIXED: Dashed Gold border
            f'border-radius: 8px;'
        ):
            # ========================================================================
            # ADVANCED SETTINGS PANEL (Level 2) - Sprint 3.2 Phase 2
            # ========================================================================
            with ui.expansion('⚙️ Advanced Settings', icon='tune').classes('w-full mt-4') as advanced_panel:
                with ui.column().classes('w-full gap-4 p-4'):
                    # Tabs for different setting categories
                    with ui.tabs().classes('w-full') as tabs:
                        models_tab = ui.tab('Models', icon='smart_toy')
                        processing_tab = ui.tab('Processing', icon='settings')
                        quality_tab = ui.tab('Quality', icon='verified')
                        routing_tab = ui.tab('Routing', icon='route')

                    # FIXED: bg-transparent prevents white block overlay on cream card
                    with ui.tab_panels(tabs, value=models_tab).classes('w-full bg-transparent'):
                        # ================================================================
                        # MODELS TAB
                        # ================================================================
                        with ui.tab_panel(models_tab):
                            with ui.column().classes('w-full gap-4'):
                                ui.label('Model Configuration').classes('text-lg font-bold')
                                ui.label('Select primary and fallback VLM models').classes('text-sm text-gray-500 mb-2')

                                # Primary Model
                                ui.label('Primary Model').classes('text-md font-semibold mt-4')
                                primary_model_adv = ui.select(
                                    AVAILABLE_MODELS,
                                    value="olmocr-2-7b",
                                    label="Primary VLM"
                                ).classes('w-full')
                                ui.label('Used for title page and body page processing').classes('text-xs text-gray-500')

                                # Fallback Model
                                ui.label('Fallback Model').classes('text-md font-semibold mt-4')
                                fallback_model_adv = ui.select(
                                    AVAILABLE_MODELS,
                                    value="olmocr-7b",
                                    label="Fallback VLM"
                                ).classes('w-full')
                                ui.label('Used when primary model fails or for Tier 5 routing').classes('text-xs text-gray-500')

                        # ================================================================
                        # PROCESSING TAB
                        # ================================================================
                        with ui.tab_panel(processing_tab):
                            with ui.column().classes('w-full gap-4'):
                                ui.label('Processing Configuration').classes('text-lg font-bold')
                                ui.label('Configure processing modes and VLM parameters').classes('text-sm text-gray-500 mb-2')

                                # Title Page Mode
                                ui.label('Title Page Mode').classes('text-md font-semibold mt-4')
                                title_mode_adv = ui.radio(
                                    {'single_shot': 'Single-Shot (~1 min, 75% accuracy)',
                                     'two_shot': 'Two-Shot (~2 min, 95% accuracy)'},
                                    value='two_shot'
                                ).props('inline').classes('mb-2')

                                # Title Resolution Slider
                                ui.label('Title Page Resolution').classes('text-md font-semibold mt-4')
                                with ui.row().classes('w-full items-center gap-4'):
                                    ui.label('512px').classes('text-xs text-gray-500')
                                    title_resolution_adv = ui.slider(min=512, max=2048, step=8, value=1288).classes('flex-grow')
                                    title_resolution_label = ui.label('1288px').classes('text-sm font-mono')
                                    title_resolution_adv.on('update:model-value', lambda e: setattr(title_resolution_label, 'text', f'{int(e.args)}px'))
                                ui.label('Higher resolution = better DOI and metadata capture').classes('text-xs text-gray-500')

                                ui.separator().classes('my-4')

                                # Body Page Mode
                                ui.label('Body Page Mode').classes('text-md font-semibold')
                                body_mode_adv = ui.radio(
                                    {'pymupdf_only': 'PyMuPDF Only (fastest, text-only)',
                                     'adaptive': 'Adaptive Routing (smart, recommended)',
                                     'vlm_only': 'VLM Only (slowest, most accurate)'},
                                    value='adaptive'
                                ).classes('mb-2')

                                ui.separator().classes('my-4')

                                # VLM Parameters (Sprint 3.4)
                                ui.label('VLM Parameters').classes('text-md font-semibold')
                                ui.label('Fine-tune model generation settings').classes('text-xs text-gray-500 mb-3')

                                # Temperature
                                ui.label('Temperature').classes('text-sm font-medium mt-3')
                                with ui.row().classes('w-full items-center gap-4'):
                                    ui.label('0.0').classes('text-xs text-gray-500')
                                    temperature_adv = ui.slider(min=0, max=1, step=0.05, value=0.4).classes('flex-grow')
                                    temperature_label = ui.label('0.40').classes('text-sm font-mono')
                                    temperature_adv.on('update:model-value', lambda e: setattr(temperature_label, 'text', f'{e.args:.2f}'))
                                ui.label('Lower = more deterministic, Higher = more creative (0.0-1.0)').classes('text-xs text-gray-500')

                                # Max Tokens
                                ui.label('Max Tokens').classes('text-sm font-medium mt-3')
                                with ui.row().classes('w-full items-center gap-4'):
                                    ui.label('1024').classes('text-xs text-gray-500')
                                    max_tokens_adv = ui.slider(min=1024, max=16384, step=512, value=4096).classes('flex-grow')
                                    max_tokens_label = ui.label('4096').classes('text-sm font-mono')
                                    max_tokens_adv.on('update:model-value', lambda e: setattr(max_tokens_label, 'text', f'{int(e.args)}'))
                                ui.label('Maximum response length (1024-16384 tokens)').classes('text-xs text-gray-500')

                                # Repeat Penalty
                                ui.label('Repeat Penalty').classes('text-sm font-medium mt-3')
                                with ui.row().classes('w-full items-center gap-4'):
                                    ui.label('1.0').classes('text-xs text-gray-500')
                                    repeat_penalty_adv = ui.slider(min=1.0, max=2.0, step=0.05, value=1.15).classes('flex-grow')
                                    repeat_penalty_label = ui.label('1.15').classes('text-sm font-mono')
                                    repeat_penalty_adv.on('update:model-value', lambda e: setattr(repeat_penalty_label, 'text', f'{e.args:.2f}'))
                                ui.label('Prevents repetition in generated text (1.0-2.0)').classes('text-xs text-gray-500')

                            
                                ui.separator().classes('my-4')

                                # ================================================================
                                # ASYNC IO PARALLELISM (Phase 4.9)
                                # ================================================================
                                with ui.expansion('Async Processing (Phase 4.9)', icon='speed').classes('w-full border rounded-lg p-2'):
                                    ui.label('Parallel VLM Execution').classes('text-md font-semibold text-blue-800')
                                    async_enabled = ui.switch('Enable AsyncIO Concurrency', value=False).props('color=blue').classes('my-2')
                                    ui.label('Process multiple pages simultaneously using AsyncIO').classes('text-xs text-gray-500 mb-2')
                                    
                                    with ui.row().classes('w-full items-center gap-4 mt-2'):
                                        vram_input = ui.number('System VRAM (GB)', value=0, min=0, max=128, step=1).classes('w-32')
                                        
                                        # Reactive Concurrency Display
                                        concurrency_info = ui.label('Est. Capacity: Sequential (Safe)').classes('text-sm font-mono text-gray-700')
                                        
                                        def get_model_vram_usage(model_name):
                                            """Read VRAM requirement from profile YAML."""
                                            try:
                                                profile_path = Path("src/processing/strategies/config/vlm_profiles") / f"{model_name}.yaml"
                                                if profile_path.exists():
                                                    with open(profile_path, 'r', encoding='utf-8') as f:
                                                        data = yaml.safe_load(f)
                                                        # Traverse: resource_requirements -> vram_per_instance_gb
                                                        req = data.get('resource_requirements', {}).get('vram_per_instance_gb', 8.0)
                                                        return float(req)
                                            except Exception as e:
                                                logger.warning(f"Failed to load profile for {model_name}: {e}")
                                            return 8.0 # Default fallback
                                        
                                        def update_concurrency_label(e=None):
                                            # Get current model requirement
                                            current_model = model_selector.value
                                            vram_per_instance = get_model_vram_usage(current_model)
                                            
                                            v_gb = float(vram_input.value or 0)
                                            
                                            # Dynamic calculation
                                            est_conc = int(v_gb / vram_per_instance)
                                            if est_conc < 1: est_conc = 0
                                            
                                            if est_conc > 0 and async_enabled.value:
                                                concurrency_info.text = f'Est. Capacity: ~{est_conc} concurrent pages ({vram_per_instance}GB/task)'
                                                concurrency_info.classes(remove='text-red-600', add='text-green-700')
                                            elif est_conc == 0 and async_enabled.value:
                                                concurrency_info.text = f'Insufficient VRAM (Need ~{vram_per_instance}GB+)'
                                                concurrency_info.classes(remove='text-green-700', add='text-red-600')
                                            else:
                                                concurrency_info.text = 'Async Disabled / Sequential'
                                                concurrency_info.classes(remove='text-red-600 text-green-700')

                                        vram_input.on('update:model-value', update_concurrency_label)
                                        async_enabled.on('update:model-value', update_concurrency_label)
                                        # Hook into model change safely
                                        model_selector.on('update:model-value', lambda e: update_concurrency_label(None))
                                        
                                    ui.label('Note: requires ~8GB VRAM per concurrent instance.').classes('text-xs text-gray-500 italic mt-1')

                        # ================================================================
                        # QUALITY TAB
                        # ================================================================
                        with ui.tab_panel(quality_tab):
                            with ui.column().classes('w-full gap-4'):
                                ui.label('Quality & Metadata Thresholds').classes('text-lg font-bold')
                                ui.label('Configure quality scores and metadata confidence').classes('text-sm text-gray-500 mb-2')

                                # Quality Scores Section
                                ui.label('Quality Thresholds').classes('text-md font-semibold mt-4')

                                # Minimum Overall Score
                                ui.label('Minimum Overall Score').classes('text-sm font-medium mt-3')
                                with ui.row().classes('w-full items-center gap-4'):
                                    ui.label('0.0').classes('text-xs text-gray-500')
                                    min_overall_score = ui.slider(min=0, max=1, step=0.05, value=0.5).classes('flex-grow')
                                    min_overall_label = ui.label('0.50').classes('text-sm font-mono')
                                    min_overall_score.on('update:model-value', lambda e: setattr(min_overall_label, 'text', f'{e.args:.2f}'))
                                ui.label('Documents below this score will be flagged').classes('text-xs text-gray-500')

                                # Minimum Page Score
                                ui.label('Minimum Page Score').classes('text-sm font-medium mt-3')
                                with ui.row().classes('w-full items-center gap-4'):
                                    ui.label('0.0').classes('text-xs text-gray-500')
                                    min_page_score = ui.slider(min=0, max=1, step=0.05, value=0.3).classes('flex-grow')
                                    min_page_label = ui.label('0.30').classes('text-sm font-mono')
                                    min_page_score.on('update:model-value', lambda e: setattr(min_page_label, 'text', f'{e.args:.2f}'))
                                ui.label('Individual pages below this will trigger warnings').classes('text-xs text-gray-500')

                                # Warning Threshold
                                ui.label('Warning Below Score').classes('text-sm font-medium mt-3')
                                with ui.row().classes('w-full items-center gap-4'):
                                    ui.label('0.0').classes('text-xs text-gray-500')
                                    warn_below_score = ui.slider(min=0, max=1, step=0.05, value=0.7).classes('flex-grow')
                                    warn_below_label = ui.label('0.70').classes('text-sm font-mono')
                                    warn_below_score.on('update:model-value', lambda e: setattr(warn_below_label, 'text', f'{e.args:.2f}'))
                                ui.label('Show warnings for scores below this threshold').classes('text-xs text-gray-500')

                                ui.separator().classes('my-4')

                                # Metadata Confidence Section
                                ui.label('Metadata Confidence').classes('text-md font-semibold')

                                # DOI Confidence Threshold
                                ui.label('DOI Confidence Threshold').classes('text-sm font-medium mt-3')
                                with ui.row().classes('w-full items-center gap-4'):
                                    ui.label('0.0').classes('text-xs text-gray-500')
                                    doi_confidence = ui.slider(min=0, max=1, step=0.05, value=0.85).classes('flex-grow')
                                    doi_confidence_label = ui.label('0.85').classes('text-sm font-mono')
                                    doi_confidence.on('update:model-value', lambda e: setattr(doi_confidence_label, 'text', f'{e.args:.2f}'))
                                ui.label('Above: use regex extraction, Below: use VLM').classes('text-xs text-gray-500')

                                # ISBN Confidence Threshold
                                ui.label('ISBN Confidence Threshold').classes('text-sm font-medium mt-3')
                                with ui.row().classes('w-full items-center gap-4'):
                                    ui.label('0.0').classes('text-xs text-gray-500')
                                    isbn_confidence = ui.slider(min=0, max=1, step=0.05, value=0.85).classes('flex-grow')
                                    isbn_confidence_label = ui.label('0.85').classes('text-sm font-mono')
                                    isbn_confidence.on('update:model-value', lambda e: setattr(isbn_confidence_label, 'text', f'{e.args:.2f}'))
                                ui.label('Threshold for ISBN hybrid extraction').classes('text-xs text-gray-500')

                                ui.separator().classes('my-4')

                                # Validation Options
                                ui.label('Validation Rules').classes('text-md font-semibold')
                                validate_doi = ui.checkbox('Validate DOI format', value=True).classes('mt-2')
                                validate_year = ui.checkbox('Validate year range (1400-2026)', value=True).classes('mt-1')
                                ui.label('Enable format validation for extracted metadata').classes('text-xs text-gray-500 ml-6')

                        # ================================================================
                        # ROUTING TAB
                        # ================================================================
                        with ui.tab_panel(routing_tab):
                            with ui.column().classes('w-full gap-4'):
                                ui.label('Adaptive Routing Configuration').classes('text-lg font-bold')
                                ui.label('Enable/disable 5-tier adaptive routing system').classes('text-sm text-gray-500 mb-2')
                                local_routing_container = ui.column().classes('w-full gap-4')
                                cloud_routing_container = ui.column().classes('w-full gap-4').style(
                                    f'background: {theme.THEME_LIGHT["canvas"]}; padding: 16px; border-radius: 8px; border: 1px solid {theme.THEME_LIGHT["indigo"]};'
                                )
                                cloud_routing_container.set_visibility(False) # Hidden by default

                                # ================= LOCAL ROUTING (5-Tier) =================
                                with local_routing_container:
                                    # Global Routing Toggle
                                    ui.label('Routing Mode').classes('text-md font-semibold mt-4')
                                    routing_enabled = ui.checkbox('Enable Adaptive Routing (5-Tier System)', value=True).classes('mt-2')
                                    ui.label('When enabled, pages are intelligently routed to optimal processing tier').classes('text-xs text-gray-500 ml-6')

                                    ui.separator().classes('my-4')

                                    # Tier Information Display
                                    ui.label('Routing Tiers').classes('text-md font-semibold')
                                    ui.label('The 5-tier system automatically selects the best processing method for each page').classes('text-xs text-gray-500 mb-3')

                                    # Tier 1
                                    with ui.card().classes('w-full p-3 mt-2').style('background: #f0f9ff; border-left: 4px solid #0284c7;'):
                                        with ui.row().classes('w-full items-center gap-2'):
                                            ui.icon('flash_on', size='sm').classes('text-blue-600')
                                            ui.label('Tier 1: PyMuPDF').classes('font-bold text-blue-800')
                                        ui.label('Pure text extraction (fastest, ~1s/page)').classes('text-sm text-gray-700 mt-1')
                                        ui.label('Used for: Text-only pages with no visual elements').classes('text-xs text-gray-600 mt-1')

                                    # Tier 2
                                    with ui.card().classes('w-full p-3 mt-2').style('background: #f0fdf4; border-left: 4px solid #16a34a;'):
                                        with ui.row().classes('w-full items-center gap-2'):
                                            ui.icon('content_cut', size='sm').classes('text-green-600')
                                            ui.label('Tier 2: Surgical VLM').classes('font-bold text-green-800')
                                        ui.label('Targeted VLM processing (~30s/page)').classes('text-sm text-gray-700 mt-1')
                                        ui.label('Used for: Pages with isolated figures or tables').classes('text-xs text-gray-600 mt-1')

                                    # Tier 3
                                    with ui.card().classes('w-full p-3 mt-2').style('background: #fef3c7; border-left: 4px solid #f59e0b;'):
                                        with ui.row().classes('w-full items-center gap-2'):
                                            ui.icon('table_chart', size='sm').classes('text-amber-600')
                                            ui.label('Tier 3: Single-Shot VLM').classes('font-bold text-amber-800')
                                        ui.label('OlmOCR single-shot (~60s/page)').classes('text-sm text-gray-700 mt-1')
                                        ui.label('Used for: Large tables without figures').classes('text-xs text-gray-600 mt-1')

                                    # Tier 4
                                    with ui.card().classes('w-full p-3 mt-2').style('background: #fae8ff; border-left: 4px solid #9333ea;'):
                                        with ui.row().classes('w-full items-center gap-2'):
                                            ui.icon('auto_awesome', size='sm').classes('text-purple-600')
                                            ui.label('Tier 4: Two-Shot VLM').classes('font-bold text-purple-800')
                                        ui.label('OlmOCR two-shot (~90s/page, most accurate)').classes('text-sm text-gray-700 mt-1')
                                        ui.label('Used for: Complex pages with mixed content').classes('text-xs text-gray-600 mt-1')

                                    # Tier 5
                                    with ui.card().classes('w-full p-3 mt-2').style('background: #fee2e2; border-left: 4px solid #dc2626;'):
                                        with ui.row().classes('w-full items-center gap-2'):
                                            ui.icon('settings_backup_restore', size='sm').classes('text-red-600')
                                            ui.label('Tier 5: Fallback').classes('font-bold text-red-800')
                                        ui.label('Tested fallback method (~60-90s/page)').classes('text-sm text-gray-700 mt-1')
                                        ui.label('Used for: Unknown page types or errors').classes('text-xs text-gray-600 mt-1')

                                # ================= CLOUD ROUTING (Lite Router) =================
                                with cloud_routing_container:
                                    ui.label('☁️ Cloud Lite Router').classes('text-lg font-bold text-blue-900')
                                    ui.label('Optimized 3-tier routing strategy for Cloud APIs').classes('text-sm text-blue-800 mb-2')
                                    
                                    # Cost Efficiency Banner
                                    with ui.row().classes('w-full p-2 bg-blue-100 rounded items-center gap-2 mb-2'):
                                        ui.icon('savings', size='sm').classes('text-blue-700')
                                        ui.label('Estimated Cost: $0.005 / document (avg)').classes('text-sm font-bold text-blue-800')

                                    # Tier A: Title Page
                                    with ui.card().classes('w-full p-3 mt-2').style('background: white; border-left: 4px solid #3b82f6;'):
                                        with ui.row().classes('w-full items-center gap-2'):
                                            ui.icon('title', size='sm').classes('text-blue-600')
                                            ui.label('Tier A: Title Page Analysis').classes('font-bold text-blue-800')
                                        ui.label('Gemini 2.0 Flash').classes('text-sm text-gray-700 mt-1')
                                        ui.label('Extracts precise metadata (DOI, Title, Authors)').classes('text-xs text-gray-500')

                                    # Tier B: Text Content
                                    with ui.card().classes('w-full p-3 mt-2').style('background: white; border-left: 4px solid #10b981;'):
                                        with ui.row().classes('w-full items-center gap-2'):
                                            ui.icon('description', size='sm').classes('text-green-600')
                                            ui.label('Tier B: Text-Heavy Pages').classes('font-bold text-green-800')
                                        ui.label('PyMuPDF (Local)').classes('text-sm text-gray-700 mt-1')
                                        ui.label('Instant text extraction for standard pages (Free)').classes('text-xs text-gray-500')

                                    # Tier C: Visual Content
                                    with ui.card().classes('w-full p-3 mt-2').style('background: white; border-left: 4px solid #8b5cf6;'):
                                        with ui.row().classes('w-full items-center gap-2'):
                                            ui.icon('image', size='sm').classes('text-purple-600')
                                            ui.label('Tier C: Complex/Visual Pages').classes('font-bold text-purple-800')
                                        ui.label('Gemini 1.5 Pro').classes('text-sm text-gray-700 mt-1')
                                        ui.label('Deep reasoning for tables, figures, and diagrams').classes('text-xs text-gray-500')

                                    ui.label('Note: Routing is handled automatically by the Cloud API.').classes('text-xs italic text-blue-800 mt-2')

                                ui.separator().classes('my-4')

                                # Future: Individual Tier Toggles
                                ui.label('🔧 Advanced Routing Controls').classes('text-md font-semibold mt-4')
                                ui.label('Individual tier enable/disable controls coming in future update').classes('text-xs text-gray-500 italic')
                                ui.label('For now, routing is all-or-nothing (adaptive mode on/off)').classes('text-xs text-gray-500 italic')

            ui.separator().classes('my-4')

            # Action Buttons
            with ui.row().classes('w-full gap-2 mt-6'):
                # Primary buttons moved to Sticky Header
                pass

                ui.space()

                btn_reset = ui.button(
                    'Reset to Defaults',
                    icon='refresh',
                    on_click=lambda: asyncio.create_task(reset_to_defaults_async())
                ).props('flat color=negative')

                btn_restore = ui.button(
                    'Restore Factory Settings',
                    icon='settings_backup_restore',
                    on_click=lambda: asyncio.create_task(restore_factory_default_async())
                ).props('flat color=warning').style('margin-left: 8px;')

            ui.separator().classes('my-6')

            # Expert Mode - YAML Editor (Sprint 3.5)
            with ui.card().classes('w-full mt-4 p-4').style('background: #FFF3E0; border-left: 4px solid #FF6B35;'):
                with ui.row().classes('w-full items-center gap-2'):
                    ui.icon('code', size='sm').classes('text-orange-600')
                    ui.label('Expert Mode').classes('text-md font-bold text-gray-800')
                    ui.space()
                    btn_yaml_editor = ui.button(
                        'Edit YAML Directly',
                        icon='edit_note',
                        on_click=lambda: open_yaml_editor()
                    ).props('color=orange-7 size=sm')

    # --- Helper Functions (Defined AFTER mount, never inside) ---

    async def load_config_async():
        """Load the current strategy configuration from YAML file."""
        nonlocal current_config, config_loaded

        try:
            logger.info("=== Starting config load ===")

            # Load YAML file
            logger.info(f"Loading YAML from: {STRATEGY_CONFIG_PATH}")
            config = load_strategy_config(STRATEGY_CONFIG_PATH)
            logger.info(f"YAML loaded successfully, version: {config.get('version', 'unknown')}")

            current_config = config
            config_loaded = True
            logger.info("Config stored in memory")

            # Update UI with loaded values
            logger.info("Updating UI from config...")
            update_ui_from_config(config)
            logger.info("UI updated successfully")

            # Enable apply and preview buttons
            logger.info("Enabling buttons...")
            btn_apply.enable()
            btn_preview.enable()
            logger.info("Buttons enabled")

            # Success notification (may not work in async - NiceGUI limitation)
            try:
                ui.notify(f'Configuration loaded successfully (v{config.get("version", "unknown")})', type='positive', icon='check_circle')
            except:
                pass  # Notifications don't work in async background tasks
            logger.info(f"✅ Successfully loaded strategy config from {STRATEGY_CONFIG_PATH}")

        except Exception as e:
            try:
                ui.notify(f'Failed to load config: {str(e)}', type='negative', icon='error')
            except:
                pass
            logger.error(f"❌ Failed to load strategy config: {e}", exc_info=True)

    async def apply_config_async():
        """Apply the selected preset and save configuration."""
        nonlocal current_config, selected_preset

        if not config_loaded:
            try:
                ui.notify('Please load configuration first!', type='warning')
            except:
                pass
            return

        try:
            # Get selected preset
            preset_key = selected_preset
            preset = PRESETS[preset_key]

            # Get selected model
            model = model_selector.value

            logger.info(f"Applying preset '{preset_key}' with model '{model}' (cloud_mode={use_cloud_mode})")

            # Update config with preset values
            updated_config = apply_preset_to_config(current_config, preset, model, use_cloud_mode)
            logger.info("Config updated with preset values")

            # Apply advanced panel overrides (Sprint 3.2 Phase 2)
            updated_config = apply_advanced_settings(updated_config)
            logger.info("Advanced settings applied")

            # Validate configuration
            validation_errors = validate_config(updated_config)
            if validation_errors:
                error_msg = f'Validation failed: {", ".join(validation_errors)}'
                try:
                    ui.notify(error_msg, type='negative')
                except:
                    pass
                logger.error(f"Validation errors: {validation_errors}")
                return

            logger.info("Config validation passed")

            # Create backup
            try:
                backup_path = create_config_backup(STRATEGY_CONFIG_PATH)
                logger.info(f"Created backup: {backup_path}")
            except Exception as backup_err:
                logger.warning(f"Backup creation failed (non-critical): {backup_err}")

            # Save updated config
            save_strategy_config(STRATEGY_CONFIG_PATH, updated_config)
            logger.info("Config saved successfully")

            # Trigger hot-reload (Sprint 3.2 implementation)
            logger.info("🔄 Triggering orchestrator hot-reload...")

            try:
                reset_orchestrator()
                # Reinitialize with fresh config
                fresh_orchestrator = get_orchestrator()
                if fresh_orchestrator:
                    logger.info("✅ Hot-reload successful - orchestrator reinitialized with new config")
                else:
                    logger.warning("⚠️ Hot-reload returned None - orchestrator may not be available")
            except ValueError as reload_err:
                # Catch configuration errors (e.g., missing API key for cloud mode)
                error_str = str(reload_err)
                if "GEMINI_API_KEY" in error_str:
                    logger.error(f"❌ Cloud mode enabled but API key missing: {reload_err}")
                    try:
                        ui.notify(
                            'Config saved, but GEMINI_API_KEY is missing. Add it to .env file to use cloud mode.',
                            type='warning',
                            icon='cloud_off',
                            timeout=5000
                        )
                    except:
                        pass
                else:
                    logger.error(f"❌ Hot-reload failed (config error): {reload_err}", exc_info=True)
                # Continue anyway - config is saved, just not hot-reloaded
            except Exception as reload_err:
                logger.error(f"❌ Hot-reload failed: {reload_err}", exc_info=True)
                # Continue anyway - config is saved, just not hot-reloaded

            # Update current config
            logger.info("Updating in-memory config...")
            current_config = updated_config
            logger.info("In-memory config updated")

            # Update preview
            logger.info("Updating config preview...")
            try:
                update_config_preview(updated_config, preset_key)
                logger.info("Config preview updated")
            except Exception as preview_err:
                logger.warning(f"Failed to update preview (non-critical): {preview_err}")

            # Success notification (may not work in async)
            try:
                ui.notify(f'Configuration saved successfully! ({preset["name"]})', type='positive', icon='check_circle')
            except:
                pass
            logger.info(f"✅ Successfully applied preset '{preset_key}' with model '{model}'")

            # ---------------------------------------------------------
            # Router Mode Integration (Phase 5.5)
            # ---------------------------------------------------------
            # If server manager is available, switch the active model immediately
            state = AppState.get_instance()
            if state.server_manager and not use_cloud_mode:
                logger.info(f"Switching VLM server to model: {model}")
                
                async def switch_task():
                    try:
                        loop = asyncio.get_running_loop()
                        success = await loop.run_in_executor(
                            None, 
                            lambda: state.server_manager.switch_model(model)
                        )
                        if success:
                            ui.notify(f"Server switched to {model}", type='positive')
                        else:
                            ui.notify(f"Failed to switch to {model}", type='negative')
                    except Exception as e:
                        logger.error(f"Error switching model: {e}")
                
                asyncio.create_task(switch_task())

        except Exception as e:
            error_msg = f'Failed to apply config: {str(e)}'
            try:
                ui.notify(error_msg, type='negative', icon='error')
            except:
                pass
            logger.error(f"Failed to apply configuration: {e}", exc_info=True)

            # Show more detailed error in notification
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

    async def reset_to_defaults_async():
        """Reset UI to balanced preset (doesn't affect saved file)."""
        nonlocal selected_preset, current_config

        # Reset to default settings
        selected_preset = "balanced"
        model_selector.value = "olmocr-2-7b"

        # Re-render radio cards to show updated selection
        render_radio_cards()

        # If config is loaded, also reset Advanced Settings to match preset
        if config_loaded and current_config:
            try:
                import copy
                preset = PRESETS["balanced"]
                # Create a config with preset defaults
                reset_config = apply_preset_to_config(copy.deepcopy(current_config), preset, "olmocr-2-7b")
                # Update Advanced Settings UI to match
                update_advanced_settings_from_config(reset_config)
                logger.info("Advanced Settings reset to Balanced preset defaults")
            except Exception as e:
                logger.warning(f"Failed to reset Advanced Settings: {e}")

        # Update preview
        update_preset_preview(selected_preset, model_selector.value)

        # User feedback
        ui.notify('Reset to default settings (Balanced mode)', type='info')
        logger.info("Reset UI to default preset: balanced")

    async def restore_factory_default_async():
        """Restore original factory default configuration from pristine backup."""
        nonlocal current_config, config_loaded

        logger.info("=== Restoring factory default configuration ===")

        try:
            # Check if factory default exists
            if not FACTORY_DEFAULT_PATH.exists():
                error_msg = f"Factory default not found: {FACTORY_DEFAULT_PATH}"
                logger.error(error_msg)
                try:
                    ui.notify('Factory default file not found!', type='negative')
                except:
                    pass
                return

            logger.info(f"Loading factory default from: {FACTORY_DEFAULT_PATH}")

            # Create backup of current config before restoring
            backup_path = create_config_backup(STRATEGY_CONFIG_PATH)
            logger.info(f"Created safety backup: {backup_path}")

            # Load factory default
            factory_config = load_strategy_config(FACTORY_DEFAULT_PATH)
            logger.info("Factory default loaded successfully")

            # Save factory default as current config
            save_strategy_config(STRATEGY_CONFIG_PATH, factory_config)
            logger.info("Factory default restored to active config")

            # Update in-memory config
            current_config = factory_config
            config_loaded = True

            # Enable apply and preview buttons
            btn_apply.enable()
            btn_preview.enable()

            # Update UI
            update_ui_from_config(factory_config)
            logger.info("UI updated with factory defaults")

            # Also update Advanced Settings UI
            update_advanced_settings_from_config(factory_config)
            logger.info("Advanced Settings UI updated with factory defaults")

            # Show success notification (may not work in async)
            try:
                ui.notify('Factory settings restored successfully!', type='positive', icon='settings_backup_restore')
            except:
                pass
            logger.info("✅ Successfully restored factory default configuration")

        except Exception as e:
            logger.error(f"❌ Failed to restore factory defaults: {e}", exc_info=True)
            try:
                ui.notify(f'Failed to restore factory settings: {str(e)}', type='negative')
            except:
                pass

    def preview_advanced_settings():
        """Preview what config would look like with current advanced settings."""
        nonlocal current_config, selected_preset

        if not config_loaded:
            ui.notify('Please load configuration first!', type='warning')
            return

        try:
            import copy
            # Start with current preset
            preset = PRESETS[selected_preset]
            model = model_selector.value

            # Apply preset to current config
            preview_config = apply_preset_to_config(copy.deepcopy(current_config), preset, model)

            # Apply advanced settings overrides
            preview_config = apply_advanced_settings(preview_config)

            # Update preview
            update_config_preview(preview_config, selected_preset)

            ui.notify('Preview updated with advanced settings', type='info', icon='visibility')
            logger.info("Preview updated with advanced settings")

        except Exception as e:
            ui.notify(f'Failed to preview: {str(e)}', type='negative')
            logger.error(f"Failed to preview advanced settings: {e}", exc_info=True)

    def apply_advanced_settings(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply settings from Advanced Panel to configuration.

        Sprint 3.2 Phase 2: Read values from advanced controls and override
        config dict. This allows fine-grained control beyond presets.

        Args:
            config: Configuration dict to update

        Returns:
            Updated configuration dict
        """
        import copy
        updated = copy.deepcopy(config)

        try:
            # Models Tab - Update profile registry
            if 'profile_registry' not in updated:
                updated['profile_registry'] = {}

            updated['profile_registry']['default'] = primary_model_adv.value
            updated['profile_registry']['fallback'] = fallback_model_adv.value
            logger.info(f"Advanced: primary={primary_model_adv.value}, fallback={fallback_model_adv.value}")

            # Processing Tab - Title page settings
            if 'vlm_provider' not in updated:
                updated['vlm_provider'] = {}
            if 'title_page' not in updated['vlm_provider']:
                updated['vlm_provider']['title_page'] = {}

            updated['vlm_provider']['title_page']['processing_mode'] = title_mode_adv.value
            updated['vlm_provider']['title_page']['resolution_override'] = int(title_resolution_adv.value)
            logger.info(f"Advanced: title_mode={title_mode_adv.value}, resolution={int(title_resolution_adv.value)}")

            # Processing Tab - Body page settings
            if 'body_pages' not in updated['vlm_provider']:
                updated['vlm_provider']['body_pages'] = {}

            updated['vlm_provider']['body_pages']['processing_mode'] = body_mode_adv.value
            logger.info(f"Advanced: body_mode={body_mode_adv.value}")

            # AsyncIO Settings (Phase 4.9)
            if 'concurrency' not in updated['vlm_provider']:
                updated['vlm_provider']['concurrency'] = {}
            
            updated['vlm_provider']['concurrency']['enabled'] = async_enabled.value
            updated['vlm_provider']['concurrency']['vram_capacity_gb'] = float(vram_input.value or 0)
            # max_concurrent is calculated at runtime by strategy, config just stores capacity
            updated['vlm_provider']['concurrency']['max_concurrent'] = "auto" 
            
            logger.info(f"Advanced: async_enabled={async_enabled.value}, vram={vram_input.value}")

            # ===================================================================
            # VLM PARAMETERS (Sprint 3.4)
            # ===================================================================

            # Create vlm_parameter_overrides section if it doesn't exist
            if 'vlm_parameter_overrides' not in updated:
                updated['vlm_parameter_overrides'] = {}

            # Store VLM parameters for profile overrides
            updated['vlm_parameter_overrides']['temperature'] = float(temperature_adv.value)
            updated['vlm_parameter_overrides']['max_tokens'] = int(max_tokens_adv.value)
            updated['vlm_parameter_overrides']['repeat_penalty'] = float(repeat_penalty_adv.value)
            logger.info(f"Advanced: VLM params temp={temperature_adv.value}, max_tokens={int(max_tokens_adv.value)}, repeat_penalty={repeat_penalty_adv.value}")

            # ===================================================================
            # QUALITY TAB (Sprint 3.3)
            # ===================================================================

            # Quality thresholds
            if 'quality' not in updated:
                updated['quality'] = {}

            updated['quality']['minimum_overall_score'] = float(min_overall_score.value)
            updated['quality']['minimum_page_score'] = float(min_page_score.value)
            updated['quality']['warn_below_score'] = float(warn_below_score.value)
            logger.info(f"Advanced: quality thresholds={min_overall_score.value}/{min_page_score.value}/{warn_below_score.value}")

            # Metadata confidence
            if 'metadata' not in updated:
                updated['metadata'] = {}

            updated['metadata']['doi_confidence_threshold'] = float(doi_confidence.value)
            updated['metadata']['isbn_confidence_threshold'] = float(isbn_confidence.value)
            updated['metadata']['validate_doi_format'] = validate_doi.value
            updated['metadata']['validate_year_range'] = validate_year.value
            logger.info(f"Advanced: metadata confidence doi={doi_confidence.value}, isbn={isbn_confidence.value}")

            # ===================================================================
            # ROUTING TAB (Sprint 3.3)
            # ===================================================================

            # Routing mode (affects body_pages processing_mode)
            if routing_enabled.value:
                # Adaptive routing is enabled
                updated['vlm_provider']['body_pages']['processing_mode'] = 'adaptive'
                logger.info("Advanced: adaptive routing ENABLED")
            else:
                # Routing disabled - default to vlm_only
                if body_mode_adv.value == 'adaptive':
                    # User had routing enabled, switch to vlm_only
                    updated['vlm_provider']['body_pages']['processing_mode'] = 'vlm_only'
                    logger.info("Advanced: adaptive routing DISABLED, using vlm_only")
                # Otherwise keep body_mode_adv value

            return updated

        except Exception as e:
            logger.error(f"Failed to apply advanced settings: {e}", exc_info=True)
            # Return original config if error
            return config

    def update_advanced_settings_from_config(config: Dict[str, Any]):
        """Update ONLY Advanced Settings panel from config (doesn't touch Quick Settings)."""
        try:
            # Models Tab
            primary_model_adv.value = config.get('profile_registry', {}).get('default', 'olmocr-2-7b')
            fallback_model_adv.value = config.get('profile_registry', {}).get('fallback', 'olmocr-7b')

            # Processing Tab
            vlm_provider = config.get('vlm_provider', {})

            # Title page settings
            title_page = vlm_provider.get('title_page', {})
            title_mode_adv.value = title_page.get('processing_mode', 'two_shot')

            title_res_val = title_page.get('resolution_override', 1288)
            title_resolution_adv.value = title_res_val
            title_resolution_label.text = f'{int(title_res_val)}px'

            # Body page settings
            body_pages = vlm_provider.get('body_pages', {})
            body_mode_value = body_pages.get('processing_mode', 'adaptive')

            # Map YAML values to UI radio options
            if body_mode_value == 'adaptive':
                body_mode_adv.value = 'adaptive'
                routing_enabled.value = True
            elif body_mode_value == 'vlm_only':
                body_mode_adv.value = 'vlm_only'
                routing_enabled.value = False
            elif body_mode_value == 'pymupdf_only':
                body_mode_adv.value = 'pymupdf_only'
                routing_enabled.value = False
            else:
                body_mode_adv.value = 'adaptive'
                routing_enabled.value = True

            # AsyncIO Settings (Phase 4.9)
            concurrency = vlm_provider.get('concurrency', {})
            async_enabled.value = concurrency.get('enabled', False)
            vram_input.value = concurrency.get('vram_capacity_gb', 0)
            update_concurrency_label(None) # Refresh label

            # VLM parameter overrides
            vlm_params = config.get('vlm_parameter_overrides', {})

            temp_val = vlm_params.get('temperature', 0.4)
            temperature_adv.value = temp_val
            temperature_label.text = f'{temp_val:.2f}'

            max_tok_val = vlm_params.get('max_tokens', 4096)
            max_tokens_adv.value = max_tok_val
            max_tokens_label.text = f'{int(max_tok_val)}'

            repeat_val = vlm_params.get('repeat_penalty', 1.15)
            repeat_penalty_adv.value = repeat_val
            repeat_penalty_label.text = f'{repeat_val:.2f}'

            # Quality thresholds
            quality = config.get('quality', {})
            min_overall_val = quality.get('minimum_overall_score', 0.5)
            min_overall_score.value = min_overall_val
            min_overall_label.text = f'{min_overall_val:.2f}'

            min_page_val = quality.get('minimum_page_score', 0.3)
            min_page_score.value = min_page_val
            min_page_label.text = f'{min_page_val:.2f}'

            warn_below_val = quality.get('warn_below_score', 0.7)
            warn_below_score.value = warn_below_val
            warn_below_label.text = f'{warn_below_val:.2f}'

            # Metadata confidence
            metadata = config.get('metadata', {})
            doi_conf_val = metadata.get('doi_confidence_threshold', 0.85)
            doi_confidence.value = doi_conf_val
            doi_confidence_label.text = f'{doi_conf_val:.2f}'

            isbn_conf_val = metadata.get('isbn_confidence_threshold', 0.85)
            isbn_confidence.value = isbn_conf_val
            isbn_confidence_label.text = f'{isbn_conf_val:.2f}'

            validate_doi.value = metadata.get('validate_doi_format', True)
            validate_year.value = metadata.get('validate_year_range', True)

            logger.info("Advanced Settings UI updated from config")

        except Exception as e:
            logger.error(f"Failed to update Advanced Settings UI: {e}", exc_info=True)

    def update_ui_from_config(config: Dict[str, Any]):
        """Update UI elements to reflect loaded configuration."""
        nonlocal selected_preset

        try:
            # Detect current preset based on config values
            detected_preset = detect_preset_from_config(config)
            if detected_preset:
                selected_preset = detected_preset
                render_radio_cards()  # Update radio card UI

            # Update model selector (Quick Settings)
            profile = config.get('profile_registry', {}).get('default', 'olmocr-2-7b')
            model_selector.value = profile
            
            # Update Server Path Input
            server_exe = config.get('vlm_provider', {}).get('local_server', {}).get('server_executable', '')
            if server_exe:
                server_path_input.value = str(server_exe)

            # Update Advanced Settings UI (refactored to use dedicated function)
            update_advanced_settings_from_config(config)

            # Update preview
            update_config_preview(config, detected_preset or "custom")

        except Exception as e:
            logger.warning(f"Failed to update UI from config: {e}")

    def open_yaml_editor():
        """Open YAML editor modal for direct configuration editing."""
        nonlocal current_config

        logger.info("Opening YAML editor modal")

        # Create modal dialog
        with ui.dialog().props('maximized') as yaml_dialog, ui.card().classes('w-full h-full'):
            with ui.column().classes('w-full h-full gap-4 p-4'):
                # Header
                with ui.row().classes('w-full items-center'):
                    ui.icon('code', size='md').classes('text-orange-600')
                    ui.label('Strategy Configuration - YAML Editor').classes('text-2xl font-bold')
                    ui.space()
                    ui.button(icon='close', on_click=yaml_dialog.close).props('flat round')

                # Info bar
                with ui.row().classes('w-full items-center gap-2 p-3').style('background: #F5F5F5; border-radius: 8px;'):
                    ui.label(f'File: {STRATEGY_CONFIG_PATH.name}').classes('text-sm font-mono')
                    ui.space()
                    validation_status = ui.label('Not validated').classes('text-sm text-gray-500')

                ui.separator()

                # Load current YAML content
                try:
                    with open(STRATEGY_CONFIG_PATH, 'r', encoding='utf-8') as f:
                        yaml_content = f.read()
                except Exception as e:
                    yaml_content = f"# Error loading YAML: {str(e)}"
                    logger.error(f"Failed to load YAML for editor: {e}")

                # YAML Editor (textarea with monospace font - light theme for readability)
                ui.label('Edit YAML directly (be careful!)').classes('text-sm text-gray-600')
                yaml_editor = ui.textarea(value=yaml_content).classes('w-full').props('rows=25 outlined').style(
                    'font-family: "Courier New", Consolas, monospace; '
                    'font-size: 14px; '
                    'line-height: 1.5;'
                )

                ui.separator()

                # Validation result area
                validation_output = ui.markdown('').classes('text-sm').style(
                    'max-height: 150px; overflow-y: auto; padding: 12px; '
                    'background: #F5F5F5; border-radius: 4px; display: none;'
                )

                # Action buttons
                with ui.row().classes('w-full gap-2 mt-4'):
                    async def validate_yaml_async():
                        """Validate YAML syntax and required fields."""
                        try:
                            validation_status.text = 'Validating...'
                            validation_status.classes(remove='text-green-600 text-red-600 text-gray-500', add='text-blue-600')
                            validation_output.style(remove='display: none;')

                            # Validate syntax
                            content = yaml_editor.value
                            try:
                                parsed = yaml.safe_load(content)
                            except yaml.YAMLError as e:
                                validation_output.content = f"**❌ YAML Syntax Error**\n\n```\n{str(e)}\n```"
                                validation_output.style(remove='display: none;')  # Make visible
                                validation_status.text = 'Invalid YAML'
                                validation_status.classes(remove='text-blue-600', add='text-red-600')
                                logger.warning(f"YAML validation failed: {e}")
                                return False

                            # Validate required fields
                            required_fields = ['strategy_name', 'version', 'vlm_provider']
                            missing = [field for field in required_fields if field not in parsed]
                            if missing:
                                validation_output.content = f"**⚠️ Missing Required Fields**\n\n{', '.join(missing)}"
                                validation_output.style(remove='display: none;')  # Make visible
                                validation_status.text = 'Missing fields'
                                validation_status.classes(remove='text-blue-600', add='text-red-600')
                                logger.warning(f"Missing required fields: {missing}")
                                return False

                            # Success
                            validation_output.content = "**✅ Valid YAML**\n\nSyntax correct, all required fields present."
                            validation_output.style(remove='display: none;')  # Make visible
                            validation_status.text = 'Valid'
                            validation_status.classes(remove='text-blue-600 text-red-600', add='text-green-600')
                            logger.info("YAML validation passed")
                            return True

                        except Exception as e:
                            validation_output.content = f"**❌ Validation Error**\n\n{str(e)}"
                            validation_output.style(remove='display: none;')  # Make visible
                            validation_status.text = 'Error'
                            validation_status.classes(remove='text-blue-600', add='text-red-600')
                            logger.error(f"Validation error: {e}")
                            return False

                    async def save_yaml_async():
                        """Save YAML with validation and backup."""
                        try:
                            # First validate
                            if not await validate_yaml_async():
                                ui.notify('Please fix validation errors before saving', type='negative')
                                return

                            # Create backup
                            backup_path = create_config_backup(STRATEGY_CONFIG_PATH)
                            logger.info(f"Created backup: {backup_path}")

                            # Save YAML
                            content = yaml_editor.value
                            with open(STRATEGY_CONFIG_PATH, 'w', encoding='utf-8') as f:
                                f.write(content)
                            logger.info(f"Saved YAML to {STRATEGY_CONFIG_PATH}")

                            # Hot-reload orchestrator
                            reset_orchestrator()
                            new_orchestrator = get_orchestrator()
                            logger.info("Hot-reload completed")

                            # Reload config in UI
                            await load_config_async()
                            logger.info("UI updated from new config")

                            # Success feedback
                            ui.notify(f'Configuration saved! Backup: {backup_path.name}', type='positive')
                            yaml_dialog.close()

                        except Exception as e:
                            ui.notify(f'Save failed: {str(e)}', type='negative')
                            logger.error(f"Failed to save YAML: {e}")

                    ui.button('Validate', icon='check_circle', on_click=lambda: asyncio.create_task(validate_yaml_async())).props('color=primary')
                    ui.button('Save & Apply', icon='save', on_click=lambda: asyncio.create_task(save_yaml_async())).props('color=positive')
                    ui.button('Cancel', icon='cancel', on_click=yaml_dialog.close).props('flat')

                    ui.space()

                    # Help text
                    ui.label('💡 Tip: Validate before saving • Automatic backup created on save').classes('text-xs text-gray-500')

        # Open dialog
        yaml_dialog.open()
        logger.info("YAML editor modal opened")

    def open_file_picker(callback, initial_path=None):
        """Open a simple local file picker dialog."""
        
        # Determine start path
        start_path = Path.cwd()
        if initial_path:
            try:
                p = Path(initial_path)
                if p.is_file():
                    start_path = p.parent
                elif p.is_dir():
                    start_path = p
            except:
                pass
        
        if not start_path.exists():
            start_path = Path.cwd()

        # Mutable state for the picker (using list to bypass closure limit)
        state = {'current_dir': start_path}

        with ui.dialog() as picker_dialog, ui.card().classes('w-[600px] h-[80vh]'):
            with ui.column().classes('w-full h-full gap-2'):
                # Header
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Select Server Executable').classes('text-xl font-bold')
                    ui.button(icon='close', on_click=picker_dialog.close).props('flat round dense')

                # Current Path Display
                path_label = ui.label(str(state['current_dir'])).classes('font-mono text-xs bg-gray-100 p-2 w-full break-all border rounded mb-2')
                
                # File List Area
                file_container = ui.scroll_area().classes('flex-grow w-full border rounded p-1')
                
                def navigate(new_path):
                    try:
                        # Permission check
                        _ = list(new_path.iterdir())
                        state['current_dir'] = new_path
                        path_label.text = str(new_path)
                        refresh_list()
                    except Exception as e:
                        ui.notify(f"Access denied: {e}", type='negative')

                def select_file(path_obj):
                    callback(str(path_obj))
                    picker_dialog.close()

                def refresh_list():
                    file_container.clear()
                    curr = state['current_dir']
                    
                    with file_container:
                        # Up Item
                        if curr.parent != curr:
                            ui.button(
                                '.. (Parent Directory)', 
                                on_click=lambda: navigate(curr.parent),
                                icon='arrow_upward'
                            ).props('flat dense align=left').classes('w-full text-gray-600 hover:bg-gray-100')

                        try:
                            # Sort: Folders first, then Files
                            items = sorted(curr.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                            
                            for p in items:
                                name = p.name
                                if p.is_dir():
                                    ui.button(
                                        f'📁 {name}', 
                                        on_click=lambda x=p: navigate(x)
                                    ).props('flat dense align=left').classes('w-full hover:bg-indigo-50 text-indigo-900')
                                else:
                                    # Highlight .exe
                                    is_exe = name.lower().endswith('.exe')
                                    icon = 'terminal' if is_exe else 'description'
                                    color = 'text-green-700 font-bold' if is_exe else 'text-gray-700'
                                    
                                    ui.button(
                                        f'{name}', 
                                        on_click=lambda x=p: select_file(x),
                                        icon=icon
                                    ).props('flat dense align=left').classes(f'w-full hover:bg-green-50 {color}')
                        
                        except Exception as e:
                             ui.label(f"Error listing files: {e}").classes('text-red-500 text-sm')

                refresh_list()
            
        picker_dialog.open()

    def update_preset_preview(preset_key: str, model: str):
        """Update preview with preset info when no config is loaded."""
        preset = PRESETS.get(preset_key, {})

        preview_text = f"""
**Preset Configuration** (Preview)

- **Preset**: {preset.get('name', 'Custom')}
- **Model**: {model}
- **Processing Mode**: {preset.get('processing_mode', 'unknown')}
- **Title Resolution**: {preset.get('title_resolution', 'unknown')}px
- **Adaptive Routing**: {'Enabled' if preset.get('adaptive_enabled', False) else 'Disabled'}
- **Two-Shot Title**: {'Enabled' if preset.get('two_shot_enabled', False) else 'Disabled'}

*Click "Load Current Config" to see full configuration details*
"""
        config_preview.set_content(preview_text)

    def update_config_preview(config: Dict[str, Any], preset_key: str):
        """Update the configuration preview markdown."""
        preset = PRESETS.get(preset_key, {})

        preview_text = f"""
**Current Configuration**

- **Preset**: {preset.get('name', 'Custom')}
- **Model**: {config.get('profile_registry', {}).get('default', 'unknown')}
- **Version**: {config.get('version', 'unknown')}
- **Processing Mode**: {config.get('vlm_provider', {}).get('body_pages', {}).get('processing_mode', 'unknown')}
- **Title Resolution**: {config.get('vlm_provider', {}).get('title_page', {}).get('resolution_override', 'unknown')}px
- **Adaptive Routing**: {'Enabled' if config.get('vlm_provider', {}).get('body_pages', {}).get('processing_mode') == 'adaptive' else 'Disabled'}

*Configuration loaded from: `{STRATEGY_CONFIG_PATH.name}`*
"""
        config_preview.set_content(preview_text)


# ============================================================================
# HELPER FUNCTIONS (Outside of mount() - clean separation)
# ============================================================================

def load_strategy_config(config_path: Path) -> Dict[str, Any]:
    """
    Load strategy configuration from YAML file.

    Args:
        config_path: Path to the strategy YAML file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Strategy config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def save_strategy_config(config_path: Path, config: Dict[str, Any]):
    """
    Save strategy configuration to YAML file.

    Preserves @ references and special YAML formatting.

    Args:
        config_path: Path to save the YAML file
        config: Configuration dictionary to save
    """
    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            # Use safe_dump with settings that preserve string formatting
            yaml.safe_dump(
                config,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                indent=2,
                width=1000,  # Prevent line wrapping which can break @ references
                default_style=None  # Don't quote strings unless necessary
            )

        logger.info(f"Saved strategy config to {config_path}")

    except Exception as e:
        logger.error(f"Error saving YAML config: {e}")
        raise


def create_config_backup(config_path: Path) -> Path:
    """
    Create a timestamped backup of the configuration file.

    Args:
        config_path: Path to the config file to backup

    Returns:
        Path to the created backup file
    """
    # Ensure backup directory exists
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Create timestamped backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{config_path.stem}_backup_{timestamp}{config_path.suffix}"
    backup_path = BACKUP_DIR / backup_filename

    # Copy file
    shutil.copy2(config_path, backup_path)

    logger.info(f"Created backup: {backup_path}")
    return backup_path


def apply_preset_to_config(
    config: Dict[str, Any],
    preset: Dict[str, Any],
    model: str,
    use_cloud: bool = False
) -> Dict[str, Any]:
    """
    Apply preset settings to configuration.

    Args:
        config: Current configuration dictionary
        preset: Preset settings to apply
        model: Selected VLM model
        use_cloud: Whether to use cloud API mode (Gemini)

    Returns:
        Updated configuration dictionary
    """
    import copy
    updated_config = copy.deepcopy(config)

    try:
        # Update profile registry based on cloud mode
        if 'profile_registry' not in updated_config:
            updated_config['profile_registry'] = {}

        updated_config['profile_registry']['default'] = model

        if use_cloud:
            # Cloud mode: Use Gemini models
            updated_config['profile_registry']['fallback'] = "gemini-2-5-flash"
            updated_config['profile_registry']['title_detection'] = model
            logger.info(f"Cloud mode enabled: default={model}, fallback=gemini-2-5-flash")
        else:
            # Local mode: Use local VLM models
            updated_config['profile_registry']['fallback'] = "olmocr-7b"
            updated_config['profile_registry']['title_detection'] = model
            logger.info(f"Local mode enabled: default={model}, fallback=olmocr-7b")

        # Update title page settings
        if 'vlm_provider' not in updated_config:
            raise ValueError("Missing 'vlm_provider' in config")
        if 'title_page' not in updated_config['vlm_provider']:
            raise ValueError("Missing 'title_page' in vlm_provider")

        title_page_config = updated_config['vlm_provider']['title_page']
        title_page_config['resolution_override'] = preset['title_resolution']
        title_page_config['processing_mode'] = "two_shot" if preset['two_shot_enabled'] else "single_shot"
        logger.info(f"Updated title_page: resolution={preset['title_resolution']}, mode={'two_shot' if preset['two_shot_enabled'] else 'single_shot'}")

        # Update body pages settings
        if 'body_pages' not in updated_config['vlm_provider']:
            raise ValueError("Missing 'body_pages' in vlm_provider")

        body_pages_config = updated_config['vlm_provider']['body_pages']
        body_pages_config['processing_mode'] = preset['processing_mode']
        logger.info(f"Updated body_pages.processing_mode to {preset['processing_mode']}")

        # Update version
        updated_config['version'] = f"{config.get('version', '1.2')}"

    except Exception as e:
        logger.error(f"Error in apply_preset_to_config: {e}")
        raise

    return updated_config


def validate_config(config: Dict[str, Any]) -> list[str]:
    """
    Validate configuration for required fields and valid values.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required top-level keys
    required_keys = ['strategy_name', 'version', 'profile_registry', 'vlm_provider']
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required field: {key}")

    # Validate profile registry
    if 'profile_registry' in config:
        if 'default' not in config['profile_registry']:
            errors.append("Missing 'default' in profile_registry")
        else:
            model = config['profile_registry']['default']
            if model not in AVAILABLE_MODELS:
                errors.append(f"Invalid model: {model}")

    # Validate VLM provider settings
    if 'vlm_provider' in config:
        vlm_config = config['vlm_provider']

        # Check title page config
        if 'title_page' not in vlm_config:
            errors.append("Missing 'title_page' in vlm_provider")
        else:
            title_config = vlm_config['title_page']
            if 'resolution_override' in title_config:
                res = title_config['resolution_override']
                if not (512 <= res <= 2048):
                    errors.append(f"Title resolution {res}px out of valid range (512-2048)")

        # Check body pages config
        if 'body_pages' not in vlm_config:
            errors.append("Missing 'body_pages' in vlm_provider")
        else:
            body_config = vlm_config['body_pages']
            if 'processing_mode' in body_config:
                mode = body_config['processing_mode']
                valid_modes = ['adaptive', 'vlm_only', 'two_shot']
                if mode not in valid_modes:
                    errors.append(f"Invalid processing_mode: {mode}")

    return errors


def detect_preset_from_config(config: Dict[str, Any]) -> Optional[str]:
    """
    Detect which preset matches the current configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Preset key ('fast', 'balanced', 'accurate') or None if custom
    """
    try:
        body_mode = config['vlm_provider']['body_pages']['processing_mode']
        title_mode = config['vlm_provider']['title_page']['processing_mode']
        title_res = config['vlm_provider']['title_page']['resolution_override']

        # Check each preset
        for preset_key, preset in PRESETS.items():
            matches = True

            # Check processing mode
            if body_mode != preset['processing_mode']:
                matches = False

            # Check two-shot enabled
            if title_mode == "two_shot" and not preset['two_shot_enabled']:
                matches = False
            if title_mode != "two_shot" and preset['two_shot_enabled']:
                matches = False

            # Check title resolution (with tolerance of ±10px)
            if abs(title_res - preset['title_resolution']) > 10:
                matches = False

            if matches:
                return preset_key

        return None  # Custom configuration

    except (KeyError, TypeError):
        return None
