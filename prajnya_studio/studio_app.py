from nicegui import ui, app
from .gui import theme, app_state, persistent_settings, health_check, components, icons
from .gui.pages import batch_commander, strategy_lab, ledger_master, help as help_page
import sys
import asyncio

# Load environment variables (for ZOTERO_API_KEY, etc.)
from dotenv import load_dotenv
load_dotenv()

# Configure Python logging
from .prajnya.logger import setup_logging
import logging
setup_logging()

# Suppress harmless Windows asyncio errors (WinError 10054)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

# Load Settings & State
settings = persistent_settings.SettingsManager.load()
state = app_state.AppState.get_instance()
state.settings = settings

# Add static file route for logo
from pathlib import Path
app.add_static_files('/static', Path(__file__).parent)

# Server Manager Integration
from src.llm.providers.server_manager import ServerManager
import yaml
from pathlib import Path

async def startup_server():
    """Start VLM server using configuration from strategy YAML."""
    logging.info("Initializing VLM Server...")

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
        logging.warning(f"Could not read strategy config: {e}")

    # Initialize ServerManager from config (or fallback to defaults)
    if server_config and server_config.get('enabled', True):
        manager = ServerManager.from_config(server_config)
        logging.info(f"Using ServerManager from config: backend={server_config.get('backend', 'router_mode')}")
    else:
        manager = ServerManager(backend="router_mode")
        logging.info("Using default ServerManager configuration")

    state.server_manager = manager

    # Run startup in thread pool to avoid blocking event loop
    loop = asyncio.get_running_loop()

    try:
        # Start server
        started = await loop.run_in_executor(None, manager.ensure_started)
        if started:
            logging.info("VLM Server ready.")
            # Load default model
            await loop.run_in_executor(None, lambda: manager.load_model(model_to_load))
        else:
            logging.error("Failed to start VLM Server.")
            ui.notify("VLM Server failed to start", type="negative")
    except Exception as e:
        logging.error(f"Error during server startup: {e}")

async def shutdown_server():
    """Stop VLM server on app shutdown."""
    if state.server_manager:
        logging.info("Stopping VLM Server...")
        # Run stop in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.server_manager.stop)

app.on_startup(startup_server)
app.on_shutdown(shutdown_server)

def run_health_checks():
    results = health_check.run_startup_checks(settings)
    if results['fatal']:
        with ui.dialog() as dlg, ui.card():
            ui.label('⚠️ Startup Health Check Failed').classes('text-xl font-bold text-red-600')
            ui.separator()
            for name, res in results['checks'].items():
                icon = '✅' if res['ok'] else '❌'
                ui.label(f"{icon} {name}: {res['message']}")
            ui.button('Exit', on_click=lambda: sys.exit(1)).props('color=negative')
        dlg.open()
    elif results['warnings']:
        ui.notify(f"Warnings: {len(results['warnings'])} issues found", type='warning')

def create_ui(current_route='/batch'):
    """Build the main UI with enhanced v1.2 design.

    Args:
        current_route: Current page route (e.g., '/batch', '/strategy')
    """
    # Inject CSS and fonts (v2.0: Cinzel, Inter, JetBrains Mono)
    ui.add_head_html(f'<style>{theme.SARASWATI_CSS}</style>')
    ui.add_head_html('''
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    ''')
    
    # Register keyboard shortcuts (per-page)
    setup_keyboard_shortcuts()

    with ui.row().classes('h-screen w-full no-wrap').style('margin:0; padding:0;'):
        
        # ============================
        # SIDEBAR (260px fixed) - Optimized for 16:10
        # ============================
        with ui.column().classes('gap-0 items-center').style(
            f'width: 260px; '
            f'height: 100vh; '
            f'padding: 16px 12px; '
            f'background: {theme.THEME_LIGHT["sidebar"]}; '
            f'border-right: 1px solid {theme.THEME_LIGHT["border"]}; '
            f'display: flex; '
            f'flex-direction: column; '
            f'overflow: hidden;'
        ):
            # Logo (Original size - 120px)
            components.flame_logo(size=120)
            # Tagline
            ui.label('Academic PDF Processor').style(
                'font-family: "Inter", sans-serif; '
                'font-size: 11px; '
                f'color: {theme.THEME_LIGHT["text_muted"]}; '
                'margin-top: 2px; '
                'margin-bottom: 16px; '
                'letter-spacing: 0.5px;'
            )

            # Navigation Group Container - EXPANDS to fill space
            with ui.column().classes('w-full gap-3 flex-1').style(
                f'background: {theme.THEME_LIGHT["canvas"]}; '
                f'border-radius: 8px; '
                f'padding: 12px 8px;'
            ):
                components.sidebar_item('Batch Commander', icons.BATCH_COMMANDER, '/batch', active=(current_route == '/batch'))
                components.sidebar_item('Strategy Lab', icons.STRATEGY_LAB, '/strategy', active=(current_route == '/strategy'))
                components.sidebar_item('Ledger Master', icons.LEDGER_MASTER, '/ledger', active=(current_route == '/ledger'), badge=state.get_pending_review_count())
                components.sidebar_item('Help', icons.HELP_DIYA, '/help', active=(current_route == '/help'))
            
            # Bottom Group: Settings + Footer (fixed at bottom)
            with ui.column().classes('w-full gap-2'):
                ui.separator().style(f'background: {theme.THEME_LIGHT["border"]}; margin: 8px 0;')
                
                # Settings Button
                with ui.button(on_click=lambda: components.open_settings_modal()).classes('w-full group').props('flat no-caps').style('padding: 0;'):
                    with ui.row().classes('w-full items-center gap-3 q-py-sm q-px-md').style(
                        'transition: all 0.15s ease;'
                    ).props('hover:bg-[#E8E4DD]'):
                        with ui.element('div').style(
                            'width: 24px; height: 24px; filter: grayscale(100%) opacity(0.7); transition: all 0.2s ease;'
                        ).classes('group-hover:filter-none group-hover:opacity-100'):
                            ui.html(icons.SETTINGS_CHAKRA, sanitize=False)
                        ui.label('Settings').style(
                            f'color: {theme.THEME_LIGHT["text_muted"]}; font-weight: 500; transition: color 0.2s ease;'
                        ).classes(f'group-hover:text-[{theme.THEME_LIGHT["indigo"]}] group-hover:font-semibold')
                        ui.space()

                # Footer (compact)
                with ui.column().classes('gap-0 items-center').style('padding-top: 4px;'):
                    ui.label('v5.0.0 • Phase 6').classes('text-xs').style(f'color: {theme.THEME_LIGHT["text_muted"]};')
                    ui.label('Copyright © Manu Unni').classes('text-xs').style(f'color: {theme.THEME_LIGHT["text_muted"]}; opacity: 0.7;')

        # ============================
        # MAIN CONTENT AREA (flex) - v2.0
        # ============================
        with ui.column().classes('flex-1 h-screen').style(
            f'background: {theme.THEME_LIGHT["canvas"]}; '
            f'overflow-y: auto; '
            f'position: relative;'
        ):
            # System Status Indicator (Top Right) - Dynamic VLM status
            with ui.row().classes('absolute top-4 right-4 gap-2 items-center').style('z-index: 10;'):
                # Check if backend is available
                from prajnya import orchestrator
                vlm_state = 'online' if orchestrator.BACKEND_AVAILABLE else 'error'
                status_dot = components.status_pulse(state=vlm_state)
                
                ui.label('VLM Online' if orchestrator.BACKEND_AVAILABLE else 'VLM Offline').classes('text-sm text-gray-600')
                latency = state.get_vlm_latency() if orchestrator.BACKEND_AVAILABLE else 0
                ui.label(f'{latency}ms' if latency > 0 else 'N/A').classes('text-xs text-gray-400')
            
            # Page content goes here (returned container)
            with ui.column().classes('w-full h-full p-8').style('padding-top: 60px;') as content:
                pass
    
    return content

def setup_keyboard_shortcuts():
    """Register global keyboard shortcuts (v1.2)."""
    @ui.keyboard
    def handle_keys(e):
        # Ctrl+B → Batch Commander
        if e.key == 'b' and e.modifiers.ctrl:
            ui.navigate.to('/batch')
        # Ctrl+L → Ledger Master
        elif e.key == 'l' and e.modifiers.ctrl:
            ui.navigate.to('/ledger')
        # Ctrl+K → Strategy Lab
        elif e.key == 'k' and e.modifiers.ctrl:
            ui.navigate.to('/strategy')
        # Ctrl+, → Settings
        elif e.key == ',' and e.modifiers.ctrl:
            components.open_settings_modal()
        # Ctrl+Shift+D → Debug (hidden)
        elif e.key == 'd' and e.modifiers.ctrl and e.modifiers.shift:
            ui.notify('Debug Panel (Not Implemented)', type='info')

# Routing
@ui.page('/')
def index():
    ui.navigate.to('/batch')

@ui.page('/batch')
def page_batch():
    content = create_ui(current_route='/batch')
    with content:
        batch_commander.mount()

@ui.page('/strategy')
def page_strategy():
    content = create_ui(current_route='/strategy')
    with content:
        strategy_lab.mount()

@ui.page('/ledger')
def page_ledger():
    content = create_ui(current_route='/ledger')
    with content:
        ledger_master.mount()

@ui.page('/help')
def page_help():
    content = create_ui(current_route='/help')
    with content:
        help_page.mount()

if __name__ in {"__main__", "__mp_main__"}:
    run_health_checks()
    # Note: Keyboard shortcuts are set up per-page in create_ui()
    ui.run(
        native=False,  # TEMP: Set to True for production (native desktop app)
        title='Prajnya Studio - v5.0.0',
        window_size=(1280, 820),
        reload=False
    )