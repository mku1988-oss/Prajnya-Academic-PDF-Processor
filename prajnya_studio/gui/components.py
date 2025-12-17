from nicegui import ui
from . import theme, persistent_settings, icons


def lotus_logo(size: int = 120, dark_mode: bool = False):
    """
    Renders the Lotus-Mandala logo (v2.0).

    Args:
        size: Logo size in pixels
        dark_mode: If True, use blue variant for dark backgrounds
    """
    return ui.html(theme.lotus_logo(size=size, dark_mode=dark_mode), sanitize=False)

def flame_logo(size: int = 64, active: bool = False):
    """
    Legacy function - redirects to lotus_logo() for backward compatibility.

    Args:
        size: Logo size in pixels
        active: Ignored (kept for API compatibility)
    """
    return lotus_logo(size=size, dark_mode=False)

def sidebar_item(label: str, icon: str, route: str, active: bool = False, badge: int = 0):
    """
    Renders a consistent sidebar navigation item (v2.0 styling).
    Supports both legacy string icons (emojis/Material) and raw SVG code.
    """
    # v2.0: Light sidebar with indigo accents
    bg_color = '#E8E4DD' if active else 'transparent'
    text_color = theme.THEME_LIGHT['indigo'] if active else theme.THEME_LIGHT['text_muted']
    border = f'3px solid {theme.THEME_LIGHT["indigo"]}' if active else '3px solid transparent'
    font_weight = '600' if active else '500'

    # Container
    with ui.row().classes('w-full items-center cursor-pointer gap-3 q-py-sm q-px-md').style(
        f'background: {bg_color}; '
        f'border-left: {border}; '
        f'border-radius: 0 8px 8px 0; '
        f'transition: all 0.15s ease;'
    ).on('click', lambda: ui.navigate.to(route)):

        # Icon Rendering Logic
        if icon.strip().startswith('<svg'):
            # SVG Icon
            filter_style = 'filter: none;' if active else 'filter: grayscale(100%) opacity(0.6);'
            
            with ui.element('div').style(f'width: 24px; height: 24px; flex-shrink: 0; {filter_style} transition: filter 0.2s ease;'):
                ui.html(icon, sanitize=False)
        else:
            # Legacy/Emoji icon support
            ui.label(icon).classes('text-xl').style(f'color: {text_color}; width: 24px; text-align: center;')

        # Label
        ui.label(label).style(f'color: {text_color}; font-weight: {font_weight};')

        ui.space()

        # Badge Logic
        if badge > 0:
            ui.label(str(badge)).classes('text-xs font-bold').style(
                f'background: {theme.THEME_LIGHT["gold"]}; '
                f'color: {theme.THEME_LIGHT["indigo"]}; '
                f'padding: 2px 8px; '
                f'border-radius: 12px;'
            )

def status_pulse(state: str = 'idle'):
    """
    Animated status indicator (v2.0 styling).
    """
    color_map = {
        'idle': '#9E9E9E',                      # Grey
        'online': theme.THEME_LIGHT['success'], # Forest Green
        'processing': theme.THEME_LIGHT['warning'], # Amber
        'error': theme.THEME_LIGHT['error']     # Error Red
    }
    color = color_map.get(state, '#9E9E9E')

    return ui.html(f'''
        <div class="status-pulse" style="width: 8px; height: 8px; border-radius: 50%; background: {color};"></div>
        <style>
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
            .status-pulse {{ animation: pulse 2s ease-in-out infinite; }}
        </style>
    ''', sanitize=False)

def radio_card(
    key: str,
    icon: str,
    title: str,
    description: str,
    selected: bool = False,
    on_select = None
):
    """
    Creates a card-based radio button per Design Bible v2.0.
    
    UPDATED: Uses Indigo/Vermillion logic instead of Green.
    - Active Border: Indigo (Structure)
    - Active Dot: Vermillion (Agni/Action)
    - Background: Transparent (Idle) / Faint Indigo (Active)
    """
    # Dynamic styling based on selection state
    if selected:
        # Selected: Indigo Border (2px), Faint Indigo Tint
        # Represents "Dharma" (Structure/Decision)
        border_style = f'border: 2px solid {theme.THEME_LIGHT["indigo"]};'
        bg_style = f'background: rgba(44, 62, 80, 0.05);' 
        radio_outer_color = theme.THEME_LIGHT['indigo']
        radio_inner_color = theme.THEME_LIGHT['vermillion'] # The "Agni" spark
    else:
        # Idle: Subtle Border, Transparent Background (blends with cream card)
        # Represents "Sattva" (Clarity/Calm)
        border_style = f'border: 1px solid {theme.THEME_LIGHT["border"]};'
        bg_style = f'background: transparent;' 
        radio_outer_color = theme.THEME_LIGHT['text_muted']
        radio_inner_color = 'transparent'

    # Card Container
    # Added 'hover:border-[#C9973F]' (Soft Gold) via props for the "Expert Touch"
    card = ui.card().classes('w-full cursor-pointer group').props(
        f'hover:border-[{theme.THEME_LIGHT["gold"]}]'
    ).style(
        f'{border_style} '
        f'{bg_style} '
        f'border-radius: 8px; '
        f'padding: 16px 20px; '
        f'transition: all 0.2s ease;'
    )

    if on_select:
        card.on('click', lambda: on_select(key))

    with card:
        with ui.row().classes('w-full items-center gap-3'):
            # Custom radio circle
            with ui.element('div').style(
                f'width: 20px; '
                f'height: 20px; '
                f'border-radius: 50%; '
                f'border: 2px solid {radio_outer_color}; '
                f'display: flex; '
                f'align-items: center; '
                f'justify-content: center; '
                f'flex-shrink: 0;'
            ):
                if selected:
                    # The "Agni" Dot
                    ui.element('div').style(
                        f'width: 10px; '
                        f'height: 10px; '
                        f'border-radius: 50%; '
                        f'background: {radio_inner_color};'
                    )

            # Icon - Always Gold (Jyoti/Illumination)
            ui.label(icon).classes('text-lg').style(
                f'color: {theme.THEME_LIGHT["gold"]}; flex-shrink: 0;'
            )

            # Text content
            with ui.column().classes('flex-1 gap-0'):
                ui.label(title).classes('text-sm font-semibold').style(
                    f'color: {theme.THEME_LIGHT["text"]}; '
                    f'font-family: var(--prajnya-font-body);'
                )
                ui.label(description).classes('text-xs').style(
                    f'color: {theme.THEME_LIGHT["text_muted"]}; '
                    f'font-family: var(--prajnya-font-body);'
                )

    return card

def open_settings_modal():
    """
    Opens the configuration modal using NiceGUI dialog.
    Polished for Saraswati v2.0 Theme.
    """
    settings = persistent_settings.SettingsManager.load()
    
    # Backdrop blur for focus
    with ui.dialog().props('backdrop-filter="blur(4px)"') as dialog, ui.card().classes('w-[600px] p-0').style(
        f'border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,0.15); border: 1px solid {theme.THEME_LIGHT["border_gold"]}; overflow: hidden;'
    ):
        # 1. Header (Cream + Gold Border)
        with ui.row().classes('w-full p-4 items-center justify-between').style(
            f'background: {theme.THEME_LIGHT["sidebar"]}; border-bottom: 2px solid {theme.THEME_LIGHT["gold"]};'
        ):
            with ui.row().classes('items-center gap-3'):
                # Small Chakra Icon
                with ui.element('div').style(f'width: 24px; height: 24px;'):
                    ui.html(icons.SETTINGS_CHAKRA, sanitize=False)
                
                ui.label('Studio Settings').style(
                    f'font-family: var(--prajnya-font-display); font-weight: 600; color: {theme.THEME_LIGHT["indigo"]}; font-size: 1.25rem; font-variant: small-caps;'
                )
            
            ui.button(icon='close', on_click=dialog.close).props('flat round dense').style(f'color: {theme.THEME_LIGHT["text_muted"]};')
            
        # 2. Body (Canvas Cream)
        with ui.column().classes('w-full p-6 gap-6').style(
            f'background: {theme.THEME_LIGHT["canvas"]}; max-height: 70vh; overflow-y: auto;'
        ):
            
            # --- Section 1: Paths ---
            with ui.column().classes('w-full gap-2'):
                ui.label('Paths & Storage').classes('text-xs font-bold uppercase tracking-wide').style(f'color: {theme.THEME_LIGHT["gold"]};')
                
                # Batch Folder Input
                default_folder = ui.input('Default Batch Folder', value=settings.get('default_batch_folder', ''))\
                    .props('outlined dense')\
                    .classes('w-full')\
                    .style('background: white;')
            
            ui.separator().style(f'background: {theme.THEME_LIGHT["border"]};')

            # --- Section 2: Processing ---
            with ui.column().classes('w-full gap-2'):
                ui.label('Processing Defaults').classes('text-xs font-bold uppercase tracking-wide').style(f'color: {theme.THEME_LIGHT["gold"]};')
                
                # Toggles (Using red-9 to approximate Vermillion since props take Quasar colors)
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Recursive Search').style(f'color: {theme.THEME_LIGHT["text"]};')
                    rec_switch = ui.switch(value=settings.get('recursive_default', False)).props('color=red-9')

                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Skip Reviewed Files').style(f'color: {theme.THEME_LIGHT["text"]};')
                    rev_switch = ui.switch(value=settings.get('skip_review_default', True)).props('color=red-9')

                with ui.row().classes('w-full justify-between items-center'):
                    with ui.column().classes('gap-0'):
                        ui.label('Enable Parallel VLM').style(f'color: {theme.THEME_LIGHT["text"]};')
                        ui.label('Experimental feature').classes('text-xs').style(f'color: {theme.THEME_LIGHT["warning"]};')
                    vlm_switch = ui.switch(value=settings.get('vlm_parallel_enabled', False)).props('color=red-9')
            
            ui.separator().style(f'background: {theme.THEME_LIGHT["border"]};')

            # --- Section 3: Appearance ---
            with ui.column().classes('w-full gap-2'):
                ui.label('Appearance').classes('text-xs font-bold uppercase tracking-wide').style(f'color: {theme.THEME_LIGHT["gold"]};')
                
                theme_select = ui.select(
                    ['saraswati_light', 'saraswati_dark'], 
                    value=settings.get('ui_theme', 'saraswati_light'), 
                    label='UI Theme'
                ).props('outlined dense').classes('w-full').style('background: white;')

        # 3. Footer (Sidebar Cream)
        with ui.row().classes('w-full p-4 border-t justify-end gap-2').style(
            f'background: {theme.THEME_LIGHT["sidebar"]}; border-color: {theme.THEME_LIGHT["border"]};'
        ):
            ui.button('Cancel', on_click=dialog.close).props('flat').style(f'color: {theme.THEME_LIGHT["indigo"]};')
            
            def save():
                new_settings = settings.copy()
                new_settings['default_batch_folder'] = default_folder.value
                new_settings['recursive_default'] = rec_switch.value
                new_settings['skip_review_default'] = rev_switch.value
                new_settings['vlm_parallel_enabled'] = vlm_switch.value
                new_settings['ui_theme'] = theme_select.value
                
                persistent_settings.SettingsManager.save(new_settings)
                ui.notify('Settings saved successfully', type='positive')
                dialog.close()
                
            # Save Button: Vermillion (Action)
            ui.button('Save Changes', on_click=save).style(
                f'background: {theme.THEME_LIGHT["vermillion"]} !important; color: white !important; font-weight: 600;'
            )

    dialog.open()