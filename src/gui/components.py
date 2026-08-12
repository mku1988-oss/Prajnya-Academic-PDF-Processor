"""
Reusable Saraswati-styled widgets: logo, sidebar nav item, status pulse,
radio card, and the quick-settings modal.

Ported from V5's gui/components.py. One control was dropped from the
settings modal: "Enable Parallel VLM" configured a concurrency path V6
does not have (see persistent_settings.py's docstring).
"""

from nicegui import ui

from . import icons, persistent_settings, theme


def lotus_logo(size: int = 120):
    """Renders the Lotus-Mandala logo.

    The `dark_mode` flag this used to take is gone: which variant shows is
    now a CSS decision, so the logo follows a live theme switch instead of
    being fixed when the page was built. See `theme.lotus_logo`.

    Args:
        size: Logo size in pixels.
    """
    return ui.html(theme.lotus_logo(size=size), sanitize=False)


def sidebar_item(label: str, icon: str, route: str, active: bool = False, badge: int = 0):
    """
    Renders a consistent sidebar navigation item. Accepts either a raw SVG
    string (used everywhere in this app) or a legacy emoji/text icon.
    """
    bg_color = theme.var('nav_hover') if active else 'transparent'
    text_color = theme.var('indigo') if active else theme.var('text_muted')
    border = f'3px solid {theme.var("indigo")}' if active else '3px solid transparent'
    font_weight = '600' if active else '500'

    with ui.row().classes('w-full items-center cursor-pointer gap-3 q-py-sm q-px-md').style(
        f'background: {bg_color}; '
        f'border-left: {border}; '
        f'border-radius: 0 8px 8px 0; '
        f'transition: all 0.15s ease;'
    ).on('click', lambda: ui.navigate.to(route)):

        if icon.strip().startswith('<svg'):
            filter_style = 'filter: none;' if active else 'filter: grayscale(100%) opacity(0.6);'
            with ui.element('div').style(
                f'width: 24px; height: 24px; flex-shrink: 0; {filter_style} transition: filter 0.2s ease;'
            ):
                ui.html(icon, sanitize=False)
        else:
            ui.label(icon).classes('text-xl').style(f'color: {text_color}; width: 24px; text-align: center;')

        ui.label(label).style(f'color: {text_color}; font-weight: {font_weight};')

        ui.space()

        if badge > 0:
            ui.label(str(badge)).classes('text-xs font-bold').style(
                f'background: {theme.var("gold")}; '
                f'color: {theme.var("indigo")}; '
                f'padding: 2px 8px; '
                f'border-radius: 12px;'
            )


def status_pulse(state: str = 'idle'):
    """Animated status indicator dot."""
    color_map = {
        'idle': theme.var('idle'),
        'online': theme.var('success'),
        'processing': theme.var('warning'),
        'error': theme.var('error'),
    }
    color = color_map.get(state, theme.var('idle'))

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
    on_select=None,
):
    """
    A card-based radio button. Active border is indigo (structure); the
    filled dot inside it is vermillion (action) — the two-colour logic
    Strategy Lab will lean on for its role -> model picker in S12.
    """
    if selected:
        border_style = f'border: 2px solid {theme.var("indigo")};'
        bg_style = f'background: {theme.var("selected_tint")};'
        radio_outer_color = theme.var('indigo')
        radio_inner_color = theme.var('vermillion')
    else:
        border_style = f'border: 1px solid {theme.var("border")};'
        bg_style = 'background: transparent;'
        radio_outer_color = theme.var('text_muted')
        radio_inner_color = 'transparent'

    card = ui.card().classes('w-full cursor-pointer group').props(
        f'hover:border-[color:{theme.var("gold")}]'
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
                    ui.element('div').style(
                        f'width: 10px; '
                        f'height: 10px; '
                        f'border-radius: 50%; '
                        f'background: {radio_inner_color};'
                    )

            ui.label(icon).classes('text-lg').style(
                f'color: {theme.var("gold")}; flex-shrink: 0;'
            )

            with ui.column().classes('flex-1 gap-0'):
                ui.label(title).classes('text-sm font-semibold').style(
                    f'color: {theme.var("text")}; '
                    f'font-family: var(--prajnya-font-body);'
                )
                ui.label(description).classes('text-xs').style(
                    f'color: {theme.var("text_muted")}; '
                    f'font-family: var(--prajnya-font-body);'
                )

    return card


def coming_soon(title: str, description: str, build_step: str):
    """
    Placeholder content for a screen whose route exists before its page
    does. S5 wires up the shell and every nav item it will ever need; the
    screens themselves land in the sessions gui_plan.md lays out for them.
    """
    with ui.column().classes('w-full h-full items-center justify-center gap-3'):
        ui.label(title).style(
            f'font-family: var(--prajnya-font-display); font-size: 1.5rem; '
            f'font-weight: 600; color: {theme.var("indigo")};'
        )
        ui.label(description).classes('text-center').style(
            f'color: {theme.var("text_muted")}; max-width: 420px;'
        )
        ui.label(f'Arrives in {build_step}').classes('text-xs uppercase tracking-wide').style(
            f'color: {theme.var("gold")}; font-weight: 600;'
        )


def open_settings_modal():
    """Opens the quick-settings dialog (paths, recursion, review skip, theme).

    The full settings surface — including the model layer, which quick
    settings deliberately does not touch — is the Settings screen (S11).
    """
    settings = persistent_settings.SettingsManager.load()

    with ui.dialog().props('backdrop-filter="blur(4px)"') as dialog, ui.card().classes('w-[600px] p-0').style(
        f'border-radius: 12px; box-shadow: 0 10px 40px {theme.var("overlay_shadow")}; border: 1px solid {theme.var("border_gold")}; overflow: hidden;'
    ):
        with ui.row().classes('w-full p-4 items-center justify-between').style(
            f'background: {theme.var("sidebar")}; border-bottom: 2px solid {theme.var("gold")};'
        ):
            with ui.row().classes('items-center gap-3'):
                with ui.element('div').style('width: 24px; height: 24px;'):
                    ui.html(icons.SETTINGS_CHAKRA, sanitize=False)

                ui.label('Studio Settings').style(
                    f'font-family: var(--prajnya-font-display); font-weight: 600; color: {theme.var("indigo")}; font-size: 1.25rem; font-variant: small-caps;'
                )

            ui.button(icon='close', on_click=dialog.close).props('flat round dense').style(f'color: {theme.var("text_muted")};')

        with ui.column().classes('w-full p-6 gap-6').style(
            f'background: {theme.var("canvas")}; max-height: 70vh; overflow-y: auto;'
        ):
            with ui.column().classes('w-full gap-2'):
                ui.label('Paths & Storage').classes('text-xs font-bold uppercase tracking-wide').style(f'color: {theme.var("gold")};')

                default_folder = ui.input('Default Batch Folder', value=settings.get('default_batch_folder', '')) \
                    .props('outlined dense') \
                    .classes('w-full') \
                    .style(f'background: {theme.var("input_bg")};')

            ui.separator().style(f'background: {theme.var("border")};')

            with ui.column().classes('w-full gap-2'):
                ui.label('Processing Defaults').classes('text-xs font-bold uppercase tracking-wide').style(f'color: {theme.var("gold")};')

                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Recursive Search').style(f'color: {theme.var("text")};')
                    rec_switch = ui.switch(value=settings.get('recursive_default', False)).props('color=red-9')

                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Skip Reviewed Files').style(f'color: {theme.var("text")};')
                    rev_switch = ui.switch(value=settings.get('skip_review_default', True)).props('color=red-9')

            ui.separator().style(f'background: {theme.var("border")};')

            with ui.column().classes('w-full gap-2'):
                ui.label('Appearance').classes('text-xs font-bold uppercase tracking-wide').style(f'color: {theme.var("gold")};')

                # 'saraswati_dark' is back, and this time there is a palette
                # behind it. S11 removed the option because selecting it did
                # nothing — the honest state of things then, in V6 and in V4
                # before it. THEME_DARK now exists, so the option means
                # something again.
                theme_select = ui.select(
                    list(theme.THEME_ATTRIBUTES),
                    value=theme.resolve_theme(settings.get('ui_theme', theme.DEFAULT_THEME)),
                    label='UI Theme',
                    # Applied on pick, not on Save. A colour scheme is the one
                    # setting whose effect you can only judge by looking at
                    # it, and Cancel still leaves the stored value untouched
                    # (the next page load reads the file, not this dropdown).
                    on_change=lambda e: ui.run_javascript(theme.apply_theme_js(e.value)),
                ).props('outlined dense').classes('w-full').style(f'background: {theme.var("input_bg")};')

        with ui.row().classes('w-full p-4 border-t justify-end gap-2').style(
            f'background: {theme.var("sidebar")}; border-color: {theme.var("border")};'
        ):
            def cancel():
                # The theme dropdown previews live, so Cancel has to put the
                # screen back as well as leave the file alone — otherwise
                # "Cancel" visibly changes the app, which is the opposite of
                # what the word promises.
                ui.run_javascript(theme.apply_theme_js(
                    settings.get('ui_theme', theme.DEFAULT_THEME)
                ))
                dialog.close()

            ui.button('Cancel', on_click=cancel).props('flat').style(f'color: {theme.var("indigo")};')

            def save():
                new_settings = settings.copy()
                new_settings['default_batch_folder'] = default_folder.value
                new_settings['recursive_default'] = rec_switch.value
                new_settings['skip_review_default'] = rev_switch.value
                new_settings['ui_theme'] = theme_select.value

                persistent_settings.SettingsManager.save(new_settings)
                ui.notify('Settings saved successfully', type='positive')
                dialog.close()

            ui.button('Save Changes', on_click=save).style(
                f'background: {theme.var("vermillion")} !important; color: {theme.var("on_accent")} !important; font-weight: 600;'
            )

    dialog.open()
