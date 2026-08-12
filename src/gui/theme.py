"""
Saraswati theme — design language constants and CSS for Prajnya Studio (V6).

For biology researchers: this file has no logic worth calling logic, only
colours, fonts and a big CSS string. It carries V5's Lotus-Mandala palette,
so a V5 user's muscle memory for the app still applies.

**How a colour reaches the screen — read this before adding one.** A page
never writes a hex code and never reads a palette dict. It calls
`theme.var('indigo')`, which returns the CSS string `var(--prajnya-indigo)`,
and the browser looks that variable up at paint time. The variables
themselves are defined once, at the top of `SARASWATI_CSS`, twice over: a
`:root` block holding `THEME_LIGHT` and a `[data-theme="dark"]` block
holding `THEME_DARK`. Switching themes therefore changes ONE attribute on
`<html>`; nothing is re-rendered and no Python runs.

That indirection is the whole feature. Before it, pages inlined 173
`THEME_LIGHT[...]` lookups into style strings at page-BUILD time, which
baked the light palette into the HTML — so a dark palette could exist and
still change nothing on screen. (That is precisely what happened in V4: it
shipped a `saraswati_dark` menu option, a `dark_mode` logo parameter and a
`logo_white.png` asset, with no dark palette anywhere. Selecting it saved a
string and changed not one pixel.)

**Adding a colour: add the key to BOTH dicts.** `var()` refuses a token
that is not in `THEME_LIGHT`, and a test asserts the two dicts have
identical key sets, so a half-added colour fails loudly here rather than
showing up as one bright panel in a dark UI three screens away.
"""

from pathlib import Path
from typing import Dict

# ============================================================
# COLOR PALETTE - Saraswati v2.0
# ============================================================

THEME_LIGHT = {
    # Primary Palette (Saraswati - Lotus/Mandala)
    'canvas': '#F5F5F0',           # Ice White (main background)
    'card': '#FAF8F5',             # Cream (card backgrounds)
    'sidebar': '#F0EDE8',          # Sidebar Cream
    'indigo': '#2C3E50',           # Primary text/headers
    'gold': '#C9973F',             # Soft Gold accents

    # Secondary Palette (Agni - Action)
    'vermillion': '#B8472A',       # Primary action buttons
    'vermillion_dark': '#9A3A22',  # Hover state for vermillion
    'success': '#2D5A3D',          # Forest Green (success states)
    'warning': '#D4A84B',          # Amber (warnings)
    'error': '#C2291F',            # Error Red

    # Neutrals
    'text': '#4A4A4A',             # Body text
    'text_muted': '#7A7A7A',       # Secondary text
    'border': '#E5E2DD',           # Card borders
    'border_gold': '#D4C4A8',      # Accent borders

    # Terminal (keep dark in BOTH themes — it imitates a console on purpose)
    'terminal_bg': '#1A1F23',
    'terminal_text': '#E0E0E0',

    # --- Surfaces and states that used to be hardcoded in pages ----------
    # Every one of these was a bare hex literal sitting in a page or in the
    # CSS below. A literal cannot respond to a theme, so each became a token.
    'nav_hover': '#E8E4DD',        # sidebar item hover + active background
    'input_bg': '#FFFFFF',         # text inputs / selects inside dialogs
    'on_accent': '#FFFFFF',        # text drawn ON vermillion (button labels)
    'row_reviewed': '#FFFFFF',     # ledger row, reviewed
    'idle': '#9E9E9E',             # status pulse, idle
    'drop_zone_border': '#CCCCCC',
    'scrollbar_thumb': '#CFD8DC',
    'scrollbar_thumb_hover': '#B0BEC5',
    'log_info': '#64B5F6',
    'log_error': '#EF5350',

    # Translucent tints. These are separate tokens rather than a hex code
    # with an alpha suffix glued on: `var(--prajnya-warning)22` is not valid
    # CSS, so the old `{THEME_LIGHT["warning"]}22` trick could not survive
    # the move to variables.
    'selected_tint': 'rgba(44, 62, 80, 0.05)',    # radio_card, selected
    'warning_tint': 'rgba(212, 168, 75, 0.13)',   # review banner background
    'row_provisional': 'rgba(255, 167, 38, 0.10)',
    'row_zebra': 'rgba(44, 62, 80, 0.03)',        # alternating table rows
    'overlay_shadow': 'rgba(0, 0, 0, 0.15)',      # dialog drop shadow

    # Legacy mappings (for backward compatibility with V5-derived CSS)
    'primary': '#2C3E50',          # Maps to indigo
    'accent': '#B8472A',           # Maps to vermillion
    'identity': '#B8472A',         # Maps to vermillion
    'status': '#2D5A3D',           # Maps to success
    'amber': '#D4A84B',            # Maps to warning
    'shadow_ink': '#1A1F23',       # Maps to terminal_bg
}

#: The dark counterpart. Same keys, same ROLES — `indigo` still means
#: "the colour of a heading", which is why it is a pale blue-grey here
#: rather than a dark blue. A token names its job, not its hue; that is what
#: lets one `var(--prajnya-indigo)` in a page be correct in both themes.
#:
#: The identity colours (gold, vermillion) are lifted rather than reused:
#: #B8472A on #14181C is legible but muddy, so dark gets brighter mixes of
#: the same hues. The palette still reads as Saraswati.
THEME_DARK = {
    'canvas': '#14181C',
    'card': '#1C2126',
    'sidebar': '#191E23',
    'indigo': '#D8E0E8',           # headings, on a dark ground
    'gold': '#D8A94F',

    'vermillion': '#D8593A',
    'vermillion_dark': '#E86B4A',  # hover goes BRIGHTER on a dark ground
    'success': '#5FBF82',
    'warning': '#E0B95E',
    'error': '#E5675C',

    'text': '#C7CCD1',
    'text_muted': '#8A9198',
    'border': '#2A3138',
    'border_gold': '#4A4030',

    'terminal_bg': '#0E1215',
    'terminal_text': '#E0E0E0',

    'nav_hover': '#232A31',
    'input_bg': '#232A31',
    'on_accent': '#FFFFFF',
    'row_reviewed': '#1C2126',
    'idle': '#6B7278',
    'drop_zone_border': '#3A424A',
    'scrollbar_thumb': '#39424A',
    'scrollbar_thumb_hover': '#4C565F',
    'log_info': '#64B5F6',
    'log_error': '#EF5350',

    'selected_tint': 'rgba(216, 169, 79, 0.10)',
    'warning_tint': 'rgba(224, 185, 94, 0.16)',
    'row_provisional': 'rgba(224, 185, 94, 0.14)',
    'row_zebra': 'rgba(255, 255, 255, 0.03)',
    'overlay_shadow': 'rgba(0, 0, 0, 0.55)',

    'primary': '#D8E0E8',
    'accent': '#D8593A',
    'identity': '#D8593A',
    'status': '#5FBF82',
    'amber': '#E0B95E',
    'shadow_ink': '#0E1215',
}

#: The value of the `data-theme` attribute for each selectable theme, keyed
#: by the string `persistent_settings` stores in `ui_theme`. "light" is the
#: absence of an override — `:root` already holds the light palette — so it
#: maps to the empty string and the attribute is removed rather than set.
THEME_ATTRIBUTES = {
    'saraswati_light': '',
    'saraswati_dark': 'dark',
}

#: The default, used whenever a stored `ui_theme` names a theme that no
#: longer exists (a settings file written by an older build, say). Falling
#: back to light is safe in a way falling back to dark is not: light is what
#: every screen was designed and verified against first.
DEFAULT_THEME = 'saraswati_light'


def var(token: str) -> str:
    """Returns the CSS `var(...)` reference for a palette token.

    This is what pages call. `theme.var('indigo')` gives the string
    `var(--prajnya-indigo)`, which is a valid CSS *value* — so it can go
    anywhere a colour goes: inside a `.style()` string, as nicegui's
    `color=` argument, or in a `stroke=` attribute on an inline SVG.

    Args:
        token: a key of THEME_LIGHT, e.g. 'indigo' or 'text_muted'.

    Returns:
        The CSS variable reference, e.g. 'var(--prajnya-text-muted)'.

    Raises:
        KeyError: if the token is not in the palette. This is the point of
            the function — a typo'd variable name in a raw CSS string is
            silently ignored by the browser and simply paints nothing,
            whereas this fails at import time with the bad name in the
            message.
    """
    if token not in THEME_LIGHT:
        raise KeyError(
            f"Unknown theme token {token!r}. "
            f"Add it to BOTH THEME_LIGHT and THEME_DARK in theme.py. "
            f"Known tokens: {', '.join(sorted(THEME_LIGHT))}"
        )
    return f"var(--prajnya-{token.replace('_', '-')})"


def resolve_theme(name: str) -> str:
    """Maps a stored `ui_theme` string to a theme this build actually has.

    Args:
        name: whatever `persistent_settings` has in `ui_theme` — which may
            be a theme name written by an older or newer build.

    Returns:
        A key of THEME_ATTRIBUTES, guaranteed. An unrecognised name gives
        DEFAULT_THEME rather than raising: a settings file is user data, and
        a stale value in it should not stop the app opening.
    """
    return name if name in THEME_ATTRIBUTES else DEFAULT_THEME


def apply_theme_js(name: str) -> str:
    """JavaScript that puts one theme into effect, for a page already built.

    This is the entire theme switch. It sets (or removes) `data-theme` on
    `<html>`, the browser re-resolves every `var(--prajnya-*)`, and the whole
    app repaints. No element is rebuilt, so it is safe to call while a batch
    is running and a live transcript is streaming.

    Args:
        name: a theme name; anything unrecognised resolves to the default.

    Returns:
        A JavaScript snippet, for `ui.run_javascript` or a `<script>` tag.
    """
    attribute = THEME_ATTRIBUTES[resolve_theme(name)]
    if not attribute:
        return "document.documentElement.removeAttribute('data-theme');"
    return f"document.documentElement.setAttribute('data-theme', '{attribute}');"


def theme_head_html(name: str) -> str:
    """The same switch as `apply_theme_js`, as a `<head>` script tag.

    Applied in the head so the attribute is set BEFORE the body paints. Done
    any later — on page-ready, say — the user sees a light flash first, which
    is the exact thing someone choosing a dark theme is trying to avoid.
    """
    return f"<script>{apply_theme_js(name)}</script>"


def _css_variable_block(palette: Dict[str, str]) -> str:
    """Renders one palette as CSS custom property declarations.

    Both themes go through this same function, so the light and dark blocks
    cannot drift apart in which variables they define — only in values.
    """
    return "\n".join(
        f"    --prajnya-{key.replace('_', '-')}: {value};"
        for key, value in palette.items()
    )

# ============================================================
# LOGO
# ============================================================

#: Where the logo PNGs are served from. app.py mounts this under
#: `/static`; keeping the constant here means a page never has to know
#: the actual filesystem location.
ASSETS_DIR = Path(__file__).parent / "assets"


def lotus_logo(size: int = 120) -> str:
    """
    Returns HTML for the Lotus-Mandala logo, in both theme variants.

    **Both images are emitted and CSS hides one.** The blue logo is for a
    light ground, the white one for dark, and which is visible is decided by
    the `.logo-light` / `.logo-dark` rules at the bottom of SARASWATI_CSS —
    not by a Python argument. That matters because the theme can change
    after the page is built (the Appearance dropdown flips one attribute on
    `<html>` and nothing re-renders); a logo chosen in Python would stay
    wrong until a reload.

    `logo_white.png` has shipped unused since V4, which had the parameter
    but never a caller that passed True. This is the first code that
    actually puts it on screen.

    Args:
        size: Logo size in pixels.

    Returns:
        HTML string for logo display.
    """
    variants = [('logo_blue.png', 'logo-light'), ('logo_white.png', 'logo-dark')]
    available = [(name, css_class) for name, css_class in variants
                 if (ASSETS_DIR / name).exists()]

    if not available:
        # Fallback if the assets are missing — never let a page fail to
        # render over a logo.
        return f'''
        <div style="width: {size}px; height: {size}px; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 48px;">\U0001fab7</span>
        </div>
        '''

    images = "\n".join(
        f'<img src="/static/{name}" class="{css_class}" width="{size}" '
        f'alt="Prajnya Studio Logo" style="object-fit: contain;"/>'
        for name, css_class in available
    )

    return f'''
    <div class="logo-container" style="
        width: {size}px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    ">
        {images}
    </div>
    '''


# ============================================================
# GLOBAL CSS
# ============================================================

SARASWATI_CSS = f"""
/* Prajnya Studio (V6) - Saraswati Theme (Lotus-Mandala) */

/* The light palette is the BASE. Every token is defined here, so a page is
   never left with an undefined variable no matter what theme is active. */
:root {{
{_css_variable_block(THEME_LIGHT)}

    /* Typography — one set, shared by every theme. */
    --prajnya-font-display: 'Cinzel', Georgia, serif;
    --prajnya-font-body: 'Inter', system-ui, sans-serif;
    --prajnya-font-mono: 'JetBrains Mono', Consolas, monospace;
}}

/* The dark palette OVERRIDES it, on one attribute set on <html>. Same
   specificity as :root (both are one attribute/pseudo-class on the same
   element), so this block must come second — it wins by source order. */
[data-theme="dark"] {{
{_css_variable_block(THEME_DARK)}
}}

/* Quasar and nicegui paint their own surfaces — cards, dialogs, inputs,
   tables, menus — from Quasar's own variables and a few hardcoded whites.
   Left alone, those stay white in dark mode, which is exactly the "one
   bright panel in a dark UI" failure this migration exists to avoid. The
   app's own styles cannot reach them, so they are re-pointed here at the
   same tokens everything else uses. */
[data-theme="dark"] .q-card,
[data-theme="dark"] .q-dialog__inner > div,
[data-theme="dark"] .q-menu,
[data-theme="dark"] .q-table,
[data-theme="dark"] .q-table__container,
[data-theme="dark"] .q-table__top,
[data-theme="dark"] .q-table__bottom,
[data-theme="dark"] .q-table thead tr,
[data-theme="dark"] .q-table tbody td,
[data-theme="dark"] .q-expansion-item__content,
/* .q-tab-panels was the one this list missed, and it is the reason the
   list is not trusted to be complete: Strategy Lab's three tabs sit in a
   1024x922 panel container that Quasar paints white, so the whole screen
   came up as a white sheet with dark cards floating on it. Nothing in the
   app's own styles touches that element. See the live-check note in
   CLAUDE.md — a new screen using a Quasar container this list does not
   name will do the same thing, and only looking at it will show that. */
[data-theme="dark"] .q-tab-panels,
[data-theme="dark"] .q-tab-panel,
[data-theme="dark"] .q-page,
[data-theme="dark"] .q-drawer,
[data-theme="dark"] .q-header,
[data-theme="dark"] .q-footer {{
    background: var(--prajnya-card);
    color: var(--prajnya-text);
}}

[data-theme="dark"] .q-field__native,
[data-theme="dark"] .q-field__input,
[data-theme="dark"] .q-field__label,
[data-theme="dark"] .q-item__label,
[data-theme="dark"] .q-tab {{
    color: var(--prajnya-text);
}}

[data-theme="dark"] .q-field__control:before {{
    border-color: var(--prajnya-border);
}}

[data-theme="dark"] .q-separator {{
    background: var(--prajnya-border);
}}

/* ========== LOGO VARIANTS ==========
   theme.lotus_logo() emits both images; these rules decide which is seen,
   so flipping data-theme swaps the logo with no page rebuild. */
.logo-dark {{ display: none; }}
[data-theme="dark"] .logo-light {{ display: none; }}
[data-theme="dark"] .logo-dark {{ display: block; }}

body {{
    background-color: var(--prajnya-canvas);
    color: var(--prajnya-text);
    font-family: var(--prajnya-font-body);
    margin: 0;
    padding: 0;
}}

/* ========== SIDEBAR ========== */
.sidebar {{
    background: var(--prajnya-sidebar);
    color: var(--prajnya-indigo);
    border-right: 1px solid var(--prajnya-border);
}}

.nav-item {{
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    transition: all 0.15s ease;
    cursor: pointer;
    color: var(--prajnya-text-muted);
    display: flex;
    align-items: center;
    gap: 12px;
}}

.nav-item:hover {{
    background: var(--prajnya-nav-hover);
    color: var(--prajnya-indigo);
}}

.nav-item-active {{
    background: var(--prajnya-nav-hover);
    border-left: 3px solid var(--prajnya-indigo);
    padding-left: 13px;
    color: var(--prajnya-indigo);
    font-weight: 600;
}}

/* ========== PROGRESS BAR ========== */
.prajnya-progress {{
    background: var(--prajnya-accent);
    border-radius: 4px;
    height: 8px;
    position: relative;
    overflow: hidden;
}}

.prajnya-progress.striped::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: linear-gradient(
        45deg,
        rgba(255, 255, 255, 0.15) 25%,
        transparent 25%,
        transparent 50%,
        rgba(255, 255, 255, 0.15) 50%,
        rgba(255, 255, 255, 0.15) 75%,
        transparent 75%,
        transparent
    );
    background-size: 20px 20px;
    animation: progress-stripes 1s linear infinite;
}}

@keyframes progress-stripes {{
    from {{ background-position: 0 0; }}
    to {{ background-position: 20px 0; }}
}}

/* ========== DROP ZONE ========== */
.drop-zone {{
    border: 2px dashed var(--prajnya-drop-zone-border);
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
    transition: all 0.3s ease;
}}

.drop-zone:hover {{
    border-color: var(--prajnya-accent);
    background: rgba(255, 179, 0, 0.05);
    cursor: pointer;
}}

.drop-zone.dragging {{
    border-color: var(--prajnya-accent);
    background: rgba(255, 179, 0, 0.1);
    border-style: solid;
    transform: scale(1.02);
}}

/* ========== TERMINAL ========== */
.terminal {{
    background: var(--prajnya-terminal-bg);
    color: var(--prajnya-terminal-text);
    font-family: var(--prajnya-font-mono);
    font-size: 13px;
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}}

.terminal-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    padding: 8px;
    background: rgba(0, 0, 0, 0.1);
    border-radius: 8px 8px 0 0;
}}

.terminal-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
}}
.terminal-dot.red {{ background: #FF5F56; }}
.terminal-dot.yellow {{ background: #FFBD2E; }}
.terminal-dot.green {{ background: #27C93F; }}

.log-entry {{ margin: 4px 0; line-height: 1.5; }}
.log-INFO {{ color: var(--prajnya-log-info); }}
.log-SUCCESS {{
    color: var(--prajnya-status);
    text-shadow: 0 0 4px rgba(16, 185, 129, 0.3);
}}
.log-WARNING {{ color: var(--prajnya-amber); }}
.log-ERROR {{ color: var(--prajnya-log-error); font-weight: 600; }}

/* ========== LEDGER GRID ========== */
.ledger-row-provisional {{
    background: var(--prajnya-row-provisional) !important;
}}

.ledger-row-reviewed {{
    background: var(--prajnya-row-reviewed) !important;
}}

/* ========== CUSTOM SCROLLBAR ========== */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}
::-webkit-scrollbar-track {{
    background: transparent;
}}
::-webkit-scrollbar-thumb {{
    background: var(--prajnya-scrollbar-thumb);
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: var(--prajnya-scrollbar-thumb-hover);
}}

/* ========== STATUS PULSE ========== */
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}

.status-pulse {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}}

.status-pulse.online {{ background: var(--prajnya-status); }}
.status-pulse.offline {{ background: var(--prajnya-log-error); }}
.status-pulse.processing {{ background: var(--prajnya-accent); }}

/* ========== BUTTONS ========== */
.btn-primary {{
    background: var(--prajnya-vermillion) !important;
    color: var(--prajnya-on-accent) !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}}

.btn-primary:hover {{
    background: var(--prajnya-vermillion-dark) !important;
    box-shadow: 0 4px 12px rgba(184, 71, 42, 0.3);
}}

.btn-secondary {{
    background: transparent !important;
    color: var(--prajnya-indigo) !important;
    border: 1px solid var(--prajnya-border) !important;
    font-weight: 500 !important;
}}

.btn-secondary:hover {{
    background: var(--prajnya-sidebar) !important;
    border-color: var(--prajnya-gold) !important;
}}
"""
