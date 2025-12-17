"""
Saraswati Theme - Design Language Constants and Styling
Version: 2.0 (Lotus-Mandala)
"""

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

    # Terminal (keep dark for contrast)
    'terminal_bg': '#1A1F23',

    # Legacy mappings (for backward compatibility)
    'primary': '#2C3E50',          # Maps to indigo
    'accent': '#B8472A',           # Maps to vermillion
    'identity': '#B8472A',         # Maps to vermillion
    'status': '#2D5A3D',           # Maps to success
    'amber_warning': '#D4A84B',    # Maps to warning
    'shadow_ink': '#1A1F23',       # Maps to terminal_bg
}

# ============================================================
# LOGO & ANIMATION
# ============================================================

def lotus_logo(size: int = 120, dark_mode: bool = False) -> str:
    """
    Returns HTML for Lotus-Mandala logo (v2.0).
    Uses PNG images for light/dark mode variants.

    Args:
        size: Logo size in pixels
        dark_mode: If True, use blue variant for dark backgrounds

    Returns:
        HTML string for logo display
    """
    from pathlib import Path

    # Determine which logo to use
    # logo_white.png = white/gold design for light sidebar
    # logo_blue.png = blue design for dark backgrounds
    logo_file = 'logo_white.png' if dark_mode else 'logo_blue.png'

    # Verify logo exists
    logo_path = Path(__file__).parent.parent / logo_file

    if not logo_path.exists():
        # Fallback if logo missing
        return f'''
        <div style="width: {size}px; height: {size}px; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 48px;">🪷</span>
        </div>
        '''

    return f'''
    <div class="logo-container" style="
        width: {size}px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    ">
        <img src="/static/{logo_file}"
             width="{size}"
             alt="Prajnya Studio Logo"
             style="object-fit: contain;"/>
    </div>
    '''

def animated_flame_logo(size: int = 64, active: bool = False) -> str:
    """Legacy function - redirects to lotus_logo() for backward compatibility."""
    return lotus_logo(size=size, dark_mode=False)

def logo_svg(size_px: int = 64) -> str:
    """Legacy function - redirects to lotus_logo() for backward compatibility."""
    return lotus_logo(size=size_px, dark_mode=False)

# ============================================================
# GLOBAL CSS
# ============================================================

SARASWATI_CSS = f"""
/* Prajnya Studio - Saraswati Theme v2.0 (Lotus-Mandala) */

:root {{
    /* Primary Palette */
    --prajnya-canvas: {THEME_LIGHT['canvas']};
    --prajnya-card: {THEME_LIGHT['card']};
    --prajnya-sidebar: {THEME_LIGHT['sidebar']};
    --prajnya-indigo: {THEME_LIGHT['indigo']};
    --prajnya-gold: {THEME_LIGHT['gold']};

    /* Secondary Palette */
    --prajnya-vermillion: {THEME_LIGHT['vermillion']};
    --prajnya-vermillion-dark: {THEME_LIGHT['vermillion_dark']};
    --prajnya-success: {THEME_LIGHT['success']};
    --prajnya-warning: {THEME_LIGHT['warning']};
    --prajnya-error: {THEME_LIGHT['error']};

    /* Neutrals */
    --prajnya-text: {THEME_LIGHT['text']};
    --prajnya-text-muted: {THEME_LIGHT['text_muted']};
    --prajnya-border: {THEME_LIGHT['border']};
    --prajnya-border-gold: {THEME_LIGHT['border_gold']};

    /* Terminal */
    --prajnya-terminal-bg: {THEME_LIGHT['terminal_bg']};

    /* Legacy variables (for backward compatibility) */
    --prajnya-primary: {THEME_LIGHT['primary']};
    --prajnya-accent: {THEME_LIGHT['accent']};
    --prajnya-identity: {THEME_LIGHT['identity']};
    --prajnya-status: {THEME_LIGHT['status']};
    --prajnya-shadow: {THEME_LIGHT['shadow_ink']};
    --prajnya-amber: {THEME_LIGHT['amber_warning']};

    /* Typography - v2.0 */
    --prajnya-font-display: 'Cinzel', Georgia, serif;
    --prajnya-font-body: 'Inter', system-ui, sans-serif;
    --prajnya-font-mono: 'JetBrains Mono', Consolas, monospace;

    /* Legacy font variables */
    --prajnya-font-serif: 'Cinzel', Georgia, serif;
    --prajnya-font-sans: 'Inter', system-ui, sans-serif;
}}

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

/* Navigation Items */
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
    background: #E8E4DD;
    color: var(--prajnya-indigo);
}}

.nav-item-active {{
    background: #E8E4DD;
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
    border: 2px dashed #CCC;
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
    color: #E0E0E0;
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
.log-INFO {{ color: #64B5F6; }}
.log-SUCCESS {{ 
    color: var(--prajnya-status); 
    text-shadow: 0 0 4px rgba(16, 185, 129, 0.3);
}}
.log-WARNING {{ color: var(--prajnya-amber); }}
.log-ERROR {{ color: #EF5350; font-weight: 600; }}

/* ========== LEDGER GRID ========== */
.ledger-row-provisional {{
    background: rgba(255, 167, 38, 0.1) !important;
}}

.ledger-row-reviewed {{
    background: white !important;
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
    background: #CFD8DC;
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: #B0BEC5;
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
.status-pulse.offline {{ background: #EF5350; }}
.status-pulse.processing {{ background: var(--prajnya-accent); }}

/* ========== BUTTONS ========== */
.btn-primary {{
    background: var(--prajnya-vermillion) !important;
    color: white !important;
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