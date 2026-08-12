"""
Prajnya Studio iconography — Vedic line-art SVGs.

Ported from V5; these are inert strings with no V5-specific behaviour to
adapt.

One change since the port: the stroke and fill colours were hardcoded
(#2C3E50 indigo, #B8472A vermillion) and are now `var(--prajnya-indigo)` /
`var(--prajnya-vermillion)`. An SVG presentation attribute is parsed as CSS,
so a variable works there exactly as it does in a style rule — and without
it, every sidebar icon would keep its dark-blue line art on the dark theme's
dark sidebar and simply disappear.
"""

BATCH_COMMANDER = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:100%;">
    <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M8 7h6" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round" opacity="0.4"/>
    <path d="M8 11h8" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round" opacity="0.4"/>
    <path d="M14.5 14.5c0-1.5-1-2-1-3.5 0-1.5 1.5-2 1.5-3.5 1.5 1.5 2.5 2 2.5 3.5 0 1.5-1.2 2-2.5 2.8" stroke="var(--prajnya-vermillion)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>
"""

STRATEGY_LAB = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:100%;">
    <circle cx="12" cy="12" r="2.5" stroke="var(--prajnya-vermillion)" stroke-width="1.5" fill="none"/>
    <circle cx="12" cy="5" r="1.5" stroke="var(--prajnya-indigo)" stroke-width="1.5"/>
    <circle cx="6" cy="16" r="1.5" stroke="var(--prajnya-indigo)" stroke-width="1.5"/>
    <circle cx="18" cy="16" r="1.5" stroke="var(--prajnya-indigo)" stroke-width="1.5"/>
    <path d="M12 7.5V9.5" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round"/>
    <path d="M10.5 13L7.5 15" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round"/>
    <path d="M13.5 13L16.5 15" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round"/>
    <path d="M20 12c0 4.4-3.6 8-8 8s-8-3.6-8-8" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-opacity="0.2" stroke-dasharray="2 2"/>
</svg>
"""

LEDGER_MASTER = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:100%;">
    <path d="M4 6h16M4 6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2M4 6v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M8 10h8" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round"/>
    <path d="M8 14h8" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round"/>
    <path d="M8 18h4" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round"/>
    <circle cx="12" cy="12" r="1" fill="var(--prajnya-vermillion)"/>
</svg>
"""

REVIEW_QUEUE = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:100%;">
    <path d="M9 11l3 3L22 4" stroke="var(--prajnya-vermillion)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

RESULTS_VIEWER = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:100%;">
    <rect x="3" y="3" width="7" height="7" rx="1" stroke="var(--prajnya-indigo)" stroke-width="1.5"/>
    <rect x="14" y="3" width="7" height="7" rx="1" stroke="var(--prajnya-vermillion)" stroke-width="1.5"/>
    <rect x="3" y="14" width="7" height="7" rx="1" stroke="var(--prajnya-vermillion)" stroke-width="1.5"/>
    <rect x="14" y="14" width="7" height="7" rx="1" stroke="var(--prajnya-indigo)" stroke-width="1.5"/>
</svg>
"""

CHAPTER_MAP = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:100%;">
    <path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15z" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linejoin="round"/>
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round"/>
    <circle cx="10.5" cy="8" r="1" fill="var(--prajnya-vermillion)"/>
    <path d="M13 8h4M8.5 11.5h8.5" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>
</svg>
"""

SETTINGS_CHAKRA = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:100%;">
    <circle cx="12" cy="12" r="3" stroke="var(--prajnya-indigo)" stroke-width="1.5"/>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linejoin="round"/>
</svg>
"""

HELP_DIYA = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:100%;">
    <path d="M4 16c0 2.5 4 4 8 4s8-1.5 8-4H4z" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-linejoin="round"/>
    <path d="M12 2C12 2 10 7 10 9s2 4 2 4 2-2 2-4-2-7-2-7z" fill="var(--prajnya-vermillion)" fill-opacity="0.3" stroke="var(--prajnya-vermillion)" stroke-width="1.5" stroke-linejoin="round"/>
    <path d="M12 22v-2" stroke="var(--prajnya-indigo)" stroke-width="1.5" stroke-opacity="0.5" stroke-linecap="round"/>
</svg>
"""
