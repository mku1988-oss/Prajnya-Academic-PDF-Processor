"""
The theme layer: two palettes, one variable indirection, and the guards
that keep a half-migrated colour from reaching a user's screen.

For biology researchers: these tests do not open a browser. They check the
*strings* the app hands the browser — that every colour a page asks for is
defined in both themes, and that no page has quietly gone back to writing a
hex code that a theme switch cannot touch.

Why that last one needs a test at all: a colour baked into the HTML at
page-build time still renders perfectly. It just renders the wrong colour in
the other theme, on one panel, three screens from where anyone is looking.
Nothing raises. The whole point of `theme.var()` is to make that class of
mistake fail here instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.gui import theme

GUI_ROOT = Path(__file__).resolve().parents[1] / "src" / "gui"

#: Every module that builds UI and therefore might name a colour.
PAGE_MODULES = sorted((GUI_ROOT / "pages").glob("*.py")) + [
    GUI_ROOT / "app.py",
    GUI_ROOT / "components.py",
]


# --- the palettes -------------------------------------------------------

class TestPalettes:

    def test_both_themes_define_the_same_tokens(self):
        """A token in one theme and not the other is the failure mode this
        whole file exists for: `var(--prajnya-x)` resolves in light, resolves
        to nothing in dark, and one element silently loses its colour."""
        assert set(theme.THEME_LIGHT) == set(theme.THEME_DARK)

    def test_no_token_is_empty(self):
        for name, palette in (("light", theme.THEME_LIGHT), ("dark", theme.THEME_DARK)):
            for token, value in palette.items():
                assert value.strip(), f"{name}/{token} is empty"

    def test_the_two_palettes_are_actually_different(self):
        """Guards against a dark theme that was added as a copy and never
        filled in — which would pass every other test in this file."""
        differing = [k for k in theme.THEME_LIGHT
                     if theme.THEME_LIGHT[k] != theme.THEME_DARK[k]]
        assert len(differing) > len(theme.THEME_LIGHT) // 2

    def test_terminal_stays_dark_in_both(self):
        """The transcript panel imitates a console on purpose. It is the one
        surface that should NOT flip when the theme does."""
        assert theme.THEME_LIGHT["terminal_text"] == theme.THEME_DARK["terminal_text"]
        for palette in (theme.THEME_LIGHT, theme.THEME_DARK):
            assert palette["terminal_bg"].lower() < "#800000"  # a dark hex


# --- var() --------------------------------------------------------------

class TestVar:

    def test_returns_a_css_variable_reference(self):
        assert theme.var("indigo") == "var(--prajnya-indigo)"

    def test_underscores_become_hyphens(self):
        """Python keys are snake_case, CSS custom properties are kebab-case.
        Getting this wrong produces a variable name that is never defined."""
        assert theme.var("text_muted") == "var(--prajnya-text-muted)"

    def test_unknown_token_raises_with_the_name_in_the_message(self):
        with pytest.raises(KeyError) as excinfo:
            theme.var("indgio")
        assert "indgio" in str(excinfo.value)

    def test_every_token_var_is_defined_in_the_css(self):
        for token in theme.THEME_LIGHT:
            reference = theme.var(token)
            declaration = reference[len("var("):-1] + ":"
            assert theme.SARASWATI_CSS.count(declaration) == 2, (
                f"{token} should be declared exactly twice — once in :root "
                f"(light) and once in [data-theme=\"dark\"]"
            )


# --- the emitted CSS ----------------------------------------------------

class TestCss:

    def test_light_block_comes_first_and_dark_overrides_it(self):
        """Both selectors have the same specificity, so the dark block wins
        only by source order. Reverse them and dark mode silently stops
        working while every other test still passes."""
        light_at = theme.SARASWATI_CSS.index(":root {")
        dark_at = theme.SARASWATI_CSS.index('[data-theme="dark"] {')
        assert light_at < dark_at

    def test_no_var_reference_in_the_css_is_undefined(self):
        """Catches a typo inside the hand-written CSS below the variable
        blocks, where nothing would otherwise complain."""
        referenced = set(re.findall(r"var\((--prajnya-[a-z-]+)\)", theme.SARASWATI_CSS))
        declared = set(re.findall(r"(--prajnya-[a-z-]+):", theme.SARASWATI_CSS))
        assert referenced <= declared, f"undefined: {sorted(referenced - declared)}"


# --- theme selection ----------------------------------------------------

class TestThemeSelection:

    def test_light_removes_the_attribute_rather_than_setting_one(self):
        """`:root` already holds the light palette, so light is the absence
        of an override — not a third state to keep in sync."""
        assert "removeAttribute" in theme.apply_theme_js("saraswati_light")

    def test_dark_sets_the_attribute(self):
        js = theme.apply_theme_js("saraswati_dark")
        assert "setAttribute" in js and "'dark'" in js

    def test_an_unknown_stored_theme_falls_back_to_the_default(self):
        """A settings file is user data and may name a theme this build
        removed. Opening the app must not depend on its contents."""
        assert theme.resolve_theme("saraswati_neon") == theme.DEFAULT_THEME
        assert theme.resolve_theme("") == theme.DEFAULT_THEME

    def test_head_html_is_a_script_tag(self):
        html = theme.theme_head_html("saraswati_dark")
        assert html.startswith("<script>") and html.endswith("</script>")

    def test_every_selectable_theme_has_an_attribute_value(self):
        for name in theme.THEME_ATTRIBUTES:
            theme.apply_theme_js(name)  # must not raise


# --- the logo -----------------------------------------------------------

class TestLogo:

    def test_both_variants_are_emitted(self):
        """V4 shipped `logo_white.png` and never once put it on screen. It is
        emitted alongside the blue one now, with CSS choosing between them —
        so a live theme switch swaps the logo too, with no page rebuild."""
        html = theme.lotus_logo()
        assert "logo_blue.png" in html and "logo_white.png" in html
        assert 'class="logo-light"' in html and 'class="logo-dark"' in html

    def test_css_hides_the_dark_logo_by_default(self):
        assert ".logo-dark { display: none; }" in theme.SARASWATI_CSS
        assert '[data-theme="dark"] .logo-light { display: none; }' in theme.SARASWATI_CSS


# --- the migration itself, pinned ---------------------------------------

class TestNoHardcodedColours:
    """Source-inspection guards. Every one of these caught something real
    during the migration; they are here so it cannot creep back."""

    def test_no_page_reads_the_palette_dict_directly(self):
        for path in PAGE_MODULES:
            source = path.read_text(encoding="utf-8")
            assert "THEME_LIGHT[" not in source and "THEME_DARK[" not in source, (
                f"{path.name} reads a palette dict directly. That inlines a "
                f"colour at page-build time, where a theme switch cannot "
                f"reach it. Use theme.var('token')."
            )

    def test_no_page_writes_a_hex_colour(self):
        hex_colour = re.compile(r"#[0-9A-Fa-f]{3}(?:[0-9A-Fa-f]{3})?\b")
        for path in PAGE_MODULES:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                assert not hex_colour.search(line), f"{path.name}:{number}: {line.strip()}"

    def test_no_page_writes_a_bare_white(self):
        """`background: white` is the specific literal that produced a glaring
        panel in the dark theme — inputs and buttons both had one."""
        bare_white = re.compile(r":\s*white\b")
        for path in PAGE_MODULES:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                assert not bare_white.search(line), f"{path.name}:{number}: {line.strip()}"

    def test_icons_use_variables_not_fixed_strokes(self):
        """The sidebar SVGs were hardcoded indigo line art. On the dark
        theme's dark sidebar that is very nearly invisible — and an icon that
        renders in the right place in the wrong colour looks like a broken
        asset, not a theming bug."""
        source = (GUI_ROOT / "icons.py").read_text(encoding="utf-8")
        # Matched as SVG attributes, not as bare text: the module docstring
        # names both old hex codes while explaining why they are gone.
        painted = re.findall(r'(?:stroke|fill)="(#[0-9A-Fa-f]{3,6})"', source)
        assert not painted, f"still hardcoded: {sorted(set(painted))}"
        assert 'stroke="var(--prajnya-indigo)"' in source
