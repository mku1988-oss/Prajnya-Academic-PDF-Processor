"""
Pytest configuration for integration tests.

Adds custom command-line options for PDF path specification and page-specific testing.
"""

import pytest
from pathlib import Path


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--pdf",
        action="store",
        default="sample_pdfs/Hydration-induced.pdf",
        help="Path to PDF file for integration testing"
    )
    parser.addoption(
        "--pages",
        action="store",
        default=None,
        help="Comma-separated page numbers to test (e.g., '1,2,3' or '5-10')"
    )


@pytest.fixture(scope="session")
def pdf_path(request):
    """Get PDF path from command line or use default."""
    path_str = request.config.getoption("--pdf")
    
    # Get project root (3 levels up from processing folder)
    project_root = Path(__file__).parent.parent.parent
    
    # Convert to absolute path
    pdf_path = Path(path_str)
    if not pdf_path.is_absolute():
        # Try relative to current working directory
        if pdf_path.exists():
            return pdf_path.absolute()
        # Try relative to project root
        alt_path = project_root / path_str
        if alt_path.exists():
            return alt_path
        # Path doesn't exist
        pytest.skip(f"PDF file not found: {path_str} (also tried {alt_path})")
    
    if not pdf_path.exists():
        pytest.skip(f"PDF file not found: {pdf_path}")
    
    return pdf_path


@pytest.fixture
def pdf_filename(request):
    """Get PDF filename from CLI argument for page-specific testing."""
    return request.config.getoption("--pdf")


@pytest.fixture
def page_numbers(request):
    """Parse page numbers from CLI argument for page-specific testing."""
    pages_arg = request.config.getoption("--pages")
    if not pages_arg:
        return None
    
    # Parse comma-separated page numbers
    pages = []
    for part in pages_arg.split(','):
        part = part.strip()
        if '-' in part:
            # Range notation: "5-10"
            start, end = map(int, part.split('-'))
            pages.extend(range(start, end + 1))
        else:
            # Single page: "3"
            pages.append(int(part))
    
    return sorted(set(pages))  # Remove duplicates and sort
