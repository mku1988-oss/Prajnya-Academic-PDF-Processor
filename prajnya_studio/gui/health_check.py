import sys
import os
from typing import Dict, Any
from .persistent_settings import SettingsManager

def run_startup_checks(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs system health checks.
    """
    results = {
        'checks': {},
        'fatal': False,
        'warnings': []
    }

    # 1. Python Version
    py_ver = sys.version_info
    if py_ver >= (3, 10):
        results['checks']['python_version'] = {'ok': True, 'message': f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"}
    else:
        results['checks']['python_version'] = {'ok': False, 'message': "Requires Python 3.10+"}
        results['fatal'] = True

    # 2. PyPDFium2 Check
    try:
        import pypdfium2
        results['checks']['pypdfium2'] = {'ok': True, 'message': "Installed"}
    except ImportError:
        results['checks']['pypdfium2'] = {'ok': False, 'message': "Not installed"}
        results['fatal'] = True

    # 3. Ledger Writable
    # (Checking a dummy path for now since we haven't set up the full persistence layer yet)
    ledger_path = SettingsManager.settings_path().parent / "semantic_ledger.csv"
    try:
        if not ledger_path.exists():
            # Try creating a dummy file to test write permissions
            with open(ledger_path, 'a') as f:
                pass
        elif not os.access(ledger_path, os.W_OK):
             raise PermissionError("Not writable")
        
        results['checks']['ledger_writable'] = {'ok': True, 'message': str(ledger_path)}
    except Exception as e:
        results['checks']['ledger_writable'] = {'ok': False, 'message': str(e)}
        # Making this a warning for Sprint 0 so app can launch
        results['warnings'].append("Ledger file not writable")

    return results