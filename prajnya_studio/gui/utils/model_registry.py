import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Default profile directory
PROFILE_DIR = Path("src/processing/strategies/config/vlm_profiles")

def discover_vlm_models(profile_dir: Path = PROFILE_DIR) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Scans the profiles directory and returns categorized model dictionaries.
    
    Returns:
        tuple: (MODELS_LOCAL, MODELS_CLOUD)
    """
    local_models = {}
    cloud_models = {}

    if not profile_dir.exists():
        logger.warning(f"Profile directory not found: {profile_dir}")
        return local_models, cloud_models

    for yaml_file in profile_dir.glob("*.yaml"):
        try:
            # Use filename stem as the primary ID (matching Orchestrator loading logic)
            model_id = yaml_file.stem
            
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            if not data:
                continue
            
            # Create a nice label
            # 1. Try notes[1] (often a punchy description)
            # 2. Try notes[0]
            # 3. Fallback to model_id capitalized
            notes = data.get('notes', [])
            if notes and len(notes) > 1:
                label = f"{model_id} - {notes[1]}"
            elif notes:
                label = f"{model_id} - {notes[0]}"
            else:
                label = model_id.replace('-', ' ').title()

            # Classification Logic
            capabilities = data.get('capabilities', [])
            model_type = data.get('model_type', '').lower()
            
            is_cloud = (
                'cloud_api' in capabilities or 
                'cloud' in model_type or 
                'gemini' in model_id.lower() or
                'gpt' in model_id.lower() or
                'claude' in model_id.lower()
            )

            if is_cloud:
                cloud_models[model_id] = label
            else:
                local_models[model_id] = label

        except Exception as e:
            logger.error(f"Error parsing model profile {yaml_file.name}: {e}")

    # Sort dictionaries by key for UI consistency
    local_models = dict(sorted(local_models.items()))
    cloud_models = dict(sorted(cloud_models.items()))

    # Fallback if empty to prevent GUI crash
    if not local_models:
        local_models = {"olmocr-2-7b": "OlmOCR 2.7B (Fallback)"}
    if not cloud_models:
        cloud_models = {"gemini-3-flash": "Gemini 3 Flash (Fallback)"}

    return local_models, cloud_models
