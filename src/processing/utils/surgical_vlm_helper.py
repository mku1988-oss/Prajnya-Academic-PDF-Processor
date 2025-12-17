"""
Surgical VLM Helper for task-based VLM calls.
Maintains modular, configuration-driven architecture.

Version: 1.1 (3-Tier Hybrid V2 Stable - Fixed API Integration)
Date: October 21, 2025
"""
import logging
import yaml
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import io

class SurgicalVLMHelper:
    """
    Helper class for surgical VLM processing.
    
    Uses task definitions to maintain separation of concerns
    and avoid hardcoded prompts in processors.
    
    INTEGRATION NOTE: Uses UniversalVLMProvider.process_page() method
    with task-specific prompts loaded from YAML definitions.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize helper with task definitions.
        
        Args:
            config_path: Path to task definitions directory
        """
        self.logger = logging.getLogger("SurgicalVLMHelper")
        self.tasks = self._load_surgical_tasks(config_path)
        
        self.logger.info(f"SurgicalVLMHelper loaded {len(self.tasks)} task definitions")
    
    def _load_surgical_tasks(self, config_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load all surgical task YAML definitions.
        
        Args:
            config_path: Path to config directory
            
        Returns:
            Dictionary of task configurations
        """
        tasks = {}
        task_dir = Path(config_path) / 'config' / 'task_definitions'
        
        task_files = [
            'figure_description.yaml',
            'equation_to_latex.yaml',
            'table_description.yaml'
        ]
        
        for task_file in task_files:
            task_path = task_dir / task_file
            
            if not task_path.exists():
                self.logger.warning(f"Task file not found: {task_path}")
                continue
            
            try:
                with open(task_path, 'r', encoding='utf-8') as f:
                    task_config = yaml.safe_load(f)
                    task_name = task_config.get('task_name')
                    tasks[task_name] = task_config
                    self.logger.debug(f"Loaded task: {task_name}")
            except Exception as e:
                self.logger.error(f"Failed to load {task_file}: {e}")
        
        return tasks
    
    def describe_figure(self, vlm_provider, image_bytes: bytes) -> str:
        """
        Generate figure description using VLM.
        
        Args:
            vlm_provider: UniversalVLMProvider instance
            image_bytes: PNG image bytes
            
        Returns:
            Description in format: **Figure:** Description text
        """
        task = self.tasks.get('figure_description')
        if not task:
            self.logger.error("figure_description task not loaded")
            return "**Figure:** [Description unavailable - task not loaded]"
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Get prompt from task definition
            prompt = task['prompt']
            
            # Create a temporary task-specific provider instance
            # This allows us to use the task's prompt with process_page()
            from src.llm.providers.universal_vlm_provider import TaskDefinition
            
            temp_task = TaskDefinition(
                task_name=task['task_name'],
                task_type=task.get('task_type', 'visual_description'),
                prompt=prompt,
                output_format=task.get('output_format', {'type': 'text'})
            )
            
            # Override the provider's task definition temporarily
            original_task = vlm_provider.task_definition
            vlm_provider.task_definition = temp_task
            vlm_provider.task_type = temp_task.task_name
            
            # Use process_page() with the task-specific prompt
            result = vlm_provider.process_page(image)
            
            # Restore original task
            vlm_provider.task_definition = original_task
            vlm_provider.task_type = original_task.task_name
            
            if result.get('success'):
                # Extract content from result
                return result.get('content', '').strip()
            else:
                self.logger.error(f"Figure description failed: {result.get('error')}")
                return "**Figure:** [Description unavailable]"
                
        except Exception as e:
            self.logger.error(f"Figure description failed: {e}", exc_info=True)
            return "**Figure:** [Description unavailable]"

    async def describe_figure_async(self, async_provider, image_bytes: bytes) -> str:
        """Async version of describe_figure."""
        task = self.tasks.get('figure_description')
        if not task:
            return "**Figure:** [Description unavailable - task not loaded]"
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            prompt = task['prompt']
            
            from src.llm.providers.universal_vlm_provider import TaskDefinition
            temp_task = TaskDefinition(
                task_name=task['task_name'],
                task_type=task.get('task_type', 'visual_description'),
                prompt=prompt,
                output_format=task.get('output_format', {'type': 'text'})
            )
            
            # Use process_page wrapper which handles semaphore and logging
            # Call _process_single_page_wrapper_async directly or expose a public async method?
            # AsyncUniversalVLMProvider doesn't have a public 'process_page_async' for single page yet?
            # It has 'process_pages_batch_async'.
            # I should use the wrapper internal method or add a public one.
            # Let's use internal wrapper for now as it's accessible.
            
            result = await async_provider._process_single_page_wrapper_async(
                page_image=image,
                task_definition=temp_task,
                page_num=0 # Unknown page num here
            )
            
            if result.get('success'):
                return result.get('content', '').strip()
            else:
                return "**Figure:** [Description unavailable]"
                
        except Exception as e:
            self.logger.error(f"Async figure description failed: {e}")
            return "**Figure:** [Description unavailable]"
    
    def convert_equation_to_latex(self, vlm_provider, image_bytes: bytes) -> str:
        """
        Convert equation image to LaTeX.
        
        Args:
            vlm_provider: UniversalVLMProvider instance
            image_bytes: PNG image bytes
            
        Returns:
            Raw LaTeX string (no $ wrappers)
        """
        task = self.tasks.get('equation_to_latex')
        if not task:
            self.logger.error("equation_to_latex task not loaded")
            return "[equation]"
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Get prompt from task definition
            prompt = task['prompt']
            
            # Create a temporary task-specific provider instance
            from src.llm.providers.universal_vlm_provider import TaskDefinition
            
            temp_task = TaskDefinition(
                task_name=task['task_name'],
                task_type=task.get('task_type', 'equation_conversion'),
                prompt=prompt,
                output_format=task.get('output_format', {'type': 'text'})
            )
            
            # Override the provider's task definition temporarily
            original_task = vlm_provider.task_definition
            vlm_provider.task_definition = temp_task
            vlm_provider.task_type = temp_task.task_name
            
            # Use process_page() with the task-specific prompt
            result = vlm_provider.process_page(image)
            
            # Restore original task
            vlm_provider.task_definition = original_task
            vlm_provider.task_type = original_task.task_name
            
            if result.get('success'):
                # Extract content and clean up any $ wrappers
                latex = result.get('content', '').strip()
                latex = latex.strip('$').strip()
                return latex
            else:
                self.logger.error(f"Equation conversion failed: {result.get('error')}")
                return "[equation]"
                
        except Exception as e:
            self.logger.error(f"Equation conversion failed: {e}", exc_info=True)
            return "[equation]"

    async def convert_equation_to_latex_async(self, async_provider, image_bytes: bytes) -> str:
        """Async version of convert_equation_to_latex."""
        task = self.tasks.get('equation_to_latex')
        if not task:
            return "[equation]"
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            prompt = task['prompt']
            
            from src.llm.providers.universal_vlm_provider import TaskDefinition
            temp_task = TaskDefinition(
                task_name=task['task_name'],
                task_type=task.get('task_type', 'equation_conversion'),
                prompt=prompt,
                output_format=task.get('output_format', {'type': 'text'})
            )
            
            result = await async_provider._process_single_page_wrapper_async(
                page_image=image,
                task_definition=temp_task,
                page_num=0
            )
            
            if result.get('success'):
                latex = result.get('content', '').strip()
                latex = latex.strip('$').strip()
                return latex
            else:
                return "[equation]"
                
        except Exception as e:
            self.logger.error(f"Async equation conversion failed: {e}")
            return "[equation]"
    
    def describe_table(self, vlm_provider, image_bytes: bytes) -> str:
        """
        Convert table image to Markdown.
        
        Args:
            vlm_provider: UniversalVLMProvider instance
            image_bytes: PNG image bytes
            
        Returns:
            Markdown table format
        """
        task = self.tasks.get('table_description')
        if not task:
            self.logger.error("table_description task not loaded")
            return "| Table | Data |\n|-------|------|\n| N/A | N/A |"
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Get prompt from task definition
            prompt = task['prompt']
            
            # Create a temporary task-specific provider instance
            from src.llm.providers.universal_vlm_provider import TaskDefinition
            
            temp_task = TaskDefinition(
                task_name=task['task_name'],
                task_type=task.get('task_type', 'table_conversion'),
                prompt=prompt,
                output_format=task.get('output_format', {'type': 'text'})
            )
            
            # Override the provider's task definition temporarily
            original_task = vlm_provider.task_definition
            vlm_provider.task_definition = temp_task
            vlm_provider.task_type = temp_task.task_name
            
            # Use process_page() with the task-specific prompt
            result = vlm_provider.process_page(image)
            
            # Restore original task
            vlm_provider.task_definition = original_task
            vlm_provider.task_type = original_task.task_name
            
            if result.get('success'):
                # Extract content
                return result.get('content', '').strip()
            else:
                self.logger.error(f"Table description failed: {result.get('error')}")
                return "| Table | Data |\n|-------|------|\n| N/A | N/A |"
                
        except Exception as e:
            self.logger.error(f"Table description failed: {e}", exc_info=True)
            return "| Table | Data |\n|-------|------|\n| N/A | N/A |"

    async def describe_table_async(self, async_provider, image_bytes: bytes) -> str:
        """Async version of describe_table."""
        task = self.tasks.get('table_description')
        if not task:
            return "| Table | Data |\n|-------|------|\n| N/A | N/A |"
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            prompt = task['prompt']
            
            from src.llm.providers.universal_vlm_provider import TaskDefinition
            temp_task = TaskDefinition(
                task_name=task['task_name'],
                task_type=task.get('task_type', 'table_conversion'),
                prompt=prompt,
                output_format=task.get('output_format', {'type': 'text'})
            )
            
            result = await async_provider._process_single_page_wrapper_async(
                page_image=image,
                task_definition=temp_task,
                page_num=0
            )
            
            if result.get('success'):
                return result.get('content', '').strip()
            else:
                return "| Table | Data |\n|-------|------|\n| N/A | N/A |"
                
        except Exception as e:
            self.logger.error(f"Async table description failed: {e}")
            return "| Table | Data |\n|-------|------|\n| N/A | N/A |"
