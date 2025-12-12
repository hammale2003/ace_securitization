"""
Utility functions for the ACE Securitization System.

Includes logging, text processing, and helper functions.
"""
import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from functools import wraps
import time


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Set up logging for the ACE system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        log_format: Log message format
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("ace_securitization")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logging()


# =============================================================================
# TEXT PROCESSING
# =============================================================================

def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain other content.
    
    Handles common cases:
    - JSON in markdown code blocks
    - JSON with leading/trailing text
    - Malformed JSON with common errors
    """
    import re
    
    text = text.strip()
    
    # Try to extract from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        text = json_match.group(1)
    
    # Find JSON object boundaries
    json_start = text.find('{')
    json_end = text.rfind('}')
    
    if json_start == -1 or json_end == -1:
        return None
    
    json_str = text[json_start:json_end + 1]
    
    # Try parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try fixing common errors
        fixed = fix_json_string(json_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def fix_json_string(json_str: str) -> str:
    """Attempt to fix common JSON formatting errors."""
    import re
    
    # Remove trailing commas before closing brackets
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # Fix single quotes (simple cases only)
    # This is risky and may break valid content, use with caution
    
    # Ensure proper escaping of quotes within strings
    # This is a heuristic and may not work in all cases
    
    return json_str


def compute_text_hash(text: str) -> str:
    """Compute a hash of the text for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    import re
    return re.sub(r'\s+', ' ', text).strip()


# =============================================================================
# TIMING AND PERFORMANCE
# =============================================================================

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.2f}s")
        return result
    return wrapper


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        logger.debug(f"{self.name} completed in {self.elapsed:.2f}s")
    
    @property
    def elapsed(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def ensure_directory(path: str) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to a JSON file."""
    ensure_directory(str(Path(path).parent))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Optional[Any]:
    """Load data from a JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load JSON from {path}: {e}")
        return None


def backup_file(path: str, backup_dir: str = "backups") -> Optional[str]:
    """Create a backup of a file."""
    import shutil
    
    source = Path(path)
    if not source.exists():
        return None
    
    backup_path = ensure_directory(backup_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{source.stem}_{timestamp}{source.suffix}"
    backup_file_path = backup_path / backup_name
    
    shutil.copy2(source, backup_file_path)
    logger.info(f"Created backup: {backup_file_path}")
    
    return str(backup_file_path)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_playbook_operation(operation: Dict[str, Any]) -> bool:
    """Validate a playbook operation dictionary."""
    required_fields = ["type", "section", "content"]
    valid_types = ["ADD"]
    valid_sections = ["strategies", "pitfalls", "templates", "definitions", "code_snippets"]
    
    # Check required fields
    for field in required_fields:
        if field not in operation:
            logger.warning(f"Operation missing required field: {field}")
            return False
    
    # Validate type
    if operation["type"].upper() not in valid_types:
        logger.warning(f"Invalid operation type: {operation['type']}")
        return False
    
    # Validate section
    if operation["section"] not in valid_sections:
        logger.warning(f"Invalid section: {operation['section']}")
        return False
    
    # Validate content
    if not operation["content"] or not operation["content"].strip():
        logger.warning("Operation has empty content")
        return False
    
    return True


def validate_bullet_tag(tag: Dict[str, str]) -> bool:
    """Validate a bullet tag dictionary."""
    if "id" not in tag:
        return False
    
    valid_tags = ["helpful", "harmful", "neutral"]
    tag_value = tag.get("tag", "").lower()
    
    return tag_value in valid_tags


# =============================================================================
# SECURITIZATION DOMAIN HELPERS
# =============================================================================

SECURITIZATION_TERMS = {
    "SPV": "Special Purpose Vehicle - A bankruptcy-remote entity created to hold assets",
    "ABS": "Asset-Backed Securities - Securities backed by a pool of assets",
    "MBS": "Mortgage-Backed Securities - ABS backed by mortgage loans",
    "CLO": "Collateralized Loan Obligation - ABS backed by a pool of loans",
    "CDO": "Collateralized Debt Obligation - ABS backed by various debt instruments",
    "Waterfall": "The priority structure for distributing cash flows to different tranches",
    "Tranche": "A slice or portion of a securitization with specific risk/return characteristics",
    "True Sale": "Transfer of assets that is legally characterized as a sale, not a loan",
    "Credit Enhancement": "Mechanisms to reduce credit risk (overcollateralization, subordination, etc.)",
    "Servicer": "Entity responsible for collecting payments and managing the assets",
    "Trustee": "Independent party that holds assets for benefit of investors",
    "Overcollateralization": "Having more assets than required to secure the obligations",
    "Subordination": "Credit enhancement where junior tranches absorb losses first",
    "Reserve Account": "Cash reserve to cover shortfalls in payments",
}


def get_term_definition(term: str) -> Optional[str]:
    """Get the definition of a securitization term."""
    return SECURITIZATION_TERMS.get(term.upper())


def identify_securitization_terms(text: str) -> List[str]:
    """Identify securitization terms mentioned in text."""
    found_terms = []
    text_upper = text.upper()
    
    for term in SECURITIZATION_TERMS:
        if term in text_upper:
            found_terms.append(term)
    
    return found_terms


# =============================================================================
# EXPORT HELPERS
# =============================================================================

def export_playbook_to_markdown(playbook_dict: Dict[str, Any], output_path: str) -> str:
    """Export playbook to a Markdown file."""
    lines = ["# ACE Securitization Playbook\n"]
    lines.append(f"*Generated: {datetime.now().isoformat()}*\n\n")
    
    sections = ["strategies", "pitfalls", "templates", "definitions", "code_snippets"]
    
    for section in sections:
        bullets = playbook_dict.get(section, [])
        if bullets:
            lines.append(f"## {section.replace('_', ' ').title()}\n")
            for bullet in bullets:
                lines.append(f"### [{bullet['id']}]")
                lines.append(f"*Helpful: {bullet.get('helpful_count', 0)} | "
                           f"Harmful: {bullet.get('harmful_count', 0)}*\n")
                lines.append(f"{bullet['content']}\n")
            lines.append("\n")
    
    content = "\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return content


def export_history_to_json(history: List[Dict[str, Any]], output_path: str) -> None:
    """Export question history to JSON."""
    save_json({
        "exported_at": datetime.now().isoformat(),
        "total_questions": len(history),
        "history": history
    }, output_path)
