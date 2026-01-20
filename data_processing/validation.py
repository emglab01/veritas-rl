"""
Input validation utilities for data processing scripts.
"""

import sys
from pathlib import Path
from typing import Optional


def validate_file_path(path: Path, must_exist: bool = True, extensions: Optional[tuple] = None) -> None:
    """
    Validate a file path.
    
    Args:
        path: Path to validate
        must_exist: If True, file must exist
        extensions: Optional tuple of allowed extensions (e.g., (".json", ".jsonl"))
    
    Raises:
        FileNotFoundError: If file doesn't exist and must_exist=True
        ValueError: If path is invalid
    """
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    if path.exists() and not path.is_file():
        raise ValueError(f"Path exists but is not a file: {path}")
    
    if extensions and path.suffix.lower() not in extensions:
        raise ValueError(
            f"File must have one of these extensions: {extensions}, "
            f"got: {path.suffix}"
        )


def validate_directory_path(path: Path, must_exist: bool = False, create: bool = True) -> None:
    """
    Validate a directory path.
    
    Args:
        path: Path to validate
        must_exist: If True, directory must exist
        create: If True, create directory if it doesn't exist
    
    Raises:
        FileNotFoundError: If directory doesn't exist and must_exist=True
        ValueError: If path is invalid or cannot be created
    """
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {path}")
    elif must_exist:
        raise FileNotFoundError(f"Directory does not exist: {path}")
    elif create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create directory {path}: {e}") from e


def validate_positive_int(value: int, name: str = "value") -> None:
    """Validate that an integer is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative_int(value: int, name: str = "value") -> None:
    """Validate that an integer is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_ratio(value: float, name: str = "ratio", min_val: float = 0.0, max_val: float = 1.0) -> None:
    """Validate that a float is within a range."""
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )

