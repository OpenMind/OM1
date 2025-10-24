# src/logger.py

"""
Centralized logging utility for OM1 project.
This is a placeholder for a future standardized logging system implementation.
"""

import logging

def get_logger(name="OM1"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
