"""
API keys configuration for external services.

Store your Gemini API key here. For security, consider using environment
variables instead of hardcoding keys.
"""

import os

# Gemini API key
# Get your API key from: https://makersuite.google.com/app/apikey
# Option 1: Set environment variable GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Option 2: Uncomment and set directly (not recommended for production)
# GEMINI_API_KEY = "your_gemini_api_key_here"

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please set it as an environment variable "
        "or update this file."
    )
