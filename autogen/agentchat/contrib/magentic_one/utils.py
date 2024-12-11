from typing import Literal, Dict, Any
import json
import re


def clean_and_parse_json(content: str) -> Dict[str, Any]:
    """Clean and parse JSON content from various formats."""
    if not content or not isinstance(content, str):
        raise ValueError("Content must be a non-empty string")

    # Extract JSON from markdown code blocks if present
    if "```json" in content:
        parts = content.split("```json")
        if len(parts) > 1:
            content = parts[1].split("```")[0].strip()
    elif "```" in content:  # Handle cases where json block might not be explicitly marked
        parts = content.split("```")
        if len(parts) > 1:
            content = parts[1].strip()  # Take first code block content
    
    # Find JSON-like structure if not in code block
    if not content.strip().startswith('{'):
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            raise ValueError(
                f"Could not find valid JSON structure in content. "
                f"Content must contain a JSON object enclosed in curly braces. "
                f"Received: {content}"
            )
        content = json_match.group(0)

    # Preserve newlines for readability in error messages
    formatted_content = content
    
    # Now clean for parsing
    try:
        # First try parsing the cleaned but formatted content
        return json.loads(formatted_content)
    except json.JSONDecodeError:
        # If that fails, try more aggressive cleaning
        cleaned_content = re.sub(r'[\n\r\t]', ' ', content)  # Replace newlines/tabs with spaces
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)  # Normalize whitespace
        cleaned_content = re.sub(r'\\(?!["\\/bfnrt])', '', cleaned_content)  # Remove invalid escapes
        cleaned_content = re.sub(r',(\s*[}\]])', r'\1', cleaned_content)  # Remove trailing commas
        cleaned_content = re.sub(r'([{,]\s*)(\w+)(?=\s*:)', r'\1"\2"', cleaned_content)  # Quote unquoted keys
        cleaned_content = cleaned_content.replace("'", '"')  # Standardize quotes
        
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON after cleaning. Error: {str(e)} Original content: {formatted_content}")