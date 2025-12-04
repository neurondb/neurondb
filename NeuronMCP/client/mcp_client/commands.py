"""
Command parsing and execution.
"""

import json
from typing import Dict, Any, Tuple, Optional


def parse_command(command_str: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse a command string into tool name and arguments.

    Format: tool_name or tool_name:arg1=val1,arg2=val2

    Args:
        command_str: Command string

    Returns:
        Tuple of (tool_name, arguments_dict)

    Examples:
        >>> parse_command("list_tools")
        ('list_tools', {})
        >>> parse_command("vector_search:table=docs,vector_column=embedding,query_vector=[0.1,0.2,0.3]")
        ('vector_search', {'table': 'docs', 'vector_column': 'embedding', 'query_vector': [0.1, 0.2, 0.3]})
    """
    command_str = command_str.strip()
    if not command_str:
        raise ValueError("Empty command")

    # Check if command has arguments
    if ':' in command_str:
        tool_name, args_str = command_str.split(':', 1)
        tool_name = tool_name.strip()
        args = _parse_arguments(args_str)
    else:
        tool_name = command_str.strip()
        args = {}

    return tool_name, args


def _parse_arguments(args_str: str) -> Dict[str, Any]:
    """
    Parse argument string into dictionary.

    Format: arg1=val1,arg2=val2,arg3=[1,2,3]

    Args:
        args_str: Argument string

    Returns:
        Dictionary of arguments
    """
    args = {}
    if not args_str.strip():
        return args

    # Simple parser for key=value pairs
    # Handles: strings, numbers, booleans, arrays, objects
    i = 0
    current_key = None
    current_value = None
    in_string = False
    in_array = False
    in_object = False
    string_char = None
    bracket_depth = 0
    brace_depth = 0

    def add_arg():
        nonlocal current_key, current_value
        if current_key is not None:
            # Try to parse value
            if current_value is None:
                args[current_key] = None
            else:
                args[current_key] = _parse_value(current_value.strip())
            current_key = None
            current_value = None

    while i < len(args_str):
        char = args_str[i]

        if char in ('"', "'") and not in_array and not in_object:
            if not in_string:
                in_string = True
                string_char = char
                if current_value is None:
                    current_value = ""
            elif char == string_char:
                in_string = False
                string_char = None
            else:
                if current_value is None:
                    current_value = ""
                current_value += char
        elif in_string:
            if current_value is None:
                current_value = ""
            current_value += char
        elif char == '[':
            in_array = True
            bracket_depth += 1
            if current_value is None:
                current_value = ""
            current_value += char
        elif char == ']':
            bracket_depth -= 1
            if current_value is None:
                current_value = ""
            current_value += char
            if bracket_depth == 0:
                in_array = False
        elif char == '{':
            in_object = True
            brace_depth += 1
            if current_value is None:
                current_value = ""
            current_value += char
        elif char == '}':
            brace_depth -= 1
            if current_value is None:
                current_value = ""
            current_value += char
            if brace_depth == 0:
                in_object = False
        elif char == '=' and not in_array and not in_object:
            if current_key is None:
                current_key = current_value.strip() if current_value else ""
                current_value = None
        elif char == ',' and not in_array and not in_object:
            add_arg()
        else:
            if current_value is None:
                current_value = ""
            current_value += char

        i += 1

    # Add last argument
    add_arg()

    return args


def _parse_value(value_str: str) -> Any:
    """
    Parse a value string into Python object.

    Args:
        value_str: Value string

    Returns:
        Parsed value (str, int, float, bool, list, dict, None)
    """
    value_str = value_str.strip()

    # None
    if value_str.lower() in ('null', 'none'):
        return None

    # Boolean
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False

    # Try JSON parsing (for arrays, objects, numbers)
    try:
        return json.loads(value_str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try number parsing
    try:
        if '.' in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Remove quotes if present
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]

    # Return as string
    return value_str


def build_tool_call_request(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a tools/call request.

    Args:
        tool_name: Name of the tool
        arguments: Tool arguments

    Returns:
        Request parameters dictionary
    """
    return {
        "name": tool_name,
        "arguments": arguments,
    }

