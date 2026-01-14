import logging
import re
from typing import Dict, List


def parse_simple_output(raw_output: str) -> List[Dict[str, str]]:
    """
    Parse agent output containing filename, optional class, and function.
    
    Format:
```
        path/to/file.py
        class: ClassName
        function: method_name
        
        path/to/other.py
        function: standalone_function
```
    
    Returns:
        List of dicts with keys: 'file', 'class', 'function'
    """
    # Extract content from triple backticks
    backtick_match = re.search(r'```(?:\w+)?\s*(.*?)\s*```', raw_output, re.DOTALL)
    if backtick_match:
        content = backtick_match.group(1)
    else:
        content = raw_output
    
    content = content.strip()
    if not content:
        return []
    
    locations = []
    current_file = None
    current_class = None
    file_added = False  # Track if current file has been added

    lines = content.split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            # Empty line - add pending file if needed, reset class context
            if current_file and not file_added:
                locations.append({
                    "file": current_file,
                    "class": None,
                    "function": None
                })
                file_added = True
            current_class = None
            continue

        # Check if this is a Python file path
        if line.endswith(".py") and not ":" in line:
            # Add previous file if not added
            if current_file and not file_added:
                locations.append({
                    "file": current_file,
                    "class": None,
                    "function": None
                })
            
            current_file = line
            current_class = None
            file_added = False
            continue

        # Parse class declaration (case-insensitive)
        class_match = re.match(r'^(?:class|Class):\s*(.+)$', line, re.IGNORECASE)
        if class_match and current_file:
            class_name = class_match.group(1).strip()
            current_class = class_name
            continue

        # Parse function/method declaration (case-insensitive)
        func_match = re.match(r'^(?:function|method|Function|Method):\s*(.+)$', line, re.IGNORECASE)
        if func_match and current_file:
            func_text = func_match.group(1).strip()
            
            # Remove parameters if present: "my_function(args)" -> "my_function"
            func_name = func_text.split("(")[0].strip()
            
            # Check if function includes class prefix: "MyClass.my_method"
            if "." in func_name:
                parts = func_name.rsplit(".", 1)
                class_name = parts[0].strip()
                method_name = parts[1].strip()

                locations.append({
                    "file": current_file,
                    "class": class_name,
                    "function": method_name
                })
            else:
                # Standalone function or method within current class context
                locations.append({
                    "file": current_file,
                    "class": current_class,
                    "function": func_name,
                })
            
            file_added = True

    # Add final pending file
    if current_file and not file_added:
        locations.append({
            "file": current_file,
            "class": None,
            "function": None
        })

    return locations


def get_simple_results_from_raw_outputs(raw_output: str) -> List[Dict[str, str]]:
    """
    Process raw output and return structured results.
    
    Returns:
        List of location dicts with keys: file, class, function
    """
    return parse_simple_output(raw_output)