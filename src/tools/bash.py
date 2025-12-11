import subprocess
from typing import Optional


def bash(command: str, cwd: Optional[str] = None) -> str:
    """
    Execute a bash command and return the results.
    Useful for finding files in the codebase and reading them.
    ALWAYS use this tool to find files irrespective of the task at hand.

    Args:
        command: The full bash command to execute
                 Example: "rg 'def main' -t py" or "ls -la" or "cat file.txt"
        cwd: Working directory to execute the command in (optional)
             If None, uses the current directory

    Returns:
        Command output as a string

    Example:
        >>> bash("rg 'def main' -t py")
        src/main.py:10:def main():
        src/utils.py:5:def main_helper():

        >>> bash("ls -la", cwd="/path/to/repo")
        >>> bash("rg --files -t py", cwd="./src")
        >>> bash("cat README.md")  # Read file contents

    Note:
        Common commands:
        - rg: ripgrep for searching code
        - ls: list directory contents
        - cat: read file contents
        - find: find files by name
        - grep: search in files
    """

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,  # 30 second timeout
            cwd=cwd,  # Set working directory
        )

        # Combine stdout and stderr
        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode == 0:
            # Command succeeded
            if not output:
                return "Command executed successfully (no output)."

            # Truncate to 200 lines and append message with remaining lines and total lines if more than 200
            if len(output) > 200:
                return f"Output truncated to 200 lines. Total lines: {len(output.splitlines())}\n\nOutput:\n{output}"

            return output
        else:
            # Command failed
            if error:
                return f"Error (exit code {result.returncode}):\n{error}"
            elif output:
                return f"Exit code {result.returncode}:\n{output}"
            else:
                return f"Command failed with exit code {result.returncode}"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    """Test the bash tool."""
    print("Testing bash tool")
    print("=" * 80)

    # Test 1: Ripgrep - Search for pattern in Python files
    print("\nTest 1: Search for 'def ' in Python files (ripgrep)")
    print("-" * 80)
    result = bash("rg '^def ' -t py --max-count 10 -n")
    print(result)

    # Test 2: Ripgrep - Case-insensitive search
    print("\n\nTest 2: Case-insensitive search for 'import' (ripgrep)")
    print("-" * 80)
    result = bash("rg import -i -t py --max-count 5 -n")
    print(result)

    # Test 3: Ripgrep - Get directory tree
    print("\n\nTest 3: Get directory tree structure (ripgrep)")
    print("-" * 80)
    result = bash("rg --files")
    print(result)

    # Test 4: ls command
    print("\n\nTest 4: List files with ls")
    print("-" * 80)
    result = bash("ls -la")
    print(result[:500])  # Truncate for readability

    # Test 5: cat command
    print("\n\nTest 5: Read file with cat")
    print("-" * 80)
    result = bash("cat ../oss_swe_grep.py | head -10")
    print(result)

    # Test 6: find command
    print("\n\nTest 6: Find Python files")
    print("-" * 80)
    result = bash("find . -name '*.py' -type f")
    print(result)

    # Test 7: grep command
    print("\n\nTest 7: Grep for 'def' in Python files")
    print("-" * 80)
    result = bash("grep -r 'def ' --include='*.py' . | head -5")
    print(result)

    # Test 8: Invalid command
    print("\n\nTest 8: Invalid command")
    print("-" * 80)
    result = bash("nonexistentcommand123")
    print(result)

    # Test 9: Execute in different directory (parent)
    print("\n\nTest 9: Execute ls in parent directory")
    print("-" * 80)
    result = bash("ls", cwd="..")
    print(result)

    # Test 10: Execute pwd to show current directory
    print("\n\nTest 10: Show current directory with pwd")
    print("-" * 80)
    result = bash("pwd")
    print(result)

    # Test 11: Execute pwd in parent directory
    print("\n\nTest 11: Show parent directory with pwd")
    print("-" * 80)
    result = bash("pwd", cwd="..")
    print(result)

    # Test 12: Search in specific directory
    print("\n\nTest 12: Search for 'def' in tools directory")
    print("-" * 80)
    result = bash("rg 'def ' -t py -n", cwd="./tools")
    print(result)

    print("\n" + "=" * 80)
    print("Tests completed!")
