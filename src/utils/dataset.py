import re


def extract_functions_from_patch(input_diff: str):
    """
    Parse a unified diff and extract, per file, the starting line of each hunk and the old line count.

    Returns: List[Tuple[str, List[int, int]]]
      Example: [("path/to/file.py", [start_line, old_count]), ...]
    """

    results: dict[str, list[list[int]]] = {}
    current_file: str | None = None
    in_hunk = False
    hunk_old_start = None
    hunk_old_count = None

    # Regex for hunk header: @@ -old_start,old_count +new_start,new_count @@ ...
    hunk_re = re.compile(r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@")

    def flush_hunk():
        nonlocal hunk_old_start, hunk_old_count, in_hunk
        if current_file is None or hunk_old_start is None:
            return
        count = hunk_old_count if hunk_old_count is not None else 1
        results.setdefault(current_file, []).append([hunk_old_start, hunk_old_start + count])
        # Reset hunk state
        in_hunk = False
        hunk_old_start = None
        hunk_old_count = None

    for raw_line in input_diff.strip().splitlines():
        line = raw_line.rstrip("\n")

        # Track current file being processed via the new file path header
        if line.startswith("+++ b/"):
            # Starting a new file ends any current hunk
            if in_hunk:
                flush_hunk()
            current_file = line[6:]
            continue

        # A new hunk header starts
        m = hunk_re.match(line)
        if m and current_file:
            # Flush any previous hunk before starting a new one
            if in_hunk:
                flush_hunk()
            in_hunk = True
            hunk_old_start = int(m.group("old_start"))
            old_count_str = m.group("old_count")
            hunk_old_count = int(old_count_str) if old_count_str is not None else 1
            continue

    # Flush any unterminated hunk at EOF
    if in_hunk:
        flush_hunk()

    targets = []
    for file, hunks in results.items():
        for hunk in hunks:
            targets.append(
                (file, hunk)
            )
    return targets
    # return results

