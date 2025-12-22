# Semantic Code Search

Search the current repository using natural language queries.

## Usage

Use the `semantic_search` tool to find code by meaning, not just keywords.

**Arguments:**
- `query` (required): Natural language description of what you're looking for
- `n_results` (optional): Number of results to return (default: 10)

**Note:** The tool automatically searches the current workspace repository. No need to specify a path.

## Examples
```json
{
  "name": "semantic_search",
  "arguments": {
    "query": "pandas to_datetime decimal division error",
    "n_results": 15
  }
}
```
```json
{
  "name": "semantic_search", 
  "arguments": {
    "query": "function that validates user input",
    "n_results": 5
  }
}
```

## When to Use

- Finding code that handles specific functionality
- Locating error handling for particular cases
- Discovering utility functions or helpers
- Understanding how a feature is implemented

The tool returns relevant code chunks with similarity scores and file paths.