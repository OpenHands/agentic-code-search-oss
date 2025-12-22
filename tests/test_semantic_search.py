"""
Tests for semantic search functionality.
"""

import tempfile
from pathlib import Path

import pytest

from src.tools.semantic_search import SemanticSearch, semantic_search


@pytest.fixture
def temp_repo():
    """Create a temporary repository with sample code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create sample Python files
        (repo_path / "utils.py").write_text(
            """
def calculate_precision(predicted, actual):
    '''Calculate precision metric.'''
    if not predicted:
        return 0.0
    correct = len(set(predicted) & set(actual))
    return correct / len(predicted)

def calculate_recall(predicted, actual):
    '''Calculate recall metric.'''
    if not actual:
        return 0.0
    correct = len(set(predicted) & set(actual))
    return correct / len(actual)
"""
        )

        (repo_path / "parser.py").write_text(
            """
import re

def parse_git_diff(diff_text):
    '''Parse a git diff and extract file paths.'''
    files = []
    for line in diff_text.split('\\n'):
        if line.startswith('diff --git'):
            match = re.search(r'b/(.+)$', line)
            if match:
                files.append(match.group(1))
    return files

def extract_hunks(diff_text):
    '''Extract hunk information from diff.'''
    hunks = []
    for line in diff_text.split('\\n'):
        if line.startswith('@@'):
            hunks.append(line)
    return hunks
"""
        )

        (repo_path / "main.py").write_text(
            """
from utils import calculate_precision, calculate_recall
from parser import parse_git_diff

def main():
    '''Main entry point.'''
    print('Hello world')
"""
        )

        yield repo_path


class TestSemanticSearch:
    """Test SemanticSearch class."""

    def test_initialization(self, temp_repo):
        """Test index initialization."""
        index = SemanticSearch(
            collection_name="test_index",
            persist_directory=str(temp_repo / ".test_index"),
        )
        assert index.collection_name == "test_index"
        assert index.collection is not None

    def test_index_code_files(self, temp_repo):
        """Test indexing code files."""
        index = SemanticSearch(
            persist_directory=str(temp_repo / ".test_index"),
        )

        stats = index.index_code_files(str(temp_repo))

        assert stats["indexed_files"] == 3
        assert stats["total_chunks"] > 0
        assert index.collection.count() > 0

    def test_search_precision_recall(self, temp_repo):
        """Test searching for precision/recall functions."""
        index = SemanticSearch(
            persist_directory=str(temp_repo / ".test_index"),
        )
        index.index_code_files(str(temp_repo))

        results = index.search("function that calculates precision and recall", n_results=5)

        assert len(results) > 0
        # Should find utils.py
        file_paths = [r["file_path"] for r in results]
        assert any("utils.py" in path for path in file_paths)

    def test_search_git_diff_parsing(self, temp_repo):
        """Test searching for git diff parsing."""
        index = SemanticSearch(
            persist_directory=str(temp_repo / ".test_index"),
        )
        index.index_code_files(str(temp_repo))

        results = index.search("code that parses git diffs", n_results=5)

        assert len(results) > 0
        # Should find parser.py
        file_paths = [r["file_path"] for r in results]
        assert any("parser.py" in path for path in file_paths)

    def test_get_unique_files(self, temp_repo):
        """Test extracting unique files from results."""
        index = SemanticSearch(
            persist_directory=str(temp_repo / ".test_index"),
        )
        index.index_code_files(str(temp_repo))

        results = index.search("function", n_results=10)
        unique_files = index.get_unique_files(results)

        assert len(unique_files) > 0
        # Should have no duplicates
        assert len(unique_files) == len(set(unique_files))

    def test_clear_index(self, temp_repo):
        """Test clearing the index."""
        index = SemanticSearch(
            persist_directory=str(temp_repo / ".test_index"),
        )
        index.index_code_files(str(temp_repo))

        assert index.collection.count() > 0

        index.clear_index()
        assert index.collection.count() == 0

    def test_get_stats(self, temp_repo):
        """Test getting index statistics."""
        index = SemanticSearch(
            persist_directory=str(temp_repo / ".test_index"),
        )
        index.index_code_files(str(temp_repo))

        stats = index.get_stats()

        assert "collection_name" in stats
        assert "total_documents" in stats
        assert "embedding_model" in stats
        assert stats["total_documents"] > 0


class TestSemanticSearchFunction:
    """Test semantic_search function."""

    def test_semantic_search_basic(self, temp_repo):
        """Test basic semantic search."""
        result = semantic_search(
            query="calculate precision and recall metrics",
            repo_path=str(temp_repo),
            n_results=5,
            rebuild_index=True,
        )

        assert isinstance(result, str)
        assert "Found" in result
        assert "utils.py" in result or "relevant" in result.lower()

    def test_semantic_search_with_existing_index(self, temp_repo):
        """Test search with existing index."""
        # First search builds index
        semantic_search(
            query="test query",
            repo_path=str(temp_repo),
            rebuild_index=True,
        )

        # Second search uses existing index
        result = semantic_search(
            query="precision recall",
            repo_path=str(temp_repo),
            rebuild_index=False,
        )

        assert isinstance(result, str)
        assert "Found" in result

    def test_semantic_search_no_results(self, temp_repo):
        """Test search with no matching results."""
        # Create index
        semantic_search(
            query="test",
            repo_path=str(temp_repo),
            rebuild_index=True,
        )

        # Search for something unlikely
        result = semantic_search(
            query="quantum computing blockchain cryptocurrency",
            repo_path=str(temp_repo),
            n_results=1,
        )

        # Should still return something (best match)
        assert isinstance(result, str)

    def test_semantic_search_invalid_path(self):
        """Test search with invalid repository path."""
        result = semantic_search(
            query="test",
            repo_path="/nonexistent/path",
            n_results=5,
        )

        assert "Error" in result or "does not exist" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])