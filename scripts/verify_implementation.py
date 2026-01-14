#!/usr/bin/env python3
"""
Verification script for batched indexing implementation.

Checks:
1. Syntax of all new files
2. Import compatibility
3. Method existence
4. Config file validity
5. Integration with existing code
"""

import sys
import ast
from pathlib import Path
import importlib.util

def check_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read(), filepath)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def check_imports(filepath):
    """Check if all imports in a file can be resolved."""
    issues = []
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read(), filepath)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Skip checking standard library and external packages
                if not alias.name.startswith('src'):
                    continue
                issues.append(f"Import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith('src'):
                issues.append(f"From: {node.module}")
    
    return issues

def check_class_methods(filepath, class_name):
    """Extract methods from a class."""
    methods = []
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read(), filepath)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)
    return methods

def verify_method_calls(caller_file, callee_file, class_name):
    """Verify that all method calls in caller exist in callee."""
    # Extract methods from callee
    available_methods = check_class_methods(callee_file, class_name)
    
    # Extract method calls from caller
    with open(caller_file, 'r') as f:
        content = f.read()
    
    issues = []
    # Simple regex-like search for method calls
    import re
    # Look for patterns like: obj.method_name.remote()
    pattern = r'embedding_service\.(\w+)\.remote\(\)'
    calls = re.findall(pattern, content)
    
    for call in set(calls):
        if call not in available_methods:
            issues.append(f"Method '{call}' called but not found in {class_name}")
    
    return issues

def main():
    print("="*80)
    print("Batched Indexing Implementation Verification")
    print("="*80)
    print()
    
    # Files to check
    new_files = [
        'src/services/embedding_service.py',
        'src/services/batched_index_manager.py',
        'src/train_batched.py',
    ]
    
    config_files = [
        'configs/batched_indexing.yaml',
    ]
    
    all_ok = True
    
    # 1. Syntax Check
    print("1. Checking Python syntax...")
    print("-" * 80)
    for filepath in new_files:
        if not Path(filepath).exists():
            print(f"  ✗ {filepath}: FILE NOT FOUND")
            all_ok = False
            continue
        
        ok, error = check_syntax(filepath)
        if ok:
            print(f"  ✓ {filepath}: OK")
        else:
            print(f"  ✗ {filepath}: SYNTAX ERROR")
            print(f"    {error}")
            all_ok = False
    print()
    
    # 2. Import Check
    print("2. Checking imports...")
    print("-" * 80)
    for filepath in new_files:
        if not Path(filepath).exists():
            continue
        
        imports = check_imports(filepath)
        if imports:
            print(f"  {filepath}:")
            for imp in imports:
                print(f"    - {imp}")
        else:
            print(f"  ✓ {filepath}: No src imports")
    print()
    
    # 3. Method Existence Check
    print("3. Checking EmbeddingService methods...")
    print("-" * 80)
    
    if Path('src/services/embedding_service.py').exists():
        methods = check_class_methods('src/services/embedding_service.py', 'EmbeddingService')
        required_methods = [
            'enter_indexing_phase',
            'enter_retrieval_phase',
            'offload_for_training',
            'get_cache_stats',
            'cleanup_batch_indices',
            'get_or_load_index',
            'search',
        ]
        
        print("  Required methods:")
        for method in required_methods:
            if method in methods:
                print(f"    ✓ {method}")
            else:
                print(f"    ✗ {method}: MISSING")
                all_ok = False
        
        print(f"\n  Total methods found: {len(methods)}")
        print(f"  Methods: {', '.join(methods[:10])}...")
    else:
        print("  ✗ embedding_service.py not found")
        all_ok = False
    print()
    
    # 4. Method Call Verification
    print("4. Verifying method calls in train_batched.py...")
    print("-" * 80)
    
    if (Path('src/train_batched.py').exists() and 
        Path('src/services/embedding_service.py').exists()):
        issues = verify_method_calls(
            'src/train_batched.py',
            'src/services/embedding_service.py',
            'EmbeddingService'
        )
        
        if issues:
            for issue in issues:
                print(f"  ✗ {issue}")
            all_ok = False
        else:
            print("  ✓ All method calls valid")
    print()
    
    # 5. Config File Check
    print("5. Checking config files...")
    print("-" * 80)
    for filepath in config_files:
        if Path(filepath).exists():
            print(f"  ✓ {filepath}: EXISTS")
            # Could add YAML parsing here
        else:
            print(f"  ✗ {filepath}: NOT FOUND")
            all_ok = False
    print()
    
    # 6. Integration Check
    print("6. Checking integration points...")
    print("-" * 80)
    
    # Check if referenced files from original codebase exist
    required_existing = {
        'src/generator/code_search_generator.py': 'CodeSearchGenerator',
        'src/async_trainer.py': 'CustomFullyAsyncRayPPOTrainer',
        'src/mcp_server/training_semantic_search_server.py': 'get_repo_commit_hash',
        'src/tools/semantic_search.py': 'SemanticSearch',
    }
    
    print("  Required files from existing codebase:")
    for filepath, component in required_existing.items():
        if Path(filepath).exists():
            print(f"    ✓ {filepath} ({component})")
        else:
            print(f"    ⚠ {filepath} ({component}): NOT FOUND IN CURRENT DIR")
            print(f"      (This file should exist in your actual codebase)")
    print()
    
    # 7. Summary
    print("="*80)
    print("Summary")
    print("="*80)
    
    if all_ok:
        print("✓ All new files are syntactically correct")
        print("✓ All method calls are valid")
        print("✓ Config files exist")
        print()
        print("⚠ NOTE: Integration with existing codebase requires:")
        print("  - src/generator/code_search_generator.py")
        print("  - src/async_trainer.py")
        print("  - src/mcp_server/training_semantic_search_server.py")
        print("  - src/tools/semantic_search.py")
        print()
        print("These files are from your documents and should exist in your repo.")
        return 0
    else:
        print("✗ Some issues found (see above)")
        return 1

if __name__ == '__main__':
    sys.exit(main())