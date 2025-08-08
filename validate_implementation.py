#!/usr/bin/env python3
"""
Validation script to check implementation completeness and basic functionality.
"""
import sys
from pathlib import Path
import ast
import importlib.util

def check_file_exists(file_path: str) -> bool:
    """Check if file exists."""
    return Path(file_path).exists()

def check_python_syntax(file_path: str) -> bool:
    """Check if Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return False

def check_imports(file_path: str) -> tuple:
    """Check what modules a file imports."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        
        return imports, None
    except Exception as e:
        return [], str(e)

def validate_implementation():
    """Validate the complete implementation."""
    print("🔍 Validating Football Highlights Generator Implementation\n")
    
    # Core files to check
    core_files = [
        "config.py",
        "audio_analysis.py", 
        "gemini_filter.py",
        "gen_highlights.py",
        "get_dataframe.py",
        "requirements.txt",
        ".python-version",
        ".gitignore"
    ]
    
    # Check file existence
    print("📁 Checking core files...")
    missing_files = []
    for file_path in core_files:
        if check_file_exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    # Check directory structure
    print("\n📂 Checking directory structure...")
    required_dirs = [
        "tools",
        "tests",
        "tests/assets", 
        ".cache",
        ".github/workflows"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ - MISSING")
            missing_dirs.append(dir_path)
    
    # Check Python syntax
    print("\n🐍 Checking Python syntax...")
    python_files = [f for f in core_files if f.endswith('.py')]
    python_files.extend([
        "tools/extract_audio.py",
        "tests/test_audio_pipeline.py"
    ])
    
    syntax_errors = []
    for file_path in python_files:
        if check_file_exists(file_path):
            if check_python_syntax(file_path):
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} - SYNTAX ERROR")
                syntax_errors.append(file_path)
        else:
            print(f"  ⏭️  {file_path} - SKIPPED (missing)")
    
    # Check imports and dependencies
    print("\n📦 Checking key imports...")
    import_checks = {
        "config.py": ["os", "pathlib"],
        "audio_analysis.py": ["numpy", "av", "librosa", "scipy"],
        "gemini_filter.py": ["asyncio", "sqlite3", "cv2", "numpy"],
        "gen_highlights.py": ["argparse", "asyncio", "pandas", "moviepy"]
    }
    
    import_issues = []
    for file_path, expected_imports in import_checks.items():
        if check_file_exists(file_path):
            imports, error = check_imports(file_path)
            if error:
                print(f"  ❌ {file_path} - ERROR: {error}")
                import_issues.append(file_path)
            else:
                missing_imports = []
                for expected in expected_imports:
                    # Check if any import contains the expected module
                    found = any(expected in imp for imp in imports if imp)
                    if not found:
                        missing_imports.append(expected)
                
                if missing_imports:
                    print(f"  ⚠️  {file_path} - Missing imports: {missing_imports}")
                else:
                    print(f"  ✅ {file_path}")
    
    # Check configuration completeness
    print("\n⚙️  Checking configuration...")
    try:
        import config
        required_configs = [
            'AUDIO_SR', 'PEAK_PROMINENCE', 'GEMINI_API_KEY', 
            'PRE_SEC', 'POST_SEC', 'CACHE_DIR'
        ]
        
        missing_configs = []
        for conf in required_configs:
            if hasattr(config, conf):
                print(f"  ✅ {conf}")
            else:
                print(f"  ❌ {conf} - MISSING")
                missing_configs.append(conf)
                
    except Exception as e:
        print(f"  ❌ Config import failed: {e}")
        missing_configs = required_configs
    
    # Summary
    print("\n" + "="*50)
    print("📊 VALIDATION SUMMARY")
    print("="*50)
    
    total_issues = len(missing_files) + len(missing_dirs) + len(syntax_errors) + len(import_issues)
    
    if total_issues == 0:
        print("🎉 All checks passed! Implementation is complete.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set GEMINI_API_KEY environment variable")
        print("3. Run tests: pytest tests/ -v")
        print("4. Test with a sample video")
        return True
    else:
        print(f"⚠️  Found {total_issues} issues:")
        
        if missing_files:
            print(f"\n📁 Missing files ({len(missing_files)}):")
            for f in missing_files:
                print(f"   - {f}")
        
        if missing_dirs:
            print(f"\n📂 Missing directories ({len(missing_dirs)}):")
            for d in missing_dirs:
                print(f"   - {d}/")
        
        if syntax_errors:
            print(f"\n🐍 Syntax errors ({len(syntax_errors)}):")
            for f in syntax_errors:
                print(f"   - {f}")
        
        if import_issues:
            print(f"\n📦 Import issues ({len(import_issues)}):")
            for f in import_issues:
                print(f"   - {f}")
        
        print("\n🔧 Please fix these issues before proceeding.")
        return False

if __name__ == "__main__":
    success = validate_implementation()
    sys.exit(0 if success else 1)