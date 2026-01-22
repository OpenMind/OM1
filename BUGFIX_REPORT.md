# Bug Fix Report

## Summary
This PR fixes critical bugs discovered during project testing and code quality checks.

## Issues Found and Fixed

### 1. **Critical: NameError in `unitree_g1_basic.py`** ✅ FIXED
- **File**: `src/inputs/plugins/unitree_g1_basic.py`
- **Issue**: `NameError: name 'dds_' is not defined` when Unitree SDK is not installed
- **Root Cause**: The `dds_` module was imported in a try-except block, but when the import failed, the code still tried to use `dds_.BmsState_` and `dds_.LowState_` in function type annotations (lines 139, 156)
- **Impact**: Tests failed during collection, preventing the test suite from running
- **Fix**: Created a placeholder `_DDSModule` class and assigned it to `dds_` when the import fails, ensuring the module is always defined
- **Test**: `pytest tests/inputs/base/test_inputs_plugins.py` now passes

### 2. **Warning: Invalid Escape Sequence** ✅ FIXED
- **File**: `src/ubtech/ubtechapi/YanAPI.py:5862`
- **Issue**: DeprecationWarning for invalid escape sequence `\.` in regex pattern
- **Root Cause**: String literal `'([0-9]{1,3}\.){3}[0-9]{1,3}'` contains unescaped backslash
- **Impact**: Deprecation warning that will become an error in future Python versions
- **Fix**: Changed to raw string: `r'([0-9]{1,3}\.){3}[0-9]{1,3}'`

## Testing Results

### Before Fixes
```
ERROR tests/inputs/base/test_inputs_plugins.py - NameError: name 'dds_' is not defined
ERROR tests/inputs/plugins/test_vlm_coco_local.py - ImportError (PyTorch/CUDA - environment issue)
ERROR tests/integration/test_case_runner.py - ImportError (PyTorch/CUDA - environment issue)
```

### After Fixes
```
✅ tests/inputs/base/test_inputs_plugins.py - All 35 tests pass
✅ Code quality checks (ruff, black, isort) - All pass
✅ Linter checks - No errors
```

## Code Quality Checks
- ✅ `ruff check` - All checks passed
- ✅ `black --check` - All files formatted correctly
- ✅ `isort --check-only` - All imports sorted correctly
- ✅ No linter errors found

## Runtime Testing
- ✅ Project starts successfully with `uv run src/run.py spot`
- ✅ No critical runtime errors (expected warnings for missing camera/hardware are normal)

## Files Modified
1. `src/inputs/plugins/unitree_g1_basic.py` - Fixed `dds_` NameError
2. `src/ubtech/ubtechapi/YanAPI.py` - Fixed invalid escape sequence

## Notes
- PyTorch/CUDA import errors in some tests are environment-specific and not code bugs
- These tests require GPU support or specific CUDA libraries that may not be available in all environments
- The fixes ensure the codebase is more robust and handles optional dependencies gracefully
