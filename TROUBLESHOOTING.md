# Troubleshooting Pixi Conda-PyPI Mapping Error

## Problem
When running `pixi shell --environment=delftblue`, you get:
```
Error: × failed to map conda packages to their PyPI equivalents
├─▶ failed to fetch conda-pypi mapping from remote source
├─▶ File still doesn't exist
╰─▶ No such file or directory (os error 2)
```

## Root Cause
Pixi needs to download a conda-pypi mapping file when mixing conda packages (from conda-forge) with PyPI dependencies (editable local packages). The download or file access is failing.

**The mapping file URL is:** `https://raw.githubusercontent.com/regro/cf-graph-countyfair/master/mappings/pypi/name_mapping.json`

**Note:** The file can be downloaded manually (verified), so network connectivity is not the issue. The problem appears to be with how pixi/rattler handles the download or cache process.

## Solutions to Try

### Solution 1: Clear Cache and Retry
```bash
# Clear the conda-pypi mapping cache
rm -rf ~/.cache/rattler/cache/conda-pypi-mapping/*

# Try again
pixi shell --environment=delftblue
```

### Solution 2: Install Environment First
Sometimes installing the environment first works better than shell:
```bash
pixi install --environment=delftblue
pixi shell --environment=delftblue
```

### Solution 3: Use Verbose Mode to See Details
```bash
pixi shell --environment=delftblue -vvv
```
This will show more details about what file pixi is trying to access.

### Solution 4: Check Network/Proxy Settings
If you're on a restricted network:
```bash
# Check if you can access prefix.dev
curl -I https://prefix.dev

# If behind a proxy, set environment variables:
export HTTP_PROXY=your_proxy_url
export HTTPS_PROXY=your_proxy_url
```

### Solution 5: Try from a Compute Node
Login nodes sometimes have restricted network access. Try running pixi from a compute node with better network connectivity.

### Solution 6: Update Pixi
```bash
pixi self-update
```

### Solution 7: Check Rattler Cache Permissions
The issue might be related to cache directory permissions or structure:
```bash
# Ensure cache directories exist and are writable
mkdir -p ~/.cache/rattler/cache/conda-pypi-mapping/tmp
chmod 755 ~/.cache/rattler/cache/conda-pypi-mapping/tmp

# Check if there are any permission issues
ls -la ~/.cache/rattler/cache/conda-pypi-mapping/
```

### Solution 8: Try Different Cache Location
If the default cache location has issues, try setting a different cache directory:
```bash
# Use a different cache location (e.g., in /scratch if available)
export RATTLER_CACHE_DIR=/scratch/$USER/.rattler_cache
pixi shell --environment=delftblue
```

### Solution 9: Use Local Mapping File (✅ IMPLEMENTED)
Since the file can be downloaded manually, we've configured pixi to use a local mapping file:

1. **The mapping file has been downloaded** to `conda-pypi-mapping.json` in your project root
2. **Configuration added to pyproject.toml:**
   ```toml
   [tool.pixi.project]
   conda-pypi-map = { "conda-forge" = "./conda-pypi-mapping.json" }
   ```

**Try running pixi again - it should now use the local mapping file instead of trying to download it!**

**Note:** Commit `conda-pypi-mapping.json` to your repository so it's available on compute nodes without internet access.

### Solution 10: Report as Bug
This appears to be a bug in pixi/rattler where the mapping file download fails even though:
- Network connectivity is fine (file can be downloaded manually)
- Cache directories exist and are writable
- The error "File still doesn't exist" suggests a race condition or temporary file issue

Consider reporting this to: https://github.com/prefix-dev/pixi/issues

## Fixed Issues
- ✅ Fixed filename mismatch: `delftblue_activation.py` → `delftblue_pyurbanair_activation.py` in pyproject.toml
- ✅ Added local conda-pypi mapping file configuration to bypass download requirement

