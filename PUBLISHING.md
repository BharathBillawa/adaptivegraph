# Publishing AdaptiveGraph to PyPI

**Complete Step-by-Step Guide**

## Prerequisites Checklist

Before publishing, ensure:
- ✅ All tests pass (`pytest tests/`)
- ✅ Code is formatted (`black src/` , `isort src/`)
- ✅ README.md is comprehensive
- ✅ LICENSE file exists
- ✅ Version number is correct in `pyproject.toml`
- ✅ CHANGELOG.md is updated
- ✅ All changes are committed to git

---

## Step 1: Install Build Tools

```bash
# Install build and publishing tools
pip install --upgrade build twine
```

**What this does:** Installs tools needed to build and upload your package.

---

## Step 2: Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf dist/ build/ src/*.egg-info

# Verify clean slate
ls dist/  # Should not exist or be empty
```

**What this does:** Ensures no old artifacts interfere with the new build.

---

## Step 3: Build the Package

```bash
# Build both source distribution and wheel
python -m build
```

**What this does:** Creates two files in `dist/`:
- `adaptivegraph-0.1.0.tar.gz` (source distribution)
- `adaptivegraph-0.1.0-py3-none-any.whl` (wheel)

**Verify the build:**
```bash
ls -lh dist/
# You should see both .tar.gz and .whl files
```

---

## Step 4: Inspect the Package (Optional but Recommended)

```bash
# Check what's inside the wheel
unzip -l dist/adaptivegraph-0.1.0-py3-none-any.whl

# Or check the source distribution
tar -tzf dist/adaptivegraph-0.1.0.tar.gz
```

**What to verify:**
- ✅ All source files are included
- ✅ README, LICENSE are present
- ✅ No unwanted files (e.g., `.pyc`, `__pycache__`)

---

## Step 5: Test the Package Locally

```bash
# Create a test virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from local wheel
pip install dist/adaptivegraph-0.1.0-py3-none-any.whl

# Test import
python -c "from adaptivegraph import LearnableEdge; print('Success!')"

# Deactivate and clean up
deactivate
rm -rf test_env
```

**What this does:** Ensures the package installs correctly before uploading.

---

## Step 6: Register on PyPI

### 6.1 Create PyPI Account
1. Go to https://pypi.org/account/register/
2. Create an account and verify your email

### 6.2 Create TestPyPI Account (for testing)
1. Go to https://test.pypi.org/account/register/
2. Create a separate account (yes, you need both!)

---

## Step 7: Create API Tokens

### 7.1 TestPyPI Token (for testing)
1. Log in to https://test.pypi.org
2. Go to Account Settings → API Tokens
3. Click "Add API token"
4. Name: `adaptivegraph-test`
5. Scope: "Entire account" (for first upload) or "Project: adaptivegraph" (after first upload)
6. **SAVE THE TOKEN** - you only see it once!

### 7.2 PyPI Token (for production)
1. Log in to https://pypi.org
2. Go to Account Settings → API Tokens
3. Click "Add API token"
4. Name: `adaptivegraph-prod`
5. Scope: "Entire account" (for first upload)
6. **SAVE THE TOKEN** - you only see it once!

### 7.3 Configure `.pypirc` (Recommended)

Create `~/.pypirc`:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-ACTUAL-PRODUCTION-TOKEN-HERE

[testpypi]
username = __token__
password = pypi-YOUR-ACTUAL-TEST-TOKEN-HERE
```

**Security:** Make sure this file has restricted permissions:
```bash
chmod 600 ~/.pypirc
```

---

## Step 8: Upload to TestPyPI (Practice Run)

```bash
# Upload to test server
python -m twine upload --repository testpypi dist/*
```

**What happens:**
- Uploads your package to https://test.pypi.org
- You'll be prompted for username (`__token__`) and password (your token) if not in `.pypirc`

**Verify the upload:**
1. Visit https://test.pypi.org/project/adaptivegraph/
2. Check that the page looks correct
3. Test installation:

```bash
# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ adaptivegraph

# Test import
python -c "from adaptivegraph import LearnableEdge; print('TestPyPI Success!')"
```

**Note:** The `--extra-index-url` is needed because TestPyPI doesn't have all dependencies (like numpy).

---

## Step 9: Upload to PyPI (Production)

**⚠️ WARNING: This step cannot be undone! You cannot re-upload the same version.**

```bash
# Final check
ls dist/

# Upload to production PyPI
python -m twine upload dist/*
```

**What happens:**
- Uploads to https://pypi.org
- Package immediately becomes publicly available
- You cannot delete or modify this version (only yank it, which hides it but keeps it)

---

## Step 10: Verify Production Installation

```bash
# Wait 1-2 minutes for PyPI to index

# Install from PyPI
pip install adaptivegraph

# Test import
python -c "from adaptivegraph import LearnableEdge; print('PyPI Success!')"

# Test optional dependencies
pip install adaptivegraph[all]
```

**Verify on PyPI:**
1. Visit https://pypi.org/project/adaptivegraph/
2. Check that all metadata is correct
3. Verify README renders correctly
4. Check that badges work

---

## Step 11: Tag the Release in Git

```bash
# Create and push git tag
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0

# Create GitHub release (optional but recommended)
# Go to https://github.com/BharathBillawa/adaptivegraph/releases
# Click "Create new release" and describe the changes
```

---

## Step 12: Post-Publication Checklist

- ✅ Package appears on https://pypi.org/project/adaptivegraph/
- ✅ `pip install adaptivegraph` works
- ✅ README renders correctly on PyPI
- ✅ Badges display correctly
- ✅ GitHub release created
- ✅ Updated CHANGELOG.md for next version
- ✅ Bumped version in `pyproject.toml` for development (e.g., `0.1.1-dev`)

---

## Troubleshooting

### Error: "File already exists"
- You cannot re-upload the same version
- Solution: Bump version in `pyproject.toml`, rebuild, and upload

### Error: "Invalid package name"
- Check that package name in `pyproject.toml` matches PyPI expectations
- Package names are case-insensitive and convert underscores to hyphens

### Error: "Missing README or LICENSE"
- Ensure files are included in `MANIFEST.in`
- Rebuild and check contents: `tar -tzf dist/*.tar.gz`

### README not rendering on PyPI
- Ensure `readme = "README.md"` in `pyproject.toml`
- Check markdown syntax is compatible with PyPI's renderer

---

## Future Releases

For subsequent releases (e.g., `0.2.0`):

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit changes
4. Run steps 2-11

Consider using CI/CD to automate this process!

---

## Best Practices

1. **Never upload sensitive data** - review files before building
2. **Test on TestPyPI first** - catch issues before production
3. **Use semantic versioning** - MAJOR.MINOR.PATCH
4. **Keep CHANGELOG** - users appreciate knowing what changed
5. **Tag releases in git** - makes it easy to track versions
6. **Use API tokens** - more secure than passwords

---

## Quick Reference

```bash
# Full publish workflow (after initial setup)
rm -rf dist/
python -m build
python -m twine upload --repository testpypi dist/*  # Test first
python -m twine upload dist/*  # Production
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

---

**Questions?** Check:
- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
