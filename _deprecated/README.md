# Deprecated Files

This folder contains files that are no longer needed or are corrupted.

## Files to Delete

- **Enhancing_diabetic_retinopathy_detection** — Corrupted folder name (leftover from initial project setup)

## How to Clean Up

After verifying locally and on GitHub that this project is working correctly, you can safely delete this entire `_deprecated/` folder.

```powershell
# Delete the deprecated folder
Remove-Item _deprecated -Recurse -Force

# Commit the cleanup
git add -A
git commit -m "Remove deprecated files"
git push origin main
```

**Status:** Safe to delete after final verification ✓
