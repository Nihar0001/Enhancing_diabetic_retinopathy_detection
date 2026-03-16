# 📥 Quick Data Download Reference

## Google Drive Links

| Resource | Link |
|----------|------|
| 📁 **Data Folder** (images, CSV, .npy files) | https://drive.google.com/drive/folders/1lKMGO2NrZ67wH5LJkvFxRpF75I89CgZt?usp=drive_link |
| 🤖 **Pre-trained Models** (optional) | https://drive.google.com/drive/folders/1AlpsNZeaCD33i0ufbsP2LSnXKM2tX7Mt?usp=drive_link |

---

## Quick Steps

1. **Download** the Data Folder from Google Drive (link above)
2. **Extract** the downloaded `.zip` into the project's `data/` folder
3. **Verify** structure:
   ```
   data/
   ├── train_images/
   ├── test_images/
   ├── train.csv
   ├── test.csv
   ├── X_train.npy, X_test.npy
   ├── X_train_scaled.npy, X_test_scaled.npy
   ├── y_train.npy, y_test.npy, y_labels.npy
   └── X_features.npy
   ```
4. **Run**: `python train_models.py`

---

> 📖 For detailed instructions, see [GOOGLE_DRIVE_SETUP.md](GOOGLE_DRIVE_SETUP.md)
