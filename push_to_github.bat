@echo off
cd /d "%~dp0"

echo ========================================
echo Cleaning up and starting fresh...
echo ========================================
rmdir /s /q .git 2>nul

echo Initializing new repository...
git init

echo Configuring git...
git config user.email "24f3004602@ds.study.iitm.ac.in"
git config user.name "24f3004602"

echo ========================================
echo Adding code files (excluding datasets)...
echo ========================================
git add .gitignore
git add feature_extractor.py
git add requirements.txt
git add slt_dataset.py
git add slt_model.py
git add train_slt.py
git add google_colab_setup.ipynb
git add push_to_github.bat
git add README.md 2>nul
git add pytorch-i3d/*.py 2>nul
git add pytorch-i3d/*.txt 2>nul
git add pytorch-i3d/*.md 2>nul

echo Committing code files...
git commit -m "Initial commit: Sign language translation training pipeline (code only)"

echo Setting branch to main...
git branch -M main

echo Adding remote...
git remote remove origin 2>nul
git remote add origin https://github.com/24f3004602/train_dataset.git

echo ========================================
echo Pushing to GitHub...
echo ========================================
git push -u origin main --force

echo.
echo ========================================
echo Done! Code pushed successfully.
echo Note: Large dataset files are excluded.
echo Upload them separately to Google Drive.
echo ========================================
echo Check: https://github.com/24f3004602/train_dataset
pause
