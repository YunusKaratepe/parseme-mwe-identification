@echo off
REM Quick start script for PARSEME 2.0 MWE identification training

echo ========================================
echo PARSEME 2.0 MWE Identification Training
echo ========================================
echo.

REM Set paths
set DATA_DIR=2.0\subtask1\FR
set TRAIN_FILE=%DATA_DIR%\train.cupt
set DEV_FILE=%DATA_DIR%\dev.cupt
set OUTPUT_DIR=models\FR

echo Training on French data...
echo Train file: %TRAIN_FILE%
echo Dev file: %DEV_FILE%
echo Output directory: %OUTPUT_DIR%
echo.

REM Run training
python src\train.py --train %TRAIN_FILE% --dev %DEV_FILE% --output %OUTPUT_DIR% --epochs 3 --batch_size 8

echo.
echo Training completed!
echo Model saved to: %OUTPUT_DIR%
echo.

pause
