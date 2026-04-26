@echo off
echo Starting Dead Internet Detective training in WSL2...
echo This will open a WSL2 terminal and begin training.
echo Make sure WSL2 + Ubuntu is installed first.
echo.
wsl -e bash -c "cd ~/dead-internet-detective && bash run_training_local.sh"
pause
