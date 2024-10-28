# echo all commands
set -x

# exit on error
set -e

bash export.sh &> 01_export.log
bash sharktank_dump.sh &> 02_sharktank_dump.log
bash iree_run.sh &> 03_iree_run.log
python shortfin_run.py &> 04_shortfin_run.log