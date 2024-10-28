# echo all commands
set -x

# exit on error
set -e

bash export.sh
bash sharktank_dump.sh
bash iree_run.sh