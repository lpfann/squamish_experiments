THREADS=$1
MODELS=$2

#!/bin/bash
## Please refer to your grid documentation for available flags. This is only an example.
#$ -S /usr/bin/bash
#$ -P fair_share
#$ -pe multislot 48
#$ -cwd
#$ -N FSBench
#$ -m e
#$ -j y
#$ -o output.$JOB_ID
PROJECT=/prj/fribench

# Path to your executable. For example, if you extracted SCOOP to $HOME/downloads/scoop
cd $PROJECT/ordregpaper/versuche/pipeline/

# Add any addition to your environment variables like PATH. For example, if your local python installation is in $HOME/python
export PATH=$PROJECT/anaconda/bin:$PATH

# If, instead, you are using the python offered by the system, you can stipulate it's library path via PYTHONPATH
#export PYTHONPATH=$HOME/wanted/path/lib/python+version/site-packages/:$PYTHONPATH
# Or use VirtualEnv via virtualenvwrapper here:
source ../ordfri/bin/activate

# Launch the remotes workers
python experiment_pipeline.py --models $2 --filename $2 --threads $THREADS
