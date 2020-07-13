#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

array=( $@ )
len=${#array[@]}
ARGS=${array[@]:0:$len}
#ARGS_SLUG=${ARGS// /_}
#ARGS_SLUG=${ARGS_SLUG//\//|}
ARGS_SLUG=${ARGS//\//_}

is_next=false
for var in "$@"
do
	if ${is_next}
	then
		EXP_DIR=${var}
		break
	fi
	if [ ${var} == "OUTPUT_DIR" ]
	then
		is_next=true
	fi
done

mkdir -p "${EXP_DIR}"
mkdir -p "${EXP_DIR}/../_logs"
BASENAME=`basename ${EXP_DIR}`
LOG="${EXP_DIR}/../_logs/${BASENAME} ${0##*/} ${ARGS_SLUG} `date +'%Y-%m-%d_%H-%M-%S'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


echo ---------------------------------------------------------------------
git log -1
git submodule foreach 'git log -1'
echo ---------------------------------------------------------------------

<<<<<<< HEAD
#export LD_LIBRARY_PATH=~/Documents/caffe2/release/lib:${LD_LIBRARY_PATH}
#export PYTHONPATH=~/Documents/caffe2/release:${PYTHONPATH}
#export PYTHONPATH=~/Documents/pytorch/build:${PYTHONPATH}
python3 tools/train_net.py --multi-gpu-testing ${ARGS}
=======
python tools/train_net.py --multi-gpu-testing ${ARGS}
>>>>>>> ecf649754cb7431cdd0f6855abdb75fd36eb88ed
