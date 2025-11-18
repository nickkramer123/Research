#!/bin/bash
# -*- coding: utf-8 -*-

#@author: tfvarley

main()
{
	iDir=${1}
	CMD_1='python EvoOpt_AllB_QUICK.py' # Runs script
	for idx in {0..1} # Submits 2 jobs, instances of EvoOpt
	do
		CMD_2="${CMD_1} ${idx}" # Command is: `python script.py i` - you can pass arguments using the sys package in Python.
		setupscript "${CMD_2}" $iDir
	done
}
setupscript()
{
	iCMD="${1}"
	echo "iCMD: $iCMD"
	iDir="${2}"
	echo "iDir: $iDir"
	arg1=$(echo ${iCMD} | awk '{print $3}')
	run_script=${PWD}/subscript_${arg1}.sh
cat > ${run_script} <<PBSHEADER
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-23:59:59
#SBATCH --mem=64gb
#SBATCH --job-name=job_${i}

PBSHEADER
 	echo "cd $iDir" >> $run_script # cd to the right directory
 	echo "source activate python3.7" >> $run_script
	echo ${iCMD} >> $run_script # put the command
	# for whatever reason... good to have newlines at end of script
	echo >> $run_script
    echo >> $run_script
}

########################################################################
# some commands yo
mkdir -p $(echo $HOME)/log/ # make a log file
# change the [cd $PWD] to change directory to where your code home dir
# is... so at the level of ./josh_examples/ 
cd ${PWD} || { echo "cannot cd there" ; exit 1 ; }
# RUN IT
main ${PWD}
mkdir submits/

rm submits/*

mv subscript* submits/
#now you'll have a bunch of scripts you'll need to qsub

for file in submits/*.sh 
do 
    sbatch $file
done

