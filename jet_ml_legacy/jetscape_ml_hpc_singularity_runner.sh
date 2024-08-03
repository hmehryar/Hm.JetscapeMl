
--author: H. Mehryar
--email: hmehryar@wayne.edu

--clone jetscape directory in the HPC Grid
--This shell file must be in parent directory of jetscape
--the number of events must be set in the jetscape-user.xml file
srun -c 6 -q debug --mem=8G -t 4:0:0 --pty bash

module load singularity
singularity --version
singularity cache clean

--jet.sif file must be already be built in the parent directory
singularity shell jet.sif

cd Projects/110_JetscapeMl/Source/JETSPACE-ML-HAYDAR-MATTER/JETSCAPE-COMP-xx/

--rm build/ -rf
--mkdir build
cd build/
--cmake ..
--make -j6

./runJetscape
     