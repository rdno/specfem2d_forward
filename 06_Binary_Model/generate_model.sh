#!/bin/sh

# original NSTEP
NSTEP=4000

# Save model as binary
../utils/parfile -s MODEL default
../utils/parfile -s SAVE_MODEL binary

# You don't need to run the whole simulation to save the model
../utils/parfile -s NSTEP 2

echo "Running simulation to generate model files"
sh ./run.sh

# Get the number of processors
NPROC=`grep ^NPROC DATA/Par_file | cut -d = -f 2 | cut -d \# -f 1 | tr -d ' '`

# Checkerboard model
generate_model_command="../utils/generate_model.py --checkerboard-vp 4 4 3000 500 --checkerboard-vs 4 4 2000 250 DATA"

# Layered model
# generate_model_command="../utils/generate_model.py --layercake-vp ../extra/vp_vel --layercake-vs ../extra/vs_vel DATA"

# Smooothed layers
# generate_model_command="../utils/generate_model.py --smooth_layercake-vp ../extra/vp_vel 5000 --smooth_layercake-vs ../extra/vs_vel 5000 DATA"

# Image 1
# generate_model_command="../utils/generate_model.py --image-vp ../extra/guernica_square.jpg 7250 2250 --image-vs ../extra/guernica_square.jpg 4170 970 DATA"

# Image 2
# generate_model_command="../utils/generate_model.py --image-vp ../extra/fault1.jpg 7250 2250 --image-vs ../extra/fault1.jpg 4170 970 DATA"

# Image 3
# generate_model_command="../utils/generate_model.py --image-vp ../extra/fault2.jpg 7250 2250 --image-vs ../extra/fault2.jpg 4170 970 DATA"

# runs the generate_model script
echo "Generating Model"
if [ "$NPROC" -eq 1 ]; then
    # This is a serial simulation
    ${generate_model_command}
else
    # This is a MPI simulation
    mpirun -np ${NPROC} ${generate_model_command}
fi

# Use binary model
../utils/parfile -s MODEL binary
../utils/parfile -s SAVE_MODEL default

# Revert back the NSTEP
../utils/parfile -s NSTEP ${NSTEP}

echo "Running simulation using the binary model"
sh run.sh



