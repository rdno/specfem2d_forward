#!/bin/sh


SPECFEM2D_BIN=${SPECFEM2D_BIN:-"../specfem2d/bin"}

rm -rf OUTPUT_FILES
mkdir OUTPUT_FILES

echo "Running mesher..."
${SPECFEM2D_BIN}/xmeshfem2D > OUTPUT_FILES/output_mesher.txt

# Get the number of processors
NPROC=`grep ^NPROC DATA/Par_file | cut -d = -f 2 | cut -d \# -f 1 | tr -d ' '`

echo "Running solver..."
# runs simulation
if [ "$NPROC" -eq 1 ]; then
    # This is a serial simulation
    ${SPECFEM2D_BIN}/xspecfem2D > OUTPUT_FILES/output_solver.txt
else
    # This is a MPI simulation
    mpirun -np ${NPROC} ${SPECFEM2D_BIN}/xspecfem2D > OUTPUT_FILES/output_solver.txt
fi



