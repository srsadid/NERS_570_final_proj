USER Documentation:

1. Installing Dependencies:

    1.1 MPI:
    Install MPI

    1.2 Petsc:
        Download and install petsc. Verifythat you have mpi inabled in your machine. 

        This is an example that you could use to configure the petsc build. 
        
        "./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --prefix=</home/sadid/Desktop/petsc/> petsc_install"

    1.3 Metis
        Download and install Metis

    1.4 Deal.ii
        Download and build deal.ii make sure you have MPI, Metis and Petsc enabled. 

        A bash script I used for our local HPC is attached here. 

        " #!/bin/bash

        #SBATCH --job-name=sadid_trial

        #SBATCH --ntasks=8

        #SBATCH --mem-per-cpu=5G
        #SBATCH --time=0-04:15:00
        #SBATCH --account=figueroc1
        #SBATCH --partition=standard
        #SBATCH --output=stdout
        #SBATCH --error=stderr

        module purge
        module load boost/1.78.0 openblas/0.3.23
        module load clang/2022.1.2 intel/2022.1.2 metis/5.1.0
        module load gcc/10.3.0
        export METIS_DIR=/sw/pkgs/coe/o/metis/intel-5.1.0
        module load openmpi/5.0.3

        cmake -DCMAKE_CXX_STANDARD=17 \
            -DCMAKE_CXX_STANDARD_REQUIRED=ON \
            -D DEAL_II_WITH_MPI=ON \
            -D MPI_C_COMPILER=mpicc \
            -D MPI_CXX_COMPILER=mpicxx \
            -D DEAL_II_WITH_PETSC=ON \
            -DPETSC_DIR=/home/sadid/Desktop/petsc/petsc_install \
            -D DEAL_II_WITH_METIS=ON \
            -D DEAL_II_WITH_TRILINOS=OFF \
            -DCMAKE_INSTALL_PREFIX=/home/sadid/Desktop/deal_ii/deal_ii_install \
            /home/sadid/Desktop/deal_ii/dealii-9.6.2

        make -j8 "

        NOTE: if you are building deal.ii locally do not use multiple processors. 

2. 
