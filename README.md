# NERS 570 Final Project: Incompressible Navier-Stokes Solver

> A parallel, PETSc + MPI-powered C++ solver for 2D incompressible flow on quadrilateral meshes (Deal.II backend).

---

## 📋 Table of Contents

1. [Features](#-features)  
2. [Dependencies](#-dependencies)  
3. [Installation](#-installation)  
4. [Building the Project](#-building-the-project)  
5. [Running Unit Tests](#-running-unit-tests)  
6. [Usage](#-usage)  
7. [Mesh File Format (`.inp`)](#-mesh-file-format-inp)  
8. [Parameter File (`.prm`)](#-parameter-file-prm)  
9. [Contributing](#-contributing)  
10. [License](#-license)  

---

## 🔧 Features

- Serial ➔ parallel implementation via PETSc + MPI  
- Rotational vs. standard formulations  
- GMRES with ILU preconditioning  
- Configurable time-stepping & output intervals  
- Deal.II mesh reader (`GridIn::read_inp`)  

---

## 🛠 Dependencies

1. **MPI** (e.g. OpenMPI or MPICH)  
2. **PETSc** (built _with_ MPI enabled)  
3. **METIS** (for partitioning)  
4. **Deal.II ≥ 9.6** (with MPI, PETSc & METIS enabled)

> On many HPCs you can load modules; locally you may install with your package manager or from source.

---


### 1.1 MPI (Message Passing Interface)

* **Requirement:** A working MPI implementation (e.g., Open MPI, MPICH) is required for parallel execution.
* **Installation:** Install MPI using your system's package manager (e.g., `sudo apt install openmpi-bin libopenmpi-dev` on Debian/Ubuntu, `sudo dnf install openmpi openmpi-devel` on Fedora) or by downloading and compiling from source. Verify your installation (e.g., `mpicc --version`).

### 1.2 PETSc (Portable, Extensible Toolkit for Scientific Computation)

* **Requirement:** PETSc is used by deal.II for linear algebra operations, especially solvers. It must be compiled with MPI support.
* **Download:** Get PETSc from the [official PETSc website](https://petsc.org/download/).
* **Configuration & Installation:**
    * Navigate to the downloaded PETSc directory.
    * Configure PETSc. Make sure to enable MPI and specify an installation prefix. **Crucially, ensure the MPI compilers (`mpicc`, `mpicxx`, `mpif90`) are found correctly.** Adapt the prefix path for your system.
        ```bash
        ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran \
                    --with-mpi-dir=/path/to/your/mpi/installation \ # Optional: Helps PETSc find MPI if needed
                    --download-fblaslapack # Recommended: Let PETSc download its own BLAS/LAPACK
                    --prefix=/path/to/your/desired/petsc_install
        ```
        *Replace `/path/to/your/desired/petsc_install` with your chosen installation location.*
    * Follow the instructions provided by `./configure` to build and install PETSc (usually involves running `make all check` and then `make install`).
    * **Note your PETSc installation path (`/path/to/your/desired/petsc_install`)**, as you will need it for deal.II.

### 1.3 Metis (Graph Partitioning Library)

* **Requirement:** Metis is used by deal.II for mesh partitioning in parallel computations.
* **Installation:**
    * **HPC/Module Systems:** Often, Metis is available as a module. Check with `module avail metis`. If found, load it (e.g., `module load metis/5.1.0`). Note the installation path provided by the module system (often needed via an environment variable like `METIS_DIR`).
    * **Manual Installation:** Download Metis from its [official source](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) and compile it according to its instructions. Note the installation path.

### 1.4 deal.II (Finite Element Library)

* **Requirement:** deal.II is the core finite element library. It needs to be configured and built with support for MPI, PETSc, and Metis.
* **Download:** Get deal.II from the [official deal.II website](https://www.dealii.org/).
* **Configuration & Installation:**
    * Create a separate build directory (recommended):
        ```bash
        mkdir dealii_build
        cd dealii_build
        ```
    * Configure deal.II using CMake, pointing it to your dependencies. The example script provided in the user guide draft is a good starting point for an HPC environment using modules. Adapt paths as necessary for your system.
        ```cmake
        # Example CMake configuration (adapt paths and module loads)

        # --- Start: Example HPC Module Loading (Adapt or remove for local install) ---
        # module purge
        # module load boost/1.78.0 openblas/0.3.23 # Example modules
        # module load clang/2022.1.2 intel/2022.1.2 # Example modules
        # module load gcc/10.3.0                   # Example modules
        # module load metis/5.1.0                  # Example Metis module
        # module load openmpi/5.0.3                # Example MPI module
        #
        # Set METIS_DIR if needed (often set by module)
        # export METIS_DIR=/path/to/your/metis_install
        # --- End: Example HPC Module Loading ---

        cmake \
          -DCMAKE_INSTALL_PREFIX=/path/to/your/desired/dealii_install \
          -DCMAKE_CXX_STANDARD=17 \
          -DCMAKE_CXX_STANDARD_REQUIRED=ON \
          -D DEAL_II_WITH_MPI=ON \
          -D MPI_C_COMPILER=mpicc \
          -D MPI_CXX_COMPILER=mpicxx \
          -D DEAL_II_WITH_PETSC=ON \
          -DPETSC_DIR=/path/to/your/desired/petsc_install \
          -D DEAL_II_WITH_METIS=ON \
          -DMETIS_DIR=/path/to/your/metis_install \ # Or let deal.II find it via module path
          -D DEAL_II_WITH_TRILINOS=OFF \
          /path/to/downloaded/dealii-source
        ```
        *Replace placeholders like `/path/to/your/desired/dealii_install`, `/path/to/your/desired/petsc_install`, `/path/to/your/metis_install`, and `/path/to/downloaded/dealii-source` with your actual paths.*
    * Build deal.II:
        ```bash
        make -jN # Replace N with the number of cores you want to use for compilation
        ```
        * **NOTE:** If building locally on a machine with limited memory, using multiple processors (`-jN`) might cause issues. Start with `-j1` or `-j2` if you encounter problems.
    * Install deal.II:
        ```bash
        make install
        ```
    * **Note your deal.II installation path (`/path/to/your/desired/dealii_install`)**.

---

## 2. Building the Project Code

Once all dependencies are installed:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/srsadid/NERS_570_final_proj.git](https://github.com/srsadid/NERS_570_final_proj.git)
    cd NERS_570_final_proj
    ```
2.  **Create Build Directory:**
    ```bash
    mkdir build
    cd build
    ```
3.  **Configure using CMake:**
    * **Load Modules (HPC Only):** If you are on an HPC system using modules, load the *same* modules you used to compile deal.II before running CMake for the project.
        ```bash
        # Example (adapt to your specific modules):
        # module purge
        # module load boost/1.78.0 openblas/0.3.23 clang/2022.1.2 intel/2022.1.2
        # module load gcc/10.3.0 metis/5.1.0 openmpi/5.0.3
        # export METIS_DIR=/path/to/your/metis_install # If needed
        ```
    * **Run CMake:** Point CMake to your deal.II installation. Choose your build type (`Release` for optimized code, `Debug` for easier debugging).
        ```bash
        # For optimized code (recommended for production runs)
        cmake -Ddeal.II_DIR=/path/to/your/desired/dealii_install/lib/cmake/deal.II \
              -DCMAKE_BUILD_TYPE=Release ..

        # OR For debugging
        # cmake -Ddeal.II_DIR=/path/to/your/desired/dealii_install/lib/cmake/deal.II \
        #       -DCMAKE_BUILD_TYPE=Debug ..

        # Optionally, specify project installation prefix:
        # cmake -Ddeal.II_DIR=... -DCMAKE_BUILD_TYPE=... -DCMAKE_INSTALL_PREFIX=/your/project/install/path ..
        ```
        *Replace `/path/to/your/desired/dealii_install` with your actual deal.II install path.*
4.  **Compile the Project:**
    ```bash
    make -jN # Use desired number of cores
    ```
5.  **Run Unit Tests (Optional but Recommended):**
    After successful compilation, run the included tests to verify basic functionality.
    ```bash
    ./run_unit_tests
    ```
    If all tests pass, the build is likely correct.

---

## 3. Running a Simulation

To run a simulation, you need:

1.  The compiled executable (e.g., `navier_stokes_projection` inside the `build` directory).
2.  A mesh file (in `.inp` format).
3.  A parameter file (in `.prm` format).

### Command Syntax

The project is run using `mpirun` for parallel execution:

```bash
mpirun -np <N_PROCS> /path/to/executable/navier_stokes_projection /path/to/parameter-file.prm
```

