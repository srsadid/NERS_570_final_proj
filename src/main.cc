/**
 * @file main.cc
 * @brief Main entry point for the 2D Navier-Stokes projection method simulation.
 *
 * This program solves the incompressible Navier-Stokes equations using a
 * projection method with Taylor-Hood finite elements (Q_{p+1}/Q_p).
 * It initializes MPI, reads runtime parameters from a file, sets up the
 * simulation object (NavierStokesProjection), runs the time-stepping loop,
 * and handles potential exceptions.
 */

#include "navier_stokes_projection.h" // Simulation class
#include "run_time_parameters.h"    // Runtime parameter handling

// deal.II includes
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/multithread_info.h> // Not strictly needed if not using WorkStream
#include <deal.II/base/thread_management.h> // Not strictly needed if not using WorkStream
#include <deal.II/base/work_stream.h>     // Not strictly needed if not using WorkStream
#include <deal.II/base/parallel.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h> // Not strictly needed here, but often useful
#include <deal.II/base/logstream.h>           // For deallog

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

// PETSc Wrappers (if needed directly in main, otherwise included via headers)
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_vector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

// Standard library includes
#include <fstream>
#include <cmath>
#include <iostream>
#include <exception> // For std::exception

// Use the namespace defined in the simulation files
using namespace NERS570_proj;

/**
 * @brief Main function.
 *
 * Initializes MPI, reads parameters, creates and runs the Navier-Stokes
 * simulation object, and catches exceptions.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return 0 on successful execution, 1 on error.
 */
int main (int argc, char **argv)
{
  // Initialize MPI environment. The MPI_InitFinalize object handles
  // MPI_Init and MPI_Finalize automatically using RAII.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1); //

  // Get MPI rank and size for informational output
  const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD); //
  const unsigned int n_procs = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); //
  // Output only from rank 0 to avoid cluttered console
  if (rank == 0)
      std::cout << "Running NavierStokesProjection on " << n_procs << " MPI processes." << std::endl; // Modified for clarity

  try // Top-level try-catch block for graceful error handling
    {
      
      // --- Get parameter filename from command line ---
      std::string parameter_filename;
      if (argc > 1) // Check if an argument was provided
        {
          parameter_filename = argv[1];
        }
      else // No argument provided, print usage and exit
        {
          // Only rank 0 prints error message
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cerr << "Usage: " << argv[0] << " <parameter_file.prm>" << std::endl;
          return 1; // Exit all processes
        }

      std::cout << parameter_filename << std::endl;
      // Object to store runtime parameters
      RunTimeParameters::Data_Storage data; //
      // Read parameters from the specified file
      //data.read_data("parameter-file.prm"); //
      data.read_data(parameter_filename); //

      // Set the depth of deal.II's logging output based on verbosity parameter
      deallog.depth_console(data.verbose ? 2 : 0); //

      // Create the main simulation object (dimension is 2)
      NavierStokesProjection<2>  simulation(data);
      // Run the simulation, passing verbosity flag and output interval
      simulation.run(data.verbose, data.output_interval); //
    }
  catch (std::exception &exc) // Catch standard exceptions
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------" << std::endl; //
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl // Print exception message
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl; //
      // Ensure MPI_Abort is called to terminate all processes cleanly in case of error
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1; // Return error code
    }
  catch (...) // Catch any other non-standard exceptions
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------" << std::endl; //
      std::cerr << "Unknown exception!" << std::endl //
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl; //
      // Ensure MPI_Abort is called
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1; // Return error code
    }

  // If execution reaches here, the simulation completed without exceptions.
  // Output only from rank 0.
  if (rank == 0) {
      std::cout << "----------------------------------------------------" << std::endl //
                << "Simulation finished successfully!" << std::endl // 
                << "----------------------------------------------------" << std::endl;
  }

  return 0; // Return success code
}