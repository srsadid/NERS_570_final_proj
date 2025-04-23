#include "../navier_stokes_projection.h" // Include the class declaration

// Include headers needed for this function
#include <deal.II/numerics/matrix_creator.h>   // For MatrixCreator::create_*_matrix
#include <deal.II/lac/petsc_sparse_matrix.h> // For PETScWrappers::MPI::SparseMatrix
#include <deal.II/base/conditional_ostream.h> // For pcout

namespace NERS570_proj {

  using namespace dealii;

  /**
   * @brief Assembles the time-independent matrices (Mass and Laplace).
   *
   * Uses the deal.II MatrixCreator functions to assemble the mass and Laplace
   * matrices for both the velocity and pressure finite element spaces.
   * These matrices (vel_Mass, vel_Laplace, pres_Mass, pres_Laplace) are constant
   * throughout the simulation. After assembly, the matrices are compressed to
   * ensure correct contributions from different MPI processes are summed for
   * shared degrees of freedom.
   */
  template <int dim>
  void NavierStokesProjection<dim>::assemble_time_independent_matrices()
  {
    pcout << "Assembling time-independent matrices (Mass, Laplace)..." << std::endl; //

    // Zero matrices before assembly (optional, but good practice)
    vel_Mass = 0; //
    vel_Laplace = 0; //
    pres_Mass = 0; //
    pres_Laplace = 0; //

    // --- Assemble Velocity Matrices ---
    pcout << "  Assembling velocity mass matrix..." << std::endl;
    MatrixCreator::create_mass_matrix(dof_handler_velocity,    // Use velocity DoF handler
                                      quadrature_velocity,     // Use velocity quadrature rule
                                      vel_Mass);               // Target PETSc matrix

    pcout << "  Assembling velocity Laplace matrix..." << std::endl;
    MatrixCreator::create_laplace_matrix(dof_handler_velocity, // Use velocity DoF handler
                                         quadrature_velocity,  // Use velocity quadrature rule
                                         vel_Laplace);         // Target PETSc matrix

    // --- Assemble Pressure Matrices ---
    pcout << "  Assembling pressure mass matrix..." << std::endl;
    MatrixCreator::create_mass_matrix(dof_handler_pressure,    // Use pressure DoF handler
                                      quadrature_pressure,     // Use pressure quadrature rule
                                      pres_Mass);              // Target PETSc matrix

    pcout << "  Assembling pressure Laplace matrix..." << std::endl;
    MatrixCreator::create_laplace_matrix(dof_handler_pressure, // Use pressure DoF handler
                                         quadrature_pressure,  // Use pressure quadrature rule
                                         pres_Laplace);        // Target PETSc matrix


    // --- Compress Matrices ---
    // This MPI communication step sums contributions to shared DoFs from different processes.
    pcout << "  Compressing assembled matrices..." << std::endl;
    vel_Mass.compress(VectorOperation::add); //
    vel_Laplace.compress(VectorOperation::add); //
    pres_Mass.compress(VectorOperation::add); //
    pres_Laplace.compress(VectorOperation::add); //

    pcout << "  ...done assembling time-independent matrices." << std::endl; //
  }

} // namespace NERS570_proj