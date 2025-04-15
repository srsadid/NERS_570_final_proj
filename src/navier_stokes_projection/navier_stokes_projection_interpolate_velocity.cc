#include "../navier_stokes_projection.h" // Include the class declaration

// Include headers needed for this function
#include <deal.II/lac/petsc_vector.h> // For PETScWrappers::MPI::Vector operations

#include <vector> // For std::vector loop

namespace Step35 {

  using namespace dealii;

  /**
   * @brief Calculates the intermediate/extrapolated velocity u_star.
   *
   * This function computes the velocity field `u_star` required for assembling
   * the advection term in the tentative velocity step.
   *
   * The current implementation calculates `u_star[d] = u_n[d] - u_n_minus_1[d]`.
   * Note: Common extrapolation methods for projection schemes often use
   * Adams-Bashforth (e.g., `u_star = 1.5*u_n - 0.5*u_n_minus_1` for AB2)
   * or simply `u_star = u_n` for stability reasons. Verify if this definition
   * is the intended one for the advection term calculation.
   *
   * The operation is performed component-wise using PETSc vector operations,
   * which handle parallel execution correctly. No explicit MPI communication
   * (like compress) is needed here.
   */
  template <int dim>
  void NavierStokesProjection<dim>::interpolate_velocity()
  {
    // Optional: Add output message for debugging/logging
    // pcout << "  Interpolating velocity (u_star = u_n - u_n_minus_1)..." << std::endl;

    for (unsigned int d = 0; d < dim; ++d) //
      {
        // u_star[d] = u_n[d];
        u_star[d] = u_n[d]; // Assign u_n to u_star (PETSc vector assignment)

        // u_star[d] += (-1.0) * u_n_minus_1[d];
        u_star[d].add(-1.0, u_n_minus_1[d]); // Add -1.0 * u_n_minus_1 (PETSc vector add)
      }

    // No .compress() needed as operations are element-wise on distributed vectors.
  }

  // Explicit Instantiation
  // If you have navier_stokes_projection_instantiation.cc, remove this line.
  // template void NavierStokesProjection::interpolate_velocity();

} // namespace Step35