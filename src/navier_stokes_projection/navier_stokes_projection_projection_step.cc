#include "../navier_stokes_projection.h" // Include the class declaration

// Include headers needed for this function
#include <deal.II/base/conditional_ostream.h> // For pcout
#include <deal.II/base/function_lib.h>     // For Functions::ZeroFunction
#include <deal.II/base/exceptions.h>       // For Assert, ExcInternalError
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_precondition.h>  // For PreconditionBlockJacobi
#include <deal.II/lac/petsc_solver.h>        // For SolverCG, SolverGMRES
#include <deal.II/numerics/matrix_tools.h>   // For MatrixTools::apply_boundary_values
#include <deal.II/numerics/vector_tools.h>   // For VectorTools::interpolate_boundary_values
#include <deal.II/lac/solver_control.h>      // For SolverControl

#include <vector> // For std::vector
#include <map>    // For std::map (boundary values)
#include <memory> // For unique_ptr access

namespace Step35 {

  using namespace dealii;

  /**
   * @brief Performs the projection step to compute the pressure correction phi_n.
   *
   * This step enforces the divergence constraint. It solves a Poisson-like
   * equation for the pressure correction term `phi_n`:
   *
   * L * phi_n = (1.5/dt) * div(u*)
   *
   * where L is the pressure Laplace matrix (`pres_Laplace`), u* is the tentative
   * velocity computed in the `diffusion_step` (and stored in `u_n`), and div
   * is approximated by the transpose of the discrete gradient operator (`-G^T`).
   * The factor (1.5/dt) comes from the BDF2 time discretization.
   *
   * The steps involved are:
   * 1. Set up the iteration matrix (`pres_iterative`) - typically just a copy of `pres_Laplace`.
   * 2. Assemble the right-hand side vector `pres_tmp` = Sum_d (Transpose(pres_Diff[d]) * u_n[d]).
   * This represents the discrete divergence of u*.
   * 3. Store the previous pressure correction: `phi_n_minus_1 = phi_n`.
   * 4. Apply homogeneous Dirichlet boundary conditions for `phi_n` on boundary ID 3
   * (modifies `pres_iterative` and `pres_tmp`).
   * 5. Optionally reinitialize the pressure preconditioner `prec_pres_Laplace`.
   * 6. Solve the linear system `pres_iterative * x = pres_tmp` for `x`, storing the result in `phi_n`.
   * 7. Scale the result: `phi_n *= (1.5 / dt)`.
   *
   * @param reinit_prec Flag indicating whether the preconditioner should be reinitialized.
   */
  template <int dim>
  void NavierStokesProjection<dim>::projection_step(const bool reinit_prec)
  {
    pcout << "  Performing projection step..." << std::endl; //

    // --- 1. Setup Iteration Matrix ---
    // For the standard projection, the matrix is the pressure Laplacian.
    pres_iterative.copy_from(pres_Laplace); // Start with L_p
    // Note: apply_boundary_values below will modify this matrix in-place.

    // --- 2. Assemble Right-Hand Side (Divergence Term) ---
    // RHS = -G^T * u* = Sum_d (Transpose(pres_Diff[d]) * u_n[d])
    // Note the sign convention difference: pres_Diff assembly included a '-' sign,
    // so we use Tvmult_add directly. If pres_Diff was positive gradient, we'd need -Tvmult_add.
    pcout << "    Assemble RHS vector pres_tmp (divergence)..." << std::endl; //
    pres_tmp = 0.; // Zero the PETSc RHS vector
    for (unsigned d = 0; d < dim; ++d) //
      {
        // Ensure pres_Diff[d] pointer is valid
        Assert(pres_Diff[d], ExcInternalError("pres_Diff pointer is null in projection_step.")); //
        // Add contribution from component d: pres_tmp += Transpose(pres_Diff[d]) * u_n[d]
        pres_Diff[d]->Tvmult_add(pres_tmp, u_n[d]); //
      }
    // Compress the assembled RHS vector
    pres_tmp.compress(VectorOperation::add); //
    pcout << "    RHS pres_tmp assembly complete (norm=" << pres_tmp.l2_norm() << ")." << std::endl; //

    // --- 3. Store Previous Phi ---
    phi_n_minus_1 = phi_n; // Store phi_n from previous step into phi_{n-1}

    // --- 4. Apply Boundary Conditions ---
    // Apply homogeneous Dirichlet BCs for phi_n on boundary ID 3 (cylinder).
    // This choice depends on the specific problem setup.
    pcout << "    Apply boundary conditions for phi_n..." << std::endl; //
    std::map<types::global_dof_index, double> boundary_values_phi; // Map for BCs

    // Use a 1-component zero function for scalar pressure/phi space
    const Functions::ZeroFunction<dim> scalar_zero_function(1); //

    VectorTools::interpolate_boundary_values(dof_handler_pressure, // Use pressure handler
                                             3,                    // Boundary ID = 3 (cylinder)
                                             scalar_zero_function, // Homogeneous Dirichlet
                                             boundary_values_phi); // Map to store BC DoFs/values

    // Apply BCs to the PETSc system (modifies matrix pres_iterative and RHS pres_tmp)
    MatrixTools::apply_boundary_values(boundary_values_phi,
                                       pres_iterative,    // The iteration matrix (L_p)
                                       phi_n,             // Solution vector (used as initial guess, modified for BCs)
                                       pres_tmp,          // RHS vector (modified for BCs)
                                       false);            // Keep diagonal entries
    pcout << "    Boundary conditions applied." << std::endl; //

    // --- 5. Initialize Preconditioner (if needed) ---
    if (reinit_prec) // Typically only need to initialize once at the start
      {
         pcout << "      Initializing preconditioner for pressure projection..." << std::endl; //
         // Initialize based on the chosen PETSc preconditioner type for prec_pres_Laplace
         // Use the matrix *after* boundary conditions have been applied.
         prec_pres_Laplace.initialize(pres_iterative); //
         // Note: Effectiveness depends heavily on the chosen preconditioner type.
         // BlockJacobi might be weak; consider AMG (via HYPRE) or ILU if available.
      }

    // --- 6. Solve the Linear System L*phi = div(u*) ---
    // System: pres_iterative * x = pres_tmp
    // Solution x is stored in phi_n

    // Set up solver control
    const double projection_solver_tol = vel_eps * pres_tmp.l2_norm() + 1e-30; // Relative tolerance
    SolverControl projection_solver_control(vel_max_its, projection_solver_tol); //

    // Create Solver (CG assumes matrix is SPD after BCs; use GMRES if unsure/non-symmetric)
    PETScWrappers::SolverCG projection_solver(projection_solver_control, mpi_communicator); //
    // PETScWrappers::SolverGMRES projection_solver(projection_solver_control, mpi_communicator); // Alternative

    pcout << "    Solving pressure projection system L*phi = div(u*)..." << std::endl; //
    projection_solver.solve(pres_iterative,    // Matrix A (L_p with BCs)
                            phi_n,             // Solution x (output)
                            pres_tmp,          // RHS b (divergence term with BCs)
                            prec_pres_Laplace); // Preconditioner
    pcout << "    Projection solve finished after "
               << projection_solver_control.last_step() << " iterations." //
               << std::endl; //


    // --- 7. Scale Result ---
    // Scale the computed phi_n by the time step factor (1.5/dt for BDF2)
    phi_n *= (1.5 / dt); //
    pcout << "  ...projection step finished (scaled phi_n norm=" << phi_n.l2_norm() << ")." << std::endl; //
  }

  // Explicit Instantiation
  // If you have navier_stokes_projection_instantiation.cc, remove this line.
  // template void NavierStokesProjection::projection_step(const bool);

} // namespace Step35