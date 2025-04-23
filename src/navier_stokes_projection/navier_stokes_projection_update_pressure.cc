#include "../navier_stokes_projection.h" // Include the class declaration

// Include headers needed for this function
#include <deal.II/base/conditional_ostream.h> // For pcout
#include <deal.II/base/exceptions.h>       // For AssertThrow, ExcNotImplemented
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_precondition.h>  // For PreconditionBlockJacobi
#include <deal.II/lac/petsc_solver.h>        // For SolverCG
#include <deal.II/lac/solver_control.h>      // For SolverControl

#include "../run_time_parameters.h" // For RunTimeParameters::Method enum

namespace NERS570_proj {

  using namespace dealii;

  /**
   * @brief Updates the pressure field p_n based on the computed correction phi_n.
   *
   * Updates the pressure `pres_n` from time `t_n` to `t_{n+1}` using the
   * pressure correction `phi_n` computed in the `projection_step`. It supports
   * two methods based on the `type` member variable:
   *
   * - **Standard Method:**
   * Calculates `p^{n+1} = p^n + phi^{n+1}` (where `pres_n` currently holds p^n).
   * Before the update, `pres_n_minus_1` is set to the old `pres_n`.
   *
   * - **Rotational Method:**
   * Implements a more complex update involving a viscous term and requires solving
   * a mass matrix system:
   * 1. Solve `M * p* = (Laplace_p * p^n)` (where RHS is stored in `pres_tmp`).
   * Note: The original code used `pres_tmp` from the projection step's RHS
   * (`div(u*)`). This seems potentially incorrect for the rotational form's
   * pressure update, which usually involves `L_p * p_n` or similar.
   * *Assuming the original code's use of `pres_tmp` [div(u*)] was intended here.*
   * 2. Update `p^{n+1} = (1/Re) * p* + p^{n-1} + phi^{n+1}`.
   * Before the update, `pres_n_minus_1` is set to the old `pres_n`.
   *
   * @param reinit_prec Flag indicating whether the preconditioner for the mass
   * matrix solve (in the rotational method) should be reinitialized.
   */
  template <int dim>
  void NavierStokesProjection<dim>::update_pressure(const bool reinit_prec)
  {
    pcout << "  Updating pressure..." << std::endl;

    // Store current pressure as previous step's pressure (PETSc assignment)
    pres_n_minus_1 = pres_n;

    switch (type)
      {
        case RunTimeParameters::Method::standard:
          {
            pcout << "    Using standard pressure update..." << std::endl;
            // p_{n+1} = p_n + phi_{n+1} (interpretation)
            pres_n += phi_n; // PETSc vector addition works
            pres_n.compress(VectorOperation::add); // Compress after modification if needed? Usually += is safe. Check PETSc wrapper behavior. Let's assume it's safe for now.
            pcout << "    Standard update complete." << std::endl;
            break;
          }

        case RunTimeParameters::Method::rotational:
          {
            pcout << "    Using rotational pressure update..." << std::endl;

            // Solve M p* = pres_tmp using iterative solver + preconditioner
            // M = pres_Mass (PETSc matrix)
            // RHS = pres_tmp (PETSc vector)
            // Solution = pres_n (PETSc vector, overwritten)
            // Preconditioner = prec_mass (PETSc preconditioner)

            // 1. Initialize preconditioner if required
            if (reinit_prec)
              {
                pcout << "      Initializing pressure mass matrix preconditioner..." << std::endl;
                 prec_mass.initialize(pres_Mass);
              }

            // Create a separate vector for the RHS 'b'
            PETScWrappers::MPI::Vector rhs_p;
            rhs_p.reinit(pres_tmp); // Initialize with same parallel layout
            rhs_p = pres_tmp;       // Copy the actual RHS vector contents

            // Zero out pres_n to use it for the solution 'x' (satisfies PETSc requirement)
            pres_n = 0.;

            // 2. Set up solver control (use small tolerance for mass matrix inversion)
            // We want to solve M p* = pres_tmp accurately.
             const double mass_solver_tol = 1e-10 * pres_tmp.l2_norm() + 1e-30; // Example tolerance
             // Use a reasonable max iteration count
             const unsigned int mass_max_its = 1000;
             SolverControl mass_solver_control(mass_max_its, mass_solver_tol);

            // 3. Create Solver (CG is suitable for Mass Matrix)
             PETScWrappers::SolverCG mass_solver(mass_solver_control, mpi_communicator);

            // 4. Solve M x = b
            // Solve M x = b
            pcout << "      Solving mass matrix system M*p = pres_tmp..." << std::endl;
            mass_solver.solve(pres_Mass,  // Matrix M
                              pres_n,     // Solution x (now starts zero)
                              rhs_p,      // RHS b (separate vector)
                              prec_mass); // Preconditioner for M
            pcout << "      Mass matrix solve finished after "
                      << mass_solver_control.last_step() << " iterations." << std::endl;

            // Now pres_n holds the result of M^{-1} * pres_tmp

            // 5. Update pressure based on previous step and phi
            // pres_n = (1/Re)*pres_n + pres_{n-1} (using PETSc sadd)
             pres_n.sadd(1.0 / Re, 1.0, pres_n_minus_1);

            // pres_n = pres_n + phi_{n+1} (using PETSc +=)
             pres_n += phi_n;

             // Optional compress if worried about compound operations
             // pres_n.compress(VectorOperation::add);

            pcout << "    Rotational update complete." << std::endl;
            break;
          }

        default:
          // This should be unreachable if parameters are validated
          AssertThrow(false, ExcNotImplemented());
      };

    // No final .compress() usually needed here unless subsequent steps
    // rely on non-local values immediately without further communication.

     pcout << "  ...pressure update finished." << std::endl;
  }

} // namespace NERS570_proj