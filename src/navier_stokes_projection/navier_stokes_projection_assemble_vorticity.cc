#include "../navier_stokes_projection.h" // Include the class declaration

// Include headers needed for this function
#include <deal.II/base/conditional_ostream.h> // For pcout
#include <deal.II/base/exceptions.h>       // For AssertThrow, ExcNotImplemented
#include <deal.II/base/quadrature_lib.h>   // For QGauss
#include <deal.II/base/tensor.h>           // For Tensor<1, dim>
#include <deal.II/base/types.h>            // For types::global_dof_index
#include <deal.II/dofs/dof_handler.h>      // For DoFHandler::active_cell_iterator
#include <deal.II/fe/fe_q.h>               // For FE_Q
#include <deal.II/fe/fe_values.h>          // For FEValues
#include <deal.II/lac/petsc_precondition.h>  // For PreconditionBlockJacobi
#include <deal.II/lac/petsc_solver.h>        // For SolverCG
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/solver_control.h>      // For SolverControl
#include <deal.II/lac/vector.h>            // For Vector<double> (local copy)


#include <vector> // For std::vector

namespace NERS570_proj {

  using namespace dealii;

  /**
   * @brief Assembles the L2 projection of vorticity onto the FE space.
   *
   * Calculates the vorticity of the velocity field `u_n` and computes its L2
   * projection onto the velocity finite element space, storing the result in `rot_u`.
   * Currently implemented only for dim=2.
   *
   * The steps are:
   * 1. Optionally reinitialize the preconditioner `prec_vel_mass` for the velocity mass matrix.
   * 2. Create local copies of the parallel `u_n[0]` and `u_n[1]` vectors.
   * 3. Iterate over cells owned by the current MPI process:
   * a. Compute gradients of `u_n[0]` and `u_n[1]` at quadrature points using the local copies.
   * b. Calculate the vorticity `w = dv/dx - du/dy` at quadrature points.
   * c. Assemble the local right-hand side vector contribution: `rhs_i = integral{ w * phi_i } dx`.
   * d. Add the local RHS contribution to the global PETSc vector `rot_u`.
   * 4. Compress the global RHS vector `rot_u`.
   * 5. Solve the mass matrix system `M * w_h = rhs`, where `M` is `vel_Mass`, `rhs` is the
   * assembled `rot_u`, and the solution `w_h` (the projected vorticity) overwrites `rot_u`.
   *
   * @param reinit_prec Flag indicating whether the velocity mass matrix preconditioner
   * (`prec_vel_mass`) should be reinitialized.
   */
  template <int dim>
  void NavierStokesProjection<dim>::assemble_vorticity(const bool reinit_prec /*=true*/)
  {
    pcout << "  Assembling vorticity..." << std::endl; //
    // Currently only implemented for 2D
    AssertThrow(dim == 2, ExcNotImplemented("assemble_vorticity is only implemented for dim=2")); //

    // --- 1. Initialize Preconditioner for Velocity Mass Matrix ---
    if (reinit_prec) //
     {
        pcout << "    Initializing preconditioner for velocity mass matrix..." << std::endl; //
        // Initialize based on the chosen PETSc preconditioner type for prec_vel_mass
        // Ensure vel_Mass is already assembled and compressed.
        prec_vel_mass.initialize(vel_Mass); //
        // Consider a better preconditioner (AMG, ILU) if BlockJacobi is too slow.
     }

    // --- 2. Create temporary LOCAL copies of u_n components ---
    // Needed for get_function_gradients if it expects Vector<double>
    pcout << "    Creating local copies of u_n for gradient calculation..." << std::endl; //
    Vector<double> local_u_n0(u_n[0]); // Create local copy of u_n[0]
    Vector<double> local_u_n1(u_n[1]); // Create local copy of u_n[1]

    // --- 3. Setup FEValues and local storage ---
    FEValues<dim> fe_val_vel(fe_velocity,           // Use velocity FE space
                             quadrature_velocity,   // Use velocity quadrature
                             update_values | update_gradients | update_JxW_values); // Need values, grads, JxW

    const unsigned int dpc = fe_velocity.n_dofs_per_cell(); // DoFs per cell for velocity space
    const unsigned int nqp = quadrature_velocity.size();    // Number of quadrature points

    Vector<double>                       local_rhs_vector(dpc); // Local RHS vector contribution
    std::vector<types::global_dof_index> local_dof_indices(dpc); // Local DoF indices
    

    // Storage for gradients at quadrature points for each component
    std::vector<Tensor<1, dim>>          grad_u0_at_qpoints(nqp);
    std::vector<Tensor<1, dim>>          grad_u1_at_qpoints(nqp);

    // --- 4. Assemble Global RHS Vector (Integral{ vorticity * phi_i }) ---
    rot_u = 0.; // Zero the global PETSc vector (will hold RHS temporarily)

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_velocity.begin_active(),
      endc = dof_handler_velocity.end();

    // --- Iterate over all active cells ---
    for (; cell != endc; ++cell) //
      {
        // --- 4a. Process only cells owned by this MPI rank ---
        if (cell->subdomain_id() == this_mpi_process) //
          {
            // --- 4b. Local RHS Calculation ---
            fe_val_vel.reinit(cell); // Reinitialize FEValues for this cell
            cell->get_dof_indices(local_dof_indices); // Get DoF indices

            // Get gradients using LOCAL copies of u_n components
            fe_val_vel.get_function_gradients(local_u_n0, grad_u0_at_qpoints); //
            fe_val_vel.get_function_gradients(local_u_n1, grad_u1_at_qpoints); //

            local_rhs_vector = 0.; // Zero local vector for this cell

            // Loop over quadrature points
            for (unsigned int q = 0; q < nqp; ++q) //
              {
                // Vorticity (2D): w = dv/dx - du/dy = grad_u1[q][0] - grad_u0[q][1]
                const double vorticity_at_q = grad_u1_at_qpoints[q][0] - grad_u0_at_qpoints[q][1]; //

                // Loop over velocity basis functions (test function phi_i)
                for (unsigned int i = 0; i < dpc; ++i) //
                  {
                    // Add contribution to local RHS: vorticity * phi_i(q) * JxW(q)
                    local_rhs_vector(i) += vorticity_at_q *
                                           fe_val_vel.shape_value(i, q) * // Test function value
                                           fe_val_vel.JxW(q);             // Integration weight
                  } // end basis function loop (i)
              } // end quadrature loop (q)

            // --- 4c. Add Local Contribution to Global PETSc Vector ---
            rot_u.add(local_dof_indices, local_rhs_vector); // Add local vector to global rot_u
          } // end if owned cell
      } // end cell loop

    // --- 5. Compress Global RHS Vector ---
    rot_u.compress(VectorOperation::add); // Sum contributions across processes
    pcout << "    RHS for vorticity projection assembled (norm=" << rot_u.l2_norm() << ")." << std::endl; //

    // --- 6. Solve the Mass Matrix system M * w = b ---
    // M = vel_Mass (Velocity Mass Matrix)
    // b = rot_u (Assembled RHS vector)
    // w = rot_u (Solution vector, overwrites RHS)
    // Preconditioner = prec_vel_mass

    // Create a separate vector to store the RHS
    PETScWrappers::MPI::Vector rhs_b; //
    rhs_b.reinit(rot_u); // Initialize with same parallel layout as rot_u
    rhs_b = rot_u;       // Copy the assembled RHS into rhs_b

    // Zero out rot_u to use it for the solution 'w' (satisfies PETSc solver requirements)
    rot_u = 0.; //

    // Set up solver control
    // Use a tolerance appropriate for projecting vorticity; may not need to be extremely small
    const double vorticity_solver_tol = 1e-8 * rhs_b.l2_norm() + 1e-30; // Relative tolerance
    const unsigned int vorticity_max_its = 1000; // Example max iterations
    SolverControl vorticity_solver_control(vorticity_max_its, vorticity_solver_tol); //

    // Create Solver (CG is suitable for SPD Mass Matrix)
    PETScWrappers::SolverCG vorticity_solver(vorticity_solver_control, mpi_communicator); //

    // Solve M * w = b
    pcout << "    Solving velocity mass matrix system M*w = b..." << std::endl; //
    vorticity_solver.solve(vel_Mass,        // Matrix M
                            rot_u,           // Solution w (now starts zero)
                            rhs_b,           // RHS b (separate vector)
                            prec_vel_mass);  // Preconditioner for M

    pcout << "    Vorticity solve finished after "
                << vorticity_solver_control.last_step() << " iterations." //
                << std::endl; //
    pcout << "  ...vorticity assembly complete (result norm=" << rot_u.l2_norm() << ")." << std::endl; //
    // rot_u now holds the projected vorticity field w_h
  }


} // namespace NERS570_proj