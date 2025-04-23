#include "../navier_stokes_projection.h" // Include the class declaration

// Include headers needed for this function
#include <deal.II/fe/fe_values.h>          // For FEValues
#include <deal.II/dofs/dof_handler.h>      // For DoFHandler::active_cell_iterator
#include <deal.II/lac/full_matrix.h>       // For FullMatrix
#include <deal.II/base/types.h>            // For types::global_dof_index
#include <deal.II/base/quadrature_lib.h>   // For QGauss
#include <deal.II/fe/fe_q.h>               // For FE_Q
#include <deal.II/base/conditional_ostream.h> // For pcout
#include <deal.II/lac/petsc_sparse_matrix.h> // For PETScWrappers::MPI::SparseMatrix
#include <deal.II/base/exceptions.h>       // For Assert, ExcInternalError

#include <vector>  // For std::vector
#include <memory>  // For unique_ptr access

namespace NERS570_proj {

  using namespace dealii;

  /**
   * @brief Assembles the discrete gradient operator matrices.
   *
   * This function assembles the matrices `pres_Diff[d]` for each dimension `d`.
   * These matrices represent the discrete gradient operator coupling the pressure
   * and velocity spaces: `G = [G_0, G_1, ..., G_{dim-1}]^T`. Specifically, the
   * entry `(i, j)` of `pres_Diff[d]` corresponds to the integral of
   * `-\partial_{x_d}(\psi^v_i) * \psi^p_j` over the domain, where `\psi^v_i` is
   * the i-th velocity basis function and `\psi^p_j` is the j-th pressure basis function.
   *
   * The assembly iterates over all cells owned by the current MPI process.
   * For each cell, it calculates the local matrix contribution using FEValues
   * and adds it to the corresponding global PETSc matrix `pres_Diff[d]`.
   * After assembling all local contributions for a component `d`, the global
   * matrix `pres_Diff[d]` is compressed to sum contributions across processes.
   *
   * Note: This implementation assumes no hanging node constraints need to be
   * applied directly during the matrix assembly (`.add` call). If constraints
   * were needed, a more complex approach involving AffineConstraints would be
   * necessary here.
   */
  template <int dim>
  void NavierStokesProjection<dim>::assemble_gradient_operator()
  {
    pcout << "Assembling gradient operator (pres_Diff)..." << std::endl; //

    // --- 1. Setup Scratch Data ---
    // Uses the InitGradScratchData struct defined in the header
    // (Requires FE_Q, QGauss, UpdateFlags)
    InitGradScratchData scratch_data(fe_velocity,
                                     fe_pressure,
                                     quadrature_velocity, // Use appropriate quadrature rule
                                     UpdateFlags(update_gradients | update_JxW_values), // Flags for velocity FE: Need gradients & JxW
                                     UpdateFlags(update_values));                      // Flags for pressure FE: Need only values

    // --- 2. Loop over dimensions and assemble each pres_Diff[d] ---
    for (unsigned int d = 0; d < dim; ++d) //
      {
        pcout << "  Assembling component d = " << d << "..." << std::endl; //

        // --- Ensure unique_ptr is valid and zero the matrix ---
        Assert(pres_Diff[d], ExcInternalError("pres_Diff pointer is null before assembly.")); //
        *pres_Diff[d] = 0; // Zero the matrix before adding contributions

        // --- 3. Manual Loop over Cells (Synchronous within each process) ---
        typename DoFHandler<dim>::active_cell_iterator
          vel_cell = dof_handler_velocity.begin_active(),
          pres_cell = dof_handler_pressure.begin_active(), // Assume iterators stay synchronized
          endc = dof_handler_velocity.end();

        // --- Local storage for assembly on one cell ---
        FullMatrix<double> local_grad_matrix(fe_velocity.n_dofs_per_cell(),
                                             fe_pressure.n_dofs_per_cell());
        std::vector<types::global_dof_index> vel_local_dof_indices(fe_velocity.n_dofs_per_cell());
        std::vector<types::global_dof_index> pres_local_dof_indices(fe_pressure.n_dofs_per_cell());

        // --- Iterate over all active cells ---
        for (; vel_cell != endc; ++vel_cell, ++pres_cell) //
          {
            // --- 4. Process only cells owned by this MPI rank ---
            if (vel_cell->subdomain_id() == this_mpi_process) //
              {
                // --- 5. Local Assembly Logic ---
                scratch_data.fe_val_vel.reinit(vel_cell);   // Reinitialize FEValues for velocity
                scratch_data.fe_val_pres.reinit(pres_cell); // Reinitialize FEValues for pressure

                vel_cell->get_dof_indices(vel_local_dof_indices); // Get DoF indices for this cell (velocity)
                pres_cell->get_dof_indices(pres_local_dof_indices); // Get DoF indices for this cell (pressure)

                local_grad_matrix = 0.; // Zero the local matrix for this cell

                // Loop over quadrature points
                for (unsigned int q = 0; q < scratch_data.nqp; ++q) //
                  {
                    // Loop over velocity basis functions (test function i)
                    for (unsigned int i = 0; i < fe_velocity.n_dofs_per_cell(); ++i) //
                      {
                        // Loop over pressure basis functions (trial function j)
                        for (unsigned int j = 0; j < fe_pressure.n_dofs_per_cell(); ++j) //
                          {
                            // Integral of: - grad(vel_test_func_i)[d] * pres_trial_func_j * JxW
                            local_grad_matrix(i, j) +=
                              -scratch_data.fe_val_vel.shape_grad(i, q)[d] * // Gradient of velocity basis i, component d
                               scratch_data.fe_val_pres.shape_value(j, q) * // Value of pressure basis j
                               scratch_data.fe_val_vel.JxW(q);              // Quadrature weight * det(Jacobian)
                          } // end pressure basis loop (j)
                      } // end velocity basis loop (i)
                  } // end quadrature loop (q)

                // --- 6. Add Local Contribution to Global PETSc Matrix ---
                // This adds the computed local_grad_matrix block into the global pres_Diff[d]
                // at the locations specified by the local DoF index vectors.
                pres_Diff[d]->add(vel_local_dof_indices,   // Row indices (velocity DoFs)
                                   pres_local_dof_indices,  // Column indices (pressure DoFs)
                                   local_grad_matrix);      // Local matrix block
              } // end if (owned cell)
          } // end cell loop

        // --- 7. Compress Matrix After Assembling All Local Contributions for component d ---
        // This performs MPI communication to sum up contributions to shared DoFs.
        pres_Diff[d]->compress(VectorOperation::add); //
        pcout << "    Compressed pres_Diff[" << d << "]" << std::endl; //
      } // end loop over dimension d

    pcout << "  ...done assembling gradient operator." << std::endl; //
  }



} // namespace NERS570_proj