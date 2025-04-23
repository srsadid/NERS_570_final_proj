#include "../navier_stokes_projection.h" // Include the class declaration

// Include headers needed for this function
#include <deal.II/fe/fe_values.h>          // For FEValues
#include <deal.II/dofs/dof_handler.h>      // For DoFHandler::active_cell_iterator
#include <deal.II/lac/full_matrix.h>       // For FullMatrix
#include <deal.II/base/point.h>            // For Point<dim>
#include <deal.II/base/tensor.h>           // For Tensor<1, dim>
#include <deal.II/base/quadrature_lib.h>   // For QGauss
#include <deal.II/fe/fe_q.h>               // For FE_Q
#include <deal.II/lac/vector.h>            // For Vector<double> (local copy)
#include <deal.II/lac/petsc_sparse_matrix.h> // For PETScWrappers::MPI::SparseMatrix
#include <deal.II/base/conditional_ostream.h> // For pcout

#include <vector>  // For std::vector

namespace NERS570_proj {

  using namespace dealii;

  /**
   * @brief Assembles the advection term matrix (vel_Advection).
   *
   * This function assembles the matrix corresponding to the nonlinear advection
   * term, using a skew-symmetric formulation:
   * A_{ij} = integral{ [(u_star · ∇)φ_j] * φ_i + 0.5 * (∇ · u_star) * φ_i * φ_j } dx
   * where u_star is the extrapolated velocity computed by interpolate_velocity(),
   * and φ_i, φ_j are velocity basis functions.
   *
   * The assembly iterates over cells owned by the current MPI process. Inside the loop:
   * 1. Local copies of the parallel u_star vectors are created.
   * 2. FEValues object is used to compute u_star values and gradients at quadrature points.
   * 3. The divergence of u_star is computed at quadrature points.
   * 4. The local matrix contribution is computed using the skew-symmetric formula.
   * 5. The local matrix is added to the global PETSc matrix vel_Advection.
   *
   * After the cell loop, the global vel_Advection matrix is compressed to sum
   * contributions across MPI processes.
   */
  template <int dim>
  void NavierStokesProjection<dim>::assemble_advection_term()
  {
    pcout << "    Assembling advection term (vel_Advection)..." << std::endl; //

    // --- 1. Setup Scratch Data ---
    // Uses AdvectionScratchData struct defined in the header
    AdvectionScratchData scratch(fe_velocity,
                                 quadrature_velocity,
                                 update_values | update_gradients | update_JxW_values); // Need values, gradients, JxW

    // --- Create temporary LOCAL copies of u_star ---
    // FEValues::get_function_values/gradients often expect a complete local vector.
    // Creating these Vector<double> copies gathers the necessary data onto each process.
    pcout << "      Creating local copies of u_star for FE value/gradient calculation..." << std::endl; //
    std::vector<Vector<double>> local_u_star(dim); //
    for (unsigned int d=0; d<dim; ++d)
        local_u_star[d] = Vector<double>(u_star[d]); // Explicit copy constructor gathers data

    // --- 2. Zero the Global Matrix ---
    vel_Advection = 0; //

    // --- 3. Manual Loop over Cells ---
    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_velocity.begin_active(),
      endc = dof_handler_velocity.end();

    // --- Local storage for assembly on one cell ---
    FullMatrix<double> local_advection(scratch.dpc, scratch.dpc);
    std::vector<types::global_dof_index> local_dof_indices(scratch.dpc);

    // --- Iterate over all active cells ---
    for (; cell != endc; ++cell) //
      {
        // --- 4. Process only cells owned by this MPI rank ---
        if (cell->subdomain_id() == this_mpi_process) //
          {
            // --- 5. Local Assembly Logic ---
            scratch.fe_val.reinit(cell); // Reinitialize FEValues for this cell
            cell->get_dof_indices(local_dof_indices); // Get DoF indices for this cell

            // --- Get u_star values at quadrature points ---
            for (unsigned int d = 0; d < dim; ++d) //
              {
                 // Use the LOCAL copy of u_star for evaluation
                 scratch.fe_val.get_function_values(local_u_star[d], scratch.u_star_tmp); // Store component d values in tmp vector
                 for (unsigned int q = 0; q < scratch.nqp; ++q)
                   scratch.u_star_local[q][d] = scratch.u_star_tmp[q]; // Copy into Point<dim> structure
              }

            // --- Get divergence term: div(u_star) at quadrature points ---
            // Re-use scratch.u_star_tmp to store the divergence
            for (unsigned int q = 0; q < scratch.nqp; ++q)
                 scratch.u_star_tmp[q] = 0.; // Zero out tmp vector

            for (unsigned int d = 0; d < dim; ++d) //
              {
                 // Use the LOCAL copy of u_star for evaluation
                 scratch.fe_val.get_function_gradients(local_u_star[d], scratch.grad_u_star); // Get gradient tensor for component d
                 for (unsigned int q = 0; q < scratch.nqp; ++q) //
                   {
                      // Add d(u_d)/dx_d to the divergence term at point q
                      scratch.u_star_tmp[q] += scratch.grad_u_star[q][d]; //
                   }
              }

            // --- Assemble local matrix using skew-symmetric form ---
            local_advection = 0.; //
            for (unsigned int q = 0; q < scratch.nqp; ++q) //
              {
                for (unsigned int i = 0; i < scratch.dpc; ++i) // Test function index
                  {
                    for (unsigned int j = 0; j < scratch.dpc; ++j) // Trial function index
                      {
                        // Term 1: (u_star . grad(phi_j)) * phi_i
                        const Tensor<1, dim> grad_phi_j = scratch.fe_val.shape_grad(j, q);
                        const double u_star_dot_grad_phi_j = scratch.u_star_local[q] * grad_phi_j;
                        const double phi_i = scratch.fe_val.shape_value(i, q);

                        // Term 2: 0.5 * div(u_star) * phi_i * phi_j (Skew-symmetric part)
                        const double div_u_star = scratch.u_star_tmp[q]; // Calculated above
                        const double phi_j = scratch.fe_val.shape_value(j, q);

                        local_advection(i, j) += (u_star_dot_grad_phi_j * phi_i +
                                                  0.5 * div_u_star * phi_i * phi_j) *
                                                 scratch.fe_val.JxW(q);
                      }
                  }
              } // end quadrature loop

            // --- 6. Add Local Contribution to Global PETSc Matrix ---
            vel_Advection.add(local_dof_indices,
                              local_dof_indices,
                              local_advection); //
          } // end if owned cell
      } // end cell loop

    // --- 7. Compress Matrix After Assembly ---
    vel_Advection.compress(VectorOperation::add); //
    pcout << "    ...done assembling advection term." << std::endl; //
  }

} // namespace NERS570_proj