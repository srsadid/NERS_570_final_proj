#include "../navier_stokes_projection.h" // Include the class declaration
#include "../equation_data.h"           // For EquationData::Velocity

// Include headers needed for this function
#include <deal.II/base/conditional_ostream.h> // For pcout
#include <deal.II/base/function_lib.h>     // For Functions::ZeroFunction
#include <deal.II/base/exceptions.h>       // For Assert, ExcInternalError, ExcMessage
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_precondition.h>  // For PreconditionBlockJacobi
#include <deal.II/numerics/matrix_tools.h>   // For MatrixTools::apply_boundary_values
#include <deal.II/numerics/vector_tools.h>   // For VectorTools::interpolate_boundary_values

#include <vector> // For std::vector
#include <map>    // For std::map (boundary values)
#include <memory> // For unique_ptr access

namespace NERS570_proj {

  using namespace dealii;

  /**
   * @brief Performs the diffusion step to compute the tentative velocity.
   *
   * This step computes an intermediate velocity field (often denoted u*) which
   * approximates the final velocity u^{n+1}. It solves a system for each
   * velocity component `d` based on a semi-implicit time discretization (e.g., BDF2/AB2):
   *
   * (1.5/dt * M + 1/Re * L + A(u_star)) * u^{n+1}_d = RHS_d
   *
   * where M is the mass matrix, L is the Laplace matrix, A is the advection matrix
   * (assembled using u_star), and RHS depends on previous time steps and pressure terms.
   *
   * The steps involved are:
   * 1. Assemble the advection term matrix `vel_Advection` using `u_star`.
   * 2. Compute an intermediate pressure term `pres_tmp` involving p_n, phi_n, phi_{n-1}.
   * 3. For each component `d`:
   * a. Assemble the right-hand side vector `force[d]`.
   * b. Update the previous velocity `u_n_minus_1[d] = u_n[d]`.
   * c. Assemble the iteration matrix `vel_it_matrix[d]` = (1.5/dt)M + (1/Re)L + A.
   * d. Apply Dirichlet boundary conditions to the system matrix and RHS vector.
   * e. Optionally reinitialize the preconditioner `prec_velocity[d]`.
   * 4. Solve the linear system for each component `d` using `diffusion_component_solve(d)`.
   * The solution `u^{n+1}_d` overwrites the current `u_n[d]`.
   *
   * @param current_time The current simulation time (t_{n+1}). Used for time-dependent BCs.
   * @param reinit_prec Flag indicating whether the preconditioner should be reinitialized.
   */
  template <int dim>
  void NavierStokesProjection<dim>::diffusion_step(const double current_time, const bool reinit_prec)
  {
    pcout << "  Performing diffusion step..." << std::endl; //

    // --- 1. Assemble Advection Term ---
    pcout << "    Assemble advection term..." << std::endl; //
    assemble_advection_term(); // Assembles and compresses vel_Advection

    // --- 2. Calculate Intermediate Pressure Term ---
    // pres_tmp = -p_n - (4/3)*phi_n + (1/3)*phi_{n-1}
    pcout << "    Calculate pressure gradient term..." << std::endl; //
    pres_tmp = 0.; //
    pres_tmp.equ(-1.0, pres_n); // pres_tmp = -1.0 * pres_n
    pres_tmp.add(-4./3., phi_n, 1./3., phi_n_minus_1); // pres_tmp += (-4/3)*phi_n + (1/3)*phi_{n-1}
    pres_tmp.compress(VectorOperation::add); // Compress after calculations involving multiple distributed vectors

    // --- 3. Loop over Velocity Components ---
    for (unsigned int d = 0; d < dim; ++d) //
      {
        pcout << "    Processing component d = " << d << "..." << std::endl; //

        // --- 3a. Assemble Right-Hand Side (force[d]) ---
        // RHS = M * (2/dt * u_n - 0.5/dt * u_{n-1}) + G_d * pres_tmp
        pcout << "      Assemble RHS vector force[" << d << "]..." << std::endl; //
        force[d] = 0.; //

        // Calculate M * ( ... ) term
        v_tmp.equ(2.0 / dt, u_n[d]);           // v_tmp = (2.0/dt) * u_n[d]
        v_tmp.add(-0.5 / dt, u_n_minus_1[d]); // v_tmp += (-0.5/dt) * u_n_minus_1[d]
        vel_Mass.vmult_add(force[d], v_tmp);   // force[d] += vel_Mass * v_tmp

        // Calculate G_d * pres_tmp term (Gradient component d)
        Assert(pres_Diff[d], ExcInternalError("pres_Diff pointer is null in diffusion step.")); //
        pres_Diff[d]->vmult_add(force[d], pres_tmp); // force[d] += pres_Diff[d] * pres_tmp

        // Compress the final RHS vector
        force[d].compress(VectorOperation::add); //
        pcout << "      RHS force[" << d << "] assembly complete (norm=" << force[d].l2_norm() << ")." //
                   << std::endl; //

        // --- 3b. Update previous velocity ---
        // Store current u_n as u_{n-1} for the *next* time step's calculation
        u_n_minus_1[d] = u_n[d]; //

        // --- 3c. Assemble Iteration Matrix (vel_it_matrix[d]) ---
        // Matrix = (1.5/dt)*M + (1/Re)*L + 1.0*A
        pcout << "      Assemble iteration matrix vel_it_matrix[" << d << "]..." << std::endl; //
        Assert(vel_it_matrix[d], ExcInternalError("vel_it_matrix pointer is null in diffusion step.")); //
        *vel_it_matrix[d] = 0; // Zero the matrix first

        // Add scaled components (M, L, A assumed already assembled & compressed)
        vel_it_matrix[d]->add(1.5 / dt, vel_Mass);      // Add (1.5/dt)*M
        vel_it_matrix[d]->add(1.0 / Re, vel_Laplace);   // Add (1.0/Re)*L
        vel_it_matrix[d]->add(1.0,      vel_Advection); // Add 1.0*A

        // --- 3d. Apply Boundary Conditions ---
        pcout << "      Apply boundary conditions..." << std::endl; //
        std::map<types::global_dof_index, double> boundary_values; // Map to store DoF indices and values for BCs

        // Set the component and time for the exact velocity function used for BCs
        vel_exact.set_component(d); //
        vel_exact.set_time( current_time ); // Use the time t_{n+1} passed to the function

        // Create a scalar zero function for homogeneous Dirichlet BCs
        const Functions::ZeroFunction<dim> scalar_zero_function(1); // For single component

        // Iterate through the boundary IDs found on the mesh
        for (const auto &boundary_id : boundary_ids) //
          {
            // Apply BC based on boundary ID (specific to the nsbench2.inp geometry)
            if (boundary_id == 2) // Dirichlet BC using vel_exact (inflow/outflow?)
              {
                VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                         boundary_id,
                                                         vel_exact, // Provides scalar value for component 'd' at current_time
                                                         boundary_values); //
              }
            else if (boundary_id == 1 || boundary_id == 4) // Dirichlet zero BC 
              {
                VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                         boundary_id,
                                                         scalar_zero_function, // Use the 1-component zero function
                                                         boundary_values); //
              }
            else if (boundary_id == 3) // Cylinder boundary - specific handling
              {
                // Original code applied ZeroFunction only if d!=0 (i.e., for y-velocity)
                if (d != 0) // Apply zero only for component d=1 (y-velocity) on boundary 3
                  {
                    VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                             boundary_id,
                                                             scalar_zero_function, // Use scalar zero function
                                                             boundary_values); //
                  }
                // For d=0 (x-velocity), no BC is applied on boundary 3 (implicitly Neumann zero?)
              }
            else
              {
                // Throw error for unexpected boundary IDs
                AssertThrow(false, ExcMessage("Unknown boundary ID encountered in diffusion_step: " + std::to_string(boundary_id))); //
              }
          } // End loop over boundary_ids

        // Apply the collected boundary values to the linear system
        // Modifies vel_it_matrix[d], u_n[d] (solution guess), and force[d] (RHS)
        MatrixTools::apply_boundary_values(boundary_values,
                                          *vel_it_matrix[d], // The iteration matrix
                                          u_n[d],            // Solution vector (used as initial guess, modified for BCs)
                                          force[d],          // RHS vector (modified for BCs)
                                          false);           // Keep diagonal entries for BC rows
        pcout << "      Boundary conditions applied." << std::endl; //

        // --- 3e. Initialize Preconditioner (if needed) ---
        if (reinit_prec) //
          {
            pcout << "      Initializing preconditioner for component " << d << "..." << std::endl; //
            // Use the potentially modified iteration matrix after BC application
            prec_velocity[d].initialize(*vel_it_matrix[d]); //
          }

      } // End loop over components d

    // --- 4. Solve for each component ---
    pcout << "    Solving linear systems for velocity components..." << std::endl; //
    for (unsigned int d = 0; d < dim; ++d) //
      {
          diffusion_component_solve(d); // Call the solver function for component d
      }

    pcout << "  ...diffusion step finished." << std::endl; //
  }


  /**
   * @brief Solves the linear system for one velocity component in the diffusion step.
   *
   * This is a helper function called by `diffusion_step`. It sets up and runs
   * the PETSc GMRES solver for a single velocity component `d`.
   *
   * System: A * x = b
   * Where:
   * A = *vel_it_matrix[d] (PETSc matrix, already assembled with BCs)
   * x = u_n[d]           (PETSc vector, solution output)
   * b = force[d]         (PETSc vector, RHS input)
   * Preconditioner = prec_velocity[d]
   *
   * @param d The index of the velocity component (0 to dim-1) to solve for.
   */
  template <int dim>
  void NavierStokesProjection<dim>::diffusion_component_solve(const unsigned int d)
  {
    pcout << "      Solving for component d = " << d << "..." << std::endl; //

    // --- Setup Solver Control ---
    // Use relative tolerance based on RHS norm
    const double rhs_norm = force[d].l2_norm(); //
    // Add small epsilon to prevent tolerance=0 if rhs_norm=0
    const double tolerance = vel_eps * rhs_norm + 1e-30; //
    SolverControl solver_control(vel_max_its, tolerance); //

    // --- Create PETSc GMRES Solver ---
    // Pass the MPI communicator
    PETScWrappers::SolverGMRES solver(solver_control, mpi_communicator); //

    // Optional: Set GMRES restart parameter if needed (using AdditionalData)
    // PETScWrappers::SolverGMRES::AdditionalData gmres_data;
    // gmres_data.gmres_restart_parameter = vel_Krylov_size; // From constructor params
    // PETScWrappers::SolverGMRES solver(solver_control, gmres_data, mpi_communicator);

    // --- Solve using PETSc objects ---
    Assert(vel_it_matrix[d], ExcInternalError("vel_it_matrix pointer is null in diffusion_component_solve.")); //
    solver.solve(*vel_it_matrix[d], // The PETSc matrix A
                 u_n[d],            // Solution PETSc vector x (output)
                 force[d],          // RHS PETSc vector b (input)
                 prec_velocity[d]); // The PETSc preconditioner object

    pcout << "      Component " << d << " solve finished after "
               << solver_control.last_step() << " iterations." //
               << std::endl; //
  }



} // namespace NERS570_proj