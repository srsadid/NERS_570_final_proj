#include "../navier_stokes_projection.h" // Include the class declaration
#include "../equation_data.h"           // For EquationData::Pressure, EquationData::Velocity

// Include headers needed for this function
#include <deal.II/numerics/vector_tools.h>   // For VectorTools::interpolate
#include <deal.II/lac/petsc_vector.h>        // For PETScWrappers::MPI::Vector assignment (=0.)
#include <deal.II/base/conditional_ostream.h> // For pcout

#include <vector> // For std::vector

namespace Step35 {

  using namespace dealii;

  /**
   * @brief Initializes the simulation state vectors.
   *
   * Sets the initial conditions for the simulation. It interpolates the
   * analytical pressure solution onto the `pres_n` and `pres_n_minus_1` vectors,
   * sets the `phi_n` and `phi_n_minus_1` vectors to zero, and interpolates the
   * analytical velocity solution onto the `u_n` and `u_n_minus_1` vector components.
   * The interpolation uses the time `t_0` for the `*_n_minus_1` fields and
   * `t_0 + dt` for the `*_n` fields, consistent with a BDF2 scheme start.
   */
  template <int dim>
  void NavierStokesProjection<dim>::initialize()
  {
    pcout << "Initializing simulation state..." << std::endl; //

    // --- Zero out Combined Matrix (Optional) ---
    // This matrix (1.5/dt*M + 1/Re*L + A) is assembled later in the diffusion step.
    // Zeroing it here ensures it starts clean, although reinit in create_triangulation_and_dofs
    // and the copy/add operations in diffusion_step should handle this.
    // vel_Laplace_plus_Mass = 0.; //

    // --- Initialize Pressure Vectors ---
    pcout << "  Interpolating initial pressure..." << std::endl; //
    EquationData::Pressure<dim> pres(t_0); // Define initial pressure function object at t_0

    // Interpolate P(t_0) onto pres_n_minus_1
    VectorTools::interpolate(dof_handler_pressure, pres, pres_n_minus_1); //
    // Advance function time to t_0 + dt for the 'n' state
    pres.advance_time(dt);
    // Interpolate P(t_0 + dt) onto pres_n
    VectorTools::interpolate(dof_handler_pressure, pres, pres_n); //

    // --- Initialize Phi Vectors ---
    // Phi represents the pressure correction, typically starts at 0.
    pcout << "  Setting initial phi fields to zero..." << std::endl; //
    phi_n         = 0.; // PETSc vector assignment works in parallel
    phi_n_minus_1 = 0.; // PETSc vector assignment works in parallel

    // --- Initialize Velocity Vectors ---
    pcout << "  Interpolating initial velocity..." << std::endl; //
    for (unsigned int d = 0; d < dim; ++d) //
      {
        // Set exact solution time to t_0 for the 'n-1' state
        vel_exact.set_time(t_0); //
        // Set the component of the velocity function to interpolate
        vel_exact.set_component(d); //

        // Interpolate U(t_0) onto u_n_minus_1[d]
        VectorTools::interpolate(dof_handler_velocity, vel_exact, u_n_minus_1[d]); //

        // Advance exact solution time to t_0 + dt for the 'n' state
        vel_exact.advance_time(dt); //
        // Interpolate U(t_0 + dt) onto u_n[d]
        // (No need to set component again if vel_exact remembers it)
        VectorTools::interpolate(dof_handler_velocity, vel_exact, u_n[d]); //
      }

    // Note: PETSc vector assignments (= 0.) and VectorTools::interpolate work correctly
    // on distributed vectors; no manual MPI communication (like compress) is needed here.
    pcout << "  ...initialization complete." << std::endl; //
  }

  // Explicit Instantiation
  // If you have navier_stokes_projection_instantiation.cc, remove this line.
  // template void NavierStokesProjection::initialize();

} // namespace Step35