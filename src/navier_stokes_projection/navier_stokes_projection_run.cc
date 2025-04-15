#include "../navier_stokes_projection.h" // Include the class declaration

// Include headers specifically needed for the run function implementation
#include <deal.II/base/timer.h>          // For Timer
#include <deal.II/base/conditional_ostream.h> // For pcout

#include <vector>                        // For std::vector
#include <cmath>                         // For static_cast

namespace Step35 {

  using namespace dealii;

  /**
   * @brief Runs the Navier-Stokes simulation.
   *
   * This function executes the main time stepping loop. It initializes the
   * simulation by assembling time-independent matrices and the gradient operator.
   * It then outputs the initial condition and enters the time loop, performing
   * the diffusion, projection, and pressure update steps sequentially until the
   * final time is reached. Results are output periodically.
   *
   * @param verbose If true, enables verbose output during the simulation run.
   * @param output_interval The frequency (in time steps) at which to output solution files.
   */
  template <int dim>
  void NavierStokesProjection<dim>::run(const bool verbose /*=false*/, const unsigned int output_interval /*= 10*/)
  {
    // Set conditional output stream based on verbose flag
    pcout.set_condition(verbose); // [cite: 277]

    // Initial Setup Steps (called from constructor)
    // create_triangulation_and_dofs()
    // initialize()

    // Assemble matrices that don't change in time loop
    pcout << "Running initial assembly..." << std::endl; // [cite: 278]
    assemble_time_independent_matrices(); // Assemble Mass, Laplace [cite: 279]
    assemble_gradient_operator();        // Assemble Gradient Operator [cite: 279]
    // Advection term is assembled inside diffusion_step

    // Time Loop Setup
    const unsigned int n_steps = static_cast<unsigned int>((T - t_0) / dt); // [cite: 279]
    double time = t_0; // Start time [cite: 280]

    //pcout << "Starting time loop: T = " << T << ", dt = " << dt << ", N_steps = " << n_steps << std::endl; // [cite: 280]
    if (this_mpi_process == 0) {
      std::cout << "Starting time loop: T = " << T << ", dt = " << dt << ", N_steps = " << n_steps << std::endl;
    }
    // Output Initial Condition (Step 0)
    // Ensure correct time for initial output if needed
    time = t_0; // [cite: 281]
    //pcout << "\nOutputting initial solution (Step 0, Time " << time << ")" << std::endl; // [cite: 282]
    if (this_mpi_process == 0) {
      std::cout << "\nOutputting initial solution (Step 0, Time " << time << ")" << std::endl;
    }
    output_results(0); // Output initial state at t_0 [cite: 283]

    Timer timer; // [cite: 283]

    // Main Time Loop
    // Start from step n=1 to reach final time T = t_0 + n_steps * dt
    for (unsigned int n = 1; n <= n_steps; ++n) // [cite: 284]
      {
        timer.restart(); // [cite: 284]
        time += dt; // Time at end of current step n [cite: 285]

        pcout << "\n====================================================" << std::endl; // [cite: 285]
        if (this_mpi_process == 0) {
          std::cout << "Step = " << n << "/" << n_steps << "   Time = " << time << std::endl;
        }
        pcout << "====================================================" << std::endl; // [cite: 287]

        // 1. Interpolate velocity (calculate u_star)
        pcout << "  Interpolating velocity..." << std::endl; // [cite: 287]
        interpolate_velocity(); // [cite: 288]

        // 2. Diffusion step (calculates tentative velocity u^{n+1})
        // Determine if preconditioners need reinitialization
        const bool reinit_diffusion_prec = (n == 1 || n % vel_update_prec == 0); // [cite: 288]
        pcout << "  Performing diffusion step (reinit prec=" << reinit_diffusion_prec << ")..." << std::endl; // [cite: 289]
        diffusion_step(time, reinit_diffusion_prec); // [cite: 290]

        // 3. Projection step (calculates pressure correction phi^{n+1})
        // Determine if preconditioner needs reinitialization (often just once)
        const bool reinit_projection_prec = (n == 1); // [cite: 290]
        pcout << "  Performing projection step (reinit prec=" << reinit_projection_prec << ")..." << std::endl; // [cite: 291]
        projection_step(reinit_projection_prec); // [cite: 292]

        // 4. Update pressure (calculates p^{n+1})
        // Determine if preconditioner needs reinitialization (often just once for rotational)
        const bool reinit_pressure_prec = (n == 1); // [cite: 292]
        pcout << "  Updating pressure (reinit prec=" << reinit_pressure_prec << ")..." << std::endl; // [cite: 293]
        update_pressure(reinit_pressure_prec); // [cite: 294]

        // 5. Update exact solution time if used for error calculation (Optional)
        // vel_exact.set_time(time); // [cite: 295]

        pcout << "Step " << n << " completed in " << timer.wall_time() << " s" << std::endl; // [cite: 295]

        // Output results periodically
        if ((n % output_interval == 0) || (n == n_steps)) // Output on interval or last step [cite: 296]
          {
             pcout << "\nOutputting solution (Step " << n << ", Time " << time << ")" << std::endl; // [cite: 296]
             output_results(n); // [cite: 297]
          }
      } // End time loop

    pcout << "\nTime loop finished." // [cite: 297]
          << std::endl; // [cite: 298]
  }

  // Explicit Instantiation (usually placed in a separate file or at the end of the main .cc if not splitting)
  // If you have navier_stokes_projection_instantiation.cc, remove this line.
  // template void NavierStokesProjection::run(const bool, const unsigned int);

} // namespace Step35