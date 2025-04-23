#include "../navier_stokes_projection.h" // Include the class declaration

// Include headers specifically needed for the run function implementation
#include <deal.II/base/timer.h>          // For Timer
#include <deal.II/base/conditional_ostream.h> // For pcout

#include <vector>                        // For std::vector
#include <cmath>                         // For static_cast

namespace NERS570_proj {

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
    pcout.set_condition(verbose); // 

    // Initial Setup Steps (called from constructor)
    // create_triangulation_and_dofs()
    // initialize()

    // Assemble matrices that don't change in time loop
    pcout << "Running initial assembly..." << std::endl; // 
    assemble_time_independent_matrices(); // Assemble Mass, Laplace 
    assemble_gradient_operator();        // Assemble Gradient Operator 
    // Advection term is assembled inside diffusion_step

    // Time Loop Setup
    const unsigned int n_steps = static_cast<unsigned int>((T - t_0) / dt); // 
    double time = t_0; // Start time 
    //pcout << "Starting time loop: T = " << T << ", dt = " << dt << ", N_steps = " << n_steps << std::endl; // 
    if (this_mpi_process == 0) {
      std::cout << "Starting time loop: T = " << T << ", dt = " << dt << ", N_steps = " << n_steps << std::endl;
    }
    // Output Initial Condition (Step 0)
    // Ensure correct time for initial output if needed
    time = t_0; // 
    //pcout << "\nOutputting initial solution (Step 0, Time " << time << ")" << std::endl; // 
    if (this_mpi_process == 0) {
      std::cout << "\nOutputting initial solution (Step 0, Time " << time << ")" << std::endl;
    }
    output_results(0); // Output initial state at t_0 

    Timer timer; // 

    // Main Time Loop
    // Start from step n=1 to reach final time T = t_0 + n_steps * dt
    for (unsigned int n = 1; n <= n_steps; ++n) // 
      {
        timer.restart(); // 
        time += dt; // Time at end of current step n 

        pcout << "\n====================================================" << std::endl; // 
        if (this_mpi_process == 0) {
          std::cout << "Step = " << n << "/" << n_steps << "   Time = " << time << std::endl;
        }
        pcout << "====================================================" << std::endl; // 

        // 1. Interpolate velocity (calculate u_star)
        pcout << "  Interpolating velocity..." << std::endl; // 
        interpolate_velocity(); // 

        // 2. Diffusion step (calculates tentative velocity u^{n+1})
        // Determine if preconditioners need reinitialization
        const bool reinit_diffusion_prec = (n == 1 || n % vel_update_prec == 0); // 
        pcout << "  Performing diffusion step (reinit prec=" << reinit_diffusion_prec << ")..." << std::endl; // 
        diffusion_step(time, reinit_diffusion_prec); // 

        // 3. Projection step (calculates pressure correction phi^{n+1})
        // Determine if preconditioner needs reinitialization (often just once)
        const bool reinit_projection_prec = (n == 1); // 
        pcout << "  Performing projection step (reinit prec=" << reinit_projection_prec << ")..." << std::endl; // 
        projection_step(reinit_projection_prec); // 

        // 4. Update pressure (calculates p^{n+1})
        // Determine if preconditioner needs reinitialization (often just once for rotational)
        const bool reinit_pressure_prec = (n == 1); // 
        pcout << "  Updating pressure (reinit prec=" << reinit_pressure_prec << ")..." << std::endl; // 
        update_pressure(reinit_pressure_prec); // 

        // 5. Update exact solution time if used for error calculation (Optional)
        // vel_exact.set_time(time); // 

        pcout << "Step " << n << " completed in " << timer.wall_time() << " s" << std::endl; // 

        // Output results periodically
        if ((n % output_interval == 0) || (n == n_steps)) // Output on interval or last step 
          {
             pcout << "\nOutputting solution (Step " << n << ", Time " << time << ")" << std::endl; // 
             output_results(n); // 
          }
      } // End time loop

    pcout << "\nTime loop finished." // 
          << std::endl; // 
  }

} // namespace NERS570_proj