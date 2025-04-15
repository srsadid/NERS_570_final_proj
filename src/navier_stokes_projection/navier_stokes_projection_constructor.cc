#include "../navier_stokes_projection.h" // Include the class declaration (adjust path if needed)

// Include headers specifically needed for the constructor implementation
#include <deal.II/base/utilities.h>        // For Utilities::MPI::*
#include <deal.II/base/conditional_ostream.h> // For ConditionalOStream
#include <deal.II/base/exceptions.h>       // For ExcMessage, AssertThrow
#include <string>                          // For std::to_string

namespace Step35 {

  using namespace dealii;

  /**
   * @brief Constructor for the NavierStokesProjection class.
   *
   * Initializes MPI communication, sets up simulation parameters based on the
   * input data, initializes finite element objects, DoF handlers, quadrature rules,
   * and resizes vectors holding PETSc objects. It also calls methods to create the
   * triangulation and DoFs, and initializes the simulation state.
   *
   * @param data A const reference to a Data_Storage object containing runtime
   * parameters read from the parameter file.
   */
  template <int dim>
  NavierStokesProjection<dim>::NavierStokesProjection(const RunTimeParameters::Data_Storage &data)
    : // MPI/Parallel Initializations
      mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      pcout(std::cout, (this_mpi_process == 0)), // Conditional output stream for rank 0

      // Simulation Parameters
      type(data.form),             // Projection method type (Standard or Rotational)
      deg(data.pressure_degree),   // Polynomial degree for pressure FE
      dt(data.dt),                 // Time step size
      t_0(data.initial_time),      // Initial time
      T(data.final_time),          // Final time
      Re(data.Reynolds),           // Reynolds number
      mesh_filename(data.mesh_filename), // Mesh File 
      vel_exact(data.initial_time),// Exact velocity function (for BC/IC)

      // FE, DoF Handlers, Quadrature
      fe_velocity(deg + 1),        // Velocity FE (Q_{p+1})
      fe_pressure(deg),            // Pressure FE (Q_p)
      dof_handler_velocity(triangulation), // Velocity DoF handler (attached to triangulation)
      dof_handler_pressure(triangulation), // Pressure DoF handler (attached to triangulation)
      quadrature_pressure(deg + 1), // Quadrature for pressure terms
      quadrature_velocity(deg + 2), // Quadrature for velocity terms

      // Resize PETSc Vector Containers (std::vector<unique_ptr> or std::vector<Vector>)
      vel_it_matrix(dim),                      // Iteration matrix per velocity component
      pres_Diff(dim),                          // Gradient operator matrix per component
      u_n(dim),                                // Velocity at time t_n
      u_n_minus_1(dim),                        // Velocity at time t_{n-1}
      u_star(dim),                             // Intermediate velocity
      force(dim),                              // RHS vector for velocity solve
      prec_velocity(dim),                       // Preconditioner per velocity component
      // Note: Actual PETSc objects (Matrices/Vectors) are initialized/reinitialized later.

      // Solver Parameters
      vel_max_its(data.vel_max_iterations),    // Max iterations for velocity solve
      vel_Krylov_size(data.vel_Krylov_size),   // GMRES Krylov subspace size
      vel_off_diagonals(data.vel_off_diagonals),// Parameter for ILU preconditioner
      vel_update_prec(data.vel_update_prec),   // Frequency for updating preconditioner
      vel_eps(data.vel_eps),                   // Relative tolerance for velocity solve
      vel_diag_strength(data.vel_diag_strength) // Parameter for ILU preconditioner


  {
    // Check for unstable FE pair (Taylor-Hood condition Q_{p+1}/Q_p is generally stable for p>=1)
    if (deg < 1)
      pcout << "WARNING: The chosen pair of finite element spaces (Q"
            << deg + 1 << "/Q" << deg << ") may not be stable. "
            << "The obtained results might be unreliable." << std::endl;

    // Validate time step
    AssertThrow(!((dt <= 0.) || (dt > T)), ExcMessage("Invalid time step dt=" + std::to_string(dt) + ". Must be 0 < dt <= T."));

    // Create mesh and distribute DoFs
    create_triangulation_and_dofs(data.n_global_refines);

    // Initialize vector fields (velocity, pressure, phi)
    initialize();

    // Log initialization details on rank 0
    pcout << "Initialized NavierStokesProjection:" << std::endl;
    pcout << "  MPI Processes: " << n_mpi_processes << std::endl;
    pcout << "  FE Spaces: V=Q" << deg+1 << ", P=Q" << deg << std::endl;
    pcout << "  Time step (dt): " << dt << std::endl;
    pcout << "  Time interval: [" << t_0 << ", " << T << "]" << std::endl;
    pcout << "  Reynolds number (Re): " << Re << std::endl;
    pcout << "  Projection Type: " << (type == RunTimeParameters::Method::rotational ? "Rotational" : "Standard") << std::endl;
  }

} // namespace Step35