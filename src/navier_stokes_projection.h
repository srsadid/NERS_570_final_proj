#ifndef NAVIER_STOKES_PROJECTION_H
#define NAVIER_STOKES_PROJECTION_H

// Base utilities
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h> // For pcout
#include <deal.II/base/timer.h>              // Added for timing in run()

// MPI/Parallel utilities
#include <deal.II/base/mpi.h>                // For MPI_Comm, Utilities::MPI::*

// PETSc Wrappers (replace serial lac)
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h> // Includes BlockJacobi etc.

// Grid
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>         // For partitioning

// DoFs
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>   // For subdomain_wise

// FE
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

// Numerics
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_component_interpretation.h> // For output

// Standard Libs
#include <fstream>
#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <memory> // For unique_ptr

// Project specific headers
#include "run_time_parameters.h"
#include "equation_data.h"

namespace NERS570_proj {

  using namespace dealii;

  template <int dim>
  class NavierStokesProjection
  {
  public:
    // Constructor takes runtime parameters
    NavierStokesProjection(const RunTimeParameters::Data_Storage &data);
    unsigned int get_n_vel_dofs() const { return dof_handler_velocity.n_dofs(); }
    unsigned int get_n_pres_dofs() const { return dof_handler_pressure.n_dofs(); }

    // Main driver function
    void run(const bool verbose = false, const unsigned int output_interval = 10);

  protected: // Changed access to protected for core methods
    // ======================================================
    // ======== MPI and Parallel Setup Members ==============
    // ======================================================
    MPI_Comm mpi_communicator;      // MPI communicator
    unsigned int n_mpi_processes;   // Total number of processes
    unsigned int this_mpi_process;  // Rank of this process
    ConditionalOStream pcout;       // Parallel output stream

    // ======================================================
    // ======== Simulation & FE Setup Members ===============
    // ======================================================
    RunTimeParameters::Method type; // Standard or Rotational projection
    const unsigned int deg;         // Polynomial degree for pressure
    const double dt;                // Time step size
    const double t_0;               // Initial time
    const double T;                 // Final time
    const double Re;                // Reynolds number

    const std::string mesh_filename; // mesh file name

    EquationData::Velocity<dim> vel_exact; // For initial/boundary conditions

    // Store boundary IDs from mesh file
    std::vector<types::boundary_id> boundary_ids;

    // Triangulation (replicated across processes)
    Triangulation<dim> triangulation;

    // Finite Elements
    const FE_Q<dim> fe_velocity;    // Velocity element (Q_{p+1})
    const FE_Q<dim> fe_pressure;    // Pressure element (Q_p)

    // DoF Handlers
    DoFHandler<dim> dof_handler_velocity;
    DoFHandler<dim> dof_handler_pressure;

    // Quadrature Rules
    const QGauss<dim> quadrature_pressure;
    const QGauss<dim> quadrature_velocity;

    // Constraints (for hanging nodes)
    AffineConstraints<double> hanging_node_constraints;

    // Sparsity Patterns (needed for PETSc matrix init with replicated mesh)
    SparsityPattern sparsity_pattern_velocity;
    SparsityPattern sparsity_pattern_pressure;
    SparsityPattern sparsity_pattern_pres_vel; // Rows: Vel, Cols: Pres

    // ======================================================
    // ======== PETSc Data Structures =======================
    // ======================================================
    // Matrices
    PETScWrappers::MPI::SparseMatrix vel_Laplace_plus_Mass; // Combined matrix for diffusion step
    std::vector< std::unique_ptr<PETScWrappers::MPI::SparseMatrix> > vel_it_matrix; // Iteration matrix per component [dim]

    PETScWrappers::MPI::SparseMatrix vel_Mass;      // Velocity mass matrix
    PETScWrappers::MPI::SparseMatrix vel_Laplace;   // Velocity Laplace matrix
    PETScWrappers::MPI::SparseMatrix vel_Advection; // Velocity advection matrix

    PETScWrappers::MPI::SparseMatrix pres_Laplace;  // Pressure Laplace matrix
    PETScWrappers::MPI::SparseMatrix pres_Mass;     // Pressure mass matrix
    std::vector< std::unique_ptr<PETScWrappers::MPI::SparseMatrix> > pres_Diff; // Gradient operator matrices [dim]
    PETScWrappers::MPI::SparseMatrix pres_iterative; // Iteration matrix for projection step

    // Vectors
    PETScWrappers::MPI::Vector pres_n;          // Pressure at time t_n
    PETScWrappers::MPI::Vector pres_n_minus_1;  // Pressure at time t_{n-1}
    PETScWrappers::MPI::Vector phi_n;           // Pressure correction at t_n
    PETScWrappers::MPI::Vector phi_n_minus_1;   // Pressure correction at t_{n-1}

    std::vector< PETScWrappers::MPI::Vector > u_n;         // Velocity at t_n [dim]
    std::vector< PETScWrappers::MPI::Vector > u_n_minus_1; // Velocity at t_{n-1} [dim]
    std::vector< PETScWrappers::MPI::Vector > u_star;      // Intermediate velocity [dim]
    std::vector< PETScWrappers::MPI::Vector > force;       // RHS for diffusion step [dim]

    PETScWrappers::MPI::Vector v_tmp;           // Temporary vector for velocity space
    PETScWrappers::MPI::Vector pres_tmp;        // Temporary vector for pressure space (e.g., projection RHS)
    PETScWrappers::MPI::Vector rot_u;           // Vorticity projected onto FE space

    // Preconditioners
    // *** IMPORTANT: Replace PreconditionBlockJacobi with more effective PETSc preconditioners ***
    // *** (e.g., wrappers for HYPRE BoomerAMG, SOR, ICC) for better performance!        ***
    std::vector< PETScWrappers::PreconditionBlockJacobi > prec_velocity; // Preconditioner for diffusion step [dim]
    PETScWrappers::PreconditionBlockJacobi prec_pres_Laplace; // Preconditioner for projection step
    PETScWrappers::PreconditionBlockJacobi prec_mass;         // Preconditioner for pressure mass matrix solve
    PETScWrappers::PreconditionBlockJacobi prec_vel_mass;     // Preconditioner for velocity mass matrix solve (vorticity)


    // ======================================================
    // ======== Method Declarations =========================
    // ======================================================

    // Setup methods (called from constructor or run)
    void create_triangulation_and_dofs(const unsigned int n_refines);
    void initialize(); // Initializes vector states
    void assemble_time_independent_matrices(); // New: Assembles M, L matrices
    void assemble_gradient_operator();        // New: Assembles pres_Diff matrices

    // Core time stepping methods
    void interpolate_velocity();              // Calculate u_star
    void assemble_advection_term();           // Assembles vel_Advection (called by diffusion_step)
    void diffusion_step(const double current_time,const bool reinit_prec); // Solve for tentative velocity
    void projection_step(const bool reinit_prec); // Solve for pressure correction phi_n
    void update_pressure(const bool reinit_prec); // Update pressure p_n

    // Helper for diffusion solve
    void diffusion_component_solve(const unsigned int d);

    // Postprocessing & Output
    void assemble_vorticity(const bool reinit_prec = true); // Calculates vorticity rot_u
    void output_results(const unsigned int step);

private: // Private members, e.g., solver parameters
    // Solver parameters (consider grouping into a struct)
    unsigned int vel_max_its;
    unsigned int vel_Krylov_size; // Used for GMRES restart
    unsigned int vel_off_diagonals; // Parameter for some preconditioners (e.g., ILU)
    unsigned int vel_update_prec; // How often to reinit preconditioner
    double       vel_eps;           // Relative tolerance for velocity solves
    double       vel_diag_strength; // Parameter for some preconditioners (e.g., ILU)

    // --- Scratch Data Structs (Used internally by assembly loops) ---
    // These are kept as they are useful for organizing FEValues etc.
    // The 'PerTask' structs are less relevant without WorkStream but harmless.

    // For Gradient Assembly
    struct InitGradPerTaskData; // Forward declaration if needed, or keep definition
    struct InitGradScratchData
    {
      unsigned int  nqp;
      FEValues<dim> fe_val_vel;
      FEValues<dim> fe_val_pres;
      InitGradScratchData(const FE_Q<dim>   &fe_v,
                          const FE_Q<dim>   &fe_p,
                          const QGauss<dim> &quad,
                          const UpdateFlags  flags_v,
                          const UpdateFlags  flags_p)
        : nqp(quad.size()),
          fe_val_vel(fe_v, quad, flags_v),
          fe_val_pres(fe_p, quad, flags_p)
      {}
      // Copy constructor needed if used in contexts requiring copies
      InitGradScratchData(const InitGradScratchData &data);
    };
    // We removed the separate assembly/copy functions for gradient

    // For Advection Assembly
    struct AdvectionPerTaskData; // Forward declaration if needed, or keep definition
    struct AdvectionScratchData
    {
      unsigned int                nqp;
      unsigned int                dpc;
      std::vector<Point<dim>>     u_star_local; // Stores u* vector at quadrature points
      std::vector<Tensor<1, dim>> grad_u_star;  // Stores gradient of one component of u*
      std::vector<double>         u_star_tmp;   // Temporary storage (e.g., for divergence)
      FEValues<dim>               fe_val;
      AdvectionScratchData(const FE_Q<dim>   &fe,
                           const QGauss<dim> &quad,
                           const UpdateFlags  flags)
        : nqp(quad.size()),
          dpc(fe.n_dofs_per_cell()),
          u_star_local(nqp),
          grad_u_star(nqp), // Size is per-component, maybe resize inside? Check usage.
          u_star_tmp(nqp),
          fe_val(fe, quad, flags)
      {}
       // Copy constructor needed if used in contexts requiring copies
      AdvectionScratchData(const AdvectionScratchData &data);
    };
     // We removed the separate assembly/copy functions for advection

  }; // end class NavierStokesProjection


   // --- Implementation details for ScratchData copy constructors ---
   // Need definitions if they are actually copied (e.g. if stored by value in std::vector)
   // Provide definitions matching the ones in your original header, likely needed by FEValues copy/move semantics.
   template <int dim>
   NavierStokesProjection<dim>::InitGradScratchData::InitGradScratchData(const InitGradScratchData &data)
       : nqp(data.nqp)
       , fe_val_vel(data.fe_val_vel.get_fe(),
                    data.fe_val_vel.get_quadrature(),
                    data.fe_val_vel.get_update_flags())
       , fe_val_pres(data.fe_val_pres.get_fe(),
                     data.fe_val_pres.get_quadrature(),
                     data.fe_val_pres.get_update_flags())
     {}

    template <int dim>
    NavierStokesProjection<dim>::AdvectionScratchData::AdvectionScratchData(const AdvectionScratchData &data)
       : nqp(data.nqp)
       , dpc(data.dpc)
       , u_star_local(data.u_star_local) // Simple copy ok?
       , grad_u_star(data.grad_u_star)   // Simple copy ok?
       , u_star_tmp(data.u_star_tmp)     // Simple copy ok?
       , fe_val(data.fe_val.get_fe(),
                data.fe_val.get_quadrature(),
                data.fe_val.get_update_flags())
     {}
//template class NavierStokesProjection<2>;
} // namespace NERS570_proj

#endif // NAVIER_STOKES_PROJECTION_H