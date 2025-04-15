#include "../navier_stokes_projection.h" // Include the class declaration

// Include headers needed for this function
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h> // For pcout
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>

#include <fstream> // For std::ifstream
#include <string>  // For std::string
#include <vector>  // For std::vector
#include <memory>  // For std::make_unique

namespace Step35 {

  using namespace dealii;

  /**
   * @brief Creates the triangulation, distributes and renumbers DoFs for parallel computation.
   *
   * Reads the mesh from a UCD file ("nsbench2.inp" by default), performs global
   * refinements, partitions the mesh among MPI processes, distributes DoFs for
   * velocity and pressure spaces, renumbers DoFs using subdomain-wise numbering,
   * determines locally owned DoFs, creates parallel sparsity patterns, and
   * initializes PETSc matrices and vectors according to the parallel distribution.
   *
   * @param n_refines The number of global refinement steps to apply to the initial mesh.
   */
  template <int dim>
  void NavierStokesProjection<dim>::create_triangulation_and_dofs(const unsigned int n_refines)
  {
    pcout << "Creating triangulation and DoFs..." << std::endl; //

    // --- 1. Create Mesh (Identical on all processes) ---
    // Check for test case signal (e.g., n_global_refines == -1)
    if (n_refines == static_cast<unsigned int>(-1)) // Check if test case signaled
    {
      pcout << "  Creating hyper_cube mesh for test setup..." << std::endl;
      GridGenerator::hyper_cube(triangulation, -1, 1);
      // No global refinement needed here, or use a different parameter if you want refinement
      // triangulation.refine_global(0); // Or read a separate test_refines parameter
    }
    else // Normal case: Read from file
    {
      const std::string filename = mesh_filename; 
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);
      // const std::string filename = "nsbench2.inp"; //
      std::ifstream file(filename); //
      
      AssertThrow(file, ExcMessage("File not found: " + filename));
      grid_in.read_ucd(file);
      pcout << "  Read mesh from: " << filename << std::endl;
      pcout << "  Number of initial global refines = " << n_refines << std::endl;
      if (n_refines > 0) // Only refine if n_refines > 0
        triangulation.refine_global(n_refines);
    }
    /*
    GridIn<dim> grid_in; //
    grid_in.attach_triangulation(triangulation); //
    {
      // Consider making the filename a parameter if needed
      const std::string filename = mesh_filename; 
      // const std::string filename = "nsbench2.inp"; //
      std::ifstream file(filename); //
      // Assert only on rank 0 to avoid filesystem checks on all nodes? Or keep for safety.
      AssertThrow(file, ExcMessage("File not found: " + filename)); //
      grid_in.read_ucd(file); //
      pcout << "  Read mesh from: " << filename << std::endl; //
    }
    pcout << "  Number of initial global refines = " << n_refines << std::endl; //
    triangulation.refine_global(n_refines); //
    */
    pcout << "  Number of active cells: " << triangulation.n_active_cells() << std::endl; //

    // Store boundary IDs present in the mesh
    boundary_ids = triangulation.get_boundary_ids(); //

    // --- 2. Partition Mesh (Assign subdomain_id to cells) ---
    // Even with replicated mesh, this helps distribute assembly work later
    GridTools::partition_triangulation(n_mpi_processes, triangulation); //

    // --- 3. Distribute DoFs (Identical on all processes) ---
    dof_handler_velocity.distribute_dofs(fe_velocity); //
    dof_handler_pressure.distribute_dofs(fe_pressure); //

    // --- 4. Renumber DoFs (Crucial for parallel efficiency) ---
    pcout << "  Renumbering DoFs subdomain-wise..." << std::endl; //
    DoFRenumbering::subdomain_wise(dof_handler_velocity); //
    DoFRenumbering::subdomain_wise(dof_handler_pressure); //

    // --- 5. Determine Locally Owned DoFs ---
    const IndexSet vel_locally_owned_dofs =
        DoFTools::locally_owned_dofs_per_subdomain(dof_handler_velocity)[this_mpi_process]; //
    const IndexSet pres_locally_owned_dofs =
        DoFTools::locally_owned_dofs_per_subdomain(dof_handler_pressure)[this_mpi_process]; //

    const unsigned int n_vel_dofs  = dof_handler_velocity.n_dofs(); //
    const unsigned int n_pres_dofs = dof_handler_pressure.n_dofs(); //

    pcout << "  Total velocity DoFs: " << n_vel_dofs
               << " (local=" << vel_locally_owned_dofs.n_elements() << ")" << std::endl; //
    pcout << "  Total pressure DoFs: " << n_pres_dofs
               << " (local=" << pres_locally_owned_dofs.n_elements() << ")" << std::endl; //

    // --- 6. Create Sparsity Patterns (Needed for PETSc Matrix Init) ---
    pcout << "  Creating sparsity patterns..." << std::endl; //

    // Velocity-Velocity Pattern
    DynamicSparsityPattern dsp_vv(n_vel_dofs, n_vel_dofs, vel_locally_owned_dofs); //
    DoFTools::make_sparsity_pattern(dof_handler_velocity, dsp_vv); //
    SparsityTools::distribute_sparsity_pattern(dsp_vv,
                                              vel_locally_owned_dofs,
                                              mpi_communicator,
                                              vel_locally_owned_dofs); // Important
    sparsity_pattern_velocity.copy_from(dsp_vv); //

    // Pressure-Pressure Pattern
    DynamicSparsityPattern dsp_pp(n_pres_dofs, n_pres_dofs, pres_locally_owned_dofs); //
    DoFTools::make_sparsity_pattern(dof_handler_pressure, dsp_pp); //
    SparsityTools::distribute_sparsity_pattern(dsp_pp,
                                              pres_locally_owned_dofs,
                                              mpi_communicator,
                                              pres_locally_owned_dofs); // Important
    sparsity_pattern_pressure.copy_from(dsp_pp); //

    // Pressure-Velocity (Coupling) Pattern (Rows: Vel, Cols: Pres)
    // Note: SparsityTools::distribute_sparsity_pattern for non-square requires both row/col partitioning.
    // The original code commented this out; assuming standard DoFTools::make_sparsity_pattern is sufficient
    // for determining the non-zero structure needed for reinit, even without full distribution info?
    // Revisit this if matrix assembly/use fails for pres_Diff.
    DynamicSparsityPattern dsp_vp(n_vel_dofs, n_pres_dofs, vel_locally_owned_dofs); //
    DoFTools::make_sparsity_pattern(dof_handler_velocity, dof_handler_pressure, dsp_vp); //
    /* // Original commented-out distribution call:
     SparsityTools::distribute_sparsity_pattern(dsp_vp,
                                               DoFTools::locally_owned_dofs_per_subdomain(dof_handler_velocity), // Row distribution (velocity)
                                               DoFTools::locally_owned_dofs_per_subdomain(dof_handler_pressure), // Column distribution (pressure)
                                               mpi_communicator,
                                               vel_locally_owned_dofs,   // Rows relevant to this process
                                               pres_locally_owned_dofs); // Columns relevant for rows on this process
    */
    sparsity_pattern_pres_vel.copy_from(dsp_vp); //

    // --- 7. Initialize PETSc Matrices ---
    // Reinitializes matrices with the correct parallel layout based on sparsity patterns
    pcout << "  Initializing PETSc matrices..." << std::endl; //
    vel_Laplace_plus_Mass.reinit(vel_locally_owned_dofs, vel_locally_owned_dofs, sparsity_pattern_velocity, mpi_communicator); //
    vel_Mass.reinit(vel_locally_owned_dofs, vel_locally_owned_dofs, sparsity_pattern_velocity, mpi_communicator); //
    vel_Laplace.reinit(vel_locally_owned_dofs, vel_locally_owned_dofs, sparsity_pattern_velocity, mpi_communicator); //
    vel_Advection.reinit(vel_locally_owned_dofs, vel_locally_owned_dofs, sparsity_pattern_velocity, mpi_communicator); //
    pres_Laplace.reinit(pres_locally_owned_dofs, pres_locally_owned_dofs, sparsity_pattern_pressure, mpi_communicator); //
    pres_Mass.reinit(pres_locally_owned_dofs, pres_locally_owned_dofs, sparsity_pattern_pressure, mpi_communicator); //
    pres_iterative.reinit(pres_locally_owned_dofs, pres_locally_owned_dofs, sparsity_pattern_pressure, mpi_communicator); //

    for (unsigned int d = 0; d < dim; ++d) {
       // If using unique_ptr (as in your header suggestion)
       vel_it_matrix[d] = std::make_unique<PETScWrappers::MPI::SparseMatrix>(); //
       vel_it_matrix[d]->reinit(vel_locally_owned_dofs, vel_locally_owned_dofs, sparsity_pattern_velocity, mpi_communicator); //

       pres_Diff[d] = std::make_unique<PETScWrappers::MPI::SparseMatrix>(); //
       pres_Diff[d]->reinit(vel_locally_owned_dofs, pres_locally_owned_dofs, sparsity_pattern_pres_vel, mpi_communicator); //
    }


    // --- 8. Initialize PETSc Vectors ---
    // Reinitializes vectors with the correct parallel layout
    pcout << "  Initializing PETSc vectors..." << std::endl; //
    pres_n.reinit(pres_locally_owned_dofs, mpi_communicator); //
    pres_n_minus_1.reinit(pres_locally_owned_dofs, mpi_communicator); //
    phi_n.reinit(pres_locally_owned_dofs, mpi_communicator); //
    phi_n_minus_1.reinit(pres_locally_owned_dofs, mpi_communicator); //
    pres_tmp.reinit(pres_locally_owned_dofs, mpi_communicator); //

    v_tmp.reinit(vel_locally_owned_dofs, mpi_communicator); //
    rot_u.reinit(vel_locally_owned_dofs, mpi_communicator); //
    for (unsigned int d = 0; d < dim; ++d) {
      u_n[d].reinit(vel_locally_owned_dofs, mpi_communicator); //
      u_n_minus_1[d].reinit(vel_locally_owned_dofs, mpi_communicator); //
      u_star[d].reinit(vel_locally_owned_dofs, mpi_communicator); //
      force[d].reinit(vel_locally_owned_dofs, mpi_communicator); //
    }

    pcout << "  ...done creating triangulation and DoFs." << std::endl; //
  }

  // Explicit Instantiation
  // If you have navier_stokes_projection_instantiation.cc, remove this line.
  // template void NavierStokesProjection::create_triangulation_and_dofs(const unsigned int);

} // namespace Step35