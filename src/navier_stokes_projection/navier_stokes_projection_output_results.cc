#include "../navier_stokes_projection.h" // Include the class declaration

// Include headers needed for this function
#include <deal.II/base/conditional_ostream.h> // For pcout
#include <deal.II/base/utilities.h>        // For Utilities::int_to_string
#include <deal.II/dofs/dof_handler.h>      // For DoFHandler
#include <deal.II/fe/fe_system.h>          // For FESystem
#include <deal.II/grid/grid_tools.h>       // For GridTools::get_subdomain_association
#include <deal.II/lac/vector.h>            // For Vector<double> (localized data)
#include <deal.II/lac/petsc_vector.h>        // For copy constructor from PETSc vector
#include <deal.II/numerics/data_out.h>     // For DataOut
#include <deal.II/numerics/data_component_interpretation.h> // For DataComponentInterpretation

#include <fstream> // For std::ofstream
#include <string>  // For std::string
#include <vector>  // For std::vector

namespace NERS570_proj {

  using namespace dealii;

  /**
   * @brief Outputs the simulation results to a VTK file.
   *
   * This function generates a VTK (.vtk) file containing the velocity vector field,
   * the pressure scalar field, and the vorticity scalar field at a given time step.
   *
   * The process involves:
   * 1. Assembling the projected vorticity field `rot_u` (if not already up-to-date).
   * 2. Creating local copies (`Vector<double>`) of the distributed PETSc solution
   * vectors (`u_n`, `pres_n`, `rot_u`). This gathers all data onto each process.
   * 3. On MPI process 0 only:
   * a. Set up a joint FESystem and DoFHandler encompassing velocity, pressure,
   * and vorticity components.
   * b. Create a single "joint" solution vector (`Vector<double>`).
   * c. Iterate through cells and copy the localized data from `localized_u_n`,
   * `localized_pres_n`, and `localized_rot_u` into the correct slots of the
   * joint solution vector based on the joint DoF handler indexing.
   * d. Set up a DataOut object attached to the joint DoF handler.
   * e. Add the joint solution vector to DataOut, providing names and data component
   * interpretation (vector for velocity, scalars for pressure/vorticity).
   * f. Optionally add MPI partitioning information as cell data.
   * g. Build patches and write the data to a VTK file named "solution-xxxxx.vtk",
   * where xxxxx is the zero-padded step number.
   *
   * @param step The current time step number, used for naming the output file.
   */
  template <int dim>
  void NavierStokesProjection<dim>::output_results(const unsigned int step)
  {
    pcout << "  Generating output for step " << step << "..." << std::endl; //

    // --- 1. Assemble Vorticity (Parallel calculation) ---
    // Ensure vorticity is computed using the current velocity u_n.
    // Decide if preconditioner needs reinit (maybe only first time or if vel_Mass changed?)
    // Assume it might need reinit if called infrequently.
    const bool reinit_vorticity_prec = (step == 0); // Reinit only for initial output? Or always true?
    assemble_vorticity(reinit_vorticity_prec); // Calculates distributed rot_u

    // --- 2. Localize Distributed Data (All processes participate) ---
    // Create local Vector<double> copies of the necessary PETSc solution vectors.
    // The Vector copy constructor implicitly handles the parallel communication (gather).
    pcout << "    Localizing data for output..." << std::endl; //
    Vector<double> localized_pres_n(pres_n); // Local copy of pressure
    Vector<double> localized_rot_u(rot_u);   // Local copy of vorticity
    std::vector< Vector<double> > localized_u_n(dim); // Local copies of velocity components
    for (unsigned int d = 0; d < dim; ++d)
        localized_u_n[d] = Vector<double>(u_n[d]); //

    // --- 3. Generate Output only on Rank 0 ---
    if (this_mpi_process == 0) //
      {
        pcout << "    Rank 0: Processing data for output..." << std::endl; //

        // --- 3a. Setup Joint FE System and DoF Handler ---
        // Create a single FE system to hold all output variables:
        // Velocity (dim components), Pressure (1 component), Vorticity (1 component)
        const FESystem<dim> joint_fe(fe_velocity, dim, // Velocity element, dim components
                                     fe_pressure, 1,  // Pressure element, 1 component
                                     fe_velocity, 1); // Vorticity element (using vel space), 1 component

        // Create a DoF handler for this combined system based on the existing triangulation
        DoFHandler<dim> joint_dof_handler(triangulation);
        joint_dof_handler.distribute_dofs(joint_fe); //

        // --- 3b. Create and Fill Joint Solution Vector ---
        // This vector will hold all data ordered according to joint_dof_handler
        Vector<double> joint_solution(joint_dof_handler.n_dofs()); //

        // Helper vectors for DoF indices
        std::vector<types::global_dof_index> joint_loc_dof_indices(joint_fe.n_dofs_per_cell());
        std::vector<types::global_dof_index> vel_loc_dof_indices(fe_velocity.n_dofs_per_cell());
        std::vector<types::global_dof_index> pres_loc_dof_indices(fe_pressure.n_dofs_per_cell());

        // --- 3c. Iterate over cells and copy data ---
        // Need iterators for the joint handler and the original handlers
        typename DoFHandler<dim>::active_cell_iterator
          joint_cell = joint_dof_handler.begin_active(),
          joint_endc = joint_dof_handler.end(),
          vel_cell   = dof_handler_velocity.begin_active(), // Assumes sync'd iteration
          pres_cell  = dof_handler_pressure.begin_active(); // Assumes sync'd iteration

        for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell, ++pres_cell) //
          {
            // Get DoF indices for the current cell from all relevant handlers
            joint_cell->get_dof_indices(joint_loc_dof_indices); //
            vel_cell->get_dof_indices(vel_loc_dof_indices);     //
            pres_cell->get_dof_indices(pres_loc_dof_indices);  //

            // Loop over DoFs within the cell for the joint system
            for (unsigned int i = 0; i < joint_fe.n_dofs_per_cell(); ++i) //
              {
                 // Figure out which original variable this joint DoF corresponds to
                 const unsigned int base_idx   = joint_fe.system_to_base_index(i).second; // Index within the base element
                 const unsigned int system_idx = joint_fe.system_to_base_index(i).first.first; // Index of the system (0=vel, 1=pres, 2=vort)
                 const unsigned int comp_idx   = joint_fe.system_to_base_index(i).first.second; // Component index within the system

                 // Copy data from the appropriate localized vector
                 switch (system_idx) //
                   {
                     case 0: // Velocity component
                       AssertIndexRange(comp_idx, dim); // Sanity check
                       // Copy from localized_u_n[component] using velocity DoF indices
                       joint_solution(joint_loc_dof_indices[i]) =
                         localized_u_n[comp_idx](vel_loc_dof_indices[base_idx]); //
                       break; //
                     case 1: // Pressure component
                       AssertIndexRange(comp_idx, 1); // Sanity check
                       // Copy from localized_pres_n using pressure DoF indices
                       joint_solution(joint_loc_dof_indices[i]) =
                         localized_pres_n(pres_loc_dof_indices[base_idx]); //
                       break; //
                     case 2: // Vorticity component
                       AssertIndexRange(comp_idx, 1); // Sanity check
                       // Copy from localized_rot_u using velocity DoF indices (since vorticity uses fe_velocity)
                       joint_solution(joint_loc_dof_indices[i]) =
                         localized_rot_u(vel_loc_dof_indices[base_idx]); //
                       break; //
                     default: // Should not happen
                       DEAL_II_ASSERT_UNREACHABLE(); //
                   } // End switch
               } // End loop over cell DoFs (i)
          } // End cell loop

        // --- 3d. Setup DataOut ---
        // Names for the data vectors in the output file
        std::vector<std::string> solution_names(dim, "velocity"); // "velocity", "velocity", ...
        solution_names.emplace_back("pressure"); //
        solution_names.emplace_back("vorticity"); //

        // Specify how components should be interpreted (vector or scalar)
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
           component_interpretation(dim + 2, DataComponentInterpretation::component_is_part_of_vector); // Default to vector parts
        component_interpretation[dim]     = DataComponentInterpretation::component_is_scalar; // Pressure is scalar
        component_interpretation[dim + 1] = DataComponentInterpretation::component_is_scalar; // Vorticity is scalar

        DataOut<dim> data_out; //
        data_out.attach_dof_handler(joint_dof_handler); // Use the combined handler

        // --- 3e. Add data vectors to DataOut ---
        data_out.add_data_vector(joint_solution,           // The combined solution data
                                 solution_names,           // Names for the fields
                                 DataOut<dim>::type_dof_data, // Data type
                                 component_interpretation); // How to interpret components

        // --- 3f. Add partitioning information (optional) ---
        // Get subdomain ID for each cell
        std::vector<unsigned int> partition_int(triangulation.n_active_cells()); //
        GridTools::get_subdomain_association(triangulation, partition_int); //
        // Convert to Vector<double> for DataOut
        const Vector<double> partitioning(partition_int.begin(), partition_int.end()); //
        data_out.add_data_vector(partitioning, "partitioning", DataOut<dim>::type_cell_data); // Add as cell data

        // --- 3g. Write Output File ---
        data_out.build_patches(deg + 1); // Build visualization patches (use velocity degree)
        // Construct filename with zero-padded step number
        const std::string filename = "solution-" + Utilities::int_to_string(step, 5) + ".vtk"; //
        std::ofstream output_file(filename); //
        data_out.write_vtk(output_file); // Write the VTK file
        pcout << "    Output written to " << filename << std::endl; //
      } // end if (this_mpi_process == 0)

    // Optional barrier: Ensure all processes wait here if subsequent steps
    // immediately depend on the output file existing (usually not needed).
    // Utilities::MPI::barrier(mpi_communicator);

    pcout << "  ...output generation finished." << std::endl; //
  }


} // namespace NERS570_proj