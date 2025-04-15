/**
 * @file run_time_parameters.cc
 * @brief Implements methods for the runtime parameter storage class.
 */

#include "run_time_parameters.h" // Class declaration
#include <fstream>                // For std::ifstream
#include <deal.II/base/utilities.h> // Not strictly needed here if only ParameterHandler uses it
#include <deal.II/base/exceptions.h> // For AssertThrow, ExcFileNotOpen

namespace Step35 {
  namespace RunTimeParameters {

    /**
     * @brief Constructor for Data_Storage.
     *
     * Declares all runtime parameters using the ParameterHandler `prm`,
     * setting default values, patterns (e.g., ranges, selections), and
     * descriptive help strings that will appear if the ParameterHandler
     * writes a template parameter file.
     */
    Data_Storage::Data_Storage()
      : // Initialize default values for member variables
        form(Method::rotational), //
        dt(5e-4), //
        initial_time(0.), //
        final_time(1.), //
        Reynolds(1.), //
        n_global_refines(0), //
        pressure_degree(1), //
        vel_max_iterations(1000), //
        vel_Krylov_size(30), //
        vel_off_diagonals(60), // (ILU parameter)
        vel_update_prec(15), //
        vel_eps(1e-12), //
        vel_diag_strength(0.01), // (ILU parameter)
        verbose(true), //
        output_interval(15) // (Original default was 1, changed to match constructor init) [cite: 365, 348]
    {
      // --- Declare parameters using ParameterHandler ---
      prm.declare_entry("Method_Form", // Parameter name in file
                        "rotational",  // Default value
                        dealii::Patterns::Selection("rotational|standard"), // Allowed values
                        "Selects the pressure update method: 'rotational' or 'standard'."); // Description

      // Group related parameters into subsections for clarity in the parameter file
      prm.enter_subsection("Physical data"); //
      {
        prm.declare_entry("initial_time", //
                          "0.", //
                          dealii::Patterns::Double(0.), // Allows non-negative doubles
                          "The initial time of the simulation."); //
        prm.declare_entry("final_time", //
                          "1.", //
                          dealii::Patterns::Double(0.), // Allows non-negative doubles
                          "The final time of the simulation."); //
        prm.declare_entry("Reynolds", //
                          "1.", //
                          dealii::Patterns::Double(0.), // Allows non-negative doubles
                          "The Reynolds number."); //
      }
      prm.leave_subsection(); //

      prm.enter_subsection("Time step data"); //
      {
        prm.declare_entry("dt", //
                          "5e-4", //
                          dealii::Patterns::Double(0.), // Allows non-negative doubles
                          "The time step size."); //
      }
      prm.leave_subsection(); //

      prm.enter_subsection("Problem definition"); // 
      {
          // Add declaration for mesh filename
          prm.declare_entry("Mesh filename",                  // Parameter name
                            "nsbench2.inp",                   // Default value
                            dealii::Patterns::FileName(),             // Input pattern (checks if it looks like a filename)
                            "Name of the mesh file to read."); // Description
      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization"); //
      {
        prm.declare_entry("n_of_refines", //
                          "0", //
                          dealii::Patterns::Integer(0, 15), // Allows integers from 0 to 15
                          "The number of global refines applied to the mesh."); //
        prm.declare_entry("pressure_fe_degree", //
                          "1", //
                          dealii::Patterns::Integer(1, 5), // Allows integers from 1 to 5
                          "The polynomial degree 'p' for the pressure space Q_p (velocity is Q_{p+1})."); //
      }
      prm.leave_subsection(); //

      prm.enter_subsection("Data solve velocity"); //
      {
        prm.declare_entry("max_iterations", //
                          "1000", //
                          dealii::Patterns::Integer(1), // Allows positive integers
                          "The maximal number of iterations for the velocity solver (GMRES)."); //
        prm.declare_entry("eps", //
                          "1e-12", //
                          dealii::Patterns::Double(0.), // Allows non-negative doubles
                          "The relative stopping tolerance for the velocity solver."); //
        prm.declare_entry("Krylov_size", //
                          "30", //
                          dealii::Patterns::Integer(1), // Allows positive integers
                          "The size of the Krylov subspace for GMRES (restart parameter)."); //
        prm.declare_entry("off_diagonals", // (Parameter for ILU-type preconditioners)
                          "60", //
                          dealii::Patterns::Integer(0), // Allows non-negative integers
                          "The number of off-diagonal elements ILU must compute (if using ILU)."); //
        prm.declare_entry("diag_strength", // (Parameter for ILU-type preconditioners)
                          "0.01", //
                          dealii::Patterns::Double(0.), // Allows non-negative doubles
                          "Diagonal strengthening coefficient (if using ILU)."); //
        prm.declare_entry("update_prec", //
                          "15", //
                          dealii::Patterns::Integer(1), // Allows positive integers
                          "Update the velocity preconditioner every 'update_prec' time steps."); //
      }
      prm.leave_subsection(); //

      prm.declare_entry("verbose", //
                        "true", //
                        dealii::Patterns::Bool(), // Allows true/false
                        "Enable verbose output during the simulation run."); //
      prm.declare_entry("output_interval", //
                        "15", // (Changed default to match constructor init) [cite: 365, 348]
                        dealii::Patterns::Integer(1), // Allows positive integers
                        "Output solution files every 'output_interval' time steps."); //
    }


    /**
     * @brief Implementation of read_data.
     *
     * Opens the specified file, uses the ParameterHandler `prm` to parse it,
     * and then retrieves the parsed values into the corresponding public
     * member variables of the Data_Storage object.
     * @param filename Path to the parameter file.
     */
    void Data_Storage::read_data(const std::string &filename)
    {
      std::ifstream file(filename); //
      // Throw exception if file cannot be opened
      AssertThrow(file, dealii::ExcFileNotOpen(filename)); //

      // Parse the file using the declared parameters
      prm.parse_input(file); //

      // --- Retrieve parsed values into member variables ---
      // Retrieve Method_Form and convert string to enum
      if (prm.get("Method_Form") == std::string("rotational")) //
        form = Method::rotational; //
      else
        form = Method::standard; //

      // Retrieve values from subsections
      prm.enter_subsection("Physical data"); //
      {
        initial_time = prm.get_double("initial_time"); //
        final_time   = prm.get_double("final_time"); //
        Reynolds     = prm.get_double("Reynolds"); //
      }
      prm.leave_subsection(); //

      prm.enter_subsection("Time step data"); //
      {
        dt = prm.get_double("dt"); //
      }
      prm.leave_subsection(); //

      prm.enter_subsection("Problem definition"); // 
      {
      mesh_filename = prm.get("Mesh filename");

      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization"); //
      {
        n_global_refines = prm.get_integer("n_of_refines"); //
        pressure_degree  = prm.get_integer("pressure_fe_degree"); //
      }
      prm.leave_subsection(); //

      prm.enter_subsection("Data solve velocity"); //
      {
        vel_max_iterations = prm.get_integer("max_iterations"); //
        vel_eps            = prm.get_double("eps"); //
        vel_Krylov_size    = prm.get_integer("Krylov_size"); //
        vel_off_diagonals  = prm.get_integer("off_diagonals"); //
        vel_diag_strength  = prm.get_double("diag_strength"); //
        vel_update_prec    = prm.get_integer("update_prec"); //
      }
      prm.leave_subsection(); //

      // Retrieve remaining top-level parameters
      verbose = prm.get_bool("verbose"); //
      output_interval = prm.get_integer("output_interval"); //
    }

  } // namespace RunTimeParameters
} // namespace Step35