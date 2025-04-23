#include "../navier_stokes_projection.h" // Or correct path to header

#include "navier_stokes_projection_constructor.cc"
#include "navier_stokes_projection_create_triangulation_and_dofs.cc"
#include "navier_stokes_projection_initialize.cc"
#include "navier_stokes_projection_assemble_time_independent_matrices.cc"
#include "navier_stokes_projection_assemble_gradient_operator.cc"
#include "navier_stokes_projection_assemble_advection_term.cc"
#include "navier_stokes_projection_diffusion_step.cc"
#include "navier_stokes_projection_interpolate_velocity.cc"
#include "navier_stokes_projection_projection_step.cc"
#include "navier_stokes_projection_update_pressure.cc"
#include "navier_stokes_projection_output_results.cc"
#include "navier_stokes_projection_assemble_vorticity.cc"
#include "navier_stokes_projection_run.cc"

template class NERS570_proj::NavierStokesProjection<2>;