/**
# Decaying two-dimensional turbulence

We solve the two-dimensional incompressible Euler equations using a
vorticity--streamfunction formulation. */

#include "grid/multigrid.h"
#include "navier-stokes/stream.h"

#include "fluidlab_pack.h"

/**
The domain is square of size unity by default. The resolution is
constant at $256^2$. */
double domain_size=1.;
int num_cells=256;

int main() {
  init_grid (256);

  OpenSimulationFolder("euler%g", domain_size);
  run();
}

/**
The initial condition for vorticity is just a white noise in the range
$[-1:1]$ .*/

event init (i = 0) {
  foreach()
    omega[] = noise();
}

/**
We generate images of the vorticity field every 4 timesteps up to
$t=1000$. We fix the colorscale to $[-0.3:0.3]$.

![Evolution of the vorticity](turbulence/omega.mp4)(autoplay loop) */

event output (i += 4; t <= 1000) {

  fprintf(stderr, "Time: %g, dt: %g\n", t, dt);
  output_ppm (omega, min = -0.3, max = 0.3, file = "omega.mp4");
}


//event output_binary_files(t+=1) {
event output_binary_files (i += 1; t <= 500) {
//event output_binary_files(t+=1) {
    // List of fields to print and the name to be given for each of them
    //scalar *list = {Ai, A2};
    //const char *list_names[] = {"Ai", "A2"};

    scalar *list = {omega, psi};
    const char *list_names[] = {"omega", "psi"};

    // Dumping the requested fields into a uniform_grid format
    PrintMeshDataDump(i, t, num_cells, domain_size, list, list_names);
}