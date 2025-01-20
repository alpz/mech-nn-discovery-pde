#include "grid/multigrid.h"
#include "navier-stokes/centered.h"

int main()
{ 
  // coordinates of lower-left corner
  origin (-0.5, -0.5);
  // number of grid points
  init_grid (64);
  // viscosity
  size(10.);

  const face vector muc[] = {1e-3,1e-3};
  mu = muc;

  // maximum timestep
  DT = 0.1;
  // CFL number
  CFL = 0.8;

  run();
}

//event init(i=0){
// boundary condition
u.t[top] = dirichlet(1);
u.t[bottom] = dirichlet(0);
u.t[left]   = dirichlet(0);
u.t[right]  = dirichlet(0);
//}

event outputfile (i += 100) 
{
  //output_matrix (u.x, stdout, N, linear = true);
}

//event movie (i += 4; t <= 15.)
event movie (i += 1; t<=64)
{
  //static FILE * fp = popen ("ppm2mpeg > norm.mpg", "w");
  scalar norme[];
  foreach()
    norme[] = norm(u);
  boundary ({norme});

  //output_ppm (norme, fp, linear = true);
  output_ppm (norme, linear = true, file="norm.mp4");
}