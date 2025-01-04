#include "grid/multigrid.h"
#include "run.h"
#include "diffusion.h"
#include "fluidlab_pack.h"

//Scalar field that I will use
scalar Ar[], Ai[], A2[];

// Parameter from the equation
// In this example we consider the case with alpha = 0
double beta;

//Then, we define the timestep
double dt;
//We create then two variables that will give us information about the simulation timesteps
mgstats mgd1, mgd2;

int num_cells = 256;
//int num_cells = 128;
double domain_size = 10.0;

int main(){
    //We initialize the parameter
    beta = 1.5;

    //Define characteristics of the grid
    size(domain_size); //This define the size in the sense of the domain
    init_grid(num_cells); //This defines the number of pixel per side, by default we are creating a square

    OpenSimulationFolder("GL_beta%g", beta);

    run();
}

//We now create the initial conditions for the sistem
event init (i = 0){
    //determine the initial conditions of the system
    foreach() {
        Ar[]= 1e-4*noise();
        Ai[]= 1e-4*noise();
    }
}

event integration (i++){
    dt=dtnext (0.05);

    foreach(){
        A2[]=sq(Ar[])+sq(Ai[]);
    }

    scalar r[], lambda[];

    foreach(){
        r[] = beta*A2[]*Ai[];
        lambda[] = 1. - A2[];
    }
    mgd1 = diffusion( Ar, dt, r=r, beta=lambda);

    foreach(){
        r[] = -beta*Ar[]*A2[];
        lambda[] = 1. - A2[];
    }
    mgd2 = diffusion (Ai, dt, r=r, beta=lambda);
}

event movies (i += 3; t <= 150){
    fprintf(stderr, "Time: %g, Maximum value of A: %g\n", t, sqrt(normf(A2).max));

    output_ppm(Ai, spread = 2, linear=true, file = "Ai_10.mp4");
    output_ppm(A2, spread = 2, linear=true, file = "A2_10.mp4");
}

event output_binary_files(t+=1) {
    // List of fields to print and the name to be given for each of them
    //scalar *list = {Ai, A2};
    //const char *list_names[] = {"Ai", "A2"};

    scalar *list = {Ai, Ar};
    const char *list_names[] = {"Ai", "Ar"};

    // Dumping the requested fields into a uniform_grid format
    PrintMeshDataDump(i, t, num_cells, domain_size, list, list_names);
}
