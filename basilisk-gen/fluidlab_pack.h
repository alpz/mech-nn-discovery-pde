/*
THINGS TO CHECK:
    -- 2D and 3D
    -- Serial and Parallel MPI
    -- ASCII and BINARY
    -- float or double

    COMBINATIONS
    1) 2D + Serial + ASCII
    2) [ok] 3D + Serial + ASCII
    3) [ok] 2D + Parallel + ASCII
    4) [ok] 3D + Parallel + ASCII

    5) 2D + Serial + BINARY + float
    6) [ok] 3D + Serial + BINARY + float
    7) 2D + Parallel + BINARY + float
    8) [ok] 3D + Parallel + BINARY + float

    9) 2D + Serial + BINARY + double
    10) [ok] 3D + Serial + BINARY + double
    11) 2D + Parallel + BINARY + double
    12) [ok] 3D + Parallel + BINARY + double

ORDER OF THINGS TO DO:
    -- Try to run a 3D version on cluster in parallel without getting that divergence things
    -- Check that the ASCII mesh function prints correctly in sequential and parallel
    -- Check that the BINARY mesh function prints correctly in sequenttial and parallel
*/

char *folder_name = NULL;
FILE *log_file = NULL;

/** 
Checks a condition and, if false, prints an error message and ends execution if requested
(Similar to the standard "assert" macro). */
void ErrorMessage(int Condition, const char *Message, const char *FunctionName, char EndExecution)
{   
  // If the condition is satisfied, we do nothing
  if( Condition )
    return;

  printf("  ===== ERROR in function %s ...\n", (FunctionName) ? FunctionName : "NOT-SPECIFIED");
  printf("  ===== %s\n", Message ? Message : "");
  if( EndExecution ) {
    printf("  ===== Ending execution...\n\n");
    exit(0);
  }

  return;
}

/**
Returns a command line argument given by the user when calling the program
Input Parameter: the index (starting from zero) of the argument to be returned
Output Parameter: string location to store the argument. */
void GetCommandLineArgument(int ArgumentIndex, char *ReturnArgument, int argc, char *argv[])
{   
  /// Checking if the user has provided the argument with the requested index
  if( argc<=ArgumentIndex+1 ) {
    printf("  ===== ERROR in function GetCommandLineArgument ...\n");
    printf("  ===== The requested parameter %d was not provided when calling the program.\n", ArgumentIndex);
    printf("  ===== Ending execution...\n\n");
    exit(0);
  }

  /// Copying the argument into the return string
  strcpy(ReturnArgument, argv[ArgumentIndex + 1]);
  return;
}

/**
Creates a folder for this simulation
All the outputs from functions "PrintLog", "PrintMeshVTK" and 
"PrintInterfaceVTK" will go into this folder

Parameter: the name of the folder to be created. You can use the same syntax as "printf" 
to format this folder name */
void OpenSimulationFolder(const char* format, ...)
{
  folder_name = (char *)malloc(1100*sizeof(char));

  // === Formatting the string with the function parameters
  char *folder_temp = (char *)malloc(1000*sizeof(char));
  va_list argptr;
  va_start(argptr, format);
  vsprintf(folder_temp, format, argptr);
  va_end(argptr);
  sprintf(folder_name, "outputs/%s/", folder_temp);
  free(folder_temp);

  // === Only root process does the rest
  if( pid() )
    return;

  

  

  // === Creating the base "outputs" folder
  system("mkdir outputs/");

  // === Creating the simulation folder
  char command[1000];
  sprintf(command, "mkdir %s", folder_name);
  system(command);

  /// === Opening the new log file
  if( log_file ) 
    fclose(log_file);
  sprintf(command, "%s/log_file.txt", folder_name);
  log_file = fopen(command, "wt");
  if( !log_file ) {
    printf("  ===== ERROR in function OpenSimulationFolder ...\n");
    printf("  ===== Unable to create the folder and log_file. Do you have permission for creating this folder?\n");
    printf("  ===== Ending execution...\n\n");
    exit(0);
  }
  
  return;
}

/**
Closes the current simulation folder and output files currently open. */
void CloseSimulationFolder() {

  // === Only root process does this
  if( pid() )
    return;

  if( log_file )
    fclose(log_file);
  log_file = NULL;

  return;
}


/**
Read parameters from a file
Parameters: 
1. The name of the file containing the parameters.
2. A list of parameter names to look for in the file. Separated by semi-colon.
3. Pointers to variables where each of the parameters will be stored
NOTE: currently I'm reading all values as "double" variables. 
I can change this in future if there is interest for it */
// void ReadParametersFromFile(const char *file_name, const char *parameters_names, ...)
// {
//   char names_copy[1000], *token;
//   int number_of_parameters = 0;
//   FILE *file;

//   // == Counting how many parameters the user wants to read
//   strcpy(names_copy, parameters_names);
//   token = strtok(names_copy, ";");
//   while( token ) {
//     number_of_parameters++;
//     token = strtok(NULL, ";");
//   }

//   // == Looking for the parameters in the text file
//   if( !(file=fopen(file_name, "rt")) ) {
//     printf("  ===== ERROR in function ReadParametersFromFile ...\n");
//     printf("  ===== Unable to open the input text file \"%s\".\n", file_name);
//     printf("  ===== Ending execution...\n\n");
//     exit(0);
//   }
//   strcpy(names_copy, parameters_names);
//   token = strtok(names_copy, ";");
//   int i_token = 0;
//   while( token ) {
//     char name_read[1000];
//     double value_read;
    
//     // Scanning the entire file from the beginning loking for this token
//     rewind(file);
//     int found_parameter = 0;
//     while( 2==fscanf(file, "%s %lf\n", name_read, &value_read) ) {
//       if( !strcmp(name_read, token) ) {
//         found_parameter = 1;
//         break;
//       }
//     }
//     if( !found_parameter ) {
//       printf("  ===== ERROR in function ReadParametrsFromFile ...");
//       printf("  ===== Could not find the parameter [%s] in the input file.\n", token);
//       printf("  ===== Ending execution...\n\n");
//       exit(0);
//     }

//     // Finding the pointer in the list of pointers given by the user in the function parameters
//     va_list ap;
//     va_start(ap, parameters_names);
//     int i;
//     double *pointer;
//     for( i=0; i<=i_token; i++ )
//       pointer = va_arg(ap, double*);
//     va_end(ap);

//     // Setting the value read in the pointer
//     *pointer = value_read;

//     token = strtok(NULL, ";");
//     i_token++;
//   }
//   fclose( file );

//   return;
// }

/**
Prints something to the log file of this simulation */
void PrintLog(const char* format, ...)
{
  // === Only root process will print
  if( pid() )
    return;
  
  if( log_file==NULL )
    return;

  // Formatted printing
  va_list argptr;
  va_start(argptr, format);
  vfprintf(log_file, format, argptr);
  va_end(argptr);
  fflush(log_file);
  
  return;
}



/**
This is basically a wrapper for the MPI_Gatherv function.
It automatically counts how much data will be sent by each processor */
#if _MPI
void MPI_Gather_Uneven(const void *sendbuf, int sendcount, MPI_Datatype sendreceivetype, void *recvbuf, int root)
{
  int array_counts[npe()];
  int displs[npe()];

  if( pid()==root ) {
    int i;
    for( i=0; i<npe(); i++ ) {
      if( i==root )
        array_counts[i] = sendcount;
      else
        MPI_Recv(array_counts+i, 1, MPI_INT, i, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      displs[i] = (i==0) ? 0 : displs[i-1] + array_counts[i-1];
    }
  }
  else
    MPI_Send(&sendcount, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD);

  MPI_Gatherv(sendbuf, sendcount, sendreceivetype, recvbuf, array_counts, displs, sendreceivetype, root, MPI_COMM_WORLD);
}
#endif

#define PRINT_CELL_CRITERIA 1 // Printing all the cells in the mesh
// #define PRINT_CELL_CRITERIA f[]>0.05 // Only printing cells with some volume in it

/**
Prints the mesh and (optionally) scalar data to an ASCII VTK file. */
void PrintMeshVTK_ASCII(int n, double time, scalar *list_scalar_data, const char **list_scalar_names)
{
    FILE *arq;
    char nomeArq[900];

    #if dimension==2
        // Each cell is a square (4 vertices)
        int vertices_per_cell = 4; 

        // VTK cell code that represents quads
        // (check Figure 2 here to understand: https://kitware.github.io/vtk-examples/site/VTKFileFormats/)
        int vtk_cell_type = 9; 
    #else
        int vertices_per_cell = 8; // Each cell is a cube

        // VTK cell code that represents voxels
        // (check Figure 2 here to understand: https://kitware.github.io/vtk-examples/site/VTKFileFormats/)
        int vtk_cell_type = 11; 
    #endif

    // Counting how many local cells we have in the mesh
    // NOTE: whenever I say "local", i mean things that are locally in this processor (in case of parallel simulation)
    int local_num_cells = 0;
    foreach(serial) {
        // if( PRINT_CELL_CRITERIA )
            local_num_cells++;
    }

    /// === Allocatting memory for the local vertex arrays
    double *local_vertices_x = (double *)malloc( vertices_per_cell*local_num_cells*sizeof(double) );
    double *local_vertices_y = (double *)malloc( vertices_per_cell*local_num_cells*sizeof(double) );
    double *local_vertices_z = (double *)malloc( vertices_per_cell*local_num_cells*sizeof(double) );

    /// === Allocating memory for ALL the local scalar data arrays
    int number_of_scalar_fields = list_len(list_scalar_data);
    typedef double** doublepp; // Hiding the double pointer with a typedef because qcc gets really annoying if i dont do this (why??)
    doublepp local_data = number_of_scalar_fields ? (double **)malloc( number_of_scalar_fields*sizeof(double *) ) : NULL;
    for( int k=0; k<number_of_scalar_fields; k++ )
        local_data[k] = (double *)malloc( local_num_cells*sizeof(double) );

    /// === Storing all the vertices coordinates in the arrays and also the scalar data to be printed data
    int i = 0, i_cell = 0;
    foreach(serial) {
        if( PRINT_CELL_CRITERIA ) {
            #if dimension==2
                local_vertices_x[i] = x-0.5*Delta; local_vertices_y[i] = y-0.5*Delta; local_vertices_z[i++] = 0.0;
                local_vertices_x[i] = x+0.5*Delta; local_vertices_y[i] = y-0.5*Delta; local_vertices_z[i++] = 0.0;
                local_vertices_x[i] = x+0.5*Delta; local_vertices_y[i] = y+0.5*Delta; local_vertices_z[i++] = 0.0;
                local_vertices_x[i] = x-0.5*Delta; local_vertices_y[i] = y+0.5*Delta; local_vertices_z[i++] = 0.0;
            #else // dimension==3
                local_vertices_x[i] = x-0.5*Delta; local_vertices_y[i] = y-0.5*Delta; local_vertices_z[i++] = z-0.5*Delta;
                local_vertices_x[i] = x+0.5*Delta; local_vertices_y[i] = y-0.5*Delta; local_vertices_z[i++] = z-0.5*Delta;
                local_vertices_x[i] = x-0.5*Delta; local_vertices_y[i] = y+0.5*Delta; local_vertices_z[i++] = z-0.5*Delta;
                local_vertices_x[i] = x+0.5*Delta; local_vertices_y[i] = y+0.5*Delta; local_vertices_z[i++] = z-0.5*Delta;

                local_vertices_x[i] = x-0.5*Delta; local_vertices_y[i] = y-0.5*Delta; local_vertices_z[i++] = z+0.5*Delta;
                local_vertices_x[i] = x+0.5*Delta; local_vertices_y[i] = y-0.5*Delta; local_vertices_z[i++] = z+0.5*Delta;
                local_vertices_x[i] = x-0.5*Delta; local_vertices_y[i] = y+0.5*Delta; local_vertices_z[i++] = z+0.5*Delta;
                local_vertices_x[i] = x+0.5*Delta; local_vertices_y[i] = y+0.5*Delta; local_vertices_z[i++] = z+0.5*Delta;
            #endif

            int list_index = 0;
            for(scalar s in list_scalar_data)
                local_data[list_index++][i_cell] = s[];

            i_cell++;
        }
    }

    /// === Getting how many cells we have in total between all processes
    /// === And then gathering all the local vertices into the root process
    int total_num_cells;
    double *vertices_x = NULL, *vertices_y = NULL, *vertices_z = NULL;
    doublepp data = NULL;
    #if _MPI
        /// Total number of cels
        MPI_Allreduce(&local_num_cells, &total_num_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        /// === Gathering all the local vertices into the root process
        if( pid()==0 ) {
            vertices_x = (double *)malloc( total_num_cells*vertices_per_cell*sizeof(double) );
            vertices_y = (double *)malloc( total_num_cells*vertices_per_cell*sizeof(double) );
            vertices_z = (double *)malloc( total_num_cells*vertices_per_cell*sizeof(double) );
            
            data = number_of_scalar_fields ? (double **)malloc( number_of_scalar_fields*sizeof(double*) ) : NULL;
            for(int k=0; k<number_of_scalar_fields; k++)
                data[k] = (double *)malloc( total_num_cells*sizeof(double) );
        }
        MPI_Gather_Uneven(local_vertices_x, vertices_per_cell*local_num_cells, MPI_DOUBLE, vertices_x, 0);
        MPI_Gather_Uneven(local_vertices_y, vertices_per_cell*local_num_cells, MPI_DOUBLE, vertices_y, 0);
        MPI_Gather_Uneven(local_vertices_z, vertices_per_cell*local_num_cells, MPI_DOUBLE, vertices_z, 0);

        int list_index = 0;
        for( list_index=0; list_index<number_of_scalar_fields; list_index++ )
            MPI_Gather_Uneven(local_data[list_index], local_num_cells, MPI_DOUBLE, (pid()==0) ? data[list_index] : NULL, 0);

        /// === Releasing local memory
        free(local_vertices_x);
        free(local_vertices_y);
        free(local_vertices_z);

        for(int k=0; k<number_of_scalar_fields; k++)
            free(local_data[k]);
        if( local_data )
            free(local_data);

    #else
        total_num_cells = local_num_cells;
        vertices_x = local_vertices_x;
        vertices_y = local_vertices_y;
        vertices_z = local_vertices_z;
        data = local_data;
    #endif

    
    /// === From now on, we only do file-printing stuff. Only the root process will do it
    if( pid()==0 ) {

        /// === Opening the file to print the mesh
        sprintf(nomeArq, "%s/Mesh-N%d.vtk", folder_name, n);
        arq = fopen(nomeArq, "wt");


        /// === Printing the VTK header information
        fprintf(arq, "# vtk DataFile Version 2.0\n");
        fprintf(arq, "MESH. step %d time %lf\n", n, time);
        fprintf(arq, "ASCII\n");
        fprintf(arq, "DATASET UNSTRUCTURED_GRID\n");

        /// === Printing all the vertices coordinates
        fprintf(arq, "POINTS %d float\n", total_num_cells*vertices_per_cell);
        int total_num_vertices = vertices_per_cell*total_num_cells;
        for( i=0; i<total_num_vertices; i++ )
            fprintf(arq, "%lf %lf %lf\n", vertices_x[i], vertices_y[i], vertices_z[i]);

        /// === Printing all the cells (each cell contains 4 (or 8) indices referring to the vertices printed above)
        fprintf(arq, "CELLS %d %d\n", total_num_cells, (vertices_per_cell + 1)*total_num_cells);
        int index = 0;
        for( i=0; i<total_num_cells; i++ ) {
            #if dimension==2
                fprintf(arq, "%d %d %d %d %d\n", vertices_per_cell, index, index+1, index+2, index+3);
            #else        
                fprintf(arq, "%d %d %d %d %d %d %d %d %d\n", vertices_per_cell, index, index+1, index+2, index+3, index+4, index+5, index+6, index+7);
            #endif
            index += vertices_per_cell;
        }

        /// === Printing the VTK cell_types (quads or voxels)
        fprintf(arq, "CELL_TYPES %d\n", total_num_cells);
        for( i=0; i<total_num_cells; i++ )
            fprintf(arq, "%d\n", vtk_cell_type);

        

        /// === Printing the actual simulation data that is stored in the cells
        if(data) {
            int list_index = 0;
            fprintf(arq, "CELL_DATA %d\n", total_num_cells);
            for( scalar s in list_scalar_data ) {
                fprintf(arq, "SCALARS %s float 1\n", list_scalar_names[list_index]);
                fprintf(arq, "LOOKUP_TABLE default\n");
                for( i=0; i<total_num_cells; i++ )
                    fprintf(arq, "%lf\n", data[list_index][i]);
                list_index++;
            }
        }


        /// === Closing the file
        fclose(arq);

        /// === Releasing memory
        free(vertices_x);
        free(vertices_y);
        free(vertices_z);
        for(int k=0; k<number_of_scalar_fields; k++)
            free(data[k]);
        if(data)
            free(data);
    }

    /// === All processes wait for the file printing to finish before continuing with the simulation
    #if _MPI
        MPI_Barrier(MPI_COMM_WORLD);
    #endif

    return;
}

// This function transforms the little-endian entries of an array into big-endian (or vice-versa). This will be used in the BINARY vtk printing functions
// IMPORTANT: currently, this function will also swap the order of the array entries as well as the byte ordering of each individual entry
void SwapArrayBytes(void *Array, size_t number_of_bytes)
{
    int i;
    size_t half_of_bytes = number_of_bytes/2;

    unsigned char *byte_array = (unsigned char *)Array;
    for( i=0; i<half_of_bytes; i++ ) {
        unsigned char temporary = byte_array[i];
        byte_array[i] = byte_array[number_of_bytes - i - 1];
        byte_array[number_of_bytes - i - 1] = temporary;
    }

    return;
}

/**
Prints mesh and (optionally) scalars to a double-precision BINARY VTK file. */
void PrintMeshVTK_Binary_Double(int n, double time, scalar *list_scalar_data, const char **list_scalar_names)
{
  FILE *arq;
  char nomeArq[900];

  // === Cell is either a square (2D) or cube (3D), meaning either 4 or 8 vertices per cell
  int vertices_per_cell = (dimension==2) ? 4 : 8;

  // === VTK cell code that represents voxels
  // === (check Figure 2 here to understand: https://kitware.github.io/vtk-examples/site/VTKFileFormats/)
  int vtk_cell_type = (dimension==2) ? 9 : 11;

  // === Counting how many local cells we have in the mesh
  // === NOTE: whenever I say "local", i mean things that are locally
  // === in this processor (in case of parallel simulation)
  int local_num_cells = 0;
  foreach(serial) {
    if( PRINT_CELL_CRITERIA )
      local_num_cells++;
  }

  // === Allocatting memory for the local vertex arrays
  // === Note: the x,y,z coordinates will all go into the same array, so the array will be [x1, y1, z1, x2, y2, z2, ...]
  double *local_vertices = (double *)malloc( 3*vertices_per_cell*local_num_cells*sizeof(double) );

  // === Allocating memory for ALL the local scalar data arrays
  int number_of_scalar_fields = list_len(list_scalar_data);
  typedef double** doublepp; // Hiding the double pointer with a typedef because qcc gets really annoying if i dont do this (why??)
  doublepp local_data = number_of_scalar_fields ? (double **)malloc( number_of_scalar_fields*sizeof(double *) ) : NULL;
  for( int k=0; k<number_of_scalar_fields; k++ )
    local_data[k] = (double *)malloc( local_num_cells*sizeof(double) );

  // === Storing all the vertices coordinates in the arrays
  int i = 0, i_cell = 0;
  foreach(serial) {
    if( PRINT_CELL_CRITERIA ) {
      // Using a macro conditional to avoid checking dimension==2 every loop during execution time...
      #if dimension==2
        local_vertices[i++] = 0.0; local_vertices[i++] = y-0.5*Delta; local_vertices[i++] = x-0.5*Delta;
        local_vertices[i++] = 0.0; local_vertices[i++] = y-0.5*Delta; local_vertices[i++] = x+0.5*Delta;
        local_vertices[i++] = 0.0; local_vertices[i++] = y+0.5*Delta; local_vertices[i++] = x+0.5*Delta;
        local_vertices[i++] = 0.0; local_vertices[i++] = y+0.5*Delta; local_vertices[i++] = x-0.5*Delta;
      #else // dimension
        local_vertices[i++] = z-0.5*Delta; local_vertices[i++] = y-0.5*Delta; local_vertices[i++] = x-0.5*Delta;
        local_vertices[i++] = z-0.5*Delta; local_vertices[i++] = y-0.5*Delta; local_vertices[i++] = x+0.5*Delta;
        local_vertices[i++] = z-0.5*Delta; local_vertices[i++] = y+0.5*Delta; local_vertices[i++] = x-0.5*Delta;
        local_vertices[i++] = z-0.5*Delta; local_vertices[i++] = y+0.5*Delta; local_vertices[i++] = x+0.5*Delta;
        
        local_vertices[i++] = z+0.5*Delta; local_vertices[i++] = y-0.5*Delta; local_vertices[i++] = x-0.5*Delta;
        local_vertices[i++] = z+0.5*Delta; local_vertices[i++] = y-0.5*Delta; local_vertices[i++] = x+0.5*Delta;
        local_vertices[i++] = z+0.5*Delta; local_vertices[i++] = y+0.5*Delta; local_vertices[i++] = x-0.5*Delta;
        local_vertices[i++] = z+0.5*Delta; local_vertices[i++] = y+0.5*Delta; local_vertices[i++] = x+0.5*Delta;
      #endif

      int list_index = 0;
      for(scalar s in list_scalar_data) {
        local_data[list_index][i_cell] = s[];
        SwapArrayBytes(&local_data[list_index][i_cell], sizeof(double));
        list_index++;
      }

      i_cell++;
    }
  }

  // === Getting how many cells we have in total between all processes
  // === And then gathering all the local vertices into the root process
  int total_num_cells;
  double *vertices = NULL;
  doublepp data = NULL;
  #if _MPI
    /// === Total number of cells
    MPI_Allreduce(&local_num_cells, &total_num_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    /// === Gathering all the local vertices and the scalar fields data into the root process
    if( pid()==0 ) {
      vertices = (double *)malloc( 3*total_num_cells*vertices_per_cell*sizeof(double) );
      data = (double **)malloc( number_of_scalar_fields*sizeof(double*) );
      for(int k=0; k<number_of_scalar_fields; k++)
        data[k] = (double *)malloc( total_num_cells*sizeof(double) );
    }
    MPI_Gather_Uneven(local_vertices, 3*vertices_per_cell*local_num_cells, MPI_DOUBLE, vertices, 0);
    int list_index = 0;
    for( list_index=0; list_index<number_of_scalar_fields; list_index++ )
      MPI_Gather_Uneven(local_data[list_index], local_num_cells, MPI_DOUBLE, (pid()==0) ? data[list_index] : NULL, 0);

    /// === Releasing local memory
    free(local_vertices);
    for(int k=0; k<number_of_scalar_fields; k++)
      free(local_data[k]);
    if( local_data )
      free(local_data);
  #else
    total_num_cells = local_num_cells;
    vertices = local_vertices;
    data = local_data;
  #endif

  /**
    From now on, we only do file-printing stuff. Only the root process will do it. */
  if( pid()==0 ) {

    // === Opening the file to print the mesh
    sprintf(nomeArq, "%s/Mesh-N%d.vtk", folder_name, n);
    arq = fopen(nomeArq, "wt");

    // === Printing the VTK header information (printed as ASCII text)
    fprintf(arq, "# vtk DataFile Version 2.0\n");
    fprintf(arq, "MESH. step %d time %lf\n", 0, 0.0);
    fprintf(arq, "BINARY\n");
    fprintf(arq, "DATASET UNSTRUCTURED_GRID\n");

    // === Printing all the vertices coordinates (as BINARY)
    fprintf(arq, "POINTS %d double\n", total_num_cells*vertices_per_cell);
    SwapArrayBytes(vertices, 3*vertices_per_cell*total_num_cells*sizeof(double));
    fwrite(vertices, sizeof(double), 3*vertices_per_cell*total_num_cells, arq);
    fprintf(arq, "\n");

    // === Printing all the cells 
    // === Each cell contains 4 (or 8) indices referring to the vertices above
    fprintf(arq, "CELLS %d %d\n", total_num_cells, (vertices_per_cell + 1)*total_num_cells);
    int *array_cell_indices = malloc( (vertices_per_cell + 1)*total_num_cells*sizeof(int) );
    int offset = 0, vertex_index = 0;
    for( i=0; i<total_num_cells; i++ ) {
      array_cell_indices[offset] = vertex_index;
      array_cell_indices[offset + 1] = vertex_index + 1;
      array_cell_indices[offset + 2] = vertex_index + 2;
      array_cell_indices[offset + 3] = vertex_index + 3;
      #if dimension==3
        array_cell_indices[offset + 4] = vertex_index + 4;
        array_cell_indices[offset + 5] = vertex_index + 5;
        array_cell_indices[offset + 6] = vertex_index + 6;
        array_cell_indices[offset + 7] = vertex_index + 7;
      #endif
      array_cell_indices[offset + vertices_per_cell] = vertices_per_cell;
      offset += vertices_per_cell + 1;
      vertex_index += vertices_per_cell;
    }
    SwapArrayBytes(array_cell_indices, (vertices_per_cell + 1)*total_num_cells*sizeof(int));
    fwrite(array_cell_indices, sizeof(int), (vertices_per_cell + 1)*total_num_cells, arq);
    fprintf(arq, "\n");
        
    // === Printing cell types (squares or cubes)
    fprintf(arq, "CELL_TYPES %d\n", total_num_cells);
    SwapArrayBytes(&vtk_cell_type, sizeof(int));
    for( i=0; i<total_num_cells; i++ )
      array_cell_indices[i] = vtk_cell_type;
    fwrite(array_cell_indices, sizeof(int), total_num_cells, arq);
    fprintf(arq, "\n");


    // === Printing the actual simulation data that is stored in the cells
    int list_index = 0;
    fprintf(arq, "CELL_DATA %d\n", total_num_cells);
    for( scalar s in list_scalar_data ) {
      fprintf(arq, "SCALARS %s double 1\n", list_scalar_names[list_index]);
      fprintf(arq, "LOOKUP_TABLE default\n");
      fwrite(data[list_index], sizeof(double), total_num_cells, arq);
      fprintf(arq, "\n");
      list_index++;
    }

    // === Releasing memory
    free(array_cell_indices);
    free(vertices);
    for( list_index=0; list_index<number_of_scalar_fields; list_index++ )
      free(data[list_index]);
    free(data);
    fclose(arq);
  }

  return;
}

/**
Prints mesh and (optionally) scalars to a single-precision BINARY VTK file. */
void PrintMeshVTK_Binary_Float(int n, double time, scalar *list_scalar_data, const char **list_scalar_names)
{
  FILE *arq;
  char nomeArq[900];

  // ===  Cell is either a square (2D) or cube (3D), meaning either 4 or 8 vertices per cell
  int vertices_per_cell = (dimension==2) ? 4 : 8;

  // === VTK cell code that represents voxels
  // === (check Figure 2 here to understand: https://kitware.github.io/vtk-examples/site/VTKFileFormats/)
  int vtk_cell_type = (dimension==2) ? 9 : 11;

  // === Counting how many local cells we have in the mesh
  // === NOTE: whenever I say "local", i mean things that are locally in this processor (in case of parallel simulation)
  int local_num_cells = 0;
  foreach(serial) {
    if( PRINT_CELL_CRITERIA )
      local_num_cells++;
  }

  // === Allocatting memory for the local vertex arrays
  // === Note: the x,y,z coordinates will all go into the same array, 
  // === so the array will be [x1, y1, z1, x2, y2, z2, ...]
  float *local_vertices = (float *)malloc( 3*vertices_per_cell*local_num_cells*sizeof(float) );

  // === Allocating memory for ALL the local scalar data arrays
  int number_of_scalar_fields = list_len(list_scalar_data);
  typedef float** floatpp; // qcc gets uncomfortable if i dont hide the double pointer (?????)
  floatpp local_data = number_of_scalar_fields ? (float **)malloc( number_of_scalar_fields*sizeof(float *) ) : NULL;
  for( int k=0; k<number_of_scalar_fields; k++ )
    local_data[k] = (float *)malloc( local_num_cells*sizeof(float) );

  // === Storing all the vertices coordinates in the arrays
  int i = 0, i_cell = 0;
  foreach(serial) {
    if( PRINT_CELL_CRITERIA ) {

      /// Using a macro conditional to avoid checking dimension==2 every loop during execution time...
      #if dimension==2
        local_vertices[i++] = 0.0; local_vertices[i++] = y-0.5*Delta; local_vertices[i++] = x-0.5*Delta;
        local_vertices[i++] = 0.0; local_vertices[i++] = y-0.5*Delta; local_vertices[i++] = x+0.5*Delta;
        local_vertices[i++] = 0.0; local_vertices[i++] = y+0.5*Delta; local_vertices[i++] = x+0.5*Delta;
        local_vertices[i++] = 0.0; local_vertices[i++] = y+0.5*Delta; local_vertices[i++] = x-0.5*Delta;
      #else // dimension
        local_vertices[i++] = z-0.5*Delta; local_vertices[i++] = y-0.5*Delta; local_vertices[i++] = x-0.5*Delta;
        local_vertices[i++] = z-0.5*Delta; local_vertices[i++] = y-0.5*Delta; local_vertices[i++] = x+0.5*Delta;
        local_vertices[i++] = z-0.5*Delta; local_vertices[i++] = y+0.5*Delta; local_vertices[i++] = x-0.5*Delta;
        local_vertices[i++] = z-0.5*Delta; local_vertices[i++] = y+0.5*Delta; local_vertices[i++] = x+0.5*Delta;
        
        local_vertices[i++] = z+0.5*Delta; local_vertices[i++] = y-0.5*Delta; local_vertices[i++] = x-0.5*Delta;
        local_vertices[i++] = z+0.5*Delta; local_vertices[i++] = y-0.5*Delta; local_vertices[i++] = x+0.5*Delta;
        local_vertices[i++] = z+0.5*Delta; local_vertices[i++] = y+0.5*Delta; local_vertices[i++] = x-0.5*Delta;
        local_vertices[i++] = z+0.5*Delta; local_vertices[i++] = y+0.5*Delta; local_vertices[i++] = x+0.5*Delta;
      #endif

      int list_index = 0;
      for(scalar s in list_scalar_data) {
        local_data[list_index][i_cell] = (float) s[];
        SwapArrayBytes(&local_data[list_index][i_cell], sizeof(float));
        list_index++;
      }

      i_cell++;
    }
  }

  // === Getting how many cells we have in total between all processes
  // === And then gathering all the local vertices into the root process
  int total_num_cells;
  float *vertices = NULL;
  floatpp data = NULL;
  #if _MPI
    // === Total number of cells
    MPI_Allreduce(&local_num_cells, &total_num_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // === Gathering all the local vertices and the scalar fields data into the root process
    if( pid()==0 ) {
      vertices = (float *)malloc( 3*total_num_cells*vertices_per_cell*sizeof(float) );
      data = (float **)malloc( number_of_scalar_fields*sizeof(float*) );
      for(int k=0; k<number_of_scalar_fields; k++)
        data[k] = (float *)malloc( total_num_cells*sizeof(float) );
    }
    MPI_Gather_Uneven(local_vertices, 3*vertices_per_cell*local_num_cells, MPI_FLOAT, vertices, 0);
    int list_index = 0;
    for( list_index=0; list_index<number_of_scalar_fields; list_index++ )
      MPI_Gather_Uneven(local_data[list_index], local_num_cells, MPI_FLOAT, (pid()==0) ? data[list_index] : NULL, 0);

    // === Releasing local memory
    free(local_vertices);
    for(int k=0; k<number_of_scalar_fields; k++)
      free(local_data[k]);
    if( local_data )
      free(local_data);
  #else
    total_num_cells = local_num_cells;
    vertices = local_vertices;
    data = local_data;
  #endif

  
  
  // === From now on, we only do file-printing stuff. Only the root process will do it
  if( pid()==0 ) {
    // === Opening the file to print the mesh
    sprintf(nomeArq, "%s/Mesh-N%d.vtk", folder_name, n);
    arq = fopen(nomeArq, "wt");

    // === Printing the VTK header information (printed as ASCII text)
    fprintf(arq, "# vtk DataFile Version 2.0\n");
    fprintf(arq, "MESH. step %d time %lf\n", 0, 0.0);
    fprintf(arq, "BINARY\n");
    fprintf(arq, "DATASET UNSTRUCTURED_GRID\n");

    // === Printing all the vertices coordinates (as BINARY)
    fprintf(arq, "POINTS %d float\n", total_num_cells*vertices_per_cell);
    SwapArrayBytes(vertices, 3*vertices_per_cell*total_num_cells*sizeof(float));
    fwrite(vertices, sizeof(float), 3*vertices_per_cell*total_num_cells, arq);
    fprintf(arq, "\n");

    // === Printing all the cells (each cell contains 4 (or 8) indices referring to the vertices printed above)
    fprintf(arq, "CELLS %d %d\n", total_num_cells, (vertices_per_cell + 1)*total_num_cells);
    int *array_cell_indices = malloc( (vertices_per_cell + 1)*total_num_cells*sizeof(int) );
    int offset = 0, vertex_index = 0;
    for( i=0; i<total_num_cells; i++ ) {
      array_cell_indices[offset] = vertex_index;
      array_cell_indices[offset + 1] = vertex_index + 1;
      array_cell_indices[offset + 2] = vertex_index + 2;
      array_cell_indices[offset + 3] = vertex_index + 3;
      #if dimension==3
        array_cell_indices[offset + 4] = vertex_index + 4;
        array_cell_indices[offset + 5] = vertex_index + 5;
        array_cell_indices[offset + 6] = vertex_index + 6;
        array_cell_indices[offset + 7] = vertex_index + 7;
      #endif
      array_cell_indices[offset + vertices_per_cell] = vertices_per_cell;
      offset += vertices_per_cell + 1;
      vertex_index += vertices_per_cell;
    }
    SwapArrayBytes(array_cell_indices, (vertices_per_cell + 1)*total_num_cells*sizeof(int));
    fwrite(array_cell_indices, sizeof(int), (vertices_per_cell + 1)*total_num_cells, arq);
    fprintf(arq, "\n");
        

    fprintf(arq, "CELL_TYPES %d\n", total_num_cells);
    SwapArrayBytes(&vtk_cell_type, sizeof(int));
    for( i=0; i<total_num_cells; i++ )
      array_cell_indices[i] = vtk_cell_type;
    fwrite(array_cell_indices, sizeof(int), total_num_cells, arq);
    fprintf(arq, "\n");


    // === Printing the actual simulation data that is stored in the cells
    int list_index = 0;
    fprintf(arq, "CELL_DATA %d\n", total_num_cells);
    for( scalar s in list_scalar_data ) {
      fprintf(arq, "SCALARS %s float 1\n", list_scalar_names[list_index]);
      fprintf(arq, "LOOKUP_TABLE default\n");
      fwrite(data[list_index], sizeof(float), total_num_cells, arq);
      fprintf(arq, "\n");
      list_index++;
    }

    free(array_cell_indices);
    free(vertices);
    for( list_index=0; list_index<number_of_scalar_fields; list_index++ )
      free(data[list_index]);
    free(data);
    fclose(arq);
  }

  return;
}

/**
Structure used to pass optional parameters to the PrintMeshVTK and PrintMeshVTK_Binary. */
typedef enum {VTK_TYPE_ASCII, VTK_TYPE_BINARY} VTK_FILE_TYPE;
typedef enum {VTK_PRECISION_FLOAT, VTK_PRECISION_DOUBLE} VTK_FILE_PRECISION;
struct StructPrintMesh {
  int n;
  double time;
  scalar *list_scalar_data;
  const char **list_scalar_names;
  VTK_FILE_TYPE vtk_type;
  VTK_FILE_PRECISION vtk_precision; // only relevant if vtk_type==binary
};

/**
  Prints mesh and (optionally) scalar data to a VTK file. */
void PrintMeshVTK(struct StructPrintMesh spm)
{
  // if( spm.vtk_type==VTK_TYPE_ASCII )
  //   PrintMeshVTK_ASCII(spm.n, spm.time, spm.list_scalar_data, spm.list_scalar_names);
  // else if( spm.vtk_type==VTK_TYPE_BINARY ) {        
  //   if( spm.vtk_precision==VTK_PRECISION_FLOAT )
  //     PrintMeshVTK_Binary_Float(spm.n, spm.time, spm.list_scalar_data, spm.list_scalar_names);
  //   else if( spm.vtk_precision==VTK_PRECISION_DOUBLE )
  //     PrintMeshVTK_Binary_Double(spm.n, spm.time, spm.list_scalar_data, spm.list_scalar_names);
  //   else
  //     ErrorMessage( 0, "The precision parameter for VTK Binary files should be either VTK_PRECISION_FLOAT or VTK_PRECISION_DOUBLE.", "PrintMeshVTK", 1 );
  // }
  // else
  //   ErrorMessage( 0, "The vtk_type parameter for VTK Binary files should be either VTK_TYPE_ASCII or VTK_TYPE_BINARY.", "PrintMeshVTK", 1 );
  return;
}

/** 
  I blatantly copied this function from here: [draw.h](http://basilisk.fr/src/draw.h)
  I just did it so I don't have to include the whole view.h and link everything just for this function
 */
static bool cfilter_hugo (Point point, scalar c, double cmin)
{
  double cmin1 = 4.*cmin;
  if (c[] <= cmin) {
    foreach_dimension()
      if (c[1] >= 1. - cmin1 || c[-1] >= 1. - cmin1)
	return true;
    return false;
  }
  if (c[] >= 1. - cmin) {
    foreach_dimension()
      if (c[1] <= cmin1 || c[-1] <= cmin1)
	return true;
    return false;
  }
  int n = 0;
  double min = HUGE, max = - HUGE;
  foreach_neighbor(1) {
    if (c[] > cmin && c[] < 1. - cmin && ++n >= (1 << dimension))
      return true;
    if (c[] > max) max = c[];
    if (c[] < min) min = c[];
  }
  return max - min > 0.5;
}

void swap_bs(double* xp, double* yp)
{
    double temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void InitializeInterfaceFromVTK(const char *file_name, double **out_centers_x, double **out_centers_y, int *out_num_centers)
{
  // === Opening the vtk file
  FILE *arq = fopen(file_name, "rt");
  if( !arq ) {
    printf("\n\n InitializeInterfaceFromVTK: Problem opening file... \n\n");
    exit(0);
  }

  // === Writing the VTK header
  double time;
  int total_count_vertices, n;
  fscanf(arq, "# vtk DataFile Version 2.0\n");
  fscanf(arq, "INTERFACE. step %d time %lf\n", &n, &time);
  fscanf(arq, "ASCII\n");
  fscanf(arq, "DATASET POLYDATA\n");
  fscanf(arq, "POINTS %d float\n", &total_count_vertices);
  
  // === Writing all the surface vertices
  double *all_vertices_x = (double *)malloc( total_count_vertices*sizeof(double) );
  double *all_vertices_y = (double *)malloc( total_count_vertices*sizeof(double) );
  double aux_vertex_z;
  for( int index_vertex=0; index_vertex<total_count_vertices; index_vertex++ )
    fscanf(arq, "%lf %lf %lf\n", &all_vertices_x[index_vertex], &all_vertices_y[index_vertex], &aux_vertex_z);
  


  // === Writing the polygons conectivity
  int total_count_polys, aux, index_center = 0;
  #if dimension==2
    fscanf(arq, "LINES %d %d\n", &total_count_polys, &aux);
  #else
    fscanf(arq, "POLYGONS %d %d\n", &total_count_polys, &aux);
  #endif

  // Allocating memory for the centers
  double *centers_x = (double *)malloc(total_count_polys*sizeof(double));
  double *centers_y = (double *)malloc(total_count_polys*sizeof(double));

  index_center = 0;
  for( int index_polygon=0; index_polygon<total_count_polys; index_polygon++ ) {
    int count_vertices_polygon, global_index_vertex[10];
    fscanf(arq, "%d", &count_vertices_polygon);
    int index_vertex;
    for( index_vertex=0; index_vertex<count_vertices_polygon; index_vertex++ )
      fscanf(arq, " %d", &global_index_vertex[index_vertex]);
    fscanf(arq, "\n");

    if( count_vertices_polygon!=2 ) {
      printf("Something wrong with number of vertices in a line: %d\n", count_vertices_polygon);
      exit(1);
    }

    centers_x[index_center] = 0.5*( all_vertices_x[global_index_vertex[0]] + all_vertices_x[global_index_vertex[1]] );
    centers_y[index_center] = 0.5*( all_vertices_y[global_index_vertex[0]] + all_vertices_y[global_index_vertex[1]] );

    if( centers_x[index_center]>0.005 ) {
      // printf("center: %lf %lf\n", centers_x[index_center], centers_y[index_center]);
      index_center++;
    }
  }

  int i, j;
  // Ordering by y-coordinate (just a bubble sort, dont care about efficiency here)
  n = index_center;
  bool swapped;
  for (i = 0; i < n - 1; i++) {
      swapped = false;
      for (j = 0; j < n - i - 1; j++) {
          if (centers_y[j] > centers_y[j + 1]) {
              swap_bs(&centers_y[j], &centers_y[j + 1]);
              swap_bs(&centers_x[j], &centers_x[j + 1]);
              swapped = true;
          }
      }

      // If no two elements were swapped by inner loop,
      // then break
      if (swapped == false)
          break;
  }

  free(all_vertices_x);
  free(all_vertices_y);

  *out_centers_x = centers_x;
  *out_centers_y = centers_y;
  *out_num_centers = index_center;
}

// void calculate_energy_budget(double *kinetic, double *surface, double *elastic, double *dissipated, double Oh, double J, double regularization)
// {
//   double kinetic_temp = 0.0;
//   double surface_temp = 0.0;
//   double elastic_temp = 0.0;
//   double dissipated_temp = 0.0;


//   foreach(reduction(+:kinetic_temp) reduction(+:surface_temp) reduction(+:elastic_temp) reduction(+:dissipated_temp)) {

//     // ==== Calculating kinetic energy
//     kinetic_temp += f[]*(2.0*pi*y)*0.5*(sq(u.x[]) + sq(u.y[]))*sq(Delta);

//     // ==== Calculating surface energy
//     double fmin = 1e-03;
//     if( cfilter_hugo (point, f, fmin) ) {
//       coord n = interface_normal (point, f);
//       double alpha = plane_alpha (f[], n);
//       coord v[2];
//       facets (n, alpha, v);
      
//       double p1x = x + v[0].x*Delta;
//       double p1y = y + v[0].y*Delta;
//       double p2x = x + v[1].x*Delta;
//       double p2y = y + v[1].y*Delta;

//       double center_y = 0.5*(p1y + p2y);
//       double center_x = 0.5*(p1x + p2x);

//       if( center_x>1e-3 )
//         surface_temp += 2.0*pi*center_y*sqrt( sq(p1x - p2x) + sq(p1y - p2y) );
//     }

//     // ==== Calculating elastic energy
//     // HOW?

//     // ==== Calculating viscously dissipated energy
//     // Note: this is only the dissipated energy at the current timestep
//     // Note: If you want the total dissipation over the simulation, you have to integrate this over time
//     double dudx = (u.x[1, 0] - u.x[-1, 0])/(2.0*Delta);
//     double dudy = (u.x[0, 1] - u.x[0, -1])/(2.0*Delta);
//     double dvdx = (u.y[1, 0] - u.y[-1, 0])/(2.0*Delta);
//     double dvdy = (u.y[0, 1] - u.y[0, -1])/(2.0*Delta);
//     double axi_term = u.y[]/max(y, 1e-12);
//     double norm_strain = sqrt( sq(dudx) + sq(dvdy) + sq(axi_term) + 0.5*sq(dudy + dvdx) );
//     double bingham_viscosity = Oh + J/(2.0*norm_strain + 1e-10)*(1.0 - exp(-norm_strain/regularization));
//     dissipated_temp += f[]*(2.0*pi*y)*sq(Delta)*bingham_viscosity*(2.0*dudx*dudx + 2.0*dvdy*dvdy + 2.0*axi_term + (dvdx  + dudy)*(dvdx  + dudy));
//   }

//   *kinetic = kinetic_temp;
//   *surface = surface_temp;
//   *elastic = elastic_temp;
//   *dissipated += dt*dissipated_temp;

//   return;
// }

void **AllocateMatrix(int Linhas, int Colunas, size_t TamanhoElemento, void *ValorInicial)
{
    void **matriz;
    int i, j, byteAtualValor, byteAtualMatriz;
    unsigned char *charValorInicial, *charMatriz;

    charValorInicial = (unsigned char *)ValorInicial;

    matriz = (void **)malloc( Linhas*sizeof(void *) );
    for( i=0; i<Linhas; i++ )
        matriz[i] = malloc( Colunas*TamanhoElemento );

    //Inicializando todos os elementos com o valor inicial (soh funciona se for um tipo int)
    if( ValorInicial!=NULL ) {
        for(i=0; i<Linhas; i++) {
            charMatriz = (unsigned char *)matriz[i];
            byteAtualMatriz = 0;
            for(j=0; j<Colunas; j++) {
                for( byteAtualValor = 0; byteAtualValor<TamanhoElemento; byteAtualValor++ )
                    charMatriz[byteAtualMatriz++] = charValorInicial[byteAtualValor];
            }
        }
    }


    return matriz;
}

void DeallocateMatrix(void **Matriz, int Linhas, int Colunas)
{
    int i;

    for( i=0; i<Linhas; i++ )
        free(Matriz[i]);
    free(Matriz);
}

void PrintMeshDataDump(int n, double time, int n_cells, double domain_size, scalar *list_scalar_data, const char **list_scalar_names)
{
  int i, j;
  double box[2][2];

  // Number of cells (uniform) and cell_size
  int base_nx = n_cells;
  double dx = domain_size/base_nx;
  
  // Getting the limits of the domain
  double xMin = 1e+10, xMax = -1e+10, yMin = 1e+10, yMax = -1e+10;
  foreach(serial) {
    if( (x - 0.5*Delta) < xMin )
      xMin = x - 0.5*Delta;
    if( (y - 0.5*Delta) < yMin )
      yMin = y - 0.5*Delta;
    if( (x + 0.5*Delta) > xMax )
      xMax = x + 0.5*Delta;
    if( (y + 0.5*Delta) > yMax )
      yMax = y + 0.5*Delta;
  }

  // Getting the actual number of cells nx, ny
  int nx = round( (xMax - xMin)/dx );
  int ny = round( (yMax - yMin)/dx );
  int num_cells = nx*ny;
  
  // Counting how many scalars we want to print
  int number_of_scalar_fields = list_len(list_scalar_data);

  // Allocating memory for all the scalar fields (plus x and y arrays)
  double **values = (double **)malloc((number_of_scalar_fields+2)*sizeof(double *));
  for(i=0; i<number_of_scalar_fields+2; i++)
    values[i] = (double *)malloc(num_cells*sizeof(double));

  int cell_index = 0;
  for(i=0; i<nx; i++) {
    for(j=0; j<ny; j++) {
      double x_grid = xMin + (i+0.5)*dx;
      double y_grid = yMin + (j+0.5)*dx;

      values[0][cell_index] = x_grid;
      values[1][cell_index] = y_grid;

      /// === Value of the velocity at the center of the channel
      int scalar_index = 2;
      for(scalar s in list_scalar_data)
        values[scalar_index++][cell_index] = interpolate(s, x_grid, y_grid);
      
      cell_index++;
    }
  }

  
  char fileName[900];
  sprintf(fileName, "%s/MeshDump-N%d.bin", folder_name, n);
  FILE *file_out = (FILE *)fopen(fileName, "w");

  // Printing a header with name of scalars and the dimensiona of the mesh (nx, ny)
  int scalar_index = 0;
  fprintf(file_out, "%d %d\n", nx, ny);
  fprintf(file_out, "x;y");
  for(scalar s in list_scalar_data)
    fprintf(file_out, ";%s", list_scalar_names[scalar_index++]);
  fprintf(file_out, "\n");

  fwrite(values[0], sizeof(double), num_cells, file_out);
  fwrite(values[1], sizeof(double), num_cells, file_out);
  scalar_index = 2;
  for(scalar s in list_scalar_data)
    fwrite(values[scalar_index++], sizeof(double), num_cells, file_out);
  fclose(file_out);


  // Releasing allocated memory
  for(i=0; i<number_of_scalar_fields+2; i++)
    free(values[i]);
  free(values);
  return;
}

