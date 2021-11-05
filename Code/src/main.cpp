/// @file

#define INTEROP       true                                                                          // "true" = use OpenGL-OpenCL interoperability.
#define SX            800                                                                           // Window x-size [px].
#define SY            600                                                                           // Window y-size [px].
#define NAME          "Neutrino - Spin-bubble"                                                      // Window name.
#define OX            0.0f                                                                          // x-axis orbit initial rotation.
#define OY            0.0f                                                                          // y-axis orbit initial rotation.
#define PX            0.0f                                                                          // x-axis pan initial translation.
#define PY            0.0f                                                                          // y-axis pan initial translation.
#define PZ            -2.0f                                                                         // z-axis pan initial translation.

#define SURFACE_TAG   2                                                                             // Surface tag.
#define BORDER_TAG    6                                                                             // Border tag.
#define SIDE_X_TAG    7                                                                             // Side "x" tag.
#define SIDE_Y_TAG    8                                                                             // Side "y" tag.
#define SURFACE_DIM   2                                                                             // Surface dimension.
#define BORDER_DIM    1                                                                             // Border dimension.
#define SIDE_X_DIM    1                                                                             // Side "x" dimension.
#define SIDE_Y_DIM    1                                                                             // Side "y" dimension.
#define DS            0.05                                                                          // vacuum elementary cell side.
#define EPSILON       0.01                                                                          // Tolerance for cell detection.
#define CELL_VERTICES 4                                                                             // Number of vertices per elementary cell.

#ifdef __linux__
  #define SHADER_HOME "../../Code/shader/"                                                          // Linux OpenGL shaders directory.
  #define KERNEL_HOME "../../Code/kernel/"                                                          // Linux OpenCL kernels directory.
  #define GMSH_HOME   "../../Code/mesh/"                                                            // Linux GMSH mesh directory.
#endif

#ifdef WIN32
  #define SHADER_HOME "..\\..\\Code\\shader\\"                                                      // Windows OpenGL shaders directory.
  #define KERNEL_HOME "..\\..\\Code\\kernel\\"                                                      // Windows OpenCL kernels directory.
  #define GMSH_HOME   "..\\..\\Code\\mesh\\"                                                        // Linux GMSH mesh directory.
#endif

#define SHADER_VERT   "voxel_vertex.vert"                                                           // OpenGL vertex shader.
#define SHADER_GEOM   "voxel_geometry.geom"                                                         // OpenGL geometry shader.
#define SHADER_FRAG   "voxel_fragment.frag"                                                         // OpenGL fragment shader.
#define KERNEL_1      "thekernel_1.cl"                                                              // OpenCL kernel source.
#define UTILITIES     "utilities.cl"                                                                // OpenCL utilities source.
#define MESH_FILE     "Square_quadrangles.msh"                                                      // GMSH mesh.
#define MESH          GMSH_HOME MESH_FILE                                                           // GMSH mesh (full path).

// INCLUDES:
#include "nu.hpp"                                                                                   // Neutrino's header file.

int main ()
{
  // INDICES:
  size_t                           i;                                                               // Index [#].
  size_t                           j;                                                               // Index [#].
  size_t                           j_min;                                                           // Index [#].
  size_t                           j_max;                                                           // Index [#].

  // MOUSE PARAMETERS:
  float                            ms_orbit_rate  = 1.0f;                                           // Orbit rotation rate [rev/s].
  float                            ms_pan_rate    = 5.0f;                                           // Pan translation rate [m/s].
  float                            ms_decaytime   = 1.25f;                                          // Pan LP filter decay time [s].

  // GAMEPAD PARAMETERS:
  float                            gmp_orbit_rate = 1.0f;                                           // Orbit angular rate coefficient [rev/s].
  float                            gmp_pan_rate   = 1.0f;                                           // Pan translation rate [m/s].
  float                            gmp_decaytime  = 1.25f;                                          // Low pass filter decay time [s].
  float                            gmp_deadzone   = 0.30f;                                          // Gamepad joystick deadzone [0...1].

  // OPENGL:
  nu::opengl*                      gl             = new nu::opengl (NAME,SX,SY,OX,OY,PX,PY,PZ);     // OpenGL context.
  nu::shader*                      S              = new nu::shader ();                              // OpenGL shader program.
  nu::projection_mode              proj_mode      = nu::MONOCULAR;                                  // OpenGL projection mode.

  // OPENCL:
  nu::opencl*                      cl             = new nu::opencl (nu::GPU);                       // OpenCL context.
  nu::kernel*                      K1             = new nu::kernel ();                              // OpenCL kernel array.
  nu::float4*                      color          = new nu::float4 (0);                             // Color [].
  nu::float4*                      position       = new nu::float4 (1);                             // Position [m].
  nu::int1*                        central        = new nu::int1 (2);                               // Central nodes.
  nu::int1*                        neighbour      = new nu::int1 (3);                               // Neighbour.
  nu::int1*                        offset         = new nu::int1 (4);                               // Offset.
  nu::int4*                        state          = new nu::int4 (5);                               // Random generator state.
  nu::float1*                      dt             = new nu::float1 (6);                             // Time step [s].

  // MESH:
  nu::mesh*                        vacuum         = new nu::mesh (MESH);                            // False vaccum domain.
  size_t                           nodes;                                                           // Number of nodes.
  size_t                           elements;                                                        // Number of elements.
  size_t                           groups;                                                          // Number of groups.
  size_t                           neighbours;                                                      // Number of neighbours.
  std::vector<size_t>              side_x;                                                          // Nodes on "x" side.
  std::vector<size_t>              side_y;                                                          // Nodes on "y" side.
  std::vector<GLint>               border;                                                          // Nodes on border.
  size_t                           side_x_nodes;                                                    // Number of nodes in "x" direction [#].
  size_t                           side_y_nodes;                                                    // Number of nodes in "x" direction [#].
  size_t                           border_nodes;                                                    // Number of border nodes.
  float                            x_min = -1.0f;                                                   // "x_min" spatial boundary [m].
  float                            x_max = +1.0f;                                                   // "x_max" spatial boundary [m].
  float                            y_min = -1.0f;                                                   // "y_min" spatial boundary [m].
  float                            y_max = +1.0f;                                                   // "y_max" spatial boundary [m].
  float                            dx;                                                              // x-axis mesh spatial size [m].
  float                            dy;                                                              // y-axis mesh spatial size [m].

  // SIMULATION PARAMETERS:
  float                            L;
  float                            hx;
  float                            hz;

  // SIMULATION VARIABLES:
  float                            dt_simulation;                                                   // Simulation time step [s].

  // BACKUP:
  std::vector<nu_float4_structure> initial_position;                                                // Backing up initial data...

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// DATA INITIALIZATION ///////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  // MESH "X" SIDE:
  vacuum->process (SIDE_X_TAG, SIDE_X_DIM, nu::MSH_PNT);                                            // Processing mesh...
  side_x_nodes    = vacuum->node.size ();                                                           // Getting number of nodes along "x" side...

  // MESH "Y" SIDE:
  vacuum->process (SIDE_Y_TAG, SIDE_Y_DIM, nu::MSH_PNT);                                            // Processing mesh...
  side_y_nodes    = vacuum->node.size ();                                                           // Getting number of nodes along "y" side...

  // COMPUTING PHYSICAL PARAMETERS:
  dx              = (x_max - x_min)/(side_x_nodes - 1);                                             // x-axis mesh spatial size [m].
  dy              = (y_max - y_min)/(side_y_nodes - 1);                                             // y-axis mesh spatial size [m].
  dt_simulation   = 1.0f;
  dt->data.push_back (dt_simulation);                                                               // Setting simulation time step...

  // MESH SURFACE:
  vacuum->process (SURFACE_TAG, SURFACE_DIM, nu::MSH_QUA_4);                                        // Processing mesh...
  position->data  = vacuum->node_coordinates;                                                       // Setting all node coordinates...
  neighbour->data = vacuum->neighbour;                                                              // Setting neighbour indices...
  offset->data    = vacuum->neighbour_offset;                                                       // Setting neighbour offsets...
  nodes           = vacuum->node.size ();                                                           // Getting the number of nodes...
  elements        = vacuum->element.size ();                                                        // Getting the number of elements...
  groups          = vacuum->group.size ();                                                          // Getting the number of groups...
  neighbours      = vacuum->neighbour.size ();                                                      // Getting the number of neighbours...
  std::cout << "nodes = " << nodes << std::endl;                                                    // Printing message...
  std::cout << "elements = " << elements/CELL_VERTICES << std::endl;                                // Printing message...
  std::cout << "groups = " << groups/CELL_VERTICES << std::endl;                                    // Printing message...
  std::cout << "neighbours = " << neighbours << std::endl;                                          // Printing message...

  // SETTING NEUTRINO ARRAYS ("surface" depending):
  for(i = 0; i < nodes; i++)
  {
    std::cout << "i = " << i << ", node index = " << vacuum->node[i] << ", neighbour indices:";     // Printing message...

    state->data.push_back ({int(234 + i), int(545 + 2*i), int(6 + 4*i), int(645 + i)});
    color->data.push_back ({0.0f, 1.0f, 0.0f, 1.0f});                                               // Setting node color...

    // Computing minimum element offset index:
    if(i == 0)
    {
      j_min = 0;                                                                                    // Setting minimum element offset index...
    }
    else
    {
      j_min = offset->data[i - 1];                                                                  // Setting minimum element offset index...
    }

    j_max = offset->data[i];                                                                        // Setting maximum element offset index...

    for(j = j_min; j < j_max; j++)
    {
      central->data.push_back (vacuum->node[i]);                                                    // Building central node tuple...

      std::cout << " " << neighbour->data[j];                                                       // Printing message...
    }

    std::cout << std::endl;                                                                         // Printing message...
  }

  // MESH BORDER:
  vacuum->process (BORDER_TAG, BORDER_DIM, nu::MSH_PNT);                                            // Processing mesh...
  border           = vacuum->node;                                                                  // Getting nodes on border...
  border_nodes     = border.size ();                                                                // Getting the number of nodes on border...

  // SETTING NEUTRINO ARRAYS ("border" depending):
  for(i = 0; i < border_nodes; i++)
  {
    // Doing nothing!
  }

  // SETTING INITIAL DATA BACKUP:
  initial_position = position->data;                                                                // Setting backup data...

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// OPENCL KERNELS INITIALIZATION //////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  K1->addsource (std::string (KERNEL_HOME) + std::string (UTILITIES));                              // Setting kernel source file...
  K1->addsource (std::string (KERNEL_HOME) + std::string (KERNEL_1));                               // Setting kernel source file...
  K1->build (nodes, 0, 0);                                                                          // Building kernel program...

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// OPENGL SHADERS INITIALIZATION //////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  S->addsource (std::string (SHADER_HOME) + std::string (SHADER_VERT), nu::VERTEX);                 // Setting shader source file...
  S->addsource (std::string (SHADER_HOME) + std::string (SHADER_GEOM), nu::GEOMETRY);               // Setting shader source file...
  S->addsource (std::string (SHADER_HOME) + std::string (SHADER_FRAG), nu::FRAGMENT);               // Setting shader source file...
  S->build (nodes);                                                                                 // Building shader program...

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////// SETTING OPENCL KERNEL ARGUMENTS //////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  cl->write ();                                                                                     // Writing OpenCL data...

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////// APPLICATION LOOP /////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  while(!gl->closed ())                                                                             // Opening window...
  {
    cl->get_tic ();                                                                                 // Getting "tic" [us]...
    cl->acquire ();                                                                                 // Acquiring OpenCL kernel...
    cl->execute (K1, nu::WAIT);                                                                     // Executing OpenCL kernel...
    cl->release ();                                                                                 // Releasing OpenCL kernel...

    gl->clear ();                                                                                   // Clearing gl...
    gl->poll_events ();                                                                             // Polling gl events...
    gl->mouse_navigation (ms_orbit_rate, ms_pan_rate, ms_decaytime);                                // Polling mouse...
    gl->gamepad_navigation (gmp_orbit_rate, gmp_pan_rate, gmp_decaytime, gmp_deadzone);             // Polling gamepad...
    gl->plot (S, proj_mode);                                                                        // Plotting shared arguments...

    ImGui_ImplOpenGL3_NewFrame ();                                                                  // Initializing ImGui...
    ImGui_ImplGlfw_NewFrame ();                                                                     // Initializing ImGui...
    ImGui::NewFrame ();                                                                             // Initializing ImGui...

    ImGui::Begin ("FALSE VACUUM PARAMETERS", NULL, ImGuiWindowFlags_AlwaysAutoResize);              // Beginning window...
    ImGui::PushItemWidth (200);                                                                     // Setting window width [px]...

    ImGui::PushStyleColor (ImGuiCol_Text, IM_COL32 (0,255,0,255));                                  // Setting text color...
    ImGui::Text ("Magnetic field:       ");                                                         // Writing text...
    ImGui::PopStyleColor ();                                                                        // Restoring text color...
    ImGui::SameLine ();                                                                             // Staying on same line...
    ImGui::Text ("hx =   ");                                                                        // Writing text...
    ImGui::SameLine ();                                                                             // Staying on same line...
    ImGui::InputFloat (" [T]", &hx);                                                                // Adding input field...

    if(ImGui::Button ("(U)pdate") || gl->key_U)
    {
      // RECOMPUTING PHYSICAL PARAMETERS:
      dt_simulation = 1.0f;                                                                         // Simulation time step [s].
      dt->data[0]   = dt_simulation;                                                                // Setting simulation time step...

      // RESETTING NEUTRINO ARRAYS ("surface" depending):
      for(i = 0; i < nodes; i++)
      {
        //mass->data[i] = m;                                                                          // Setting mass...

        // Computing minimum element offset index:
        if(i == 0)
        {
          j_min = 0;                                                                                // Setting minimum element offset index...
        }
        else
        {
          j_min = offset->data[i - 1];                                                              // Setting minimum element offset index...
        }

        j_max = offset->data[i];                                                                    // Setting maximum element offset index...

        for(j = j_min; j < j_max; j++)
        {
          //stiffness->data[j] = K;                                                                   // Setting link stiffness...
        }
      }

      cl->write (6);                                                                                // Writing OpenCL data...
      //cl->write (7);                                                                                // Writing OpenCL data...
      //cl->write (9);                                                                                // Writing OpenCL data...
      //cl->write (10);                                                                               // Writing OpenCL data...
      //cl->write (15);                                                                               // Writing OpenCL data...
    }

    ImGui::SameLine (100);

    if(ImGui::Button ("(R)estart") || gl->button_TRIANGLE || gl->key_R)
    {
      position->data = initial_position;                                                            // Restoring backup...
      cl->write (1);                                                                                // Writing data...
      cl->write (2);                                                                                // Writing data...
      cl->write (3);                                                                                // Writing data...
      cl->write (4);                                                                                // Writing data...
      cl->write (5);                                                                                // Writing data...
    }

    ImGui::SameLine (200);

    if(ImGui::Button ("(M)onocular") || gl->key_M)
    {
      proj_mode = nu::MONOCULAR;                                                                    // Setting monocular projection...
    }

    ImGui::SameLine (300);

    if(ImGui::Button ("(B)inocular") || gl->key_B)
    {
      proj_mode = nu::BINOCULAR;                                                                    // Setting binocular projection...
    }

    ImGui::SameLine (400);

    if(ImGui::Button ("(E)xit") || gl->button_CIRCLE || gl->key_E)
    {
      gl->close ();                                                                                 // Closing gl...
    }

    ImGui::End ();                                                                                  // Finishing window...

    ImGui::Render ();                                                                               // Rendering windows...
    ImGui_ImplOpenGL3_RenderDrawData (ImGui::GetDrawData ());                                       // Rendering windows...

    gl->refresh ();                                                                                 // Refreshing gl...

    cl->get_toc ();                                                                                 // Getting "toc" [us]...
  }

  ImGui_ImplOpenGL3_Shutdown ();                                                                    // Deinitializing ImGui...
  ImGui_ImplGlfw_Shutdown ();                                                                       // Deinitializing ImGui...
  ImGui::DestroyContext ();                                                                         // Deinitializing ImGui...

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////// CLEANUP /////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  delete cl;                                                                                        // Deleting OpenCL context...
  delete gl;                                                                                        // Deleting OpenGL context...
  delete S;                                                                                         // Deleting shader...
  delete color;                                                                                     // Deleting color data...
  delete position;                                                                                  // Deleting position data...
  delete central;                                                                                   // Deleting centrals...
  delete neighbour;                                                                                 // Deleting neighbours...
  delete offset;                                                                                    // Deleting offset...
  delete state;                                                                                     // Deleting random generator state...
  delete dt;                                                                                        // Deleting time step data...
  delete K1;                                                                                        // Deleting OpenCL kernel...
  delete vacuum;                                                                                    // deleting vacuum mesh...

  return 0;
}
