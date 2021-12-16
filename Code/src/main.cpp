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

#define SURFACE_TAG   1                                                                             // Surface tag.
#define BORDER_TAG    9                                                                             // Border tag.
#define SIDE_X_TAG    10                                                                            // Side "x" tag.
#define SIDE_Y_TAG    11                                                                            // Side "y" tag.
#define CURVE_DIM     1                                                                             // Curve dimension.
#define SURFACE_DIM   2                                                                             // Surface dimension.
#define BORDER_DIM    1                                                                             // Border dimension.
#define SIDE_X_DIM    1                                                                             // Side "x" dimension.
#define SIDE_Y_DIM    1                                                                             // Side "y" dimension.
#define DS            0.05f                                                                         // vacuum elementary cell side.
#define EPSILON       0.01f                                                                         // Tolerance for cell detection.
#define CELL_VERTICES 4                                                                             // Number of vertices per elementary cell.

#define M_MAX         100                                                                           // Maximum allowed number of rejections.
#define HX_INIT       0.8f                                                                          // Longitudinal magnetic field.
#define HZ_INIT       0.01f                                                                         // Transverse magnetic field.
#define T_INIT        0.0125f                                                                       // Temperature.
#define ALPHA_INIT    1.0f                                                                          // Radial exponent.
#define THETA_INIT    1.5f*M_PI                                                                     // Theta angle.

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
#define KERNEL_0      "ramp_up.cl"                                                                  // OpenCL kernel source.
#define KERNEL_1      "thekernel_1.cl"                                                              // OpenCL kernel source.
#define KERNEL_2      "thekernel_2.cl"                                                              // OpenCL kernel source.
#define UTILITIES     "utilities.cl"                                                                // OpenCL utilities source.
#define MESH_FILE     "Periodic_square.msh"                                                         // GMSH mesh.
//#define MESH_FILE     "Periodic_segment.msh"                                                           // GMSH mesh.
#define MESH          GMSH_HOME MESH_FILE                                                           // GMSH mesh (full path).

// INCLUDES:
#include "nu.hpp"                                                                                   // Neutrino's header file.

// utility structure for realtime plot
struct ScrollingBuffer {
  int MaxSize;
  int Offset;
  ImVector<ImVec2> Data;
  ScrollingBuffer(
                  int max_size = 2000
                 ) {
    MaxSize = max_size;
    Offset  = 0;
    Data.reserve (MaxSize);
  }
  void AddPoint (
                 float x,
                 float y
                ) {
    if(Data.size () < MaxSize)
      Data.push_back (ImVec2 (x,y));
    else {
      Data[Offset] = ImVec2 (x,y);
      Offset       = (Offset + 1) % MaxSize;
    }
  }
  void Erase () {
    if(Data.size () > 0)
    {
      Data.shrink (0);
      Offset = 0;
    }
  }
};

// utility structure for realtime plot
struct RollingBuffer {
  float Span;
  ImVector<ImVec2> Data;
  RollingBuffer() {
    Span = 10.0f;
    Data.reserve (2000);
  }
  void AddPoint (
                 float x,
                 float y
                ) {
    float xmod = fmodf (x, Span);
    if(!Data.empty () && xmod < Data.back ().x)
      Data.shrink (0);
    Data.push_back (ImVec2 (xmod, y));
  }
};

void ShowDemo_RealtimePlots () {
  ImGui::BulletText ("Move your mouse to change the data!");
  ImGui::BulletText ("This example assumes 60 FPS. Higher FPS requires larger buffer size.");
  static ScrollingBuffer sdata1, sdata2;
  static RollingBuffer   rdata1, rdata2;
  ImVec2                 mouse   = ImGui::GetMousePos ();
  static float           t       = 0;
  t          += ImGui::GetIO ().DeltaTime;
  sdata1.AddPoint (t, mouse.x * 0.0005f);
  rdata1.AddPoint (t, mouse.x * 0.0005f);
  sdata2.AddPoint (t, mouse.y * 0.0005f);
  rdata2.AddPoint (t, mouse.y * 0.0005f);

  static float           history = 10.0f;
  ImGui::SliderFloat ("History",&history,1,30,"%.1f s");
  rdata1.Span = history;
  rdata2.Span = history;

  static ImPlotAxisFlags flags   = ImPlotAxisFlags_NoTickLabels;

  if(ImPlot::BeginPlot ("##Scrolling", ImVec2 (-1,150)))
  {
    ImPlot::SetupAxes (NULL, NULL, flags, flags);
    ImPlot::SetupAxisLimits (ImAxis_X1,t - history, t, ImGuiCond_Always);
    ImPlot::SetupAxisLimits (ImAxis_Y1,0,1);
    ImPlot::SetNextFillStyle (IMPLOT_AUTO_COL,0.5f);
    ImPlot::PlotShaded (
                        "Mouse X",
                        &sdata1.Data[0].x,
                        &sdata1.Data[0].y,
                        sdata1.Data.size (),
                        -INFINITY,
                        sdata1.Offset,
                        2 * sizeof(float)
                       );
    ImPlot::PlotLine (
                      "Mouse Y",
                      &sdata2.Data[0].x,
                      &sdata2.Data[0].y,
                      sdata2.Data.size (),
                      sdata2.Offset,
                      2*sizeof(float)
                     );
    ImPlot::EndPlot ();
  }
  if(ImPlot::BeginPlot ("##Rolling", ImVec2 (-1,150)))
  {
    ImPlot::SetupAxes (NULL, NULL, flags, flags);
    ImPlot::SetupAxisLimits (ImAxis_X1,0,history, ImGuiCond_Always);
    ImPlot::SetupAxisLimits (ImAxis_Y1,0,1);
    ImPlot::PlotLine (
                      "Mouse X",
                      &rdata1.Data[0].x,
                      &rdata1.Data[0].y,
                      rdata1.Data.size (),
                      0,
                      2 * sizeof(float)
                     );
    ImPlot::PlotLine (
                      "Mouse Y",
                      &rdata2.Data[0].x,
                      &rdata2.Data[0].y,
                      rdata2.Data.size (),
                      0,
                      2 * sizeof(float)
                     );
    ImPlot::EndPlot ();
  }
}

int main ()
{
  // INDICES:
  size_t                           i;                                                               // Index [#].
  size_t                           j;                                                               // Index [#].
  size_t                           j_min;                                                           // Index [#].
  size_t                           j_max;                                                           // Index [#].

  // MOUSE PARAMETERS:
  float                            ms_orbit_rate   = 1.0f;                                          // Orbit rotation rate [rev/s].
  float                            ms_pan_rate     = 5.0f;                                          // Pan translation rate [m/s].
  float                            ms_decaytime    = 1.25f;                                         // Pan LP filter decay time [s].

  // GAMEPAD PARAMETERS:
  float                            gmp_orbit_rate  = 1.0f;                                          // Orbit angular rate coefficient [rev/s].
  float                            gmp_pan_rate    = 1.0f;                                          // Pan translation rate [m/s].
  float                            gmp_decaytime   = 1.25f;                                         // Low pass filter decay time [s].
  float                            gmp_deadzone    = 0.30f;                                         // Gamepad joystick deadzone [0...1].

  // OPENGL:
  nu::opengl*                      gl              = new nu::opengl (NAME,SX,SY,OX,OY,PX,PY,PZ);    // OpenGL context.
  nu::shader*                      S               = new nu::shader ();                             // OpenGL shader program.
  nu::projection_mode              proj_mode       = nu::MONOCULAR;                                 // OpenGL projection mode.

  // OPENCL:
  nu::opencl*                      cl              = new nu::opencl (nu::GPU);                      // OpenCL context.
  nu::kernel*                      K0              = new nu::kernel ();                             // OpenCL kernel array.
  nu::kernel*                      K1              = new nu::kernel ();                             // OpenCL kernel array.
  nu::kernel*                      K2              = new nu::kernel ();                             // OpenCL kernel array.
  nu::float4*                      color           = new nu::float4 (0);                            // Color [].
  nu::float4*                      position        = new nu::float4 (1);                            // Position [m].
  nu::int1*                        central         = new nu::int1 (2);                              // Central nodes.
  nu::int1*                        neighbour       = new nu::int1 (3);                              // Neighbour.
  nu::int1*                        offset          = new nu::int1 (4);                              // Offset.
  nu::float1*                      theta           = new nu::float1 (5);                            // Theta.
  nu::float1*                      theta_int       = new nu::float1 (6);                            // Theta (intermediate value).
  nu::int4*                        state_theta     = new nu::int4 (7);                              // Random generator state.
  nu::int4*                        state_threshold = new nu::int4 (8);                              // Random generator state.
  nu::int1*                        max_rejections  = new nu::int1 (9);                              // Maximum allowed number of rejections.
  nu::float1*                      longitudinal_H  = new nu::float1 (10);                           // Longitudinal magnetic field.
  nu::float1*                      transverse_H    = new nu::float1 (11);                           // Transverse magnetic field.
  nu::float1*                      temperature     = new nu::float1 (12);                           // Temperature.
  nu::float1*                      radial_exponent = new nu::float1 (13);                           // Radial exponent.
  nu::float1*                      ds              = new nu::float1 (14);                           // Mesh side.
  nu::float1*                      dt              = new nu::float1 (15);                           // Time step [s].

  // MESH:
  nu::mesh*                        vacuum          = new nu::mesh (MESH);                           // False vacuum domain.
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
  float                            x_min       = -1.0f;                                             // "x_min" spatial boundary [m].
  float                            x_max       = +1.0f;                                             // "x_max" spatial boundary [m].
  float                            y_min       = -1.0f;                                             // "y_min" spatial boundary [m].
  float                            y_max       = +1.0f;                                             // "y_max" spatial boundary [m].
  float                            dx;                                                              // x-axis mesh spatial size [m].
  float                            dy;                                                              // y-axis mesh spatial size [m].

  // SIMULATION VARIABLES:
  float                            Hx          = HX_INIT;                                           // Longitudinal magnetic field.
  float                            Hz          = HZ_INIT;                                           // Transverse magnetic field.
  float                            T           = T_INIT;                                            // Temperature.
  float                            alpha       = ALPHA_INIT;                                        // Radial exponent.
  float                            theta_angle = THETA_INIT;                                        // Theta angle.
  float                            dt_simulation;                                                   // Simulation time step [s].

  // BACKUP:
  std::vector<nu_float4_structure> initial_position;                                                // Backing up initial data...
  std::vector<float>               initial_theta;                                                   // Backing up initial data...
  std::vector<float>               initial_theta_int;                                               // Backing up initial data...

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// DATA INITIALIZATION //////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
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
  //vacuum->process (SIDE_X_TAG, SURFACE_DIM, nu::MSH_LIN_2);                                         // Processing mesh...
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

    state_theta->data.push_back ({rand (), rand (), rand (), rand ()});                             // Setting state_sz seed...
    state_threshold->data.push_back ({rand (), rand (), rand (), rand ()});                         // Setting state_th seed...
    color->data.push_back ({0.0f, 1.0f, 0.0f, 1.0f});                                               // Setting node color...
    theta->data.push_back (theta_angle);                                                            // Setting initial theta...
    theta_int->data.push_back (0.0f);                                                               // Setting initial theta (intermediate value)...

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
  border       = vacuum->node;                                                                      // Getting nodes on border...
  border_nodes = border.size ();                                                                    // Getting the number of nodes on border...

  // SETTING NEUTRINO ARRAYS ("border" depending):
  for(i = 0; i < border_nodes; i++)
  {
    // Doing nothing!
  }

  // SETTING INITIAL PARAMETERS:
  max_rejections->data.push_back (M_MAX);                                                           // Setting maximum allowed number of rejections...
  temperature->data.push_back (T);                                                                  // Setting temperature...
  longitudinal_H->data.push_back (Hx);                                                              // Setting longitudinal magnetic field...
  transverse_H->data.push_back (Hz);                                                                // Setting transverse magnetic field...
  radial_exponent->data.push_back (alpha);                                                          // Setting radial exponent...
  ds->data.push_back (dx);                                                                          // Setting mesh side...

  // SETTING INITIAL DATA BACKUP:
  initial_position  = position->data;                                                               // Setting backup data...
  initial_theta     = theta->data;                                                                  // Setting backup data...
  initial_theta_int = theta_int->data;                                                              // Setting backup data...

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// OPENCL KERNELS INITIALIZATION /////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  K0->addsource (std::string (KERNEL_HOME) + std::string (UTILITIES));                              // Setting kernel source file...
  K0->addsource (std::string (KERNEL_HOME) + std::string (KERNEL_0));                               // Setting kernel source file...
  K0->build (nodes, 0, 0);                                                                          // Building kernel program...
  K1->addsource (std::string (KERNEL_HOME) + std::string (UTILITIES));                              // Setting kernel source file...
  K1->addsource (std::string (KERNEL_HOME) + std::string (KERNEL_1));                               // Setting kernel source file...
  K1->build (nodes, 0, 0);                                                                          // Building kernel program...
  K2->addsource (std::string (KERNEL_HOME) + std::string (UTILITIES));                              // Setting kernel source file...
  K2->addsource (std::string (KERNEL_HOME) + std::string (KERNEL_2));                               // Setting kernel source file...
  K2->build (nodes, 0, 0);                                                                          // Building kernel program...

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// OPENGL SHADERS INITIALIZATION /////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  S->addsource (std::string (SHADER_HOME) + std::string (SHADER_VERT), nu::VERTEX);                 // Setting shader source file...
  S->addsource (std::string (SHADER_HOME) + std::string (SHADER_GEOM), nu::GEOMETRY);               // Setting shader source file...
  S->addsource (std::string (SHADER_HOME) + std::string (SHADER_FRAG), nu::FRAGMENT);               // Setting shader source file...
  S->build (nodes);                                                                                 // Building shader program...

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////// SETTING OPENCL KERNEL ARGUMENTS /////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  cl->write ();                                                                                     // Writing OpenCL data...

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// CHARGING RANDOM GENERATORS ////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  cl->acquire ();                                                                                   // Acquiring OpenCL kernel...
  cl->execute (K0, nu::WAIT);                                                                       // Executing OpenCL kernel...
  cl->release ();                                                                                   // Releasing OpenCL kernel...

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////// APPLICATION LOOP ////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  while(!gl->closed ())                                                                             // Opening window...
  {
    cl->get_tic ();                                                                                 // Getting "tic" [us]...
    cl->acquire ();                                                                                 // Acquiring OpenCL kernel...
    cl->execute (K1, nu::WAIT);                                                                     // Executing OpenCL kernel...
    cl->execute (K2, nu::WAIT);                                                                     // Executing OpenCL kernel...
    cl->release ();                                                                                 // Releasing OpenCL kernel...

    gl->begin ();                                                                                   // Beginning gl...
    gl->poll_events ();                                                                             // Polling gl events...
    gl->mouse_navigation (ms_orbit_rate, ms_pan_rate, ms_decaytime);                                // Polling mouse...
    gl->gamepad_navigation (gmp_orbit_rate, gmp_pan_rate, gmp_decaytime, gmp_deadzone);             // Polling gamepad...
    gl->plot (S, proj_mode);                                                                        // Plotting shared arguments...

    ImGui::Begin ("FALSE VACUUM PARAMETERS", NULL, ImGuiWindowFlags_AlwaysAutoResize);              // Beginning window...
    ImGui::PushItemWidth (200);                                                                     // Setting window width [px]...

    ImGui::PushStyleColor (ImGuiCol_Text, IM_COL32 (0,255,0,255));                                  // Setting text color...
    ImGui::Text ("Angle:                             ");                                            // Writing text...
    ImGui::PopStyleColor ();                                                                        // Restoring text color...
    ImGui::SameLine ();                                                                             // Staying on same line...
    ImGui::Text ("[rad] ");                                                                         // Writing text...
    ImGui::SameLine ();                                                                             // Staying on same line...
    ImGui::SliderFloat (" = theta ", &theta_angle, 0.0f, 2.0f*M_PI);
    //ImGui::InputFloat (" = theta ", &theta_angle);                                                  // Adding input field...

    ImGui::PushStyleColor (ImGuiCol_Text, IM_COL32 (0,255,0,255));                                  // Setting text color...
    ImGui::Text ("Temperature:                       ");                                            // Writing text...
    ImGui::PopStyleColor ();                                                                        // Restoring text color...
    ImGui::SameLine ();                                                                             // Staying on same line...
    ImGui::Text ("[K]   ");                                                                         // Writing text...
    ImGui::SameLine ();                                                                             // Staying on same line...
    ImGui::InputFloat (" = T ", &T);                                                                // Adding input field...

    ImGui::PushStyleColor (ImGuiCol_Text, IM_COL32 (0,255,0,255));                                  // Setting text color...
    ImGui::Text ("Radial exponent:                   ");                                            // Writing text...
    ImGui::PopStyleColor ();                                                                        // Restoring text color...
    ImGui::SameLine ();                                                                             // Staying on same line...
    ImGui::Text ("[]    ");                                                                         // Writing text...
    ImGui::SameLine ();                                                                             // Staying on same line...
    ImGui::InputFloat (" = alpha ", &alpha);                                                        // Adding input field...

    ImGui::PushStyleColor (ImGuiCol_Text, IM_COL32 (0,255,0,255));                                  // Setting text color...
    ImGui::Text ("Longitudinal magnetic field:       ");                                            // Writing text...
    ImGui::PopStyleColor ();                                                                        // Restoring text color...
    ImGui::SameLine ();                                                                             // Staying on same line...
    ImGui::Text ("[T]   ");                                                                         // Writing text...
    ImGui::SameLine ();                                                                             // Staying on same line...
    ImGui::InputFloat (" = Hx", &Hx);                                                               // Adding input field...

    ImGui::PushStyleColor (ImGuiCol_Text, IM_COL32 (0,255,0,255));                                  // Setting text color...
    ImGui::Text ("Transverse magnetic field:         ");                                            // Writing text...
    ImGui::PopStyleColor ();                                                                        // Restoring text color...
    ImGui::SameLine ();                                                                             // Staying on same line...
    ImGui::Text ("[T]   ");                                                                         // Writing text...
    ImGui::SameLine ();                                                                             // Staying on same line...
    ImGui::InputFloat (" = Hz", &Hz);                                                               // Adding input field...

    if(ImGui::Button ("(U)pdate") || gl->key_U)
    {
      // UPDATING PHYSICAL PARAMETERS:
      longitudinal_H->data[0]  = Hx;                                                                // Setting longitudinal magnetic field...
      transverse_H->data[0]    = Hz;                                                                // Setting transverse magnetic field...
      temperature->data[0]     = T;                                                                 // Setting temperature...
      radial_exponent->data[0] = alpha;                                                             // Setting radial exponent...
      dt->data[0]              = dt_simulation;                                                     // Setting simulation time step...

      // RESETTING NEUTRINO ARRAYS ("surface" depending):
      for(i = 0; i < nodes; i++)
      {
        // NODE PROPERTIES:
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
          // LINK PROPERTIES:
          //stiffness->data[j] = K;                                                                   // Setting link stiffness...
        }
      }

      cl->write (10);                                                                               // Writing OpenCL data...
      cl->write (11);                                                                               // Writing OpenCL data...
      cl->write (12);                                                                               // Writing OpenCL data...
      cl->write (13);                                                                               // Writing OpenCL data...
      cl->write (15);                                                                               // Writing OpenCL data...
    }

    ImGui::SameLine (100);

    if(ImGui::Button ("(R)estart") || gl->button_TRIANGLE || gl->key_R)
    {
      position->data = initial_position;                                                            // Restoring backup...

      // Resetting theta for all nodes:
      for(i = 0; i < nodes; i++)
      {
        std::cout << "i = " << i << ", node index = " << vacuum->node[i] << std::endl;              // Printing message...

        color->data.clear ();                                                                       // Clearing color vector...
        theta->data.clear ();                                                                       // Clearing theta vector...
        theta_int->data.clear ();                                                                   // Clearing theta (intermediate) vector...

        color->data.push_back ({0.0f, 1.0f, 0.0f, 1.0f});                                           // Setting node color...
        theta->data.push_back (theta_angle);                                                        // Setting initial theta...
        theta_int->data.push_back (0.0f);                                                           // Setting initial theta (intermediate value)...
      }

      cl->write (1);                                                                                // Writing data...
      cl->write (5);                                                                                // Writing data...
      cl->write (6);                                                                                // Writing data...
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

    ShowDemo_RealtimePlots ();

    ImGui::End ();                                                                                  // Finishing window...

    gl->end ();                                                                                     // Ending gl...
    cl->get_toc ();                                                                                 // Getting "toc" [us]...
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////// CLEANUP ////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  delete cl;                                                                                        // Deleting OpenCL context...
  delete gl;                                                                                        // Deleting OpenGL context...
  delete S;                                                                                         // Deleting shader...
  delete color;                                                                                     // Deleting color data...
  delete position;                                                                                  // Deleting position data...
  delete central;                                                                                   // Deleting centrals...
  delete neighbour;                                                                                 // Deleting neighbours...
  delete offset;                                                                                    // Deleting offset...
  delete theta;                                                                                     // Deleting theta...
  delete theta_int;                                                                                 // Deleting theta (intermediate)...
  delete state_theta;                                                                               // Deleting random generator state...
  delete state_threshold;                                                                           // Deleting random generator state...
  delete max_rejections;                                                                            // Deleting max_rejections...
  delete longitudinal_H;                                                                            // Deleting longitudinal magnetic field...
  delete transverse_H;                                                                              // Deleting tranverse magnetic field...
  delete temperature;                                                                               // Deleting temperature...
  delete radial_exponent;                                                                           // Deleting radial exponent...
  delete ds;                                                                                        // Deleting mesh side...
  delete dt;                                                                                        // Deleting time step data...
  delete K1;                                                                                        // Deleting OpenCL kernel...
  delete K2;                                                                                        // Deleting OpenCL kernel...
  delete vacuum;                                                                                    // deleting vacuum mesh...

  return 0;
}
