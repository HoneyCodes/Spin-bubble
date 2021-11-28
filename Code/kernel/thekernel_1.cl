/// @file

// Central energy function.
float E_central(float Hx, float Hz, float sz_central)
{
  float E;
  
  E = -(Hx*sqrt(1 - pow(sz_central, 2)) + Hz*sz_central);

  return E;
}

// Neighbour energy function.
float E_neighbour(float C_radial, float sz_neighbour, float sz_central)
{
  float E;
  
  E = -(C_radial*sz_neighbour*sz_central);

  return E;
}

__kernel void thekernel(__global float4*    color,                              // Color.
                        __global float4*    position,                           // Position.
                        __global int*       central,                            // Node.
                        __global int*       nearest,                            // Neighbour.
                        __global int*       offset,                             // Offset.
                        __global int4*      state_x,                            // Random number generator state.
                        __global int4*      state_y,                            // Random number generator state.  
                        __global float*     longitudinal_H,                     // Longitudinal magnetic field.
                        __global float*     transverse_H,                       // Transverse magnetic field.
                        __global float*     sz,                                 // z-component of the spin.  
                        __global float*     dt_simulation)                      // Simulation time step.
{ 
  ////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// INDICES ///////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  uint         i = get_global_id(0);                                            // Global index [#].
  uint         j = 0;                                                           // Neighbour stride index.
  uint         j_min = 0;                                                       // Neighbour stride minimun index.
  uint         j_max = offset[i];                                               // Neighbour stride maximum index.
  uint         k = 0;                                                           // Neighbour tuple index.
  uint         n = central[j_max - 1];                                          // Node index.               
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////// CELL VARIABLES //////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  float4       p                 = position[n];                                 // Central node position.
  float4       neighbour         = (float4)(0.0f, 0.0f, 0.0f, 1.0f);            // Neighbour node position.
  float4       link              = (float4)(0.0f, 0.0f, 0.0f, 1.0f);            // Neighbour link.
  float        L                 = 0.0f;                                        // Neighbour link length.
  float        dt                = dt_simulation[0];                            // Simulation time step [s].
  uint4        st_x              = convert_uint4(state_x[n]);                   // Random generator state.
  uint4        st_y              = convert_uint4(state_y[n]);                   // Random generator state.
  float        Hx                = longitudinal_H[0];
  float        Hz                = transverse_H[0];
  float        E                 = 0.0f;
  float4       c                 = color[n];                                    // Node color.
 
  // COMPUTING STRIDE MINIMUM INDEX:
  if (i == 0)
  {
    j_min = 0;                                                                  // Setting stride minimum (first stride)...
  }
  else
  {
    j_min = offset[i - 1];                                                      // Setting stride minimum (all others)...
  }

  E = E_central(Hx, Hz, sz[n]);

  // COMPUTING ENERGY:
  for (j = j_min; j < j_max; j++)
  {
    k = nearest[j];                                                             // Computing neighbour index...
    neighbour = position[k];                                                    // Getting neighbour position...
    link = neighbour - p;                                                       // Getting neighbour link vector...
    L = length(link.xyz);                                                       // Computing neighbour link length...
    E += E_neighbour(0.5f/L, sz[k], sz[n]);
  }
  
  //p.z = uint_to_float(xoshiro128pp(&sx), -0.05f, +0.05f);                     // Setting z position...
  p.z = 0.1f*rejection (E_central,
                        E_neighbour,
                        sz,
                        j_min,
                        j_max,
                        0.0f,
                        M_PI_F,
                        0.0f,
                        1.0f,
                        &st_x,
                        &st_y,
                        1000);        // Setting z position...
  state_x[n] = convert_int4(st_x);                                              // Updating random generator state...
  state_y[n] = convert_int4(st_y);                                              // Updating random generator state...
  c.xyz = colormap(0.5f*(20.0f*p.z + 1.0f));                                    // Setting color...
  color[n] = c;                                                                 // Updating color...
  position[n] = p;                                                              // Updating position...
}
