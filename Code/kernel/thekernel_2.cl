/// @file

__kernel void thekernel(__global float4*    color,                              // Color.
                        __global float4*    position,                           // Position.
                        __global int*       central,                            // Node.
                        __global int*       nearest,                            // Neighbour.
                        __global int*       offset,                             // Offset.
                        __global float*     sz,                                 // z-component of the spin.  
                        __global float*     sz_int,                             // z-component of the spin (intermediate value). 
                        __global int4*      state_sz,                           // Random number generator state.
                        __global int4*      state_th,                           // Random number generator state. 
                        __global int*       max_rejections,                     // Maximum allowed number of rejections. 
                        __global float*     longitudinal_H,                     // Longitudinal magnetic field.
                        __global float*     transverse_H,                       // Transverse magnetic field.
                        __global float*     temperature,                        // Temperature.
                        __global float*     radial_exponent,                    // Radial exponent.
                        __global float*     ds_simulation,                      // Mesh side.
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
  uint         m = 0;                                                           // Rejection index.           
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////// CELL VARIABLES //////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  float4       p                 = position[n];                                 // Central node position.
  float4       neighbour         = (float4)(0.0f, 0.0f, 0.0f, 1.0f);            // Neighbour node position.
  float4       link              = (float4)(0.0f, 0.0f, 0.0f, 1.0f);            // Neighbour link.
  float        L                 = 0.0f;                                        // Neighbour link length.
  float        dt                = dt_simulation[0];                            // Simulation time step [s].
  float        ds                = ds_simulation[0];                            // Simulation space step.
  uint4        st_sz             = convert_uint4(state_sz[n]);                  // Random generator state.
  uint4        st_th             = convert_uint4(state_th[n]);                  // Random generator state.
  float        Hx                = longitudinal_H[0];                           // Longitudinal magnetic field.
  float        Hz                = transverse_H[0];                             // Transverse magnetic field.
  float        E                 = 0.0f;                                        // Energy function.
  float        En                = 0.0f;                                        // Energy of central node.
  float        sz_rand           = 0.0f;                                        // Flat tandom z-spin.
  float        th_rand           = 0.0f;                                        // Flat random threshold.
  float        T                 = temperature[0];                              // Temperature.
  float        alpha             = radial_exponent[0];                          // Radial exponent.
  float        D                 = 0.0f;                                        // Distributed random z-spin.
  uint         m_max             = max_rejections[0];                           // Maximum allowed number of rejections.
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

  sz[n] = sz_int[n];                                                            // Setting new z-spin...
  p.z = 0.05f*sz[n];                                                            // Setting new z position...
  c.xyz = colormap(0.5f*(20.0f*p.z + 1.0f));                                    // Setting color...
  color[n] = c;                                                                 // Updating color...
  position[n] = p;                                                              // Updating position...
}
