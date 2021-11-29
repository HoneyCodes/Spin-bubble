/// @file

// Central node energy function:
float E_central(float Hx, float Hz, float sz_central)
{
  float E;                                                                      // Energy.
  
  E = -(Hx*sqrt(1 - pow(sz_central, 2)) + Hz*sz_central);                       // Computing energy...

  return E;
}

// Neighbour node energy function:
float E_neighbour(float C_radial, float sz_neighbour, float sz_central)
{
  float E;                                                                      // Energy.
  
  E = -(C_radial*sz_neighbour*sz_central);                                      // Computing energy...

  return E;
}

__kernel void thekernel(__global float4*    color,                              // Color.
                        __global float4*    position,                           // Position.
                        __global int*       central,                            // Node.
                        __global int*       nearest,                            // Neighbour.
                        __global int*       offset,                             // Offset.
                        __global int4*      state_sz,                           // Random number generator state.
                        __global int4*      state_th,                           // Random number generator state.  
                        __global float*     longitudinal_H,                     // Longitudinal magnetic field.
                        __global float*     transverse_H,                       // Transverse magnetic field.
                        __global float*     sz,                                 // z-component of the spin.  
                        __global float*     temperature,                        // Temperature.
                        __global int*       max_rejections,                     // Maximum allowed number of rejections.
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
  uint4        st_sz             = convert_uint4(state_sz[n]);                  // Random generator state.
  uint4        st_th             = convert_uint4(state_th[n]);                  // Random generator state.
  float        Hx                = longitudinal_H[0];                           // Longitudinal magnetic field.
  float        Hz                = transverse_H[0];                             // Transverse magnetic field.
  float        E                 = 0.0f;                                        // Energy function.
  float        En                = 0.0f;                                        // Energy of central node.
  float        sz_rand           = 0.0f;                                        // Flat tandom z-spin.
  float        th_rand           = 0.0f;                                        // Flat random threshold.
  float        T                 = temperature[0];                              // Temperature.
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

  // COMPUTING RANDOM Z-SPIN FROM DISTRIBUTION (rejection sampling):
  do
  {
    sz_rand = uint_to_float(xoshiro128pp(state_sz), -1.0f, +1.0f);              // Generating random z-spin (flat distribution)...
    th_rand = uint_to_float(xoshiro128pp(state_th), -1.0f, +1.0f);              // Generating random threshold (flat distribution)...
    En = E_central(Hx, Hz, sz[n]);                                              // Computing central energy term on central z-spin...
    E = E_central(Hx, Hz, sz_rand);                                             // Computing central energy term on random z-spin...

    // COMPUTING ENERGY:
    for (j = j_min; j < j_max; j++)
    {
      k = nearest[j];                                                           // Computing neighbour index...
      neighbour = position[k];                                                  // Getting neighbour position...
      link = neighbour - p;                                                     // Getting neighbour link vector...
      L = length(link.xyz);                                                     // Computing neighbour link length...
      En += E_neighbour(0.5f/L, sz[k], sz[n]);                                  // Accumulating neighbour energy terms on central z-spin...
      E += E_neighbour(0.5f/L, sz[k], sz_rand);                                 // Accumulating neighbour energy terms on random z-spin...          
    }
    
    D = 1.0f/(1.0f + exp((E - En)/T));                                          // Computing new z-spin candidate from distribution...
    m++;                                                                        // Updating rejection index...
  }
  while ((th_rand > D) && (m < m_max));                                         // Evaluating new z-spin candidate (discarding if not found before m_max iterations)...
  
  sz[n] = D;                                                                    // Setting new z-spin...
  p.z = 0.05f*D;                                                                // Setting new z position...
  state_sz[n] = convert_int4(st_sz);                                            // Updating random generator state...
  state_th[n] = convert_int4(st_th);                                            // Updating random generator state...
  c.xyz = colormap(0.5f*(20.0f*p.z + 1.0f));                                    // Setting color...
  color[n] = c;                                                                 // Updating color...
  position[n] = p;                                                              // Updating position...
}
