/// @file
#define RAMP_UP_CYCLES 1000

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
                        __global int*       rows,                               // Number of rows in mesh.
                        __global float*     spin_z_row_sum,                     // z-spin row summation.
                        __global float*     spin_z2_row_sum,                    // z-spin square row summation.
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
  uint         r = 0;                                                           // Ramp-up index.

  ////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////// RANDOM GENERATOR //////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  float        sz_rand           = 0.0f;                                        // Flat tandom z-spin.
  float        th_rand           = 0.0f;                                        // Flat random threshold.
  uint4        st_sz             = convert_uint4(state_sz[n]);                  // Random generator state.
  uint4        st_th             = convert_uint4(state_th[n]);                  // Random generator state.

  // RAMPING UP RANDOM GENERATORS:
  for (r = 0; r < RAMP_UP_CYCLES; r++)
  {
    sz_rand = uint_to_float(xoshiro128pp(&st_sz), -1.0f, +1.0f);                // Generating random z-spin (flat distribution)...
    th_rand = uint_to_float(xoshiro128pp(&st_th), 0.0f, +1.0f);                 // Generating random threshold (flat distribution)...  
    state_sz[n] = convert_int4(st_sz);                                          // Updating random generator state...
    state_th[n] = convert_int4(st_th);                                          // Updating random generator state...
  }
}
