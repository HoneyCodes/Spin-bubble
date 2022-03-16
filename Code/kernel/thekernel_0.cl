/// @file
#define RAMP_UP_CYCLES 1000

__kernel void thekernel(__global float4*    color,                              // Color.
                        __global float4*    position,                           // Position.
                        __global int*       central,                            // Node.
                        __global int*       neighbour,                          // Neighbour.
                        __global int*       offset,                             // Offset. 
                        __global float*     theta,                              // Theta.  
                        __global float*     theta_int,                          // Theta (intermediate value). 
                        __global int4*      state_theta,                        // Random number generator state.
                        __global int4*      state_threshold,                    // Random number generator state. 
                        __global float*     spin_z_row_sum,                     // z-spin row summation.
                        __global float*     spin_z2_row_sum,                    // z-spin square row summation.
                        __global int*       m_overflow,                         // Rejection sampling overflow.
                        __global int*       m_overflow_sum,                     // Rejection sampling overflow sum.
                        __global float*     parameter)                          // Parameters.
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
  uint4        st_sz             = convert_uint4(state_theta[n]);               // Random generator state.
  uint4        st_th             = convert_uint4(state_threshold[n]);           // Random generator state.

  // RAMPING UP RANDOM GENERATORS:
  for (r = 0; r < RAMP_UP_CYCLES; r++)
  {
    sz_rand = uint_to_float(xoshiro128pp(&st_sz), -1.0f, +1.0f);                // Generating random z-spin (flat distribution)...
    th_rand = uint_to_float(xoshiro128pp(&st_th), 0.0f, +1.0f);                 // Generating random threshold (flat distribution)...  
    state_theta[n] = convert_int4(st_sz);                                       // Updating random generator state...
    state_threshold[n] = convert_int4(st_th);                                   // Updating random generator state...
  }
}
