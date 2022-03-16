/// @file

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
  uint         j = 0;                                                           // Row stride index.
  uint         j_min = i*(uint)parameter[5];                                    // Row stride minimum index (based on number of columns).
  uint         j_max = (i + 1)*(uint)parameter[5] - 1;                          // Row stride maximum index (based on number of columns).
  float        spin_z_partial_sum = 0.0f;                                       // z_spin partial summation.
  float        spin_z2_partial_sum = 0.0f;                                      // z_spin square partial summation.
  int          m_overflow_partial_sum = 0;                                      // Rejection sampling overflow partial summation.

  // Summating all z-spin in a row:
  for (j = j_min; j < j_max; j++)
  {
    spin_z_partial_sum += sin(theta[j]);                                        // Accumulating z-spin partial summation...
    spin_z2_partial_sum += pown(sin(theta[j]), 2);                              // Accumulating z-spin square partial summation...
    m_overflow_partial_sum += m_overflow[j];                                    // Accumulating rejection sampling partial overflows...
  }

  spin_z_row_sum[i] = spin_z_partial_sum;                                       // Setting z-spin row summation...
  spin_z2_row_sum[i] = spin_z2_partial_sum;                                     // Setting z-spin square row summation...
  m_overflow_sum[i] = m_overflow_partial_sum;                                   // Setting rejection sampling overflow row summation...
}