/// @file

__kernel void thekernel(__global float4*    color,                              // Color.
                        __global float4*    position,                           // Position.
                        __global int*       central,                            // Node.
                        __global int*       nearest,                            // Neighbour.
                        __global int*       offset,                             // Offset. 
                        __global float*     theta,                              // Theta.  
                        __global float*     theta_int,                          // Theta (intermediate value). 
                        __global int4*      state_theta,                        // Random number generator state.
                        __global int4*      state_threshold,                    // Random number generator state. 
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
  uint         j = 0;                                                           // Row stride index.
  uint         j_min = i*rows[0];                                               // Row stride minimun index.
  uint         j_max = (i + 1)*rows[0];                                         // Row stride maximum index.
  float        spin_z_partial_sum = 0.0f;                                       // z_spin partial summation.
  float        spin_z2_partial_sum = 0.0f;                                      // z_spin square partial summation.

  // Summating all z-spin in a row:
  for (j = j_min; j < j_max; j++)
  {
    spin_z_partial_sum += sin(theta[j]);                                        // Accumulating z-spin partial summation...
    spin_z2_partial_sum += pown(sin(theta[j]), 2);                              // Accumulating z-spin square partial summation...
  }

  spin_z_row_sum[i] = spin_z_partial_sum;                                       // Setting z-spin row summation...
  spin_z2_row_sum[i] = spin_z2_partial_sum;                                     // Setting z-spin square row summation...
}