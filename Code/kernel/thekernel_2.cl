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

  theta[n] = theta_int[n];                                                      // Setting new theta...
  p.z = 0.05f*sin(theta[n]);                                                    // Setting new z position...
  c.xyz = colormap(0.5f*(20.0f*p.z + 1.0f));                                    // Setting color...
  color[n] = c;                                                                 // Updating color...
  position[n] = p;                                                              // Updating position...
}
