/// @file

__kernel void thekernel(__global float4*    color,                              // Color.
                        __global float4*    position,                           // Position.
                        __global int*       central,                            // Node.
                        __global int*       nearest,                            // Neighbour.
                        __global int*       offset,                             // Offset.
                        __global int4*      state,                              // Random number generator state.     
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
  uint4        s                 = convert_uint4(state[n]);                     // Random generator state.
 
  // COMPUTING STRIDE MINIMUM INDEX:
  if (i == 0)
  {
    j_min = 0;                                                                  // Setting stride minimum (first stride)...
  }
  else
  {
    j_min = offset[i - 1];                                                      // Setting stride minimum (all others)...
  }

  // COMPUTING ENERGY:
  for (j = j_min; j < j_max; j++)
  {
    k = nearest[j];                                                             // Computing neighbour index...
    neighbour = position[k];                                                    // Getting neighbour position...
    link = neighbour - p;                                                       // Getting neighbour link vector...
    L = length(link.xyz);                                                       // Computing neighbour link length...
  }
  //printf("n = %u\n", xoshiro128pp(&s));
  p.z = uint_to_float(xoshiro128pp(&s), -0.05f, +0.05f);                         // Setting z position...
  //printf("f = %f\n", p.z);
  state[n] = convert_int4(s);                                                   // Updating random generator state...
  position[n] = p;                                                              // Updating position...
}
