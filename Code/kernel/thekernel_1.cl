/// @file

__kernel void thekernel(__global float4*    color,                              // Color.
                        __global float4*    position,                           // Position.
                        __global int*       central,                            // Node.
                        __global int*       nearest,                            // Neighbour.
                        __global int*       offset,                             // Offset.
                        __global float*     spin_z,                             // z-component of the spin.  
                        __global float*     energy,
                        __global float*     energyi,
                        __global float*     probability,                        // Probability distrubution.
                        __global float*     Hx;                                 // Transverse magnetic field.
                        __global float*     Hz;                                 // Longitudinal magnetic field.
                        __global float*     dt_simulation)                      // Simulation time step.
{
  ////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// INDICES ///////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  unsigned int i = get_global_id(0);                                            // Global index [#].
  unsigned int j = 0;                                                           // Neighbour stride index.
  unsigned int j_min = 0;                                                       // Neighbour stride minimun index.
  unsigned int j_max = offset[i];                                               // Neighbour stride maximum index.
  unsigned int k = 0;                                                           // Neighbour tuple index.
  unsigned int n = central[j_max - 1];                                          // Node index.
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////// CELL VARIABLES //////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  float4        p                 = position[n];                                // Central node position.
  float4        neighbour         = (float4)(0.0f, 0.0f, 0.0f, 1.0f);           // Neighbour node position.
  float4        link              = (float4)(0.0f, 0.0f, 0.0f, 1.0f);           // Neighbour link.
  float         L                 = 0.0f;                                       // Neighbour link length.
  float         sz                = spin_z[i];                                  // z-component of the spin.
  float         dt                = dt_simulation[0];                           // Simulation time step [s].
  float         D                 = probability[i];
  float         E                 = 0.0f;

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
    E += -0.5f*(1/pown(L, alpha)*sz[k]);                                        // Accumulating energy...
  }

  E += -(Hx*sqrt(1 - sz[i]*sz[i]) + Hz*sz[i]);                                  // Adding magnetic terms...
}
