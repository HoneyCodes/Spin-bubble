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
                        __global float*     dt_simulation)                      // Simulation time step.
{
  ////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// INDEXES ///////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  unsigned int i = get_global_id(0);                                            // Global index [#].
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////// CELL VARIABLES //////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  float4        p                 = position[i];                                // Central node position.
  float         sz                = spin_z[i];                                  // z-component of the spin.
  float         dt                = dt_simulation[0];                           // Simulation time step [s].
  float         D                 = probability[i];


 
}
