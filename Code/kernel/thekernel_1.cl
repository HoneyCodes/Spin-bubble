/// @file

// Central node energy function:
float E_central(float Hx, float Hz, float theta_central)
{
  float E;                                                                      // Energy.
  
  E = -(Hx*cos(theta_central) + Hz*sin(theta_central));                         // Computing energy...

  return E;
}

// Neighbour node energy function:
float E_neighbour(float C_radial, float theta_neighbour, float theta_central)
{
  float E;                                                                      // Energy.
  
  E = -(C_radial*sin(theta_neighbour)*sin(theta_central));                      // Computing energy...

  return E;
}

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
  float2       link              = (float2)(0.0f, 0.0f);                        // Neighbour link.
  float        L                 = 0.0f;                                        // Neighbour link length.
  float        dt                = dt_simulation[0];                            // Simulation time step [s].
  float        ds                = ds_simulation[0];                            // Simulation space step.
  uint4        st_theta          = convert_uint4(state_theta[n]);               // Random generator state.
  uint4        st_threshold      = convert_uint4(state_threshold[n]);           // Random generator state.
  float        Hx                = longitudinal_H[0];                           // Longitudinal magnetic field.
  float        Hz                = transverse_H[0];                             // Transverse magnetic field.
  float        E                 = 0.0f;                                        // Energy function.
  float        En                = 0.0f;                                        // Energy of central node.
  float        theta_rand        = 0.0f;                                        // Flat random theta.
  float        threshold_rand    = 0.0f;                                        // Flat random threshold.
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

  // COMPUTING RANDOM Z-SPIN FROM DISTRIBUTION (rejection sampling):
  do
  {
    theta_rand = uint_to_float(xoshiro128pp(&st_theta), 0.0f, 2.0f*M_PI_F);     // Generating random theta (flat distribution)...
    threshold_rand = uint_to_float(xoshiro128pp(&st_threshold), 0.0f, +1.0f);   // Generating random threshold (flat distribution)...
    En = E_central(Hx, Hz, theta[n]);                                           // Computing central energy term on central theta...
    E = E_central(Hx, Hz, theta_rand);                                          // Computing central energy term on random theta...

    // COMPUTING ENERGY:
    for (j = j_min; j < j_max; j++)
    {
      k = nearest[j];                                                           // Computing neighbour index...
      neighbour = position[k];                                                  // Getting neighbour position...
      link = neighbour.xy - p.xy;                                               // Getting neighbour link vector...
      L = length(link);                                                         // Computing neighbour link length...

      if (L == (2.0f + ds))
      {
        L = ds;
      }

      if (L > (2.0f + ds))
      {
        L = sqrt(2.0f)*ds;
      }
      
      En += E_neighbour(0.5f/pow(L/ds, alpha), theta[k], theta[n]);             // Accumulating neighbour energy terms on central z-spin...
      E += E_neighbour(0.5f/pow(L/ds, alpha), theta[k], theta_rand);            // Accumulating neighbour energy terms on random z-spin...          
    }
    
    D = 1.0f/(1.0f + exp((E - En)/T));                                          // Computing new z-spin candidate from distribution...
    m++;                                                                        // Updating rejection index...
  }
  while ((threshold_rand > D) && (m < m_max));                                  // Evaluating new z-spin candidate (discarding if not found before m_max iterations)...

  theta_int[n] = theta_rand;                                                    // Setting new z-spin (intermediate value)...
  state_theta[n] = convert_int4(st_theta);                                      // Updating random generator state...
  state_threshold[n] = convert_int4(st_threshold);                              // Updating random generator state...
}
