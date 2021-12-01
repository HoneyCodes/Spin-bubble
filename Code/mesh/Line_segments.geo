//////////////////////////////////////////////////////////////////
//
//      A---------B
//
//
//      y
//      |
//      |
//      o -----x
//     /
//    /
//   z
//
//////////////////////////////////////////////////////////////////

ds = 0.02;                                                      // Setting side discretization length...
x_min = -1.0;                                                   // Setting "x_min"...
x_max = +1.0;                                                   // Setting "x_max"...

// Nodes:
A = 1;                                                          // Setting point "A" tag...
B = 2;                                                          // Setting point "B" tag...

// Sides:
AB = 1;                                                         // Setting side "AB" tag...

// Physical groups:
x_side = 7;                                                     // Setting "x-side" tag (physical group)...

Point(A) = {x_min, 0.0, 0.0, ds};                               // Setting point "A"...
Point(B) = {x_max, 0.0, 0.0, ds};                               // Setting point "B"...

Line(AB) = {A, B};                                              // Setting side "AB"...

Physical Curve(x_side) = {AB};                                  // Setting group: side "AB"...

Mesh 1;                                                         // Setting mesh type: lines...

Mesh.SaveAll = 1;                                               // Saving all mesh nodes...