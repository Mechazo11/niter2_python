#ifndef TRIANGULATE_H
#define TRIANGULATE_H

#include <cmath>
#include "vec4.h"

/*
Implementation of the niter2 non-iterative two-view triangulation method
described in

  Peter Lindstrom
  Triangulation Made Easy
  IEEE Computer Vision and Pattern Recognition 2010, pp. 1554-1561

The function triangulate_niter2 performs optimal two-view triangulation
of a pair of point correspondences in calibrated cameras.  Given measured
projections u = (u1, u2, -1) and v = (v1, v2, -1) of a 3D point, u and v
are minimally corrected so that they (to near machine precision) satisfy
the epipolar constraint u' E v = 0.  The corrected points on the image
plane are returned as x = (x1, x2, -1) and y = (y1, y2, -1).

The remaining function arguments encode the 3x3 essential matrix

              (e11 e12 e13)
  E = [t] R = (e21 e22 e23)
              (e31 e32 e33)

as two 4-vectors e = (e11, e22, e21, e12) and f = (e13, e23, e31, e32) and
a scalar g = e33.  This peculiar encoding of E is used in order to exploit
SSE vectorization.  Here t and R denote the relative position and
orientation of the cameras, and [t] is the cross product matrix 

        ( 0  -t3  t2)
  [t] = ( t3  0  -t1)
        (-t2  t1  0 )

The 3D point coordinates p expressed in the first camera can be recovered
using

  z = [x] R y
  p = (z' E y) x / (z' z)

A right-handed coordinate system is assumed, with the image plane at
z = -1.  If a left-handed coordinate system is used, where the image plane
is at z = +1, then the argument f should be negated (alternatively, all
subtractions of f in the function could be changed to additions).

The method should also handle calibrated cameras, where E is replaced by
the fundamental matrix (but see caveats in Section 2 of the paper).  The
template argument Float should be float or double.

For questions or comments on this code, please contact the author by email
at pl@llnl.gov.
*/

template <typename Float>
inline void
triangulate_niter2(
  Vec4<Float>& z,       // output coordinates: x1 x2 y1 y2
  const Vec4<Float>& w, // input coordinates: u1 u2 v1 v2
  const Vec4<Float>& e, // essential matrix: e11 e22 e21 e12
  const Vec4<Float>& f, // essential matrix: e13 e23 e31 e32
  Float g               // essential matrix: e33
)
{
  z = w;
  Vec4<Float> n = e.mul(w.zwyx()) + e.mul(w).wzxy() - f;
  Vec4<Float> m = e.mul(n.zwyx()) + e.mul(n).wzxy();
  Float a = n * m;
  Float b = n * n;
  Float c = (n - f) * w + 2 * g;
  Float d = b * b - a * c;
  if (d > 0) {
    d = sqrt(d);
    Float t = c / (b + d);
    m = t * m - n;
    t *= d / (m * m);
    m *= t;
    z += m;
  }
}

#endif
