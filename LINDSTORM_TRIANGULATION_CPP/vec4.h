#ifndef VEC4_H
#define VEC4_H

template <typename Float>
class Vec4 {
public:
  // constructors
  Vec4() {}
  Vec4(Float x, Float y, Float z, Float w) { a[0] = x; a[1] = y; a[2] = z; a[3] = w; }

  // accessors
  Float& operator[](int i) { return a[i]; }
  const Float& operator[](int i) const { return a[i]; }

  // vector addition
  Vec4& operator*=(Float s)
  {
    a[0] *= s;
    a[1] *= s;
    a[2] *= s;
    a[3] *= s;
    return *this;
  }

  // vector addition
  Vec4& operator+=(const Vec4& v)
  {
    a[0] += v[0];
    a[1] += v[1];
    a[2] += v[2];
    a[3] += v[3];
    return *this;
  }

  // vector subtraction
  Vec4& operator-=(const Vec4& v)
  {
    a[0] -= v[0];
    a[1] -= v[1];
    a[2] -= v[2];
    a[3] -= v[3];
    return *this;
  }

  // component-wise multiplication
  Vec4 mul(const Vec4& v) const { return Vec4(a[0] * v.a[0], a[1] * v.a[1], a[2] * v.a[2], a[3] * v.a[3]); }

  // shuffling
  Vec4 wzxy() const { return Vec4(a[3], a[2], a[0], a[1]); }
  Vec4 zwyx() const { return Vec4(a[2], a[3], a[1], a[0]); }

private:
  Float a[4]; // vector (x, y, z, w)
};

// vector scaling
template <typename Float>
inline Vec4<Float>
operator*(Float s, const Vec4<Float>& v)
{
  return Vec4<Float>(v) *= s;
}

// vector sum
template <typename Float>
inline Vec4<Float>
operator+(const Vec4<Float>& u, const Vec4<Float>& v)
{
  return Vec4<Float>(u) += v;
}

// vector difference
template <typename Float>
inline Vec4<Float>
operator-(const Vec4<Float>& u, const Vec4<Float>& v)
{
  return Vec4<Float>(u) -= v;
}

// dot product
template <typename Float>
inline Float
operator*(const Vec4<Float>& u, const Vec4<Float>& v)
{
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3];
}

#endif
