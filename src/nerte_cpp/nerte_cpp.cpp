/*
 *  Module for accelerated nerte code.
 *  WARNING: The current state of this module is EXPERIMENTAL (proof of concept)
 *           and therefore the code is quite 'hacky'.
 *  REQUIREMENTS: BOOST and BOOST/PYTHON
 */

#include <boost/python/numpy.hpp>
#include <cmath>
#include <iostream>

namespace p = boost::python;
namespace np = boost::python::numpy;

constexpr inline double powi(const double base, const long exp) {
  double res = 1.0;
  for (long i = 0; i < exp; ++i) {
    res *= base;
  }
  return res;
}

inline double *carray_from_ndarray(const np::ndarray &ndarray) {
  return reinterpret_cast<double *>(ndarray.get_data());
}

inline double dot(const double *const v1, const double *const v2) {
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

inline void cross(double *dest, const double *const v1,
                  const double *const v2) {
  dest[0] = v1[1] * v2[2] - v1[2] * v2[1];
  dest[1] = v1[2] * v2[0] - v1[0] * v2[2];
  dest[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

inline void normalize(double *dest) {
  const double norm_sq = dot(dest, dest);
  const double fac = 1 / std::sqrt(norm_sq);
  dest[0] *= fac;
  dest[1] *= fac;
  dest[2] *= fac;
}

inline void vec_add(double *dest, const double *const v1,
                    const double *const v2) {
  dest[0] = v1[0] + v2[0];
  dest[1] = v1[1] + v2[1];
  dest[2] = v1[2] + v2[2];
}

inline void vec_sub(double *dest, const double *const v1,
                    const double *const v2) {
  dest[0] = v1[0] - v2[0];
  dest[1] = v1[1] - v2[1];
  dest[2] = v1[2] - v2[2];
}

inline void vec_mult(double *dest, const double fac, const double *const v) {
  dest[0] = fac * v[0];
  dest[1] = fac * v[1];
  dest[2] = fac * v[2];
}

inline void mat_add_mult(double *dest, const double *mat, const double fac) {
  for (std::size_t i = 0; i < 9; ++i) {
    dest[i] = mat[i] * fac;
  }
}

bool in_triangle(const double *b1, const double *b2, const double *x) {
  const double b1b1 = dot(b1, b1);
  const double b1b2 = dot(b1, b2);
  const double b2b2 = dot(b2, b2);
  const double D = b1b1 * b2b2 - b1b2 * b1b2;
  const double b1x = dot(x, b1);
  const double b2x = dot(x, b2);
  const double f1 = (b1b1 * b2x - b1b2 * b1x) / D;
  const double f2 = (b2b2 * b1x - b1b2 * b2x) / D;

  // test if x is inside the triangle
  return f1 >= 0 && f2 >= 0 && f1 + f2 <= 1;
}

double intersection_ray_depth(const np::ndarray &ray_base,
                              const np::ndarray &ray_direction,
                              const bool ray_is_finite,
                              const np::ndarray &face_p0,
                              const np::ndarray &face_p1,
                              const np::ndarray &face_p2) {
  // assert:
  // all objects are 3D vectors
  // ray_direction is non-zero
  // dtype is double
  const double *const s = carray_from_ndarray(ray_base);
  const double *const u = carray_from_ndarray(ray_direction);
  const double *const v0 = carray_from_ndarray(face_p0);
  const double *const v1 = carray_from_ndarray(face_p1);
  const double *const v2 = carray_from_ndarray(face_p2);

  double b1[3];
  double b2[3];

  vec_sub(b1, v1, v0);
  vec_sub(b2, v2, v0);

  double n[3];
  cross(n, b1, b2);
  normalize(n);

  const double l = dot(n, v0);
  const double a = l - dot(s, n);
  const double b = dot(u, n);

  constexpr double inf = std::numeric_limits<double>::infinity();

  if (b == 0) {
    // ray is parallel to plane
    if (a == 0) {
      // ray starts inside plane
      return 0.0; // this value is somewhat arbitrary
    }
    // ray starts outside of plane
    return inf; // no intersection possible
  }
  const double t = a / b;

  if (t < 0) {
    // intersection is before ray segment started
    return inf;
  }
  if (ray_is_finite && t > 1) {
    // intersection after ray segment ended
    return inf;
  }
  // x = intersection point with respect to the triangles origin
  // return if x lies in the triangle spanned by b1 and b2
  double q[3];
  vec_mult(q, t, u);
  vec_add(q, s, q);
  vec_sub(q, q, v0);
  if (in_triangle(b1, b2, q)) {
    return t;
  }
  return inf;
}

std::tuple<np::ndarray, np::ndarray, np::ndarray>
internal_christoffel_1(const double swirl, const double *const cs) {
  const double a = swirl;
  const double u = cs[0];
  const double v = cs[1];
  const double z = cs[2];
  // frequent factors
  const double r = std::sqrt(u * u + v * v);
  const double arz = a * r * z;
  const double alpha = std::atan2(v, u);
  const double cos_alpha = std::cos(alpha);
  const double sin_alpha = std::sin(alpha);
  const double cos_2alpha = std::cos(2 * alpha);
  const double sin_2alpha = std::sin(2 * alpha);
  const double cos_3alpha = std::cos(3 * alpha);
  const double sin_3alpha = std::sin(3 * alpha);

  p::tuple shape = p::make_tuple(3, 3);
  np::dtype dtype = np::dtype::get_builtin<double>();
  np::ndarray res0 = np::zeros(shape, dtype);
  np::ndarray res1 = np::zeros(shape, dtype);
  np::ndarray res2 = np::zeros(shape, dtype);

  double *const chris1_0 = carray_from_ndarray(res0);
  double *const chris1_1 = carray_from_ndarray(res1);
  double *const chris1_2 = carray_from_ndarray(res2);

  chris1_0[0] = a * z * (arz * cos_alpha - powi(sin_alpha, 3));
  chris1_0[1] = -a * z * powi(cos_alpha, 3);
  chris1_0[2] = a * r * cos_alpha * (arz * cos_alpha - sin_alpha);

  chris1_0[3] = -a * z * powi(cos_alpha, 3);
  chris1_0[4] =
      -0.25 * a * z * (-4.0 * arz * cos_alpha + 9.0 * sin_alpha + sin_3alpha);
  chris1_0[5] = 0.5 * a * r * (-3.0 + cos_2alpha + arz * sin_2alpha);

  chris1_0[6] = a * r * cos_alpha * (arz * cos_alpha - sin_alpha);
  chris1_0[7] = 0.5 * a * r * (-3.0 + cos_2alpha + arz * sin_2alpha);
  chris1_0[8] = -(a * a) * powi(r, 3) * cos_alpha;

  chris1_1[0] = 1.0 / 4.0 * a * z *
                (9.0 * cos_alpha - cos_3alpha + 4.0 * arz * sin_alpha);
  chris1_1[1] = a * z * sin_alpha * sin_alpha * sin_alpha;
  chris1_1[2] = 1.0 / 2.0 * a * r * (3 + cos_2alpha + arz * sin_2alpha);

  chris1_1[3] = a * z * powi(sin_alpha, 3);
  chris1_1[4] = a * z * (powi(cos_alpha, 3) + arz * sin_alpha);
  chris1_1[5] = a * r * sin_alpha * (cos_alpha + arz * sin_alpha);

  chris1_1[6] = 1.0 / 2.0 * a * r * (3.0 + cos_2alpha + arz * sin_2alpha);
  chris1_1[7] = a * r * sin_alpha * (cos_alpha + arz * sin_alpha);
  chris1_1[8] = -(a * a) * powi(r, 3) * sin_alpha;

  chris1_2[0] = 1.0 / 2.0 * a * a * r * r * z * (3.0 + cos_2alpha);
  chris1_2[1] = a * a * r * r * z * cos_alpha * sin_alpha;
  chris1_2[2] = 2.0 * a * a * r * r * r * cos_alpha;

  chris1_2[3] = a * a * r * r * z * cos_alpha * sin_alpha;
  chris1_2[4] = -(1.0 / 2.0) * a * a * r * r * z * (-3.0 + cos_2alpha);
  chris1_2[5] = 2.0 * a * a * r * r * r * sin_alpha;

  chris1_2[6] = 2.0 * a * a * r * r * r * cos_alpha;
  chris1_2[7] = 2.0 * a * a * r * r * r * sin_alpha;
  chris1_2[8] = 0.0;

  return std::tuple(res0, res1, res2);
}

p::tuple christoffel_1(const double swirl, const np::ndarray &coords) {
  const double a = swirl;
  const double *const cs = carray_from_ndarray(coords);
  const double u = cs[0];
  const double v = cs[1];
  const double z = cs[2];
  // frequent factors
  const double r = std::sqrt(u * u + v * v);
  const double arz = a * r * z;
  const double alpha = std::atan2(v, u);
  const double cos_alpha = std::cos(alpha);
  const double sin_alpha = std::sin(alpha);
  const double cos_2alpha = std::cos(2 * alpha);
  const double sin_2alpha = std::sin(2 * alpha);
  const double cos_3alpha = std::cos(3 * alpha);
  const double sin_3alpha = std::sin(3 * alpha);

  p::tuple shape = p::make_tuple(3, 3);
  np::dtype dtype = np::dtype::get_builtin<double>();
  np::ndarray res0 = np::zeros(shape, dtype);
  np::ndarray res1 = np::zeros(shape, dtype);
  np::ndarray res2 = np::zeros(shape, dtype);

  double *const chris1_0 = carray_from_ndarray(res0);
  double *const chris1_1 = carray_from_ndarray(res1);
  double *const chris1_2 = carray_from_ndarray(res2);

  chris1_0[0] = a * z * (arz * cos_alpha - powi(sin_alpha, 3));
  chris1_0[1] = -a * z * powi(cos_alpha, 3);
  chris1_0[2] = a * r * cos_alpha * (arz * cos_alpha - sin_alpha);

  chris1_0[3] = -a * z * powi(cos_alpha, 3);
  chris1_0[4] =
      -0.25 * a * z * (-4.0 * arz * cos_alpha + 9.0 * sin_alpha + sin_3alpha);
  chris1_0[5] = 0.5 * a * r * (-3.0 + cos_2alpha + arz * sin_2alpha);

  chris1_0[6] = a * r * cos_alpha * (arz * cos_alpha - sin_alpha);
  chris1_0[7] = 0.5 * a * r * (-3.0 + cos_2alpha + arz * sin_2alpha);
  chris1_0[8] = -(a * a) * powi(r, 3) * cos_alpha;

  chris1_1[0] = 1.0 / 4.0 * a * z *
                (9.0 * cos_alpha - cos_3alpha + 4.0 * arz * sin_alpha);
  chris1_1[1] = a * z * sin_alpha * sin_alpha * sin_alpha;
  chris1_1[2] = 1.0 / 2.0 * a * r * (3 + cos_2alpha + arz * sin_2alpha);

  chris1_1[3] = a * z * powi(sin_alpha, 3);
  chris1_1[4] = a * z * (powi(cos_alpha, 3) + arz * sin_alpha);
  chris1_1[5] = a * r * sin_alpha * (cos_alpha + arz * sin_alpha);

  chris1_1[6] = 1.0 / 2.0 * a * r * (3.0 + cos_2alpha + arz * sin_2alpha);
  chris1_1[7] = a * r * sin_alpha * (cos_alpha + arz * sin_alpha);
  chris1_1[8] = -(a * a) * powi(r, 3) * sin_alpha;

  chris1_2[0] = 1.0 / 2.0 * a * a * r * r * z * (3.0 + cos_2alpha);
  chris1_2[1] = a * a * r * r * z * cos_alpha * sin_alpha;
  chris1_2[2] = 2.0 * a * a * r * r * r * cos_alpha;

  chris1_2[3] = a * a * r * r * z * cos_alpha * sin_alpha;
  chris1_2[4] = -(1.0 / 2.0) * a * a * r * r * z * (-3.0 + cos_2alpha);
  chris1_2[5] = 2.0 * a * a * r * r * r * sin_alpha;

  chris1_2[6] = 2.0 * a * a * r * r * r * cos_alpha;
  chris1_2[7] = 2.0 * a * a * r * r * r * sin_alpha;
  chris1_2[8] = 0.0;

  return p::make_tuple(res0, res1, res2);
}

np::ndarray metric_inverted(const double swirl, const np::ndarray &coords) {
  const double a = swirl;
  const double *const cs = carray_from_ndarray(coords);
  const double u = cs[0];
  const double v = cs[1];
  const double z = cs[2];
  // frequent factors
  const double r = std::sqrt(u * u + v * v);
  const double s = u * u - v * v;
  const double aur = a * u * r;
  const double avr = a * v * r;
  const double u2v2z2 = u * u + v * v + z * z;

  p::tuple shape = p::make_tuple(3, 3);
  np::dtype dtype = np::dtype::get_builtin<double>();
  np::ndarray res = np::zeros(shape, dtype);

  double *const gi = carray_from_ndarray(res);

  gi[0] = 1.0 + a * v * ((2.0 * u * z) / r + a * v * (r * r + z * z));

  gi[1] = a * ((-s * z) / r - a * u * v * u2v2z2);
  gi[2] = avr;

  gi[3] = a * ((-s * z) / r - a * u * v * u2v2z2);
  gi[4] = 1.0 + a * u * ((-2.0 * v * z) / r + a * u * u2v2z2);
  gi[5] = -aur;

  gi[6] = avr;
  gi[7] = -aur;
  gi[8] = 1.0;

  return res;
}

p::tuple christoffel_2(const double swirl, const np::ndarray &coords) {
  const double *const cs = carray_from_ndarray(coords);
  std::tuple<np::ndarray, np::ndarray, np::ndarray> chris_1 =
      internal_christoffel_1(swirl, cs);

  const double *chris_1_0 = carray_from_ndarray(std::get<0>(chris_1));
  const double *chris_1_1 = carray_from_ndarray(std::get<1>(chris_1));
  const double *chris_1_2 = carray_from_ndarray(std::get<2>(chris_1));

  np::ndarray met_inv = metric_inverted(swirl, coords);
  const double *const gi = carray_from_ndarray(met_inv);

  p::tuple shape = p::make_tuple(3, 3);
  np::dtype dtype = np::dtype::get_builtin<double>();
  np::ndarray res0 = np::zeros(shape, dtype);
  np::ndarray res1 = np::zeros(shape, dtype);
  np::ndarray res2 = np::zeros(shape, dtype);

  double *chris_2_0 = carray_from_ndarray(res0);
  double *chris_2_1 = carray_from_ndarray(res1);
  double *chris_2_2 = carray_from_ndarray(res2);

  const double a = swirl;
  const double u = cs[0];
  const double v = cs[1];
  const double z = cs[2];
  // frequent factors;
  const double r = std::sqrt(u * u + v * v);
  const double arz = a * r * z;
  const double arz2 = arz * arz;
  const double ar = a * r;
  const double az = a * z;

  const double alpha = std::atan2(v, u);
  const double c_alpha = std::cos(alpha);
  const double s_alpha = std::sin(alpha);
  const double c_2alpha = std::cos(2 * alpha);
  const double s_2alpha = std::sin(2 * alpha);
  const double c_3alpha = std::cos(3 * alpha);
  const double s_3alpha = std::sin(3 * alpha);

  const double fac1 =
      (az * ((3.0 + arz2) * c_alpha + c_3alpha -
             arz * (arz * c_3alpha + s_alpha - 3.0 * s_3alpha))) /
      4.0;
  const double fac2 =
      (ar * (s_2alpha + arz * (2.0 * c_2alpha + arz * s_2alpha))) / 2.0;
  const double fac3 =
      (ar * (-3.0 - arz2 + (1.0 + arz2) * c_2alpha - 2.0 * arz * s_2alpha)) /
      2.0;
  const double fac4 =
      (az * (2.0 * powi(s_alpha, 3) +
             arz * c_alpha * (-1.0 + 3.0 * c_2alpha + arz * s_2alpha))) /
      2.0;
  const double fac5 =
      (ar * (3.0 + arz2 + (1.0 + arz2) * c_2alpha - 2.0 * arz * s_2alpha)) /
      2.0;
  const double fac6 =
      (az *
       (-(arz * (c_alpha + 3.0 * c_3alpha)) -
        2.0 * (1.0 + arz2 + (-1.0 + arz2) * c_2alpha) * s_alpha) /
       4.0);
  const double fac7 =
      (az * s_alpha *
       (-5.0 - arz2 + (-1.0 + arz2) * c_2alpha - 3.0 * arz * s_2alpha)) /
      2.0;
  const double fac8 = -(a * a * powi(r, 3) * (c_alpha + arz * s_alpha));
  const double fac9 =
      (az * c_alpha *
       (5.0 + arz2 + (-1.0 + arz2) * c_2alpha - 3.0 * arz * s_2alpha)) /
      2.0;
  const double fac10 = a * a * powi(r, 3) * (arz * c_alpha - s_alpha);

  chris_2_0[0] = fac6;
  chris_2_0[1] = -fac1;
  chris_2_0[2] = -fac2;

  chris_2_0[3] = -fac1;
  chris_2_0[4] = fac7;
  chris_2_0[5] = fac3;

  chris_2_0[6] = -fac2;
  chris_2_0[7] = fac3;
  chris_2_0[8] = fac8;

  chris_2_1[0] = fac9;
  chris_2_1[1] = fac4;
  chris_2_1[2] = fac5;

  chris_2_1[3] = fac4;
  chris_2_1[4] = fac1;
  chris_2_1[5] = fac2;

  chris_2_1[6] = fac5;
  chris_2_1[7] = fac2;
  chris_2_1[8] = fac10;

  // chris_2_2 = ZERO

  return p::make_tuple(res0, res1, res2);
}

BOOST_PYTHON_MODULE(nerte_cpp) {
  using namespace boost::python;
  np::initialize();
  def("intersection_ray_depth", intersection_ray_depth);
  def("christoffel_1", christoffel_1);
  def("metric_inverted", metric_inverted);
  def("christoffel_2", christoffel_2);
}
