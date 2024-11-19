// ISPC code
// Interleaved assignment
export void ispc_sinx (
  uniform int N,            // N = 1024 as input value
  uniform int terms,
  uniform float* X,
  uniform float* result) {
  // assume N % programCount = 0
  // Interleave assignment
  // "Gang" of ISPC program instances, Gang contains programCount = 8 instances
  for (uniform int i = 0; i < N; i += programCount) {
    int idx = i + programIndex;             // Local variable
    float value = x[idx];                   // Local variable
    float numer = x[idx] * x[idx] *x[idx];  // Local variable
    uniform int denom = 6;
    uniform int sign = -1;
    for (uniform int j = 1; j <= terms; ++j) {
      value += sign * numer / denom;
      numer *= x[idx] * x[idx];
      denom *= (2 * j + 2) * (2 * j + 3);
      sign *= -1;
    }
    result[idx] = value;
  }
}
