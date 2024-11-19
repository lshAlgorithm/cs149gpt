// Blocked assingment of array elements to program instances
// Assign multiple elements to each instance
export void ispc_sinx (
  uniform int N,
  uniform int terms,
  uniform float x[],
  uniform float result[]) {
  // Assume N % programCount = 0;
  // Block assingment
  uniform int count = N / programCount;
  int start = programIndex * count;
  for (uniform int i = 0; i < count; ++i) {
    int idx = start + i;
    float value = x[idx];
    float numer = x[idx] * x[idx] * x[idx];
    uniform int denom = 6;
    uniform int sign = -1;
    for (uniform int j = 1; j < terms; ++j) {
      value += sign * numer / denom;
      numer *= x[idx] * x[idx];
      denom *= (j + 3) * (j + 4); // block assignment
      sign *= -1;
    }
    result[idx] = value;
  }
}
