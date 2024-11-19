#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
#include "ispc.h"
using namespace ispc;

int main() {
    float x[10], res[10];
    for (int i = 0; i < 10; i++) {
        x[i] = i * 1.0;
    }
    ispc_sinx(10, 10, x, res);
    for (int i = 0; i < 10; i++) {
        printf(" %d ", res[i]);
    }
    return 0;
}