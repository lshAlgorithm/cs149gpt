extern float fourDimRead(uniform float tensor[], int x, int y, int z, int b, 
        uniform int &sizeX, uniform int &sizeY, uniform int &sizeZ) {
            return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
        }

extern float fourDimWrite(uniform float tensor[], int x, int y, int z, int b, 
        uniform int &sizeX, uniform int &sizeY, uniform int &sizeZ, float val) {
            tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
        }

extern float twoDimRead(uniform float tensor[], int x, int y, uniform int &sizeX) {
    return tensor[x * (sizeX)+ y];
}
extern float twoDimWrite(uniform float tensor[], int x, int y, uniform int &sizeX, float val){
    tensor[x * (sizeX) + y] = val;
}

export void NavieAttention_ispc(uniform float O[], 
                                uniform float Q[], 
                                uniform float K[], 
                                uniform float V[], 
                                uniform float QK_t[],
                                uniform int B,
                                uniform int H,
                                uniform int N, 
                                uniform int d) 
{
    foreach (b = 0...B, h = 0...H, j = 0...N) {

        for (int h = 0; h < H; h++) {

            // step1: calculate QK_t
            for (int i = 0; i < N; i++) {

                for (int j = 0; j < N; j++) {
                    
                    // QK_t(N * N) = Q(i, q) * K_t(q, j)
                    // K_t(q, j) = K(j, q)
                    float QK_ij = .0;
                    for (int q = 0; q < d; q++) {
                        QK_ij += fourDimRead(Q, b, h, i, q, H, N, d)
                            * fourDimRead(K, b, h, j, q, H, N, d);
                    }
                    
                    twoDimWrite(QK_t, i, j, N, QK_ij);
                }
            }

            // step2: Softmax in around
            for (int i = 0; i < N; i++) {
                float sum = .0;

                // sum = exp(a[i]) (i = 0, 1, 2, ...)
                for (int j = 0; j < N; j++) {
                    float tmp = twoDimRead(QK_t, i, j, N);
                    tmp = exp(tmp);
                    twoDimWrite(QK_t, i, j, N, tmp);
                    sum += tmp;
                }

                for (int j = 0; j < N; j++) {
                    float sm = twoDimRead(QK_t, i, j, N);
                    sm /= sum;
                    twoDimWrite(QK_t, i, j, N, sm);
                }
            }
            
            // step3: multiply V and get O
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < d; j++) {
                    float O_ij = .0;

                    for (int k = 0; k < N; k++) {
                        O_ij += twoDimRead(QK_t, i, k, N)
                            * fourDimRead(V, b, h, k, j, H, N, d);
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, O_ij);
                }
            }
        }
    }
}