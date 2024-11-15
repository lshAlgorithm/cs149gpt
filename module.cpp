#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int x, int y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int x, int y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int x, int y, int z, int b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int x, int y, int z, int b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //

    for (int b = 0; b < B; b++) {

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

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked_lsh(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //

    // N = 1024, d = 32
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {

            // Divide the matrix into 16 * 16
            // ps: 16 is the cache line in my machine
            const int GAP = 16;

            // Cut it into blocks, *first matrix oriented*
            for (int block_i = 0; block_i < N; block_i += GAP) {

                for (int block_j = 0; block_j < d; block_j += GAP) {
                    for (int K_id = 0; K_id < N; K_id += GAP) 

                    // Iterate over the block, get the coordinate(row, col) in the result matrix
                    for (int i = 0; i < GAP; i++) {
                        for (int j = 0; j < GAP; j++) {
                            int row = block_i + i, col = block_j + j;

                            // To get the addition, find the targeted column to read.
                            for (int q = 0; q < GAP; q++) {
                                //K_id means the offset of the GAP-wide traversion of matrix K.
                                int idx = K_id + q;

                                if (idx >= N || row >= N || col >= d) continue;

                                float QK_ij = twoDimRead(QK_t, row, idx, N);

                                QK_ij += fourDimRead(Q, b, h, row, col, H, N, d)
                                    * fourDimRead(K, b, h, idx, col, H, N, d);
                                
                                twoDimWrite(QK_t, row, idx, N, QK_ij);
                            }
                            
                        }
                    }
                }
            }

            // Softmax
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

            // Multiply V, *first matrix oriented*
            for (int block_i = 0; block_i < N; block_i += GAP) {
                for (int block_j = 0; block_j < N; block_j += GAP) {
                    for (int K_id = 0; K_id < d; K_id += GAP) 
                    // Iterate over the block, get the coordinate(row, col) in the result matrix
                    for (int i = 0; i < GAP; i++) {
                        for (int j = 0; j < GAP; j++) {
                            int row = block_i + i, col = block_j + j;

                            // To get the addition, find the targeted column to read.
                            for (int q = 0; q < GAP; q++) {
                                int idx = K_id + q;
                                if (idx >= d || row >= N || col >= N) continue;

                                float O_ij = fourDimRead(O, b, h, row, idx, H, N, d);

                                O_ij += twoDimRead(QK_t, row, col, N)
                                    * fourDimRead(V, b, h, col, idx, H, N, d);

                                fourDimWrite(O, b, h, row, idx, H, N, d, O_ij);
                            }

                        }
                    }
                }
            }

        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/// @brief This function is refered to BoxuanYang, much faster and more decent implementation than mine. The overhead of mine is 
/// reading and writing in every inner loop during the iteration due to my *first-matrix oriented* strategy, which is highly
/// unreconmmendened.
/// @param QTensor 
/// @param KTensor 
/// @param VTensor 
/// @param QK_tTensor 
/// @param B 
/// @param H 
/// @param N 
/// @param d 
/// @return O, the result of `(softmax(Q * K_t)) * V`
torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    int blocksize = 16;

    // -------- YOUR CODE HERE  -------- //

    for(int b = 0; b < B; b++){
        for(int h = 0; h < H; h++){

        // 1: Multiply Q, K^T and store in QK_t. Q, K are N by d in size.

        // start row index of Z-block
        for(int i = 0; i < N; i += blocksize){
            // size of Z-block in rows
            int QK_t_block_rows = std::min(blocksize, N - i);
            
            // start col index of QK_t block
            for(int j = 0; j < N; j += blocksize){
                // size of QK_t block in cols
                int QK_t_block_cols = blocksize < N - j ? blocksize : N - j; // std::min(blocksize, N - j);
                
                // start index of X's blocks(row) and Y's block(col)
                for(int k = 0; k < d; k += blocksize){
                    int d_size = blocksize < d - k ? blocksize : d - k;
                    // i-offset within block of Z 
                    for(int ii = 0; ii < QK_t_block_rows; ii++){
                        // j-offset within block of Z
                        for(int jj = 0; jj < QK_t_block_cols; jj++){
                            float QK_t_i_ii_j_jj = twoDimRead(QK_t, i + ii, j + jj, N);
                            
                            // k-offset within block of X(row-wise) and Y(col-wise)
                            for(int kk = 0; kk < d_size; kk++){
                                
                                // read Q[i + ii][k + kk]
                                float Q_ik = fourDimRead(Q, b, h, i + ii, k + kk, H, N, d);

                                // read K^T[k + kk][j + jj], i.e., K[j + jj][k + kk]
                                float K_jk = fourDimRead(K, b, h, j + jj, k + kk, H, N, d);

                                // Compute QK_t[i + ii][j + jj] += Q[i + ii][k + kk] * K^T[k + kk][j + jj]
                                QK_t_i_ii_j_jj += Q_ik * K_jk;
                            }
                            
                            twoDimWrite(QK_t, i + ii, j + jj, N, QK_t_i_ii_j_jj);
                        }                  
                    }
                } 
            }
        }


            // 2: Softmax Q K^T, in QK_t, QK_t is N by N in size
            for(int i = 0; i < N; i++){
                // 2.1: Transform i-th row of QK_t into exp and compute the norm
                float row_i_norm = 0;
                for(int j = 0; j < N; j++){ 
                    float QK_t_ij = twoDimRead(QK_t, i, j, N);
                    QK_t_ij = exp(QK_t_ij);
                    twoDimWrite(QK_t, i, j, N, QK_t_ij);

                    // Accumulate row norm in row_i_norm
                    row_i_norm += QK_t_ij;
                }

                // 2.2: Divide i-th row by QK_t_i_sum
                for(int j = 0; j < N; j++){ // divide by row sum
                    float QK_t_ij = twoDimRead(QK_t, i, j, N);
                    QK_t_ij = QK_t_ij / row_i_norm;
                    twoDimWrite(QK_t, i, j, N, QK_t_ij);
                }
            }


           // 3: Multiply QK_t, V and store in O, O is in shape: (B, H, N, d)

           // start row index of O block
           for(int i = 0; i < N; i += blocksize){
            // size of O-block in rows
            int O_block_rows = blocksize < N - i ? blocksize : N - i;
            
            // start col index of O block
            for(int j = 0; j < d; j += blocksize){
                // size of O block in cols
                int O_block_cols = blocksize < d - j ? blocksize : d - j; // std::min(blocksize, N - j);
                
                // start index of QK_t's blocks(row) and V's block(col)
                for(int k = 0; k < N; k += blocksize){
                    int d_size = blocksize < N - k ? blocksize : N - k;
                    // i-offset within block of O
                    for(int ii = 0; ii < O_block_rows; ii++){
                        // j-offset within block of O
                        for(int jj = 0; jj < O_block_cols; jj++){
                            // read O[i + ii][j + jj]
                            float O_i_ii_j_jj = fourDimRead(O, b, h, i + ii, j + jj, H, N, d);
                            
                            // k-offset within block of QK_t(row-wise) and V(col-wise)
                            for(int kk = 0; kk < d_size; kk++){
                                // read QK_t[i + ii][k + kk] 
                                float QK_t_ik = twoDimRead(QK_t, i + ii, k + kk, N);

                                // read V[k + kk][j + jj]
                                float V_jk = fourDimRead(V, b, h, k + kk, j + jj, H, N, d);

                                // Compute O[i + ii][j + jj] += QK_t[i + ii][k + kk] * V[k + kk][j + jj]
                                O_i_ii_j_jj += QK_t_ik * V_jk;
                            }
                            fourDimWrite(O, b, h, i + ii, j + jj, H, N, d, O_i_ii_j_jj);
                        }                  
                    }
                } 
            }
        }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    #pragma omp parallel for collapse(3)
    // We give you a template of the first three loops for your convenience
    //loop over batch
    for (int b = 0; b < B; b++){

        //loop over heads
        for (int h = 0; h < H; h++){
            for (int i = 0; i < N ; i++){

		// ORowTensor is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);
		//YOUR CODE HERE
                float sm_sum = .0;
                for (int j = 0; j < N; j++) {

                    float Row_j = .0;
                    for (int q = 0; q < d; q++) {
                        Row_j += fourDimRead(Q, b, h, i, q, H, N, d)
                            * fourDimRead(K, b, h, j, q, H, N, d);
                    }

                    float Row_j_exp = exp(Row_j);
                    ORow[j] = Row_j_exp;
                    sm_sum += Row_j_exp;
                }

                for (int j = 0; j < N; j++) {
                    ORow[j] /= sm_sum;
                }

                for (int j = 0; j < d; j++) {
                    // Get O_ij to put it into ultimate matrix
                    float O_ij = .0;
                    for (int k = 0; k < N; k++) {
                        O_ij += ORow[k] * fourDimRead(V, b, h, k, j, H, N, d);
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, O_ij);
                }
            }
	}
    }
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //


/// @brief Implement flash attention in C++ where you can specify the `Bc` and `Br`.
/// @param QTensor 
/// @param KTensor 
/// @param VTensor 
/// @param QiTensor 
/// @param KjTensor 
/// @param VjTensor 
/// @param SijTensor 
/// @param PijTensor 
/// @param PVTensor 
/// @param OiTensor 
/// @param LTensor 
/// @param LiTensor 
/// @param LijTensor 
/// @param LnewTensor 
/// @param Bc 
/// @param Br 
/// @param B 
/// @param H 
/// @param N 
/// @param d 
/// @return 
torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //

    // The number of blocks among rows(Q)
    int Tr = (N + Br - 1) / Br;
    // The number of blocks among cols(K)

    int Tc = (N + Bc - 1) / Bc;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            
            
            // Initialize l with 0's
            for(int ii = 0; ii < N; ii++){
                l[ii] = 0.0;
            }


            for(int j = 0; j < Tc; j++){
                // 1. Load Kj, Vj --CHECKED
                // Kj, Vj are passed in with Shape: (Bc, d)

                // start row index of Kj, Vj
                int j_start = j * Bc; 
                // size of the Kj & Vj
                int j_size = Bc < N - j_start ? Bc : N - j_start;

                for(int ii = 0; ii < j_size; ii++){
                    for(int jj = 0; jj < d; jj++){
                        float Kj_ii_jj = fourDimRead(K, b, h, ii + j_start, jj, H, N, d);
                        twoDimWrite(Kj, ii, jj, d, Kj_ii_jj);

                        float Vj_ii_jj = fourDimRead(V, b, h, ii + j_start, jj, H, N, d);
                        twoDimWrite(Vj, ii, jj, d, Vj_ii_jj);
                    }
                }

                for(int i = 0; i < Tr; i++){
                    // 2. Load Qi, Oi, li --CHECKED
                    // Qi:  (Br,d)  = (i_size,d)
                    // Oi:  (Br,d)  = (i_size,d)
                    // li:   Br

                    int i_start = i * Br;
                    int i_size = Br < N - i_start ? Br : N - i_start;

                    for(int ii = 0; ii < i_size; ii++){
                        // load li
                        li[ii] = l[ii + i_start];
                        for(int jj = 0; jj < d; jj++){
                            // load Qi
                            float Qi_ii_jj = fourDimRead(Q, b, h, ii + i_start, jj, H, N, d);
                            twoDimWrite(Qi, ii, jj, d, Qi_ii_jj);

                            // load Oi
                            float Oi_ii_jj = fourDimRead(O, b, h, ii + i_start, jj, H, N, d);
                            twoDimWrite(Oi, ii, jj, d, Oi_ii_jj);
                        }
                    }

                    
                    
                    // 3. Compute Sij = Qi * Kj^T
                    // Qi   : (Br,d)  = (i_size,d)
                    // Kj   : (Bc,d)  = (j_size,d)
                    // Sij  : (Br,Bc), passed in with Shape: (Br, Bc)
                    // Sij[ii][jj] += Qi[ii][kk] * Kj[jj][kk](Kj^T[kk][jj])
                    for(int ii = 0; ii < i_size; ii++){
                        for(int jj = 0; jj < j_size; jj++){
                            float Sij_ii_jj = 0;
                            for(int kk = 0; kk < d; kk++){
                                
                                Sij_ii_jj += twoDimRead(Qi, ii, kk, d) * twoDimRead(Kj, jj, kk, d);
                            }
                            
                            twoDimWrite(Sij, ii, jj, Bc, Sij_ii_jj);

                            // 4. Compute Pij = exp(Sij)
                            float Pij_ii_jj = exp(Sij_ii_jj);
                            twoDimWrite(Pij, ii, jj, Bc, Pij_ii_jj);
                        }
                    }

                    // 5. Compute lij = rowsum(Pij)
                    // Pij: (Br,Bc)
                    // lij: Br
                    for(int ii = 0; ii < i_size; ii++){
                        float lij_ii = 0;
                        for(int jj = 0; jj < j_size; jj++){
                            lij_ii += twoDimRead(Pij, ii, jj, Bc);
                        }
                        lij[ii] = lij_ii;
                    }

                    
                    // 6. Compute lnew = li + lij
                    // li, lij, and lnew are passed in with shape (Br)
                    for(int ii = 0; ii < i_size; ii++){
                        lnew[ii] = li[ii] + lij[ii];
                    }

                    
                    

                    // 7. Compute Oi = (li * Oi + Pij * Vj) / lnew. Note: li * Oi is elementwise multiply
                    // Oi:  (Br,d)  = (i_size,d),       passed in with shape: (Br, d)
                    // Pij: (Br,Bc) = (i_size,j_size),  passed in with shape: (Br, Bc)
                    // Vj:  (Bc,d)  = (j_size,d),       passed in with shape: (Bc, d)
                    // li:   Br     =  i_size,          passed in with shape  (Br)
                    // lnew: Br     =  i_size,          passed in with shape  (Br)
                    for(int ii = 0; ii < i_size; ii++){
                        float li_ii = li[ii];
                        for(int jj = 0; jj < d; jj++){
                            // read Oi[ii][jj]
                            float Oi_ii_jj = twoDimRead(Oi, ii, jj, d);

                            // Oi[ii][jj] = li[ii] * Oi[ii][jj]
                            Oi_ii_jj = li_ii * Oi_ii_jj;


                            for(int kk = 0; kk < j_size; kk++){
                                
                                Oi_ii_jj += twoDimRead(Pij, ii, kk, Bc) * twoDimRead(Vj, kk, jj, d);

                            }
                            
                            Oi_ii_jj = Oi_ii_jj / lnew[ii];
                            twoDimWrite(Oi, ii, jj, d, Oi_ii_jj);
                        }
                        
                    }

                    

                    
                   

                    // 8. Write Oi and lnew back to O and l
                    // Oi:  (Br,d)  = (i_size,d),       passed in with shape: (Br, d)
                    // lnew: Br     =  i_size,          passed in with shape  (Br)
                    for(int ii = 0; ii < i_size; ii++){
                        float lnew_ii = lnew[ii];
                        l[i_start + ii] = lnew_ii;
                        for(int jj = 0; jj < d; jj++){
                            float Oi_ii_jj = twoDimRead(Oi, ii, jj, d);
                            fourDimWrite(O, b, h, ii + i_start, jj, H, N, d, Oi_ii_jj);
                        }
                    }
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
