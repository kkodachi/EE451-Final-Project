#include "../include/module.h"
#include "../include/rand.h"
#include "../include/timer.h"
#include <cmath>

namespace Parallelize {
    int blockSize = 256;
    int gridSize = 8;
    int sharedSize = 32;
    int ceil_division(int n, int d) {
        return (n + d - 1) / d;
    }
}

// ################################################################################################################
/**
 * Dense matrix multiplication layer. 
*/
Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p) :
        a(a), b(b), c(c), m(m), n(n), p(p) {}

__global__ void forwardKernelMatmul(float* A,float* B,float* C,int m,int n,int p,int shared,int it){    
    // allocate shared memory
    __shared__ float A_S[32][32];
    __shared__ float B_S[32][32];
    // calculate necessary indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int r = blockIdx.y * blockDim.x + ty;
    int c = blockIdx.x * shared + tx;
    float sum = 0;
    // perform necessary iterations
    for (int i=0;i<it;i++){
        // copy to shared
        if (r < m && i * shared + tx < n) A_S[ty][tx] = A[r * n + i * shared + tx];
        else A_S[ty][tx] = 0;
        if (c < p && i * shared + ty < n) B_S[ty][tx] = B[(i * shared + ty) * p + c];
        else B_S[ty][tx] = 0;

        __syncthreads();
        
        for (int j = 0; j < shared; j++)
            sum += A_S[ty][j] * B_S[j][tx];

        __syncthreads();
    }
    if (r < m && c < p) C[r * p + c] = sum;
}

void Matmul::forward(bool training) {
    timer_start(TMR_MATMUL_FW);
    c->zero();
    // for (int i = 0; i < m; i++)
    //     for (int j = 0; j < n; j++) {
    //         for (int k = 0; k < p; k++)
    //             c->data[i * p + k] += a->data[i * n + j] * b->data[j * p + k];
    //     }
    float *gpu_A, *gpu_B, *gpu_C;
    // allocate memory on gpu
    cudaMalloc(&gpu_A, m * n * sizeof(float));
    cudaMalloc(&gpu_B, n * p * sizeof(float));
    cudaMalloc(&gpu_C, m * p * sizeof(float));
    
    // copy A and B from host to device
    cudaMemcpy(gpu_A, a->data.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, b->data.data(), n * p * sizeof(float), cudaMemcpyHostToDevice);

    // call kernel
    dim3 dimGrid(Parallelize::ceil_division(p,Parallelize::sharedSize),Parallelize::ceil_division(m,Parallelize::sharedSize));
    dim3 dimBlock(Parallelize::sharedSize,Parallelize::sharedSize);
    forwardKernelMatmul<<<dimGrid,dimBlock>>>(gpu_A,gpu_B,gpu_C,m,n,p,Parallelize::sharedSize,Parallelize::ceil_division(n,Parallelize::sharedSize));
    
    // copy result back
    cudaMemcpy(c->data.data(), gpu_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    // deallocate on gpu
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    
    timer_stop(TMR_MATMUL_FW);
}

__global__ void backwardKernelMatMulA(float* A,float* B,float* C,int m,int n,int p,int shared,int it){
    // allocate shared memory
    __shared__ float C_S[32][32];
    __shared__ float B_S[32][32];
    //calculate necessary indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int r = blockIdx.y * blockDim.x + ty;
    int c = blockIdx.x * shared + tx;
    float sum = 0;
    // perform necessary iterations
    for (int i = 0; i < it; i++){
        // copy to shared
        if (r < m && i * shared + tx < p) C_S[ty][tx] = C[r * p + i * shared + tx];
        else C_S[ty][tx] = 0;
        if (c < n && i * shared + ty < p) B_S[ty][tx] = B[c * p + i * shared + ty];
        else B_S[ty][tx] = 0;
            
        __syncthreads();

        for (int k = 0; k < shared; k++)
            sum += C_S[ty][k] * B_S[k][tx];

        __syncthreads();
    }
    if (r < m && c < n) A[r * n + c] = sum;
}

__global__ void backwardKernelMatMulB(float* A,float* B,float* C,int m,int n,int p,int shared,int it){
    // allocate shared memory
    __shared__ float A_S[32][32];
    __shared__ float C_S[32][32];
    // compute necessary indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int r = blockIdx.y * blockDim.x + ty;
    int c = blockIdx.x * shared + tx;
    float sum = 0;
    // perform necessary iterations
    for (int i = 0; i < it; i++) {
        if (r < n && i * shared + tx < m) A_S[ty][tx] = A[(i * shared + tx) * n + r];
        else A_S[ty][tx] = 0;
        if (c < p && i * shared + ty < m) C_S[ty][tx] = C[(i * shared + ty) * p + c];
        else C_S[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < shared; k++)
            sum += A_S[ty][k] * C_S[k][tx];

        __syncthreads();
    }
    if (r < n && c < p) B[r * p + c] = sum;
}

void Matmul::backward() {
    timer_start(TMR_MATMUL_BW);
    a->zero_grad();
    b->zero_grad();
    // for (int i = 0; i < m; i++)
    //     for (int j = 0; j < n; j++) {
    //             float tmp = 0;
    //             for (int k = 0; k < p; k++) {
    //                 tmp += c->grad[i * p + k] * b->data[j * p + k];
    //                 b->grad[j * p + k] += c->grad[i * p + k] * a->data[i * n + j];
    //             }
	// 	a->grad[i * n + j] = tmp;
    //     }
    float* gpu_A, *gpu_B, *gpu_C;
    // allocate memory on gpu
    cudaMalloc(&gpu_A, m * n * sizeof(float));
    cudaMalloc(&gpu_B, n * p * sizeof(float));
    cudaMalloc(&gpu_C, m * p * sizeof(float));
    
    // copy memory to gpu
    cudaMemcpy(gpu_A, a->data.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, b->data.data(), n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_C, c->data.data(), m * p * sizeof(float), cudaMemcpyHostToDevice);
    // call kernel function for A
    dim3 dimGridA(Parallelize::ceil_division(n,Parallelize::sharedSize),Parallelize::ceil_division(m,Parallelize::sharedSize));
    dim3 dimBlockA(Parallelize::sharedSize,Parallelize::sharedSize);
    backwardKernelMatMulA<<<dimGridA,dimBlockA>>>(gpu_A,gpu_B,gpu_C,m,n,p,Parallelize::sharedSize,Parallelize::ceil_division(p,Parallelize::sharedSize));
    // call kernel function for B
    dim3 dimGridB(Parallelize::ceil_division(p,Parallelize::sharedSize),Parallelize::ceil_division(n,Parallelize::sharedSize));
    dim3 dimBlockB(Parallelize::sharedSize,Parallelize::sharedSize);
    backwardKernelMatMulB<<<dimGridB,dimBlockB>>>(gpu_A,gpu_B,gpu_C,m,n,p,Parallelize::sharedSize,Parallelize::ceil_division(m,Parallelize::sharedSize));
    // copy both results
    cudaMemcpy(a->data.data(), gpu_A, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b->data.data(), gpu_B, n * p * sizeof(float), cudaMemcpyDeviceToHost);
    // deallocate on gpu
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);

    timer_stop(TMR_MATMUL_BW);
}

// ################################################################################################################

/**
 * A sparse matrix multiplication layer.
*/
SparseMatmul::SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p) :
        a(a), b(b), c(c), sp(sp), m(m), n(n), p(p) {}

// m = rows a,c
// n = cols a, rows b
// p = cols b,c

__global__ void forwardSparseMatmulKernel(float* A, float* B, float* C,int* indptr,int* indices,int m,int n,int p){
    // compute necessary indices
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int incr = blockDim.x * gridDim.x;
    // perform necessary iterations
    for (int i = ind; i < m * p; i += incr){
        int r = i / p;
        int c = i % p;
        float sum = 0;
        for (int j = indptr[r]; j < indptr[r + 1]; j++)
            sum += A[j] * B[indices[j] * p + c];
        C[i] = sum;
    }
    // for (int i = ind; i < m * p; i += incr) {
    //     int r = i / p;
    //     int c = i % p;
    //     float sum = 0;
    //     for (int k = 0; k < n; k++)
    //         sum += A[r * n + k] * B[k * p + c];
    //     C[i] = sum;
    // }
}



void SparseMatmul::forward(bool training) {
    timer_start(TMR_SPMATMUL_FW);
    c->zero();
    // printf("m,p,n,indptrsize : %d,%d,%d,%d",m,p,n,sp->indptr.size());
    // std::cout<< "indptr[0-4] "<< sp->indptr[0] << " " << sp->indptr[1] << " " << sp->indptr[2] << " " << sp->indptr[3] << std::endl;
    // for (int i = 0; i < sp->indptr.size() - 1; i++)
    //     for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
    //         int j = sp->indices[jj];
    //         for (int k = 0; k < p; k++)
    //             c->data[i * p + k] += a->data[jj] * b->data[j * p + k];
    //     }
    float* gpu_A, *gpu_B, *gpu_C;
    int *gpu_indptr, *gpu_indices;
    // allocate memory on gpu
    cudaMalloc(&gpu_A, sp->indptr.back() * sizeof(float)); // Allocate for the maximum size of A
    cudaMalloc(&gpu_B, b->data.size() * sizeof(float));
    cudaMalloc(&gpu_C, m * p * sizeof(float));
    cudaMalloc(&gpu_indptr, sp->indptr.size() * sizeof(int));
    cudaMalloc(&gpu_indices, sp->indices.size() * sizeof(int));
    // copy to gpu
    cudaMemcpy(gpu_A, a->data.data(), sp->indptr.back() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, b->data.data(), b->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_indptr, sp->indptr.data(), sp->indptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_indices, sp->indices.data(), sp->indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    // call kernel
    dim3 dimGrid(Parallelize::ceil_division(m * p, Parallelize::blockSize));
    dim3 dimBlock(Parallelize::blockSize);
    forwardSparseMatmulKernel<<<dimGrid, dimBlock>>>(gpu_A, gpu_B, gpu_C, gpu_indptr, gpu_indices, m, n, p);
    // copy result back
    cudaMemcpy(c->data.data(), gpu_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    // deallocate on gpu
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    cudaFree(gpu_indptr);
    cudaFree(gpu_indices);
    
    timer_stop(TMR_SPMATMUL_FW);
}

__global__ void backwardSparseMatmulKernel(float* A, float* B, float* C,int* indptr,int* indices,int m,int n,int p){
    // compute necessary indices
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int incr = blockDim.x * gridDim.x;
    // perform necessary iterations
    for (int i = ind; i < m * p; i += incr) {
        int r = i / p;
        int c = i % p;
        float val = C[i];
        for (int j = indptr[r]; j < indptr[r + 1]; j++) {
            atomicAdd(&B[indices[j] * p + c], A[j] * val); // prevents data race
        }
    }
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // if (row < m && col < p) {
    //     float sum = 0.0f;
    //     for (int k = 0; k < n; ++k) {
    //         sum += A[row * n + k] * C[k * p + col];
    //     }
    //     atomicAdd(&B[row * p + col], sum);
    // }
}

void SparseMatmul::backward() {
    timer_start(TMR_SPMATMUL_BW);
    b->zero_grad();
    // int row = 0;
    // printf("m,p,n,indptrsize : %d,%d,%d,%d",m,p,n,sp->indptr.size());
    // for (int i = 0; i < sp->indptr.size() - 1; i++)
    //     for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
    //         int j = sp->indices[jj];
    //         for (int k = 0; k < p; k++)
    //             b->grad[j * p + k] += c->grad[i * p + k] * a->data[jj];
    //     }

    float* gpu_A, *gpu_B, *gpu_C;
    int *gpu_indptr, *gpu_indices;
    // allocate on gpu
    cudaMalloc(&gpu_A, sp->indptr.back() * sizeof(float)); // Allocate for the maximum size of A
    cudaMalloc(&gpu_B, b->data.size() * sizeof(float));
    cudaMalloc(&gpu_C, m * p * sizeof(float));
    cudaMalloc(&gpu_indptr, sp->indptr.size() * sizeof(int));
    cudaMalloc(&gpu_indices, sp->indices.size() * sizeof(int));
    // copy to gpu
    cudaMemcpy(gpu_A, a->data.data(), sp->indptr.back() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, b->data.data(), b->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_indptr, sp->indptr.data(), sp->indptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_indices, sp->indices.data(), sp->indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    // call kernel
    dim3 dimGrid(Parallelize::ceil_division(m * p, Parallelize::blockSize));
    dim3 dimBlock(Parallelize::blockSize);
    forwardSparseMatmulKernel<<<dimGrid, dimBlock>>>(gpu_A, gpu_B, gpu_C, gpu_indptr, gpu_indices, m, n, p);
    // copy results
    cudaMemcpy(c->data.data(), gpu_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    // deallocate gpu
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    cudaFree(gpu_indptr);
    cudaFree(gpu_indices);


    timer_stop(TMR_SPMATMUL_BW);
}

// ################################################################################################################

/**
 * A specialized sparse matrix multiplication for graphs.
*/
GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim) :
        in(in), out(out), graph(graph), dim(dim) {}

__global__ void GraphSumKernel(float* in, float* out, int* indptr, int* indices, float* coefs, int size, int dim) {
    // compute necessary indices
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int inc = blockDim.x * gridDim.x;
    // perform necessary iterations
    for (int src = ind; src < size; src += inc) {
        for (int i = indptr[src]; i < indptr[src + 1]; i++) {
            int dst = indices[i];
            float coef = coefs[src];
            for (int j = 0; j < dim; j++) {
                atomicAdd(&out[dst * dim + j], coef * in[src * dim + j]); // prevents data race
            }
        }
    }
}


void GraphSum::forward(bool training) {
    timer_start(TMR_GRAPHSUM_FW);
    out->zero();
    // for (int src = 0; src < graph->indptr.size() - 1; src++)
    //     for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
    //         int dst = graph->indices[i];
    //         float coef = 1.0 / sqrtf(
    //                 (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
    //         );
    //         for (int j = 0; j < dim; j++)
    //             // This only works for undirected graphs. Should be out[dst] += coef * in[src]
    //             out->data[src * dim + j] += coef * in->data[dst * dim + j];
    //     }
    int size = graph->indptr.size() - 1;
    int *gpu_indptr, *gpu_indices;
    float *gpu_in, *gpu_out, *gpu_coef;
    // allocate on gpu
    cudaMalloc(&gpu_in, in->data.size() * sizeof(float));
    cudaMalloc(&gpu_out, out->data.size() * sizeof(float));
    cudaMalloc(&gpu_indptr, graph->indptr.size() * sizeof(int));  // Use .size() instead of ->data().size()
    cudaMalloc(&gpu_indices, graph->indices.size() * sizeof(int)); // Use .size() instead of ->data().size()
    cudaMalloc(&gpu_coef, size *sizeof(float));
    // copy to gpu
    cudaMemcpy(gpu_in, in->data.data(), in->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_out, out->data.data(), out->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_indptr, graph->indptr.data(), graph->indptr.size() * sizeof(int), cudaMemcpyHostToDevice); // Use .data() instead of ->data().data()
    cudaMemcpy(gpu_indices, graph->indices.data(), graph->indices.size() * sizeof(int), cudaMemcpyHostToDevice); // Use .data() instead of ->data().data()
    // compute coefficents
    std::vector<float> coefs(size);
    for (int i = 0; i < size; i++) {
        coefs[i] = 1.0f / sqrtf((graph->indptr[i + 1] - graph->indptr[i]) * (graph->indptr[graph->indices[graph->indptr[i + 1] - 1] + 1] - graph->indptr[graph->indices[graph->indptr[i + 1] - 1]]));
    }
    cudaMemcpy(gpu_coef, coefs.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    // call kernel
    GraphSumKernel<<<Parallelize::gridSize,Parallelize::blockSize>>>(gpu_in, gpu_out, gpu_indptr, gpu_indices, gpu_coef, size, dim);
    // copy result
    cudaMemcpy(gpu_out, out->data.data(), out->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    // deallocate on gpu
    cudaFree(gpu_in);
    cudaFree(gpu_out);
    cudaFree(gpu_indptr);
    cudaFree(gpu_indices);
    cudaFree(gpu_coef);

    timer_stop(TMR_GRAPHSUM_FW);
}

void GraphSum::backward() {
    timer_start(TMR_GRAPHSUM_BW);
    in->zero_grad();
    // for (int src = 0; src < graph->indptr.size() - 1; src++)
    //     for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
    //         int dst = graph->indices[i];
    //         float coef = 1.0 / sqrtf(
    //                 (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
    //         );
    //         for (int j = 0; j < dim; j++)
    //             in->grad[src * dim + j] += coef * out->grad[dst * dim + j];
    //     }
    int size = graph->indptr.size() - 1;
    int *gpu_indptr, *gpu_indices;
    float *gpu_in, *gpu_out, *gpu_coef;
    // allocate on gpu
    cudaMalloc(&gpu_in, in->data.size() * sizeof(float));
    cudaMalloc(&gpu_out, out->data.size() * sizeof(float));
    cudaMalloc(&gpu_indptr, graph->indptr.size() * sizeof(int));  // Use .size() instead of ->data().size()
    cudaMalloc(&gpu_indices, graph->indices.size() * sizeof(int)); // Use .size() instead of ->data().size()
    cudaMalloc(&gpu_coef, size *sizeof(float));
    // copy to gpu
    cudaMemcpy(gpu_in, in->data.data(), in->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_out, out->data.data(), out->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_indptr, graph->indptr.data(), graph->indptr.size() * sizeof(int), cudaMemcpyHostToDevice); // Use .data() instead of ->data().data()
    cudaMemcpy(gpu_indices, graph->indices.data(), graph->indices.size() * sizeof(int), cudaMemcpyHostToDevice); // Use .data() instead of ->data().data()
    // compute coefficents
    std::vector<float> coefs(size);
    for (int i = 0; i < size; i++) {
        coefs[i] = 1.0f / sqrtf((graph->indptr[i + 1] - graph->indptr[i]) * (graph->indptr[graph->indices[graph->indptr[i + 1] - 1] + 1] - graph->indptr[graph->indices[graph->indptr[i + 1] - 1]]));
    }
    cudaMemcpy(gpu_coef, coefs.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    // call kernel
    GraphSumKernel<<<Parallelize::gridSize,Parallelize::blockSize>>>(gpu_out, gpu_in, gpu_indptr, gpu_indices, gpu_coef, size, dim);
    // copy results
    cudaMemcpy(gpu_out, out->data.data(), out->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    // deallocate on gpu
    cudaFree(gpu_in);
    cudaFree(gpu_out);
    cudaFree(gpu_indptr);
    cudaFree(gpu_indices);
    cudaFree(gpu_coef);

    timer_stop(TMR_GRAPHSUM_BW);
}

// ################################################################################################################

/**
 * Each predicted class probability is compared to the actual class desired and a loss is computed to penalize the proabability based on how far it is with respect to the actual expected value.
 * Also called logaritmic loss. 
*/
CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes) :
        logits(logits), truth(truth), loss(loss), num_classes(num_classes) {}

void CrossEntropyLoss::forward(bool training) {
    timer_start(TMR_LOSS_FW);
    float total_loss = 0;
    int count = 0;
    if (training) logits->zero_grad();
    for (int i = 0; i < logits->data.size() / num_classes; i++) {
        if (truth[i] < 0) continue;
        count++;
        float *logit = &logits->data[i * num_classes];
        float max_logit = -1e30, sum_exp = 0;
        for (int j = 0; j < num_classes; j++)
            max_logit = fmax(max_logit, logit[j]);
        for (int j = 0; j < num_classes; j++) {
            logit[j] -= max_logit;
            sum_exp += expf(logit[j]);
        }
        total_loss += logf(sum_exp) - logit[truth[i]];

        if (training) {
            for (int j = 0; j < num_classes; j++) {
                float prob = expf(logit[j]) / sum_exp;
                logits->grad[i * num_classes + j] = prob;
            }
            logits->grad[i * num_classes + truth[i]] -= 1.0;
        }
    }
    *loss = total_loss / count;
    if (training)
        for (float & i : logits->grad)
            i /= count;
    timer_stop(TMR_LOSS_FW);
}

void CrossEntropyLoss::backward() {
}

// ################################################################################################################

/**
 * Rectified Linear Unit activation function.
 * If input is negative it will output 0.
*/
ReLU::ReLU(Variable *in) {
    this->in = in;
    mask = new bool[in->data.size()];
}

ReLU::~ReLU() {
    delete[] mask;
}

__global__ void forwardReLuKernel(float* in,unsigned char* mask,int size,bool training){
    // compute necessary indices
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int incr = blockDim.x * gridDim.x;
    // perform necessary iterations
    for (int i=ind;i<size;i+=incr){
        bool keep = in[i] > 0;
        if (training) mask[i] = keep ? 1 : 0;
        if (!keep) in[i] = 0;
    }
}

void ReLU::forward(bool training) {
    timer_start(TMR_RELU_FW);
    // for (int i = 0; i < in->data.size(); i++) {
    //     bool keep = in->data[i] > 0;
    //     if (training) mask[i] = keep;
    //     if (!keep) in->data[i] = 0;
    // }
    float* gpu_in;
    unsigned char* gpu_mask;
    // allocate on gpu
    cudaMalloc(&gpu_in, in->data.size() * sizeof(float));
    cudaMalloc(&gpu_mask, in->data.size() * sizeof(unsigned char));
    // copy to gpu
    cudaMemcpy(gpu_in, in->data.data(), in->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    // convert bool to unsigned char because CUDA had issues with bools
    std::vector<unsigned char> gpu_mask_char;
    for (size_t i = 0; i < in->data.size(); ++i) {
        if (mask[i]) gpu_mask_char.push_back(1);
        else gpu_mask_char.push_back(0);
    }
    cudaMemcpy(gpu_mask, gpu_mask_char.data(), in->data.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // call kernel
    int numBlocks = Parallelize::ceil_division(in->data.size(),Parallelize::blockSize);
    forwardReLuKernel<<<numBlocks,Parallelize::blockSize>>>(gpu_in,gpu_mask,in->data.size(),training);
    // copy results
    cudaMemcpy(in->data.data(), gpu_in, in->data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_mask_char.data(), gpu_mask, in->data.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // convert unsigned char back to bool
    for (size_t i = 0; i < in->data.size(); ++i) {
        mask[i] = gpu_mask_char[i] ? 1 : 0;
    }
    // deallocate on gpu
    cudaFree(gpu_in);
    cudaFree(gpu_mask);

    timer_stop(TMR_RELU_FW);
}

__global__ void backwardReLuKernel(float* in,unsigned char* mask,int size){
    // compute necessary indices
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int incr = blockDim.x * gridDim.x;
    // perform necessary iterations
    for (int i=ind;i<size;i+=incr){
        if (!mask[i]) in[i] = 0;
    }
}

void ReLU::backward() {
    timer_start(TMR_RELU_BW);
    // for (int i = 0; i < in->data.size(); i++)
    //     if (!mask[i]) in->grad[i] = 0;
    float* gpu_in;
    unsigned char* gpu_mask;
    // allocate on gpu
    cudaMalloc(&gpu_in, in->data.size() * sizeof(float));
    cudaMalloc(&gpu_mask, in->data.size() * sizeof(unsigned char));
    // copy to gpu
    cudaMemcpy(gpu_in, in->data.data(), in->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    // convert bool to unsigned char because CUDA had issues with bools
    std::vector<unsigned char> gpu_mask_char;
    for (size_t i = 0; i < in->data.size(); ++i) {
        if (mask[i]) gpu_mask_char.push_back(1);
        else gpu_mask_char.push_back(0);
    }
    cudaMemcpy(gpu_mask, gpu_mask_char.data(), in->data.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // call kernel
    int numBlocks = Parallelize::ceil_division(in->data.size(),Parallelize::blockSize);
    backwardReLuKernel<<<numBlocks,Parallelize::blockSize>>>(gpu_in,gpu_mask,in->data.size());
    // copu results back
    cudaMemcpy(in->data.data(), gpu_in, in->data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_mask_char.data(), gpu_mask, in->data.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // convert unsigned char back to bool
    for (size_t i = 0; i < in->data.size(); ++i) {
        mask[i] = gpu_mask_char[i] ? 1 : 0;
    }
    // deallocate on gpu
    cudaFree(gpu_in);
    cudaFree(gpu_mask);

    timer_stop(TMR_RELU_BW);
}

// ################################################################################################################

/**
 * The dropout layer randomly sets input units to 0 with a frequency of P at each step during training time to prevent overfitting. 
 * Inputs that are not set to 0 are scaled up by 1/(1-P).
*/
Dropout::Dropout(Variable *in, float p) {
    this->in = in;
    this->p = p;
    if (!in->grad.empty()) 
        mask = new int[in->data.size()];
    else mask = nullptr;
}

Dropout::~Dropout() {
    if (mask) delete[] mask;
}

void Dropout::forward(bool training) {
    if (!training) return;
    timer_start(TMR_DROPOUT_FW);
    const int threshold = int(p * MY_RAND_MAX);
    float scale = 1 / (1 - p);
    for (int i = 0; i < in->data.size(); i++) {
        bool keep = (int)RAND() >= threshold;
        in->data[i] *= keep ? scale : 0;
        if (mask) mask[i] = keep;
    }
    timer_stop(TMR_DROPOUT_FW);
}

void Dropout::backward() {
    if (!mask) return;
    timer_start(TMR_DROPOUT_BW);
    float scale = 1 / (1 - p);
    for (int i = 0; i < in->data.size(); i++)
        in->grad[i] *= mask[i] ? scale : 0;
    timer_stop(TMR_DROPOUT_BW);
}

// ################################################################################################################