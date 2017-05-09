#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

/* TODO: Your code here */
__global__ void array_set_kernel(float *arr, const float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    arr[idx] = value;
}

__global__ void broadcast_to(int size, const float *input, float *output) {
    output += size * threadIdx.x;
    memcpy(output, input, size * sizeof(float));
}

__global__ void reduce_sum_axiszero_kernel(int nx, int size,
                                           const float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int x = 0; x < nx; ++ x)
        output[idx] += input[x * size + idx];
}

__global__ void matrix_elementwise_add(const float *matA, const float *matB, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = matA[idx] + matB[idx];
}

__global__ void matrix_elementwise_addbyconst(float value, const float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx] + value;
}

__global__ void matrix_elementwise_multiply(const float *matA, const float *matB, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = matA[idx] * matB[idx];
}

__global__ void matrix_elementwise_multiplybyconst(float value, const float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx] * value;
}

__global__ void relu(const float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx] > 0. ? input[idx] : 0.;
}

__global__ void relu_gradient(const float *input, const float *in_grad, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx] > 0. ? in_grad[idx]  : 0.;
}

__global__ void matrix_softmax(int nrow, int ncol,
                               const float *input, float *output) {
    // Two dimensional thread blocks.
    int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
            threadIdx.x;
    if (y >= nrow) {
        return;
    }
    input += y * ncol;
    output += y * ncol;
    float maxval = *input;
    // Find max for a row.
    for (int x = 1; x < ncol; ++x) {
        maxval = max(maxval, input[x]);
    }
    // Deduct by max for a row, and raise to exp.
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
        sum += exp(input[x] - maxval);
    }
    for (int x = 0; x < ncol; ++x) {
        output[x] = exp(input[x] - maxval) / sum;
    }
}


/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
    // Dynamic shared memory, size provided at kernel launch.
    extern __shared__ float loss_per_row[];
    // Two dimensional thread blocks.
    int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
            threadIdx.x;
    if (y >= nrow) {
        return;
    }
    input_a += y * ncol;
    input_b += y * ncol;
    float maxval = *input_a;
    // Find max for a row.
    for (int x = 1; x < ncol; ++x) {
        maxval = max(maxval, input_a[x]);
    }
    // Deduct by max for a row, and raise to exp.
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
        sum += exp(input_a[x] - maxval);
    }
    // Compute per-row loss.
    float loss = 0;
    for (int x = 0; x < ncol; ++x) {
        loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
    }
    loss_per_row[y] = loss;
    __syncthreads();
    // Compute reduce_mean across rows.
    float mean_loss = 0;
    // Use a single thread to reduce mean across rows.
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        for (int i = 0; i < nrow; ++i) {
            mean_loss += loss_per_row[i];
        }
        mean_loss /= nrow;
        output[0] = mean_loss;
    }
}

int DLGpuArraySet(DLArrayHandle arr, float value) {
    /* TODO: Your code here */
    int size = arr->shape[0];
    for (int i = 1; i < arr->ndim; ++ i) size *= arr->shape[i];
    float *arr_data = (float *) arr->data;
    dim3 block, grid;
    if (size <= 1024) {
        block.x = size;
        grid.x = 1;
    } else {
        block.x = 1024;
        grid.x = (size + 1023) >> 10;
    }
    array_set_kernel <<< grid, block >>> (arr_data, value);
    return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    int size = input->shape[0], nx = output->shape[0];
    for (int i = 1; i < input->ndim; ++ i) size *= input->shape[i];
    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    dim3 block;
    block.x = nx;
    broadcast_to <<< 1, block >>> (size, input_data, output_data);
    return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    int size = output->shape[0];
    for (int i = 1;i < output->ndim; ++ i) size *= output->shape[i];
    int nx = input->shape[0];
    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    cudaMemset((void *) output_data, .0, size * sizeof(float));
    dim3 block, grid;
    if (size <= 1024) {
        block.x = size;
        grid.x = 1;
    } else {
        block.x = 1024;
        grid.x = (size + 1023) >> 10;
    }
    reduce_sum_axiszero_kernel <<< grid, block >>> (nx, size, input_data, output_data);
    return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
    /* TODO: Your code here */
    int size = output->shape[0];
    for (int i = 1;i < output->ndim; ++ i) size *= output->shape[i];
    const float *matA_data = (const float *) matA->data;
    const float *matB_data = (const float *) matB->data;
    float *output_data = (float *) output->data;
    dim3 block, grid;
    if (size <= 1024) {
        block.x = size;
        grid.x = 1;
    } else {
        block.x = 1024;
        grid.x = (size + 1023) >> 10;
    }
    matrix_elementwise_add <<< grid, block >>> (matA_data, matB_data, output_data);
    return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
    /* TODO: Your code here */
    int size = output->shape[0];
    for (int i = 1;i < output->ndim; ++ i) size *= output->shape[i];
    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    dim3 block, grid;
    if (size <= 1024) {
        block.x = size;
        grid.x = 1;
    } else {
        block.x = 1024;
        grid.x = (size + 1023) >> 10;
    }
    matrix_elementwise_addbyconst <<< grid, block >>> (val, input_data, output_data);
    return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
    /* TODO: Your code here */
    int size = output->shape[0];
    for (int i = 1;i < output->ndim; ++ i) size *= output->shape[i];
    const float *matA_data = (const float *) matA->data;
    const float *matB_data = (const float *) matB->data;
    float *output_data = (float *) output->data;
    dim3 block, grid;
    if (size <= 1024) {
        block.x = size;
        grid.x = 1;
    } else {
        block.x = 1024;
        grid.x = (size + 1023) >> 10;
    }
    matrix_elementwise_multiply <<< grid, block >>> (matA_data, matB_data, output_data);
    return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
    /* TODO: Your code here */
    int size = output->shape[0];
    for (int i = 1;i < output->ndim; ++ i) size *= output->shape[i];
    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    dim3 block, grid;
    if (size <= 1024) {
        block.x = size;
        grid.x = 1;
    } else {
        block.x = 1024;
        grid.x = (size + 1023) >> 10;
    }
    matrix_elementwise_multiplybyconst <<< grid, block >>> (val, input_data, output_data);
    return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
    /* TODO: Your code here */
    // Hint: use cublas
    // cublas assume matrix is column major
    assert(matA->ndim == 2);
    assert(matB->ndim == 2);
    assert(matC->ndim == 2);
    int mA = matA->shape[0], nA = matA->shape[1];
    int mB = matB->shape[0], nB = matB->shape[1];
    assert((transposeA ? mA : nA) == (transposeB ? nB : mB));

    const float *matA_data = (const float *) matA->data;
    const float *matB_data = (const float *) matB->data;
    float *matC_data = (float *) matC->data;

    static cublasHandle_t handle;
    static cublasStatus_t status = cublasCreate(&handle);
    assert(status == CUBLAS_STATUS_SUCCESS);

    int lB = nB, fA = mA, fB = mB;
    if (transposeA) {
        fA = nA;
    }
    if (transposeB) {
        lB = mB;
        fB = nB;
    }
    cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    float alpha = 1.0f, beta = 0.0f;
    status = cublasSgemm_v2(handle, opB, opA,
                         lB, fA, fB, &alpha,
                         matB_data, nB,
                         matA_data, nA, &beta,
                         matC_data, lB);

    assert(status == CUBLAS_STATUS_SUCCESS);
    return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    assert(input->ndim == output->ndim);
    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    int n_units = input->shape[0];
    for (int i = 1;i < input->ndim; ++ i) n_units *= input->shape[i];
    dim3 block, grid;
    if (n_units <= 1024) {
        block.x = n_units;
        grid.x = 1;
    } else {
        block.x = 1024;
        grid.x = (n_units + 1023) >> 10;
    }
    relu <<< grid, block >>> (input_data, output_data);
    return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
    /* TODO: Your code here */
    assert(input->ndim == output->ndim);
    const float *input_data = (const float *) input->data;
    const float *in_grad_data = (const float *) in_grad->data;
    float *output_data = (float *) output->data;
    int n_units = input->shape[0];
    for (int i = 1;i < input->ndim; ++ i) n_units *= input->shape[i];
    dim3 block, grid;
    if (n_units <= 1024) {
        block.x = n_units;
        grid.x = 1;
    } else {
        block.x = 1024;
        grid.x = (n_units + 1023) >> 10;
    }
    relu_gradient <<< grid, block >>> (input_data, in_grad_data, output_data);
    return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    int nrow = input->shape[0];
    // Maximum x- or y-dimension of a block = 1024
    // But we need 'nrow' shared memory, and max shared memory is 48KB.
    // Conservatively allow max 16KB shared memory.
    assert(nrow <= 1024 * 4);
    int ncol = input->shape[1];
    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    } else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_softmax << < 1, threads, nrow * sizeof(float) >> > (
            nrow, ncol, input_data, output_data);
    return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
    assert(input_a->ndim == 2);
    assert(input_b->ndim == 2);
    assert(output->ndim == 1);
    assert(input_a->shape[0] == input_b->shape[0] &&
           input_a->shape[1] == input_b->shape[1]);
    int nrow = input_a->shape[0];
    // Maximum x- or y-dimension of a block = 1024
    // But we need 'nrow' shared memory, and max shared memory is 48KB.
    // Conservatively allow max 16KB shared memory.
    assert(nrow <= 1024 * 4);
    int ncol = input_a->shape[1];
    const float *input_data_a = (const float *) input_a->data;
    const float *input_data_b = (const float *) input_b->data;
    float *output_data = (float *) output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    } else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_softmax_cross_entropy_kernel << < 1, threads, nrow * sizeof(float) >> > (
            nrow, ncol, input_data_a, input_data_b, output_data);
    return 0;
}
