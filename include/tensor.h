#ifndef TENSOR_H
#define TENSOR_H

typedef struct Tensor {
    float *data;
    int ndim;           
    int *shape;
    int size;   
    
    int requires_grad;
    struct Tensor *grad;
} Tensor;

Tensor* tensor_create(int ndim, int *shape);
void tensor_free(Tensor *t);

#ifdef __cplusplus
extern "C" {
#endif
    // convolution operations
    void conv2d_forward_pass(const Tensor *input, const Tensor *filters, int padding, Tensor *output);
    void conv2d_backward_pass_weight(const Tensor *input, const Tensor *dout, int padding, Tensor *grad_w);
    void conv2d_backward_pass_input(const Tensor *filters, const Tensor *dout, int padding, Tensor *grad_x);
    //matmul ops
    Tensor* matmul(const Tensor *A, const Tensor *B);
    void matmul_forward_pass(const Tensor *A, const Tensor *B, Tensor *C);
    void matmul_backward_pass(const Tensor *A, const Tensor *B, const Tensor *dC, Tensor *dA, Tensor *dB);
#ifdef __cplusplus
}
#endif
#endif


