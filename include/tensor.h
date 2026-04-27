#ifndef TENSOR_H
#define TENSOR_H

struct Node;

typedef struct Tensor {
    float *data;
    int ndim;
    int *shape;
    int size;   
    Node *prev;

    int on_gpu;
    int requires_grad;
    struct Tensor *grad;
} Tensor;

#ifdef __cplusplus
extern "C" {
#endif
    // convolution operations
    void conv2d_forward_pass(const Tensor *input, const Tensor *filters, int padding, Tensor *output);
    void conv2d_backward_pass_weight(const Tensor *input, const Tensor *dout, int padding, Tensor *grad_w);
    void conv2d_backward_pass_input(const Tensor *filters, const Tensor *dout, int padding, Tensor *grad_x);

    //matmul ops
    void matmul_forward_pass(const Tensor *A, const Tensor *B, Tensor *C);
    void matmul_backward_pass_A(const Tensor *A, const Tensor *B, const Tensor *dC, Tensor *dA);
    void matmul_backward_pass_B(const Tensor *A, const Tensor *B, const Tensor *dC, Tensor *dA, Tensor *dB);

    // global average pooling
    void global_pooling_forward_pass(Tensor *input, Tensor *output);
    void global_pooling_backward_pass(Tensor *dout, Tensor *grad_input);

    // softmax-ce fused kernel
    void softmax_ce_forward(Tensor *logits, Tensor *labels, Tensor *loss);
    void softmax_ce_backward(Tensor *logits, Tensor *labels, Tensor *grad_logits);

    void normal_xavier_init(Tensor *weight, int in_fan, int out_fan);

    Tensor* tensor_create(int ndim, int *shape, int requires_grad, int on_gpu);
    void tensor_free(Tensor *t);
    void tensor_to_gpu(Tensor *t);
    void tensor_to_cpu(Tensor *t);

#ifdef __cplusplus
}
#endif
#endif


