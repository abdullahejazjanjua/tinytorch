#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

typedef struct Tensor Tensor;
typedef struct Node Node;

// global pooling wrappers
Tensor* global_pooling_functional_forward(Tensor *input, int ndim, int *expected_shape, int requires_grad);
void global_pooling_functional_backward(Node *node, Tensor *dout);

void conv2d_functional_backward(Node *node, Tensor *dout);
Tensor* conv2d_functional_forward(Tensor *input, Tensor *weights, int padding, int ndim, int *expected_shape, int requires_grad);


#endif