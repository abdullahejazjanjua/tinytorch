#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

typedef struct Tensor Tensor;
typedef struct Node Node;

// global pooling wrappers
Tensor* global_pooling_functional_forward(Tensor *input, int ndim, int *expected_shape, int requires_grad);
void global_pooling_functional_backward(Node *node, Tensor *dout);

// convolution layer wrappers
Tensor* conv2d_functional_forward(Tensor *input, Tensor *weights, int padding, int requires_grad);
void conv2d_functional_backward(Node *node, Tensor *dout);

// linear layer wrappers
Tensor* linear_functional_forward(Tensor *input, Tensor *weights, int requires_grad);
void linear_functional_backward(Node *node, Tensor *dout);

// cross-entropy layer wrappers
Tensor* cross_entropy_functional_forward(Tensor *logits, Tensor *labels, int ndim, int *expected_shape, int requires_grad);
void cross_entropy_functional_backward(Node *node, Tensor *dout);

#endif