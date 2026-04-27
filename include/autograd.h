#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

typedef void (*backward_fn)(Node *node, Tensor *dout);

typedef struct Node {
    Tensor **inputs; // array of tensors that were used to create this node
    void **ctx; // all ctx needed
    
    int num_inputs;
    int num_ctx;
    
    // Generic func pointer
    backward_fn backward;
} Node;

#endif