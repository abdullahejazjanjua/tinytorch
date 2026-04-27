#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../../include/tensor.h"
#include "../../include/autograd.h"

void cross_entropy_functional_backward(Node *node, Tensor *dout) {
    Tensor *logits = node->inputs[0];
    Tensor *labels = node->inputs[1];
    softmax_ce_backward(logits, labels, logits->grad);   
}


Tensor* cross_entropy_functional_forward(Tensor *logits, Tensor *labels, int ndim, int *expected_shape, int requires_grad) {
    Tensor *loss = tensor_create(ndim, expected_shape, 0, 1); // create on gpu
    cudaMemset(loss->data, 0, loss->size * sizeof(float));
    
    // call the forward function
    softmax_ce_forward(logits, labels, loss);

    // As our kernel is fused, we directly compute grads for logits, skipping grad for loss. So, loss doesn't have a grad field
    if (logits->requires_grad) {
        Node *_prev = (Node *) malloc(sizeof(Node));
        if (_prev == nullptr) {
            std :: cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't allocate space for Node\n";
            return nullptr;
        }

        _prev->inputs = (Tensor**) malloc(2 * sizeof(Tensor*));
        if (_prev->inputs == nullptr) {
            std :: cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't allocate space for pointer to inputs\n";
            return nullptr;
        }
        _prev->inputs[0] = logits;
        _prev->inputs[1] = labels;
        _prev->num_inputs = 2;

        // placeholder to put denominator, max from forward pass
        _prev->ctx =  nullptr;//(void**) malloc(2 * sizeof(void*));; // no ctx needed for backprop
        // float val = 0.0f;
        // _prev->ctx[0] = (void*)&val;
        // _prev->ctx[1] = (void*)&val;
        _prev->num_ctx = 0;

        _prev->backward = cross_entropy_functional_backward;

        loss->prev = _prev;
    }
    return loss;
}