#ifndef OPTIM_H
#define OPTIM_H

typedef struct Tensor Tensor;

#ifdef __cplusplus
extern "C" {
#endif
    void sgd_step_pass(Tensor *param, float lr);
#ifdef __cplusplus
}
#endif

#endif
