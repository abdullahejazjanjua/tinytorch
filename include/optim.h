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

#ifdef __cplusplus
#include <vector>

class SGD {
    private:
        std::vector<Tensor*> params;
        float lr;
    public:
        SGD(std::vector<Tensor*> params, float lr);
        void step();
        void zero_grad();
};
#endif

#endif
