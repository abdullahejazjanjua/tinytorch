#include <curand.h>
#include <curand_kernel.h>

#include "../include/tensor.h"
#include "../include/common.cuh"


void normal_xavier_init(Tensor *weight, int in_fan, int out_fan) {
    curandGenerator_t randGen;
    CHECK_CURAND( curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT) );
    CHECK_CURAND( curandSetPseudoRandomGeneratorSeed(randGen, 1234ULL) );

    float standard_dev = sqrt(2.0f/(in_fan + out_fan));
    CHECK_CURAND( curandGenerateNormal(randGen,  weight->data, weight->size, 0.0f, standard_dev) );  

    CHECK_CURAND( curandDestroyGenerator(randGen) );

}