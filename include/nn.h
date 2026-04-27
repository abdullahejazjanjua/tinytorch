#ifndef NN_H
#define NN_H

typedef struct Tensor Tensor;

class GlobalPooling {
    private:
        int requires_grad;
    public:
        GlobalPooling(int requires_grad = 1);

    Tensor* forward(Tensor* input);
};

class Conv2D {
    private:
        int padding;
        int in_channels;
        int out_channels;
        int kernel_size;
        int requires_grad;
        Tensor *weights;
    public:
        Conv2D(int in_channels, int out_channels, int kernel_size, int padding, int requires_grad);
    
    Tensor *forward(Tensor *input);

};

#endif