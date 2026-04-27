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

class CrossEntropy {
    private:
        int requires_grad;
    public:
        CrossEntropy(int requires_grad = 1);

    Tensor* forward(Tensor* logits, Tensor *labels);
};

class Conv2D {
    public:
        int padding;
        int in_channels;
        int out_channels;
        int kernel_size;
        int requires_grad;
        Tensor *weights;

        Conv2D(int in_channels, int out_channels, int kernel_size, int padding, int requires_grad);
    
    Tensor *forward(Tensor *input);

};

class Linear {
    public:
        int in_features;
        int out_features;
        int requires_grad;
        Tensor *weights;

        Linear(int in_features, int out_features, int requires_grad);
    
    Tensor *forward(Tensor *input);

};

#endif