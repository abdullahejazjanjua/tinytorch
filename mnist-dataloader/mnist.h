#ifndef MNIST_H
#define MNIST_H

#include "../include/tensor.h"

#define IMAGE_SIZE 28 * 28

struct MNISTData {
    unsigned char* images;
    unsigned char* labels;
};

int* create_indices(int num_images);
void free_mnist_data(MNISTData* data);
MNISTData* load_dataset_in_ram(const char *images_path, const char *labels_path, int num_images);
void load_batch_to_tensor(MNISTData *dataset, int batch_start, int batch_end, int *indices, Tensor *img_batch, Tensor *labels_batch);
#endif