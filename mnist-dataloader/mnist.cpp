#include <iostream>
#include <fstream>

#include "mnist.h"
#include "../include/tensor.h"

int* create_indices(int num_images) {
    int *indices = (int*) malloc(num_images * sizeof(num_images));
    if (!indices) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't allocate" << num_images << " indices array\n";
        return nullptr;
    }
    for (int i = 0; i < num_images; i++) {
        indices[i] = i;
    }
    return indices;
}

void shuffle_indices(int *indices, int num_images) {
    for (int i = num_images - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

MNISTData* load_dataset_in_ram(char *images_path, char *labels_path, int num_images) {
    MNISTData* data = (MNISTData*) malloc(sizeof(MNISTData));    
    data->images = (unsigned char*) malloc(num_images * IMAGE_SIZE * sizeof(unsigned char));
    data->labels = (unsigned char*) malloc(num_images * sizeof(unsigned char));

    if (data->images == nullptr) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't allocate a "<< num_images << " images array\n";
        free(data);
        return nullptr;
    }
    if (data->labels == nullptr) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't allocate a"<< num_images << " labels array\n";
        free(data->images);
        free(data);
        return nullptr;
    }

    std::ifstream ifile(images_path, std::ios::binary);
    if (!ifile.is_open()) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't open" << images_path << "\n";
        free(data->images);
        free(data->labels);
        free(data);
        return nullptr;
    }
    // HEADER: 4 bytes (magic number) + 4 bytes (num_images) + 4 bytes (num_rows) + 4 bytes (num_cols) = 16 bytes
    ifile.seekg(16, std::ios::beg); // skip header
    ifile.read((char*)data->images, num_images * IMAGE_SIZE);
    ifile.close();

    std::ifstream lfile(labels_path, std::ios::binary);
    if (!lfile.is_open()) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't open" << labels_path << "\n";
        free(data->images);
        free(data->labels);
        free(data);
        return nullptr;
    }
    // HEADER: 4 bytes (magic number) + 4 bytes (num_images) = 8 bytes
    lfile.seekg(8, std::ios::beg); // skip header
    lfile.read((char*)data->labels, num_images);
    lfile.close();
    
    return data;
}

void load_batch_to_tensor(MNISTData *dataset, int batch_start, int batch_end, int *indices, Tensor *img_batch, Tensor *labels_batch) {
    int batch_size = img_batch->shape[0];;
    int num_channels = img_batch->shape[1];
    int image_height = img_batch->shape[2];
    int image_width = img_batch->shape[3];;
    int local_batch = 0;
    for (int bs = batch_start; bs < batch_end; bs++) {
        int img_id = indices[bs];

        for (int i = 0; i < image_height; i++) {
            for (int j = 0; j < image_width; j++) {
                // load one pixel
                img_batch->data[local_batch * (num_channels * image_height * image_width) + 
                                0 * (image_height * image_width) + // 0 because we have only one channel (included for completeness)
                                i * (image_width) + 
                                j] = (float) dataset->images[img_id * (image_height * image_width) + 
                                            i * (image_width) + j] / 255.0f;
            }
        }
        labels_batch->data[local_batch] = (float) dataset->labels[img_id];
        local_batch++;
    }
}

void free_mnist_data(MNISTData* data) {
    if (data) {
        if (data->images) free(data->images);
        if (data->labels) free(data->labels);
        free(data);
    }
}

int main() {
    const int NUM_IMAGES = 60000;
    const int BATCH_SIZE = 4;
    char images_path[] = "train-images-idx3-ubyte";
    char labels_path[] = "train-labels-idx1-ubyte";

    // 1. Load Dataset
    MNISTData* dataset = load_dataset_in_ram(images_path, labels_path, NUM_IMAGES);
    if (!dataset) {
        std::cerr << "Failed to load dataset.\n";
        return 1;
    }

    // 2. Initialize and Shuffle Indices
    int* indices = create_indices(NUM_IMAGES);
    if (!indices) {
        free_mnist_data(dataset);
        return 1;
    }
    
    srand(42);
    shuffle_indices(indices, NUM_IMAGES);

    // 3. Allocate Tensors
    int img_shape[] = {BATCH_SIZE, 1, 28, 28};
    Tensor* img_batch = tensor_create(4, img_shape, 0);

    int label_shape[] = {BATCH_SIZE};
    Tensor* labels_batch = tensor_create(1, label_shape, 0);

    // 4. Load a Single Batch
    load_batch_to_tensor(dataset, 0, BATCH_SIZE, indices, img_batch, labels_batch);

    // 5. Verification Output
    std::cout << "--- Batch Verification ---\n";
    for (int b = 0; b < BATCH_SIZE; b++) {
        std::cout << "Batch Index: " << b 
                  << " | Original Image ID: " << indices[b] 
                  << " | Label: " << labels_batch->data[b] << "\n";
    }

    std::cout << "\nCenter 5x5 pixel patch of the first batch image (normalized):\n";
    for (int i = 12; i < 17; i++) {
        for (int j = 12; j < 17; j++) {
            // Index calculation for [Batch=0, Channel=0, Height=i, Width=j]
            int pixel_idx = 0 * (1 * 28 * 28) + 0 * (28 * 28) + i * 28 + j;
            std::cout << std::fixed << std::setprecision(2) << img_batch->data[pixel_idx] << " ";
        }
        std::cout << "\n";
    }

    // 6. Cleanup
    free_mnist_data(dataset);
    free(indices);
    tensor_free(img_batch);
    tensor_free(labels_batch);

    return 0;
}