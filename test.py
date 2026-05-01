import nn
import optim
import base
import mnist_io as data
import numpy as np

def test_tensor_basics():
    print("Testing Tensor creation and NumPy integration...")
    shape = [2, 3]
    t = base.tensor_create(shape, 1, 0)
    
    # Check properties
    assert t.ndim == 2
    assert t.shape == [2, 3]
    assert t.size == 6
    
    # Write data via NumPy view
    data = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float32)
    t.data[:] = data
    
    # Read data back
    assert np.allclose(t.data, data)
    base.tensor_free(t)
    print("Tensor basics passed.")

def test_layers_forward():
    print("Testing ReLU forward pass...")
    t = base.tensor_create([4], 0, 0)
    t.data[:] = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    base.tensor_to_gpu(t)

    relu = nn.ReLU()
    output = relu.forward(t)
    base.tensor_to_cpu(output)
    
    # Expected: [1.0, 0.0, 2.0, 0.0]
    expected = np.array([1.0, 0.0, 2.0, 0.0], dtype=np.float32)
    assert np.allclose(output.data, expected)
    
    base.tensor_free(t)
    base.tensor_free(output)
    # Note: If your C++ ReLU creates a new Tensor, it should be freed too
    print("ReLU forward passed.")

def test_linear_layer():
    print("Testing Linear layer initialization...")
    in_features, out_features = 10, 5
    linear = nn.Linear(in_features, out_features, 1, 1)
    
    assert linear.in_features == 10
    assert linear.out_features == 5
    assert linear.weights.shape == [in_features, out_features]
    assert linear.bias.shape == [out_features]
    
    print("Linear layer passed.")

def test_sgd_optimizer():
    print("Testing SGD optimizer...")
    w = base.tensor_create([5], 1, 1)
    b = base.tensor_create([1], 1, 1)
    # base.tensor_to_gpu(w)
    # base.tensor_to_gpu(b)
    
    params = [w, b]
    optimizer = optim.SGD(params, lr=0.01)
    
    optimizer.zero_grad()
    optimizer.step()
    
    base.tensor_free(w)
    base.tensor_free(b)
    print("SGD passed.")

def test_data():
    dataset = data.load_dataset_in_ram("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 60000)

    indices = data.create_indices(60000)
    np.random.shuffle(indices)
    
    img_batch = base.tensor_create([32, 784], 0, 0)
    label_batch = base.tensor_create([32, 10], 0, 0)

    data.load_batch_to_tensor(dataset, 0, 32, indices, img_batch, label_batch)
    print(label_batch.data)
    
    data.free_mnist_data(dataset)
    base.tensor_free(img_batch)
    base.tensor_free(label_batch)

if __name__ == "__main__":
    try:
        test_tensor_basics()
        test_layers_forward()
        test_linear_layer()
        test_sgd_optimizer()
        test_data()
        print("\nAll tests passed successfully.")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")