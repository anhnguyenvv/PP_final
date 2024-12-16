#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <fstream>

#define INPUT_SIZE 784
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 128
#define OUTPUT_SIZE 10
#define TRAIN_DATA_SIZE 10000
#define TEST_DATA_SIZE 1000
#define BATCH_SIZE 4
#define EPOCHS 10
#define LEARNING_RATE 0.01

typedef struct {
    float *weights_input_hidden1;
    float *weights_hidden1_hidden2;
    float *weights_hidden2_output;
    float *bias_hidden1;
    float *bias_hidden2;
    float *bias_output;
    float *grad_weights_input_hidden1;
    float *grad_weights_hidden1_hidden2;
    float *grad_weights_hidden2_output;
    float *grad_bias_hidden1;
    float *grad_bias_hidden2;
    float *grad_bias_output;
} NeuralNetwork;


#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start,0);
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

void initialize_weights(float *weights, int size) {
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

void initialize_neural_network(NeuralNetwork *nn) {
    CHECK(cudaMalloc(&nn->weights_input_hidden1, HIDDEN1_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&nn->weights_hidden1_hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&nn->weights_hidden2_output, OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&nn->bias_hidden1, HIDDEN1_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&nn->bias_hidden2, HIDDEN2_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&nn->bias_output, OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&nn->grad_weights_input_hidden1, HIDDEN1_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&nn->grad_weights_hidden1_hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&nn->grad_weights_hidden2_output, OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&nn->grad_bias_hidden1, HIDDEN1_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&nn->grad_bias_hidden2, HIDDEN2_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&nn->grad_bias_output, OUTPUT_SIZE * sizeof(float)));

    // Allocate temporary host memory
    float *h_weights_input_hidden1 = (float *)malloc(HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
    float *h_weights_hidden1_hidden2 = (float *)malloc(HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
    float *h_weights_hidden2_output = (float *)malloc(OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));
    float *h_bias_hidden1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    float *h_bias_hidden2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    float *h_bias_output = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    // Initialize weights and biases on the host
    initialize_weights(h_weights_input_hidden1, HIDDEN1_SIZE * INPUT_SIZE);
    initialize_weights(h_weights_hidden1_hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE);
    initialize_weights(h_weights_hidden2_output, OUTPUT_SIZE * HIDDEN2_SIZE);
    initialize_bias(h_bias_hidden1, HIDDEN1_SIZE);
    initialize_bias(h_bias_hidden2, HIDDEN2_SIZE);
    initialize_bias(h_bias_output, OUTPUT_SIZE);

    // Copy initialized values to device
    CHECK(cudaMemcpy(nn->weights_input_hidden1, h_weights_input_hidden1, HIDDEN1_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nn->weights_hidden1_hidden2, h_weights_hidden1_hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nn->weights_hidden2_output, h_weights_hidden2_output, OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nn->bias_hidden1, h_bias_hidden1, HIDDEN1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nn->bias_hidden2, h_bias_hidden2, HIDDEN2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nn->bias_output, h_bias_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Free temporary host memory
    free(h_weights_input_hidden1);
    free(h_weights_hidden1_hidden2);
    free(h_weights_hidden2_output);
    free(h_bias_hidden1);
    free(h_bias_hidden2);
    free(h_bias_output);
}

void load_data(const char *filename, float *data, int size) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
      printf("Không thể mở file\n");
      return;
  }
  // Tạo bộ nhớ để lưu dữ liệu
  size_t elements_read = fread(data, sizeof(float), size, file);

  // Kiểm tra số lượng phần tử đọc được
  if (elements_read != size) {
      printf("Số phần tử đọc được không khớp\n");
  }
  // Đóng file
  fclose(file);
}

void load_labels(const char *filename, int *labels, int size) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
      printf("Không thể mở file\n");
      return;
  }
  // Tạo bộ nhớ để lưu dữ liệu
  size_t elements_read = fread(labels, sizeof(int), size, file);

  // Kiểm tra số lượng phần tử đọc được
  if (elements_read != size) {
      printf("Số phần tử đọc được không khớp\n");
  }
  // Đóng file
  fclose(file);
}



void softmax(float *x, int batch_size, int size) {
  for (int b = 0; b < batch_size; b++) {
      float max_val = x[b * size];
      for (int i = 1; i < size; i++) {
          if (x[b * size + i] > max_val) max_val = x[b * size + i];
      }
      float sum = 0.0f;
      for (int i = 0; i < size; i++) {
          x[b * size + i] = expf(x[b * size + i] - max_val);
          sum += x[b * size + i];
      }
      for (int i = 0; i < size; i++) {
          x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
      }
  }
}
__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x;
    int tid = threadIdx.x;

    // Each thread processes one element
    if (b < batch_size && tid < size) {
        __shared__ float max_val;
        __shared__ float sum;

        // Find the maximum value in the batch
        if (tid == 0) {
            max_val = x[b * size];
            for (int i = 1; i < size; i++) {
                if (x[b * size + i] > max_val) max_val = x[b * size + i];
            }
        }
        __syncthreads();

        // Compute the exponentials and sum them up
        float exp_val = expf(x[b * size + tid] - max_val);
        atomicAdd(&sum, exp_val);
        __syncthreads();

        // Compute the softmax values
        x[b * size + tid] = fmaxf(exp_val / sum, 1e-7f);
    }
}
__global__ void reluKernel(float* x, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
      x[idx] = fmaxf(0.0f, x[idx]);
  }
}
void relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

void forwardLayer(float *input, float *weights, float *bias, float *output, int input_size,
                   int output_size, int batch_size, bool use_relu) { 
  
  for (int i = 0; i < batch_size; i++) { 
    for (int j = 0; j < output_size; j++) {
      output[i*output_size + j] = 0.0f;
      for (int k = 0; k < input_size; k++) {
          output[i*output_size + j] += input[i*input_size + k] * weights[k * output_size + j];
      }
      output[i] += bias[i]; // Cộng bias
    }
  }

  if (use_relu) {
      relu(output, batch_size*output_size); 
  }
}


float calculate_loss(float *output, int *labels, int batch_size)
{
  float total_loss = 0.0f;
  for (int b = 0; b < batch_size; b++) {
    if(labels[b]>0.0f)
      total_loss -= labels[b]*log(output[b* OUTPUT_SIZE + labels[b]]);
  }
  return total_loss / batch_size;
};

// Hàm tính đạo hàm của ReLU
void relu_derivative(float *x, int size, float *grad) {
  for (int i = 0; i < size; i++) {
      grad[i] *= (x[i] > 0.0f) ? 1.0f : 0.0f;
  }
}
__global__ void compute_delta_relu_kernel(float *relu_del_out, float *weights, float *input_layer, float *relu_del, int batch_size, int output_size, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * input_size) {
        int i = idx / input_size;
        int j = idx % input_size;
        
        relu_del[idx] = 0.0f;
        for (int k = 0; k < output_size; k++) {
            relu_del[idx] += relu_del_out[i * output_size + k] * weights[j * output_size + k];
        }

        relu_del[idx] *= (input_layer[idx] > 0.0f);
    }
}
void compute_delta_relu(float *relu_del_out, float *weights, float *input_layer, float *relu_del, 
                        int batch_size, int output_size, int input_size) {
  for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < input_size; j++) {
        int idx = i * input_size + j;
        relu_del[idx]  = 0.0f;
        for (int k = 0; k < output_size; k++) {
            relu_del[idx] += relu_del_out[i * output_size + k] * weights[j* output_size + k];
        }
        //  tính đạo hàm của ReLU

        relu_del[idx] *= (input_layer[idx] > 0.0f);
      }
  }
}
//  function to compute gradients for weights and biases
void compute_gradients(float *input, float *grad_output, float *grad_weights, 
                       float *grad_bias, int batch_size, int input_size, int output_size) {
  // Compute weight gradients
  for (int b = 0; b < batch_size; b++) {
      for (int j = 0; j < output_size; j++) {
        for (int i = 0; i < input_size; i++) {
          grad_weights[i * output_size + j] += input[b * input_size + i] * grad_output[b * output_size + j];
        }
        grad_bias[j] += grad_output[b * output_size + j];
      }        
  }
}

// Hàm cập nhật trọng số và bias
void update_weights(float * weights, float * grad_weights, float * bias, float * grad_bias, int output_size, int input_size) {
  // Cập nhật trọng số và bias
  for (int i = 0; i < input_size * output_size; i++) {
      weights[i] -= LEARNING_RATE *grad_weights[i];
  }
  for (int i = 0; i < output_size; i++) {
      bias[i] -= LEARNING_RATE * grad_bias[i];
  }
}

__global__ void forwardLayerKernel(float *input, float *weights, float *bias, float *output, int input_size, int output_size, int batch_size, bool use_relu) {
    int batch = blockIdx.x;
    int neuron = threadIdx.x;

    if (neuron < output_size && batch < batch_size) {
        float sum = 0.0f;

        for (int i = 0; i < input_size; ++i) {
            sum += input[batch * input_size + i] * weights[i * output_size + neuron];
        }

        output[batch * output_size + neuron] = sum + bias[neuron];
        if (use_relu) {
            output[batch * output_size + neuron] = fmaxf(0.0f, output[batch * output_size + neuron]);
        }
    }
}

__global__ void compute_gradients_kernel(float *input, float *grad_output, float *grad_weights, float *grad_bias, int batch_size, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_size * output_size) {
        int i = idx / output_size;
        int j = idx % output_size;

        float grad_weight = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad_weight += input[b * input_size + i] * grad_output[b * output_size + j];
        }
        grad_weights[idx] = grad_weight;

        if (i == 0) {
            float grad_bias_val = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                grad_bias_val += grad_output[b * output_size + j];
            }
            grad_bias[j] = grad_bias_val;
        }
    }
}

__global__ void update_weights_kernel(float *weights, float *grad_weights, float *bias, float *grad_bias, int output_size, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_size * output_size) {
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
    }

    if (idx < output_size) {
        bias[idx] -= LEARNING_RATE * grad_bias[idx];
    }
}

void train(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *d_X_train, *d_hidden1, *d_hidden2, *d_output, *d_del_output, *d_d_ReLU_out2, *d_d_ReLU_out1;
    int *d_y_train;
    float *hidden1 = (float *)malloc(BATCH_SIZE * HIDDEN1_SIZE * sizeof(float));
    float *hidden2 = (float *)malloc(BATCH_SIZE * HIDDEN2_SIZE * sizeof(float));
    float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    CHECK(cudaMalloc(&d_X_train, TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_hidden1, BATCH_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_hidden2, BATCH_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_del_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_d_ReLU_out2, BATCH_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_d_ReLU_out1, BATCH_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_y_train, TRAIN_DATA_SIZE * sizeof(int)));

    CHECK(cudaMemcpy(d_X_train, X_train, TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y_train, y_train, TRAIN_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    int num_batches = TRAIN_DATA_SIZE / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;

        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;

            // Forward pass for layer 1
            forwardLayerKernel<<<BATCH_SIZE, HIDDEN1_SIZE>>>(d_X_train + start_idx * INPUT_SIZE, nn->weights_input_hidden1, nn->bias_hidden1, d_hidden1, INPUT_SIZE, HIDDEN1_SIZE, BATCH_SIZE, true);

            // Forward pass for layer 2
            forwardLayerKernel<<<BATCH_SIZE, HIDDEN2_SIZE>>>(d_hidden1, nn->weights_hidden1_hidden2, nn->bias_hidden2, d_hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE, BATCH_SIZE, true);

            // Forward pass for output layer
            forwardLayerKernel<<<BATCH_SIZE, OUTPUT_SIZE>>>(d_hidden2, nn->weights_hidden2_output, nn->bias_output, d_output, HIDDEN2_SIZE, OUTPUT_SIZE, BATCH_SIZE, false);

            // Apply softmax
            softmax_kernel<<<BATCH_SIZE, OUTPUT_SIZE>>>(d_output, BATCH_SIZE, OUTPUT_SIZE);

            CHECK(cudaMemcpy(output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            float loss = calculate_loss(output, &y_train[start_idx], BATCH_SIZE);
            total_loss += loss;

            for (int i = 0; i < BATCH_SIZE; i++) {
                int predicted = 0;
                for (int j = 1; j < OUTPUT_SIZE; j++) {
                    if (output[i * OUTPUT_SIZE + j] > output[i * OUTPUT_SIZE + predicted]) {
                        predicted = j;
                    }
                }
                if (predicted == y_train[start_idx + i]) {
                    correct++;
                }
            }

            // Backpropagation
            CHECK(cudaMemset(nn->grad_weights_input_hidden1, 0, HIDDEN1_SIZE * INPUT_SIZE * sizeof(float)));
            CHECK(cudaMemset(nn->grad_weights_hidden1_hidden2, 0, HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float)));
            CHECK(cudaMemset(nn->grad_weights_hidden2_output, 0, OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float)));
            CHECK(cudaMemset(nn->grad_bias_hidden1, 0, HIDDEN1_SIZE * sizeof(float)));
            CHECK(cudaMemset(nn->grad_bias_hidden2, 0, HIDDEN2_SIZE * sizeof(float)));
            CHECK(cudaMemset(nn->grad_bias_output, 0, OUTPUT_SIZE * sizeof(float)));

            // Compute gradient at output layer
            for (int b = 0; b < BATCH_SIZE; b++) {
                for (int i = 0; i < OUTPUT_SIZE; i++) {
                    output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i] - (i == y_train[start_idx + b] ?  1.0f : 0.0f);
                }
            }
            CHECK(cudaMemcpy(d_del_output, output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            compute_gradients_kernel<<<(HIDDEN2_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(d_hidden2, d_del_output, nn->grad_weights_hidden2_output, nn->grad_bias_output, BATCH_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);

            compute_delta_relu_kernel<<<(BATCH_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(d_del_output, nn->weights_hidden2_output, d_hidden2, d_d_ReLU_out2, BATCH_SIZE, OUTPUT_SIZE, HIDDEN2_SIZE);
            compute_gradients_kernel<<<(HIDDEN1_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(d_hidden1, d_d_ReLU_out2, nn->grad_weights_hidden1_hidden2, nn->grad_bias_hidden2, BATCH_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE);

            compute_delta_relu_kernel<<<(BATCH_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(d_d_ReLU_out2, nn->weights_hidden1_hidden2, d_hidden1, d_d_ReLU_out1, BATCH_SIZE, HIDDEN2_SIZE, HIDDEN1_SIZE);
            compute_gradients_kernel<<<(INPUT_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(d_X_train + start_idx * INPUT_SIZE, d_d_ReLU_out1, nn->grad_weights_input_hidden1, nn->grad_bias_hidden1, BATCH_SIZE, INPUT_SIZE, HIDDEN1_SIZE);

            update_weights_kernel<<<(INPUT_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(nn->weights_input_hidden1, nn->grad_weights_input_hidden1, nn->bias_hidden1, nn->grad_bias_hidden1, HIDDEN1_SIZE, INPUT_SIZE);
            update_weights_kernel<<<(HIDDEN1_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(nn->weights_hidden1_hidden2, nn->grad_weights_hidden1_hidden2, nn->bias_hidden2, nn->grad_bias_hidden2, HIDDEN2_SIZE, HIDDEN1_SIZE);
            update_weights_kernel<<<(HIDDEN2_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(nn->weights_hidden2_output, nn->grad_weights_hidden2_output, nn->bias_output, nn->grad_bias_output, OUTPUT_SIZE, HIDDEN2_SIZE);
        }

        printf("Epoch %d/%d completed, Loss: %.4f, Accuracy: %.2f%%\n", epoch + 1, EPOCHS, total_loss / num_batches, 100.0f * correct / TRAIN_DATA_SIZE);
    }

    free(hidden1);
    free(hidden2);
    free(output);
    CHECK(cudaFree(d_X_train));
    CHECK(cudaFree(d_hidden1));
    CHECK(cudaFree(d_hidden2));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_del_output));
    CHECK(cudaFree(d_d_ReLU_out2));
    CHECK(cudaFree(d_d_ReLU_out1));
    CHECK(cudaFree(d_y_train));
}
void test(NeuralNetwork *nn, float *X_test, int *y_test) {
    float *d_X_test, *d_hidden1, *d_hidden2, *d_output;
    CHECK(cudaMalloc(&d_X_test, TEST_DATA_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_hidden1, BATCH_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_hidden2, BATCH_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    CHECK(cudaMemcpy(d_X_test, X_test, TEST_DATA_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int num_batches = TEST_DATA_SIZE / BATCH_SIZE;
    int correct = 0;

    for (int batch = 0; batch < num_batches; batch++) {
        int start_idx = batch * BATCH_SIZE;

        // Forward pass for layer 1
        forwardLayerKernel<<<BATCH_SIZE, HIDDEN1_SIZE>>>(d_X_test + start_idx * INPUT_SIZE, nn->weights_input_hidden1, nn->bias_hidden1, d_hidden1, INPUT_SIZE, HIDDEN1_SIZE, BATCH_SIZE, true);

        // Forward pass for layer 2
        forwardLayerKernel<<<BATCH_SIZE, HIDDEN2_SIZE>>>(d_hidden1, nn->weights_hidden1_hidden2, nn->bias_hidden2, d_hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE, BATCH_SIZE, true);

        // Forward pass for output layer
        forwardLayerKernel<<<BATCH_SIZE, OUTPUT_SIZE>>>(d_hidden2, nn->weights_hidden2_output, nn->bias_output, d_output, HIDDEN2_SIZE, OUTPUT_SIZE, BATCH_SIZE, false);

        // Apply softmax
        softmax_kernel<<<BATCH_SIZE, OUTPUT_SIZE>>>(d_output, BATCH_SIZE, OUTPUT_SIZE);

        float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
        CHECK(cudaMemcpy(output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        // Check prediction accuracy
        for (int i = 0; i < BATCH_SIZE; i++) {
            int predicted = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (output[i * OUTPUT_SIZE + j] > output[i * OUTPUT_SIZE + predicted]) {
                    predicted = j;
                }
            }
            if (predicted == y_test[start_idx + i]) {
                correct++;
            }
        }
        free(output);
    }

    float accuracy = 100.0f * correct / TEST_DATA_SIZE;
    printf("Test Accuracy: %.2f%%\n", accuracy);

    CHECK(cudaFree(d_X_test));
    CHECK(cudaFree(d_hidden1));
    CHECK(cudaFree(d_hidden2));
    CHECK(cudaFree(d_output));
}
int main(int argc, char **argv) {
    srand(time(NULL));

    NeuralNetwork nn;
    initialize_neural_network(&nn);

    float *X_train = (float *)malloc(TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int *)malloc(TRAIN_DATA_SIZE * sizeof(int));

    float *X_test =  (float *)malloc(TEST_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = (int *)malloc(TEST_DATA_SIZE * sizeof(int));

    load_data("x_train.bin", X_train, TRAIN_DATA_SIZE * INPUT_SIZE);
    load_labels("y_train.bin", y_train, TRAIN_DATA_SIZE);
    load_data("x_test.bin", X_test, TEST_DATA_SIZE * INPUT_SIZE);
    load_labels("y_test.bin", y_test, TEST_DATA_SIZE);

    // Training
    train(&nn, X_train, y_train);

    // Testing
    test(&nn, X_test, y_test);

    CHECK(cudaFree(nn.weights_input_hidden1));
    CHECK(cudaFree(nn.weights_hidden1_hidden2));
    CHECK(cudaFree(nn.weights_hidden2_output));
    CHECK(cudaFree(nn.bias_hidden1));
    CHECK(cudaFree(nn.bias_hidden2));
    CHECK(cudaFree(nn.bias_output));
    CHECK(cudaFree(nn.grad_weights_input_hidden1));
    CHECK(cudaFree(nn.grad_weights_hidden1_hidden2));
    CHECK(cudaFree(nn.grad_weights_hidden2_output));
    CHECK(cudaFree(nn.grad_bias_hidden1));
    CHECK(cudaFree(nn.grad_bias_hidden2));
    CHECK(cudaFree(nn.grad_bias_output));
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}
