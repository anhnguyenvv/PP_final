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
#define TILE_WIDTH 32

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
        float random = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Uniform [-1, 1]
        weights[i] = random * scale; // Áp dụng scale He
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

__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        int idx = b * size;
        float max_val = x[idx];
        
        for (int i = 1; i < size; i++) {
            max_val = fmaxf(max_val, x[idx + i]);
        }

        float sum = 0.0f;
        
        for (int i = 0; i < size; i++) {
            x[idx + i] = expf(x[idx + i] - max_val);
            sum += x[idx + i];
        }
        
        for (int i = 0; i < size; i++) {
            x[idx + i] = fmaxf(x[idx + i] / sum, 1e-7f);
        }
    }
}
__global__ void matrix_multiplication_kernel1(float* A, float* B, float* C, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k) {
        float value = 0;
        for (int e = 0; e < n; ++e) {
            value += A[row * n + e] * B[e * k + col];
        }
        C[row * k + col] = value;
    }
}

__global__ void matrix_multiplication_kernel2(float* A, float* B, float* C, int m, int n, int k)
{
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float value = 0;

    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (Row < m && t * TILE_WIDTH + tx < n)
            s_A[ty][tx] = A[Row * n + t * TILE_WIDTH + tx];
        else
            s_A[ty][tx] = 0.0;

        if (Col < k && t * TILE_WIDTH + ty < n)
            s_B[ty][tx] = B[(t * TILE_WIDTH + ty) * k + Col];
        else
            s_B[ty][tx] = 0.0;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += s_A[ty][i] * s_B[i][tx];
        }

        __syncthreads();
    }

    if (Row < m && Col < k) {
        C[Row * k + Col] = value;
    }
}
// CUDA kernel for bias addition
__global__ void bias_forward_kernel(float *x, float *bias, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * size) {
        int i = idx % size;
        x[idx] += bias[i];
    }
}
__global__ void relu_kernel(float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void forwardLayerKernel(float *input, float *weights, float *bias, float *output, int input_size, int output_size, int batch_size, bool use_relu) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && output_idx < output_size) {
        float value = 0;
        for (int i = 0; i < input_size; ++i) {
            value += input[batch_idx * input_size + i] * weights[i * output_size + output_idx];
        }
        value += bias[output_idx];
        if (use_relu) {
            value = fmaxf(0.0f, value);
        }
        output[batch_idx * output_size + output_idx] = value;
    }
}
__global__ void compute_gradients_kernel(float *input, float *delta, float *grad_weights, float *grad_bias, int batch_size, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size * output_size) {
      int i = idx / output_size;
      int j = idx % output_size;

      float grad_w = 0.0f;
      if (i == 0) grad_bias[j] =0.0f;
      for (int b = 0; b < batch_size; b++) {
          grad_w += input[b * input_size + i] * delta[b * output_size + j];
          if (i == 0) grad_bias[j] += delta[b * output_size + j];
      }
      grad_weights[idx] = grad_w;
      
    }
}
float compute_loss(float *output, int *labels, int batch_size)
{
  float total_loss = 0.0f;
  for (int b = 0; b < batch_size; b++) {
    if(labels[b]>0.0f)
      total_loss -= labels[b]*log(output[b* OUTPUT_SIZE + labels[b]]);
  }
  return total_loss / batch_size;
};

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



__global__ void update_weights_kernel(float *weights, float *grad_weights, float *bias, float *grad_bias, int output_size, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_size * output_size) {
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
    }

    if (idx < output_size) {
        bias[idx] -= LEARNING_RATE * grad_bias[idx];
    }
}

int checkPredictions(float *output, int *labels, int batch_size, int output_size) {
    int correct = 0;
    for (int i = 0; i < batch_size; i++) {              
        int predicted = 0;
        // Tìm lớp có xác suất cao nhất
        for (int j = 1; j < output_size; j++) {
            if (output[i * output_size + j] > output[i * output_size + predicted]) {
                predicted = j;
            }
        }
        if (predicted == labels[i]) {
            correct++;
        }
    }
    return correct;
}

// Hàm tính toán gradient tại lớp đầu ra
void compute_output_gradient(float* grad_output, float* output, int* labels, int batch_size, int output_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < output_size; i++) {
            grad_output[b * output_size + i] = output[b * output_size + i] - (i == labels[b] ? 1.0f : 0.0f);
        }
    }
}
// Hàm tính toán gradient tại lớp đầu ra
__global__ void compute_output_gradient_kernel(float* grad_output, float* output, int* labels, int batch_size, int output_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b < batch_size){ 
        for (int i = 0; i < output_size; i++) {
            grad_output[b * output_size + i] = output[b * output_size + i] - (i == labels[b] ? 1.0f : 0.0f);
        }
    }
}

void train(NeuralNetwork *nn, float *X_train, int *y_train, int version) {
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
    // Define CUDA kernel dimensions
    dim3 blockDim(32, 32);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;
        float total_layer1_time = 0.0f, total_layer2_time = 0.0f, total_output_time = 0.0f;

        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;
            if(version==0)
            {
              // Forward pass for layer 1
              dim3 gridDim((HIDDEN1_SIZE + blockDim.x - 1) / blockDim.x, (BATCH_SIZE + blockDim.y - 1) / blockDim.y);
              GpuTimer timer1;
              timer1.Start();
              matrix_multiplication_kernel1<<<gridDim, blockDim>>>(d_X_train + start_idx * INPUT_SIZE, nn->weights_input_hidden1, d_hidden1, BATCH_SIZE, INPUT_SIZE, HIDDEN1_SIZE);
              bias_forward_kernel<<<(BATCH_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(d_hidden1, nn->bias_hidden1, BATCH_SIZE, HIDDEN1_SIZE);
              relu_kernel<<<(BATCH_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(d_hidden1, BATCH_SIZE * HIDDEN1_SIZE);
              timer1.Stop();
              total_layer1_time += timer1.Elapsed();
              
              // Forward pass for layer 2
              dim3 gridDim2((HIDDEN2_SIZE + blockDim.x - 1) / blockDim.x, (BATCH_SIZE + blockDim.y - 1) / blockDim.y);
              GpuTimer timer2;
              timer2.Start();
              matrix_multiplication_kernel1<<<gridDim2, blockDim>>>(d_hidden1, nn->weights_hidden1_hidden2, d_hidden2, BATCH_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE);
              bias_forward_kernel<<<(BATCH_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(d_hidden2, nn->bias_hidden2, BATCH_SIZE, HIDDEN2_SIZE);
              relu_kernel<<<(BATCH_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(d_hidden2, BATCH_SIZE * HIDDEN2_SIZE);
              timer2.Stop();
              total_layer2_time += timer2.Elapsed();

              // Forward pass for output layer
              dim3 gridDim3((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (BATCH_SIZE + blockDim.y - 1) / blockDim.y);

              GpuTimer timer3;
              timer3.Start();
              matrix_multiplication_kernel1<<<gridDim3, blockDim>>>(d_hidden2, nn->weights_hidden2_output, d_output, BATCH_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
              bias_forward_kernel<<<(BATCH_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(d_output, nn->bias_output, BATCH_SIZE, OUTPUT_SIZE);

              // Apply softmax on the output layer
              softmax_kernel<<<(BATCH_SIZE + 31) / 32, 32>>>(d_output, BATCH_SIZE, OUTPUT_SIZE);
              timer3.Stop();
              total_output_time += timer3.Elapsed();
            }
            else if(version==1)
            {
                // Forward pass for layer 1
              dim3 gridDim((HIDDEN1_SIZE + blockDim.x - 1) / blockDim.x, (BATCH_SIZE + blockDim.y - 1) / blockDim.y);
              GpuTimer timer1;
              timer1.Start();
              matrix_multiplication_kernel2<<<gridDim, blockDim>>>(d_X_train + start_idx * INPUT_SIZE, nn->weights_input_hidden1, d_hidden1, BATCH_SIZE, INPUT_SIZE, HIDDEN1_SIZE);
              bias_forward_kernel<<<(BATCH_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(d_hidden1, nn->bias_hidden1, BATCH_SIZE, HIDDEN1_SIZE);
              relu_kernel<<<(BATCH_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(d_hidden1, BATCH_SIZE * HIDDEN1_SIZE);
              timer1.Stop();
              total_layer1_time += timer1.Elapsed();
              
              // Forward pass for layer 2
              dim3 gridDim2((HIDDEN2_SIZE + blockDim.x - 1) / blockDim.x, (BATCH_SIZE + blockDim.y - 1) / blockDim.y);
              GpuTimer timer2;
              timer2.Start();
              matrix_multiplication_kernel2<<<gridDim2, blockDim>>>(d_hidden1, nn->weights_hidden1_hidden2, d_hidden2, BATCH_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE);
              bias_forward_kernel<<<(BATCH_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(d_hidden2, nn->bias_hidden2, BATCH_SIZE, HIDDEN2_SIZE);
              relu_kernel<<<(BATCH_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(d_hidden2, BATCH_SIZE * HIDDEN2_SIZE);
              timer2.Stop();
              total_layer2_time += timer2.Elapsed();

              // Forward pass for output layer
              dim3 gridDim3((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (BATCH_SIZE + blockDim.y - 1) / blockDim.y);

              GpuTimer timer3;
              timer3.Start();
              matrix_multiplication_kernel2<<<gridDim3, blockDim>>>(d_hidden2, nn->weights_hidden2_output, d_output, BATCH_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
              bias_forward_kernel<<<(BATCH_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(d_output, nn->bias_output, BATCH_SIZE, OUTPUT_SIZE);

              // Apply softmax on the output layer
              softmax_kernel<<<(BATCH_SIZE + 31) / 32, 32>>>(d_output, BATCH_SIZE, OUTPUT_SIZE);
              timer3.Stop();
              total_output_time += timer3.Elapsed();
            }
            else{
              // Forward pass for layer 1
               dim3 gridDim((HIDDEN1_SIZE + blockDim.x - 1) / blockDim.x, (BATCH_SIZE + blockDim.y - 1) / blockDim.y);
              GpuTimer timer1;
              timer1.Start();
              forwardLayerKernel<<<gridDim, blockDim>>>(d_X_train + start_idx * INPUT_SIZE, nn->weights_input_hidden1, nn->bias_hidden1, d_hidden1, INPUT_SIZE, HIDDEN1_SIZE, BATCH_SIZE, true);
              timer1.Stop();
              total_layer1_time += timer1.Elapsed();

              // Forward pass for layer 2
              dim3 gridDim2((HIDDEN2_SIZE + blockDim.x - 1) / blockDim.x, (BATCH_SIZE + blockDim.y - 1) / blockDim.y);
              GpuTimer timer2;
              timer2.Start();
              forwardLayerKernel<<<gridDim2, blockDim>>>(d_hidden1, nn->weights_hidden1_hidden2, nn->bias_hidden2, d_hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE, BATCH_SIZE, true);
              timer2.Stop();
              total_layer2_time += timer2.Elapsed();

              // Forward pass for output layer
              dim3 gridDim3((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (BATCH_SIZE + blockDim.y - 1) / blockDim.y);
              GpuTimer timer3;
              timer3.Start();
              forwardLayerKernel<<<gridDim3, blockDim>>>(d_hidden2, nn->weights_hidden2_output, nn->bias_output, d_output, HIDDEN2_SIZE, OUTPUT_SIZE, BATCH_SIZE, false);
              softmax_kernel<<<(BATCH_SIZE + 31) / 32, 32>>>(d_output, BATCH_SIZE, OUTPUT_SIZE);
              timer3.Stop();
              total_output_time += timer3.Elapsed();
            }
            
            CHECK(cudaMemcpy(output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            float loss = compute_loss(output, &y_train[start_idx], BATCH_SIZE);
            total_loss += loss;
            // Check prediction accuracy
            correct += checkPredictions(output, &y_train[start_idx], BATCH_SIZE, OUTPUT_SIZE);

            // Backpropagation
            CHECK(cudaMemset(nn->grad_weights_input_hidden1, 0, HIDDEN1_SIZE * INPUT_SIZE * sizeof(float)));
            CHECK(cudaMemset(nn->grad_weights_hidden1_hidden2, 0, HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float)));
            CHECK(cudaMemset(nn->grad_weights_hidden2_output, 0, OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float)));
            CHECK(cudaMemset(nn->grad_bias_hidden1, 0, HIDDEN1_SIZE * sizeof(float)));
            CHECK(cudaMemset(nn->grad_bias_hidden2, 0, HIDDEN2_SIZE * sizeof(float)));
            CHECK(cudaMemset(nn->grad_bias_output, 0, OUTPUT_SIZE * sizeof(float)));

            //GpuTimer backprop_timer3;
            //backprop_timer3.Start();
            // Compute gradient at output layer
            compute_output_gradient_kernel<<<(BATCH_SIZE + 31) / 32, 32>>>(d_del_output, d_output, d_y_train + start_idx, BATCH_SIZE, OUTPUT_SIZE);

            compute_gradients_kernel<<<(HIDDEN2_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(d_hidden2, d_del_output, nn->grad_weights_hidden2_output, nn->grad_bias_output, BATCH_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
            //backprop_timer3.Stop();
            //total_output_time += backprop_timer3.Elapsed();

            //GpuTimer backprop_timer2;
            //backprop_timer2.Start();
            compute_delta_relu_kernel<<<(BATCH_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(d_del_output, nn->weights_hidden2_output, d_hidden2, d_d_ReLU_out2, BATCH_SIZE, OUTPUT_SIZE, HIDDEN2_SIZE);
            compute_gradients_kernel<<<(HIDDEN1_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(d_hidden1, d_d_ReLU_out2, nn->grad_weights_hidden1_hidden2, nn->grad_bias_hidden2, BATCH_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE);
            //backprop_timer2.Stop();
            //total_layer2_time += backprop_timer2.Elapsed();

            //GpuTimer backprop_timer1;
            //backprop_timer1.Start();
            compute_delta_relu_kernel<<<(BATCH_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(d_d_ReLU_out2, nn->weights_hidden1_hidden2, d_hidden1, d_d_ReLU_out1, BATCH_SIZE, HIDDEN2_SIZE, HIDDEN1_SIZE);
            compute_gradients_kernel<<<(INPUT_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(d_X_train + start_idx * INPUT_SIZE, d_d_ReLU_out1, nn->grad_weights_input_hidden1, nn->grad_bias_hidden1, BATCH_SIZE, INPUT_SIZE, HIDDEN1_SIZE);
            //backprop_timer1.Stop();
            //total_layer1_time += backprop_timer1.Elapsed();  
            //
            update_weights_kernel<<<(INPUT_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(nn->weights_input_hidden1, nn->grad_weights_input_hidden1, nn->bias_hidden1, nn->grad_bias_hidden1, HIDDEN1_SIZE, INPUT_SIZE);
            update_weights_kernel<<<(HIDDEN1_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(nn->weights_hidden1_hidden2, nn->grad_weights_hidden1_hidden2, nn->bias_hidden2, nn->grad_bias_hidden2, HIDDEN2_SIZE, HIDDEN1_SIZE);
            update_weights_kernel<<<(HIDDEN2_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(nn->weights_hidden2_output, nn->grad_weights_hidden2_output, nn->bias_output, nn->grad_bias_output, OUTPUT_SIZE, HIDDEN2_SIZE);          
        }

        printf("Epoch %d/%d completed, Loss: %.4f, Accuracy: %.2f%%\n", epoch + 1, EPOCHS, total_loss / num_batches, 100.0f * correct / TRAIN_DATA_SIZE);
        
        printf("    Layer 1 time: %.6f s", total_layer1_time/1000);
        printf("    Layer 2 time: %.6f s", total_layer2_time/1000);
        printf("    Output layer time: %.6f s\n", total_output_time/1000);
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
    // Define CUDA kernel dimensions
    dim3 blockDim(32, 32);
    for (int batch = 0; batch < num_batches; batch++) {
        int start_idx = batch * BATCH_SIZE;

        // Forward pass for layer 1
       dim3 gridDim((HIDDEN1_SIZE + blockDim.x - 1) / blockDim.x, (BATCH_SIZE + blockDim.y - 1) / blockDim.y);

      forwardLayerKernel<<<gridDim, blockDim>>>(d_X_train + start_idx * INPUT_SIZE, nn->weights_input_hidden1, nn->bias_hidden1, d_hidden1, INPUT_SIZE, HIDDEN1_SIZE, BATCH_SIZE, true);

      // Forward pass for layer 2
      dim3 gridDim2((HIDDEN2_SIZE + blockDim.x - 1) / blockDim.x, (BATCH_SIZE + blockDim.y - 1) / blockDim.y);

      forwardLayerKernel<<<gridDim2, blockDim>>>(d_hidden1, nn->weights_hidden1_hidden2, nn->bias_hidden2, d_hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE, BATCH_SIZE, true);

      // Forward pass for output layer
      dim3 gridDim3((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (BATCH_SIZE + blockDim.y - 1) / blockDim.y);

      forwardLayerKernel<<<gridDim3, blockDim>>>(d_hidden2, nn->weights_hidden2_output, nn->bias_output, d_output, HIDDEN2_SIZE, OUTPUT_SIZE, BATCH_SIZE, false);
      softmax_kernel<<<(BATCH_SIZE + 31) / 32, 32>>>(d_output, BATCH_SIZE, OUTPUT_SIZE);

float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
CHECK(cudaMemcpy(output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        // Kiểm tra kết quả dự đoán
        correct += checkPredictions(output, &y_test[start_idx], BATCH_SIZE, OUTPUT_SIZE);

        free(output);
    }

    float accuracy = 100.0f * correct / TEST_DATA_SIZE;
    printf("Test Accuracy: %.2f%%\n", accuracy);

    CHECK(cudaFree(d_X_test));
    CHECK(cudaFree(d_hidden1));
    CHECK(cudaFree(d_hidden2));
    CHECK(cudaFree(d_output));
}
void freeNN(NeuralNetwork* nn)
{
    CHECK(cudaFree(nn->weights_input_hidden1));
    CHECK(cudaFree(nn->weights_hidden1_hidden2));
    CHECK(cudaFree(nn->weights_hidden2_output));
    CHECK(cudaFree(nn->bias_hidden1));
    CHECK(cudaFree(nn->bias_hidden2));
    CHECK(cudaFree(nn->bias_output));
    CHECK(cudaFree(nn->grad_weights_input_hidden1));
    CHECK(cudaFree(nn->grad_weights_hidden1_hidden2));
    CHECK(cudaFree(nn->grad_weights_hidden2_output));
    CHECK(cudaFree(nn->grad_bias_hidden1));
    CHECK(cudaFree(nn->grad_bias_hidden2));
    CHECK(cudaFree(nn->grad_bias_output));
}
int main(int argc, char **argv) {
    srand(time(NULL));

    

    float *X_train = (float *)malloc(TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int *)malloc(TRAIN_DATA_SIZE * sizeof(int));

    float *X_test =  (float *)malloc(TEST_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = (int *)malloc(TEST_DATA_SIZE * sizeof(int));

    load_data("x_train.bin", X_train, TRAIN_DATA_SIZE * INPUT_SIZE);
    load_labels("y_train.bin", y_train, TRAIN_DATA_SIZE);
    load_data("x_test.bin", X_test, TEST_DATA_SIZE * INPUT_SIZE);
    load_labels("y_test.bin", y_test, TEST_DATA_SIZE);
    NeuralNetwork nn;
    printf("\nBasic GPU kernel \n");
    initialize_neural_network(&nn);
    // Training
    train(&nn, X_train, y_train, 0);

    // Testing
    test(&nn, X_test, y_test);  
    freeNN(&nn);
    printf("==============================\n");
    
    printf("TiledMatrixMultiplication sử dụng shared memory \n");
    initialize_neural_network(&nn);
    // Training
    train(&nn, X_train, y_train, 1);
    // Testing
    test(&nn, X_test, y_test);  
    freeNN(&nn);
    printf("==============================\n");
    
    printf("Kernel fusion for add bias, relu and matrix-multiplication\n");
    initialize_neural_network(&nn);
    // Training
    train(&nn, X_train, y_train, 2);
    // Testing
    test(&nn, X_test, y_test);  
    freeNN(&nn);
    

    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}
