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
    nn->weights_input_hidden1 = (float *)malloc(HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
    nn->weights_hidden1_hidden2 = (float *)malloc(HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
    nn->weights_hidden2_output = (float *)malloc(OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));
    nn->bias_hidden1 =(float *) malloc(HIDDEN1_SIZE * sizeof(float));
    nn->bias_hidden2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    nn->bias_output = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    nn->grad_weights_input_hidden1 = (float *)malloc(HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
    nn->grad_weights_hidden1_hidden2 = (float *)malloc(HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
    nn->grad_weights_hidden2_output = (float *)malloc(OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));
    nn->grad_bias_hidden1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    nn->grad_bias_hidden2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    nn->grad_bias_output = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    initialize_weights(nn->weights_input_hidden1, HIDDEN1_SIZE * INPUT_SIZE);
    initialize_weights(nn->weights_hidden1_hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE);
    initialize_weights(nn->weights_hidden2_output, OUTPUT_SIZE * HIDDEN2_SIZE);
    initialize_bias(nn->bias_hidden1, HIDDEN1_SIZE);
    initialize_bias(nn->bias_hidden2, HIDDEN2_SIZE);
    initialize_bias(nn->bias_output, OUTPUT_SIZE);
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
      int idx = b*size;
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

void relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}


void matrix_multiplication(float *A, float *B, float *C, int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
          for (int col = 0; col < k; ++col) {
              float value = 0;
              for (int e = 0; e < n; ++e) {
                  value += A[row * n + e] * B[e * k + col];
              }
              C[row * k + col] = value;
          }
      }
}
// Add bias
void bias_forward(float *x, float *bias, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < size; i++) {
            x[b * size + i] += bias[i];
        }
    }
}
void forwardLayer(float *input, float *weights, float *bias, float *output, int input_size,
                   int output_size, int batch_size, bool use_relu) { 
  
  matrix_multiplication(input, weights, output, batch_size, input_size, output_size);    
  // Add bias1
  bias_forward(output, bias, batch_size, output_size);
  if (use_relu) {
      relu(output, batch_size*output_size); 
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
// Helper function to compute gradients for weights and biases
void compute_gradients(float *input, float *delta, float *grad_weights, 
                       float *grad_bias, int batch_size, int input_size, int output_size) {
    // Compute weight gradients
    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < output_size; j++) {
          for (int i = 0; i < input_size; i++) {
            grad_weights[i * output_size + j] += input[b * input_size + i] * delta[b * output_size + j];
          }
          grad_bias[j] += delta[b * output_size + j];
        }        
    }
}

void compute_delta_relu(float *relu_del_next, float *weights, float *input_layer, float *relu_del, 
                        int batch_size, int next_size, int input_size) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < input_size; j++) {
          int idx = i * input_size + j;
          relu_del[idx]  = 0.0f;
          // Tính tổng các delta từ lớp sau truyền về
          for (int k = 0; k < next_size; k++) {
              relu_del[idx] += relu_del_next[i * next_size + k] * weights[j* next_size + k];
          }
          //  tính đạo hàm của ReLU
          relu_del[idx] *= (input_layer[idx] > 0.0f);
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

void train(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *hidden1 = (float *)malloc(BATCH_SIZE * HIDDEN1_SIZE * sizeof(float));
    float *hidden2 = (float *)malloc(BATCH_SIZE * HIDDEN2_SIZE * sizeof(float));
    float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TRAIN_DATA_SIZE / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;
        double layer1_time = 0.0, layer2_time = 0.0, output_time = 0.0;
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;

            // Forward pass for layer 1
            clock_t start = clock();
            forwardLayer(&X_train[start_idx * INPUT_SIZE], nn->weights_input_hidden1, nn->bias_hidden1, hidden1, INPUT_SIZE, HIDDEN1_SIZE, BATCH_SIZE, true);
            clock_t end = clock();
            layer1_time += (double)(end - start) / CLOCKS_PER_SEC;

            // Forward pass for layer 2
            start = clock();
            forwardLayer(hidden1, nn->weights_hidden1_hidden2, nn->bias_hidden2, hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE, BATCH_SIZE, true);
            end = clock();
            layer2_time += (double)(end - start) / CLOCKS_PER_SEC;

            // Forward pass for output layer (no ReLU on output layer)
            start = clock();
            forwardLayer(hidden2, nn->weights_hidden2_output, nn->bias_output, output, HIDDEN2_SIZE, OUTPUT_SIZE, BATCH_SIZE, false);
            
            // Apply softmax to output
            softmax(output, BATCH_SIZE, OUTPUT_SIZE);
            end = clock();
            output_time += (double)(end - start) / CLOCKS_PER_SEC;
            
            float loss = compute_loss(output, &y_train[start_idx], BATCH_SIZE);
            total_loss += loss;

            // Check prediction accuracy
            correct += checkPredictions(output, &y_train[start_idx], BATCH_SIZE, OUTPUT_SIZE);

            // Backpropagation
            memset(nn->grad_weights_input_hidden1, 0, HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
            memset(nn->grad_weights_hidden1_hidden2, 0, HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
            memset(nn->grad_weights_hidden2_output, 0, OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));
            memset(nn->grad_bias_hidden1, 0, HIDDEN1_SIZE * sizeof(float));
            memset(nn->grad_bias_hidden2, 0, HIDDEN2_SIZE * sizeof(float));
            memset(nn->grad_bias_output, 0, OUTPUT_SIZE * sizeof(float));

            start = clock();
            float *grad_out = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
            // Compute gradient at output layer
            compute_output_gradient(grad_out, output, &y_train[start_idx], BATCH_SIZE, OUTPUT_SIZE);            
            // Compute gradients for weights and biases between Hidden2 -> Output
            compute_gradients(hidden2, grad_out, nn->grad_weights_hidden2_output, nn->grad_bias_output, BATCH_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
            end = clock();
            output_time += (double)(end - start) / CLOCKS_PER_SEC;
            
            start = clock();  
            
            float *d_ReLU_out2 = (float *)malloc(BATCH_SIZE * HIDDEN2_SIZE * sizeof(float));
            compute_delta_relu(grad_out, nn->weights_hidden2_output, hidden2, d_ReLU_out2, 
                               BATCH_SIZE, OUTPUT_SIZE, HIDDEN2_SIZE);

            // Compute gradients for weights and biases between Hidden1 -> Hidden2
            compute_gradients(hidden1, d_ReLU_out2, nn->grad_weights_hidden1_hidden2, nn->grad_bias_hidden2, BATCH_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE);
            
            end = clock(); 
            layer2_time += (double)(end - start) / CLOCKS_PER_SEC;
            
            start = clock();       
            float *d_ReLU_out1 = (float *)malloc(BATCH_SIZE * HIDDEN1_SIZE * sizeof(float));    
            compute_delta_relu(d_ReLU_out2, nn->weights_hidden1_hidden2, hidden1, d_ReLU_out1, 
                               BATCH_SIZE, HIDDEN2_SIZE, HIDDEN1_SIZE);
            
            // Compute gradients for weights and biases between Input -> Hidden1
            compute_gradients(&X_train[start_idx * INPUT_SIZE], d_ReLU_out1, nn->grad_weights_input_hidden1, nn->grad_bias_hidden1, BATCH_SIZE, INPUT_SIZE, HIDDEN1_SIZE);
            end = clock();
            layer1_time += (double)(end - start) / CLOCKS_PER_SEC;
            update_weights(nn->weights_hidden2_output, nn->grad_weights_hidden2_output, 
                            nn->bias_output, nn->grad_bias_output, OUTPUT_SIZE, HIDDEN2_SIZE);
            update_weights(nn->weights_hidden1_hidden2, nn->grad_weights_hidden1_hidden2, 
                              nn->bias_hidden2, nn->grad_bias_hidden2, HIDDEN2_SIZE, HIDDEN1_SIZE);
            update_weights(nn->weights_input_hidden1, nn->grad_weights_input_hidden1, nn->bias_hidden1, 
                                  nn->grad_bias_hidden1, HIDDEN1_SIZE, INPUT_SIZE);
            // Free temporary variables
            free(grad_out);
            free(d_ReLU_out2);
            free(d_ReLU_out1);            
        }

        // Print information after each epoch
        printf("Epoch %d/%d completed, Loss: %.4f, Accuracy: %.2f%%\n",
               epoch + 1, EPOCHS, total_loss / num_batches, 100.0f * correct / TRAIN_DATA_SIZE);

        printf("    Layer 1 time: %.6f seconds", layer1_time);
        printf("    Layer 2 time: %.6f seconds", layer2_time);
        printf("    Output layer time: %.6f seconds\n", output_time);
    }

    free(hidden1);
    free(hidden2);
    free(output);
}

void test(NeuralNetwork *nn, float *X_test, int *y_test) {
    float *hidden1 = (float *)malloc(BATCH_SIZE * HIDDEN1_SIZE * sizeof(float));
    float *hidden2 = (float *)malloc(BATCH_SIZE * HIDDEN2_SIZE * sizeof(float));
    float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    int num_batches = TEST_DATA_SIZE / BATCH_SIZE;
    int correct = 0;

    for (int batch = 0; batch < num_batches; batch++) {
        int start_idx = batch * BATCH_SIZE;

        // Forward pass cho layer 1
        forwardLayer(&X_test[start_idx * INPUT_SIZE], nn->weights_input_hidden1, nn->bias_hidden1, hidden1, INPUT_SIZE, HIDDEN1_SIZE, BATCH_SIZE, true);

        // Forward pass cho layer 2
        forwardLayer(hidden1, nn->weights_hidden1_hidden2, nn->bias_hidden2, hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE, BATCH_SIZE, true);

        // Forward pass cho layer output (không dùng ReLU)
        forwardLayer(hidden2, nn->weights_hidden2_output, nn->bias_output, output, HIDDEN2_SIZE, OUTPUT_SIZE, BATCH_SIZE, false);

        // Áp dụng softmax
        softmax(output, BATCH_SIZE, OUTPUT_SIZE);

        // Kiểm tra kết quả dự đoán
        correct += checkPredictions(output, &y_test[start_idx], BATCH_SIZE, OUTPUT_SIZE);
    }

    float accuracy = 100.0f * correct / TEST_DATA_SIZE;
    printf("Test Accuracy: %.2f%%\n", accuracy);

    free(hidden1);
    free(hidden2);
    free(output);
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

    free(nn.weights_input_hidden1);
    free(nn.weights_hidden1_hidden2);
    free(nn.weights_hidden2_output);
    free(nn.bias_hidden1);
    free(nn.bias_hidden2);
    free(nn.bias_output);
    free(nn.grad_weights_input_hidden1);
    free(nn.grad_weights_hidden1_hidden2);
    free(nn.grad_weights_hidden2_output);
    free(nn.grad_bias_hidden1);
    free(nn.grad_bias_hidden2);
    free(nn.grad_bias_output);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}
