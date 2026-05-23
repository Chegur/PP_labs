#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>  

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                 << " - " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

bool read_matrix(const string& filename, vector<double>& matrix, int& n) {
    ifstream file(filename);
    if (!file.is_open()) return false;
    file >> n;
    matrix.resize(n * n);
    for (int i = 0; i < n * n; ++i) {
        if (!(file >> matrix[i])) return false;
    }
    return true;
}

void generate_matrix(vector<double>& matrix, int n) {
    matrix.resize(n * n);
    srand(42 + n);
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = static_cast<double>(rand() % 1000) / 100.0;
    }
}

bool write_matrix(const string& filename, const vector<double>& matrix, int n) {
    ofstream file(filename);
    if (!file.is_open()) return false;
    file << n << '\n';
    file << fixed << setprecision(6);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << matrix[i * n + j] << " ";
        }
        file << '\n';
    }
    return true;
}

__global__ void matMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "[ERROR] CUDA-capable GPU not found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "Using GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")\n";

    const vector<int> SIZES = { 200, 400, 600, 800, 1200, 1600, 2000 };
    const vector<pair<int,int>> BLOCK_CONFIGS = { {8,8}, {16,16}, {32,32} };
    const bool GENERATE_IF_MISSING = true;
    const bool SAVE_RESULTS = true;  

    cout << "N,BlockSize,Time_sec,GFLOPS,Output_File\n";
    cout.flush();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int n : SIZES) {
        string FILE_A = "matrix_a" + to_string(n) + ".txt";
        string FILE_B = "matrix_b" + to_string(n) + ".txt";
        
        vector<double> A(n * n), B(n * n);
        
        bool files_ok = !GENERATE_IF_MISSING &&
            read_matrix(FILE_A, A, n) &&
            read_matrix(FILE_B, B, n);

        if (!files_ok) {
            cout << "[GPU] Generating " << n << "x" << n << " matrices...\n";
            generate_matrix(A, n);
            generate_matrix(B, n);
        }

        double *d_A, *d_B, *d_C;
        size_t bytes = n * n * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));
        
        A.clear(); A.shrink_to_fit();
        B.clear(); B.shrink_to_fit();

        for (auto [bx, by] : BLOCK_CONFIGS) {
            string block_tag = to_string(bx) + "x" + to_string(by);
            string FILE_C = "matrix_c" + to_string(n) + "_" + block_tag + ".txt";

            dim3 block(bx, by);
            dim3 grid((n + bx - 1) / bx, (n + by - 1) / by);

            CUDA_CHECK(cudaEventRecord(start));
            matMulKernel<<<grid, block>>>(d_A, d_B, d_C, n);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            double exec_time = ms / 1000.0;

            vector<double> C;
            if (SAVE_RESULTS) {
                C.resize(n * n);
                CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
                write_matrix(FILE_C, C, C, n);
            }

            double flops = 2.0 * n * n * n; 
            double gflops = (flops / exec_time) / 1e9;

            cout << n << ","
                 << block_tag << ","
                 << fixed << setprecision(6) << exec_time << ","
                 << setprecision(3) << gflops << ","
                 << (SAVE_RESULTS ? FILE_C : "N/A") << "\n";
            cout.flush();
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cout << "\n[DONE] All benchmarks completed.\n";
    return 0;
}