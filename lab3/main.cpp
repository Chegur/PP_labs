#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <mpi.h>
#include <format>

using namespace std;

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const vector<int> SIZES = { 200, 400, 800, 1200, 1600, 2000 };
    const bool GENERATE_IF_MISSING = true;

    if (rank == 0) {
        cout << "N,Procs,Time_sec,GFLOPS,Output_File\n";
        cout.flush();
    }

    for (int n : SIZES) {
        if (n % size != 0) {
            if (rank == 0) {
                cerr << "[SKIP] N=" << n << " not divisible by MPI processes=" << size << "\n";
            }
            continue;
        }

        string FILE_A = format("matrix_a{}.txt", n);
        string FILE_B = format("matrix_b{}.txt", n);
        string FILE_C = format("matrix_c{}.txt", n);

        int local_n = n / size;
        vector<double> local_A(local_n * n);
        vector<double> local_B(n * n);
        vector<double> local_C(local_n * n, 0.0);
        vector<double> A_full;

        if (rank == 0) {
            A_full.resize(n * n);
            bool files_ok = !GENERATE_IF_MISSING &&
                read_matrix(FILE_A, A_full, n) &&
                read_matrix(FILE_B, local_B, n);

            if (!files_ok) {
                if (rank == 0) {
                    cout << "[Rank 0] Generating " << n << "x" << n << " matrices...\n";
                    cout.flush();
                }
                generate_matrix(A_full, n);
                generate_matrix(local_B, n);
            }
        }

        MPI_Scatter(rank == 0 ? A_full.data() : nullptr, local_n * n, MPI_DOUBLE,
            local_A.data(), local_n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(local_B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            A_full.clear(); A_full.shrink_to_fit();
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        for (int i = 0; i < local_n; ++i) {
            for (int k = 0; k < n; ++k) {
                double r = local_A[i * n + k];
                for (int j = 0; j < n; ++j) {
                    local_C[i * n + j] += r * local_B[k * n + j];
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t_end = MPI_Wtime();
        double exec_time = t_end - t_start;

        vector<double> C_full;
        if (rank == 0) C_full.resize(n * n);
        MPI_Gather(local_C.data(), local_n * n, MPI_DOUBLE,
            C_full.data(), local_n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            write_matrix(FILE_C, C_full, n);
            double flops = 2.0 * n * n * n;
            double gflops = (flops / exec_time) / 1e9;

            cout << n << ","
                << size << ","
                << fixed << setprecision(6) << exec_time << ","
                << setprecision(3) << gflops << ","
                << FILE_C << "\n";
            cout.flush();
        }

        local_A.clear(); local_B.clear(); local_C.clear();
        if (rank == 0) { C_full.clear(); A_full.clear(); }
    }

    MPI_Finalize();
    return 0;
}
