/**********************************************************************************************************************
 * 
 * - Program task: explore different matrix multiplication approaches and compare their respective performance for 
 *                 random floating-point value square matrices of different dimensions. 
 * 
 * - Data structure: Matrices are stored as flat 1D arrays for better cache locality, when compared to the naive 
 *                   array-of-pointers implementation. 
 * 
 * 
 ***********************************************************************************************************************/

// Libraries
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include <stdint.h>

// Helper function to generate a random floating-point value within a given range
// ARGUMENTS: 
// - double lo --> lower bound for random number generation
// - double hi --> upper bound for random number generation
// RETURN TYPE: double --> random floating-point value within the given range
static inline double random_float(double lo, double hi) {
    return lo + (hi - lo) * ((double)arc4random() / ((double)UINT32_MAX + 1.0));
}

// Helper function to randomize matrix values
// ARGUMENTS: 
// - double* A --> pointer to square matrix stored as a flat 1D array
// - size_t n --> dimension of the square matrix 
// - double min --> lower bound for random number generation
// - double max --> upper bound for random number generation
// RETURN TYPE: none
static inline void randomize(double* A, size_t n, float min, float max)
{
    for(size_t i = 0; i < n * n; i++) A[i] = random_float(min, max);
}

// Helper function to access a particular index of a matrix stored as a flat 1D array
// ARGUMENTS: 
// - size_t n --> square matrix dimension
// - size_t i --> row index of the entry
// - size_t j --> column index of the entry
// RETURN TYPE: size_t --> array index corresponding to the specified matrix position
static inline size_t idx(size_t n, size_t i, size_t j) {return n * i + j;}

// Helper function to print the matrix contents 
// ARGUMENTS: 
// - double* A --> square matrix stored as a flat 1D array
// - size_t n --> dimension of the square matrix
// RETURN TYPE: none
static inline void print_mat(double* A, size_t n)
{
    for(size_t i = 0; i < n; i++)
    {
        for(size_t j = 0; j < n; j++)
        {
            // Print values with 6 decimal places of precision
            printf("%10.6lf ", A[idx(n, i, j)]);
        }
        printf("\n");
    }
    printf("\n");
}

// Helper function to initialize a flat 1D array with zeroes
// ARGUMENTS: 
// - size_t n --> dimension of the square matrix
// RETURN TYPE: double pointer to a zero matrix, represented as a flat 1D array
static inline double* make_mat(size_t n) {return calloc(n*n, sizeof(double));}

// Helper function to deallocate a matrix
// ARGUMENTS:
// - double* A --> matrix pointer 
// RETURN TYPE: none
static inline void kill_mat(double* A) {free(A);}

// Helper function to read a matrix in from user
// ARGUMENTS: 
// - double* A --> pointer to an initialized matrix
// - size_t n --> dimension of the square matrix 
// RETURN TYPE: none
static inline void read_mat(double* A, size_t n)
{
    for(size_t i = 0; i < n * n; i++) scanf("%lf", &A[i]);
}

// Multiply two matrices together using the naive approach of the standard definition
// ARGUMENTS: 
// - double* A --> pointer to an initialized matrix A
// - double* B --> pointer to an initialized matrix B
// - size_t n --> dimension of both square matrices
// RETURN TYPE: double* --> product of matrix multiplication C
static inline double* matmul_naive(double* A, double* B, size_t n)
{
    // Create a blank matrix
    double* C = make_mat(n); 

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < n; k++) {
                C[idx(n, i, j)] += A[idx(n, i, k)] * B[idx(n, k, j)];
            }
        }
    }

    return C;
}

// Function to transpose a given matrix into another blank matrix
// (Original matrix will not be affected)
// ARGUMENTS: 
// - double* A --> matrix to transpose
// - size_t n --> dimension of the square matrix
// RETURN TYPE: double* --> transposed matrix T
double* transpose(double* A, size_t n) {
    double *T = make_mat(n);

    for (size_t i = 0; i < n; i++) 
    {
        for (size_t j = 0; j < n; j++) 
        {
            T[idx(n, j, i)] = A[idx(n, i, j)];
        }
    }

    return T;
}

// Multiply two matrices together using the improved approach of the standard definition
// Matrix B has been transposed for better cache-locality 
// ARGUMENTS: 
// - double* A --> pointer to an initialized matrix A
// - double* T --> pointer to an initialized matrix B
// - size_t n --> dimension of both square matrices
// RETURN TYPE: double* --> product of matrix multiplication C
static inline double* matmul_trans(double* A, double* T, size_t n)
{
    // Create a blank matrix
    double* C = make_mat(n); 

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            const double *a_row  = &A[i * n];
            const double *t_row = &T[j * n];
            for (size_t k = 0; k < n; k++)
            { 
                C[idx(n, i, j)] += a_row[k] * t_row[k];
            }
        }
    }

    return C;
}

/******************************************* BLOCKING APPROACH ********************************************/

#define BLOCK_THRESHOLD (1 << 3)  // switch to naive below this size

/* mat_add, mat_sub, and mat_mul_naive are unchanged from before */

static void mat_add(const double *A, size_t strideA,
                     const double *B, size_t strideB,
                     double *C, size_t strideC, size_t n) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            C[i*strideC + j] = A[i*strideA + j] + B[i*strideB + j];
}

static void mat_mul_naive(const double *A, size_t strideA,
                           const double *B, size_t strideB,
                           double *C, size_t strideC, size_t n) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++)
                sum += A[i*strideA + k] * B[k*strideB + j];
            C[i*strideC + j] = sum;
        }
}

/*
 * Recursive blocking engine: partitions A, B, C into quadrants and
 * computes each output quadrant as the sum of two sub-products
 * (the standard 8-multiply/4-add definition), recursing until the
 * submatrix size drops to BLOCK_THRESHOLD, then uses the naive kernel.
 */
static void block_rec(const double *A, size_t strideA,
                       const double *B, size_t strideB,
                       double *C, size_t strideC, size_t n) {
    if (n <= BLOCK_THRESHOLD) {
        mat_mul_naive(A, strideA, B, strideB, C, strideC, n);
        return;
    }

    size_t m = n / 2;

    const double *A11 = A,              *A12 = A + m;
    const double *A21 = A + m*strideA,  *A22 = A + m*strideA + m;
    const double *B11 = B,              *B12 = B + m;
    const double *B21 = B + m*strideB,  *B22 = B + m*strideB + m;
    double       *C11 = C,              *C12 = C + m;
    double       *C21 = C + m*strideC,  *C22 = C + m*strideC + m;

    double *T = malloc(m * m * sizeof(double));  // scratch for the 2nd product per quadrant

    /* C11 = A11*B11 + A12*B21 */
    block_rec(A11, strideA, B11, strideB, C11, strideC, m);
    block_rec(A12, strideA, B21, strideB, T, m, m);
    mat_add(C11, strideC, T, m, C11, strideC, m);

    /* C12 = A11*B12 + A12*B22 */
    block_rec(A11, strideA, B12, strideB, C12, strideC, m);
    block_rec(A12, strideA, B22, strideB, T, m, m);
    mat_add(C12, strideC, T, m, C12, strideC, m);

    /* C21 = A21*B11 + A22*B21 */
    block_rec(A21, strideA, B11, strideB, C21, strideC, m);
    block_rec(A22, strideA, B21, strideB, T, m, m);
    mat_add(C21, strideC, T, m, C21, strideC, m);

    /* C22 = A21*B12 + A22*B22 */
    block_rec(A21, strideA, B12, strideB, C22, strideC, m);
    block_rec(A22, strideA, B22, strideB, T, m, m);
    mat_add(C22, strideC, T, m, C22, strideC, m);

    free(T);
}

/*
 * Public entry point: allocates a fresh, tightly-packed n x n matrix C
 * (stride n), fills it with A * B via recursive blocking, and returns it.
 * Caller is responsible for freeing the returned pointer.
 */
double *block_multiply(const double *A, size_t strideA,
                        const double *B, size_t strideB,
                        size_t n) {
    double *C = malloc(n * n * sizeof(double));
    if (!C) return NULL;

    block_rec(A, strideA, B, strideB, C, n, n);

    return C;
}

/* Thin wrapper: A and B are standalone n x n matrices (stride == n). */
double* matmul_block(double *A, double *B, size_t n) {
    return block_multiply(A, n, B, n, n);
}

/******************************************* STRASSEN'S APPROACH ********************************************/

#define STRASSEN_THRESHOLD (1 << 3) // switch to naive below this size

static inline void mat_sub(const double *A, size_t strideA,
                     const double *B, size_t strideB,
                     double *C, size_t strideC, size_t n) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            C[i*strideC + j] = A[i*strideA + j] - B[i*strideB + j];
}

/* Internal recursive engine: still writes into a caller-provided C,
   since quadrant views of C need their own stride during recursion. */
static inline void strassen_rec(const double *A, size_t strideA,
                          const double *B, size_t strideB,
                          double *C, size_t strideC, size_t n) {
    if (n <= STRASSEN_THRESHOLD) {
        mat_mul_naive(A, strideA, B, strideB, C, strideC, n);
        return;
    }

    size_t m = n / 2;

    const double *A11 = A,              *A12 = A + m;
    const double *A21 = A + m*strideA,  *A22 = A + m*strideA + m;
    const double *B11 = B,              *B12 = B + m;
    const double *B21 = B + m*strideB,  *B22 = B + m*strideB + m;
    double       *C11 = C,              *C12 = C + m;
    double       *C21 = C + m*strideC,  *C22 = C + m*strideC + m;

    double *M1 = malloc(m*m*sizeof(double));
    double *M2 = malloc(m*m*sizeof(double));
    double *M3 = malloc(m*m*sizeof(double));
    double *M4 = malloc(m*m*sizeof(double));
    double *M5 = malloc(m*m*sizeof(double));
    double *M6 = malloc(m*m*sizeof(double));
    double *M7 = malloc(m*m*sizeof(double));
    double *T1 = malloc(m*m*sizeof(double));
    double *T2 = malloc(m*m*sizeof(double));

    mat_add(A11, strideA, A22, strideA, T1, m, m);
    mat_add(B11, strideB, B22, strideB, T2, m, m);
    strassen_rec(T1, m, T2, m, M1, m, m);                 // M1 = (A11+A22)(B11+B22)

    mat_add(A21, strideA, A22, strideA, T1, m, m);
    strassen_rec(T1, m, B11, strideB, M2, m, m);           // M2 = (A21+A22)B11

    mat_sub(B12, strideB, B22, strideB, T2, m, m);
    strassen_rec(A11, strideA, T2, m, M3, m, m);           // M3 = A11(B12-B22)

    mat_sub(B21, strideB, B11, strideB, T2, m, m);
    strassen_rec(A22, strideA, T2, m, M4, m, m);           // M4 = A22(B21-B11)

    mat_add(A11, strideA, A12, strideA, T1, m, m);
    strassen_rec(T1, m, B22, strideB, M5, m, m);           // M5 = (A11+A12)B22

    mat_sub(A21, strideA, A11, strideA, T1, m, m);
    mat_add(B11, strideB, B12, strideB, T2, m, m);
    strassen_rec(T1, m, T2, m, M6, m, m);                  // M6 = (A21-A11)(B11+B12)

    mat_sub(A12, strideA, A22, strideA, T1, m, m);
    mat_add(B21, strideB, B22, strideB, T2, m, m);
    strassen_rec(T1, m, T2, m, M7, m, m);                  // M7 = (A12-A22)(B21+B22)

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < m; j++) {
            C11[i*strideC+j] = M1[i*m+j] + M4[i*m+j] - M5[i*m+j] + M7[i*m+j];
            C12[i*strideC+j] = M3[i*m+j] + M5[i*m+j];
            C21[i*strideC+j] = M2[i*m+j] + M4[i*m+j];
            C22[i*strideC+j] = M1[i*m+j] - M2[i*m+j] + M3[i*m+j] + M6[i*m+j];
        }

    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(T1); free(T2);
}

/*
 * Public entry point: allocates a fresh, tightly-packed n x n matrix C
 * (stride n), fills it with A * B via Strassen's algorithm, and returns it.
 * Caller is responsible for freeing the returned pointer.
 */
static inline double *strassen(const double *A, size_t strideA,
                  const double *B, size_t strideB,
                  size_t n) {
    double *C = malloc(n * n * sizeof(double));
    if (!C) return NULL;

    strassen_rec(A, strideA, B, strideB, C, n, n);

    return C;
}

/*
 * Thin wrapper: A and B are standalone n x n matrices (stride == n).
 */
static inline double* matmul_strass(double *A, double *B, size_t n) {
    return strassen(A, n, B, n, n);
}

/******************************************* TIMING ALGORITHM ********************************************/

// Function pointer type matching the signature of the algorithm to be timed
typedef double* (*func)(double* A, double* B, size_t n);

// Times `num_runs` calls to `algo`, returns the average elapsed time in seconds.
// The result pointer from the final run is written to *out_result (can be NULL to ignore).
static inline double time_algorithm(func algo, double* A, double* B, size_t n,
                       size_t num_runs, size_t warm_up) {
    struct timespec start, end;
    double total_elapsed = 0.0;

    for(size_t w = 0; w < warm_up; w++) kill_mat(algo(A, B, n));

    for (size_t run = 0; run < num_runs; run++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        kill_mat(algo(A, B, n));
        clock_gettime(CLOCK_MONOTONIC, &end);

        total_elapsed += (end.tv_sec - start.tv_sec)  + (end.tv_nsec - start.tv_nsec) / 1e9;
    }

    return total_elapsed / num_runs;
}

void run_trials(func algo)
{
    size_t dim = 1 << 4; 
    double* A = make_mat(dim); 
    double* B = make_mat(dim);
    randomize(A, dim, -100, 100);
    randomize(B, dim, -100, 100);

    printf("Avg runtime at input size %lu: %.9lf sec.\n", dim, time_algorithm(algo, A, B, dim, 500, 250));

    kill_mat(A);
    kill_mat(B);

    dim = 1 << 7;
    A = make_mat(dim); 
    B = make_mat(dim);
    randomize(A, dim, -100, 100);
    randomize(B, dim, -100, 100);

    printf("Avg runtime at input size %lu: %.9lf sec.\n", dim, time_algorithm(algo, A, B, dim, 100, 50));

    kill_mat(A);
    kill_mat(B);

    dim = 1 << 10; 
    A = make_mat(dim); 
    B = make_mat(dim);
    randomize(A, dim, -100, 100);
    randomize(B, dim, -100, 100);

    printf("Avg runtime at input size %lu: %.9lf sec.\n", dim, time_algorithm(algo, A, B, dim, 20, 10));
    kill_mat(A);
    kill_mat(B);

}

int main()
{
    printf("\nNaive approach of the standard matrix multiplication algorithm: \n");
    run_trials(matmul_naive);

    printf("\nImproved approach of the standard algorithm with transposition: \n");
    run_trials(matmul_trans);

    printf("\nRecursive blocking approach of the standard definition: \n");
    run_trials(matmul_block);

    printf("\nStrassen's recursive approach: \n");
    run_trials(matmul_strass);

    return 0;
}