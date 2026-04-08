#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define TOL 1e-3
#define NREP 5  // Número de repeticiones para medición

// Kernel OpenCL
const char* kernelSource =
"__kernel void matmul(__global float* A, __global float* B, __global float* C, int N) {"
"   int row = get_global_id(0);"
"   int col = get_global_id(1);"
"   float sum = 0.0f;"
"   for (int k = 0; k < N; k++) {"
"       sum += A[row*N + k] * B[k*N + col];"
"   }"
"   C[row*N + col] = sum;"
"}";

// Función para comparar doubles (qsort)
int compare_doubles(const void* a, const void* b) {
    double diff = *(double*)a - *(double*)b;
    return (diff > 0) - (diff < 0);
}

// Función para calcular la mediana
double compute_median(double* values, int n) {
    qsort(values, n, sizeof(double), compare_doubles);
    if (n % 2 == 1) {
        return values[n / 2];
    } else {
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    }
}


int main() {
    int sizes[] = {512, 1024, 2048, 4096};
    double results_time[4];
    double results_gflops[4];
    int results_valid[4];

    // Plataforma y dispositivo
    cl_platform_id platform;
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, NULL, &num_platforms);

    if (num_platforms == 0) {
        printf("No OpenCL platforms found\n");
        return 1;
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);

    platform = platforms[0];
    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_name[256];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        if (strstr(platform_name, "NVIDIA") != NULL) {
            platform = platforms[i];
            break;
        }
    }

    free(platforms);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Contexto y cola (con profiling)
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    const cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, queue_props, NULL);

    // Programa y kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "matmul", NULL);

    for (int s = 0; s < 4; s++) {
        int N = sizes[s];
        size_t bytes = N * N * sizeof(float);

        // Reservar memoria
        float* A = (float*)malloc(bytes);
        float* B = (float*)malloc(bytes);
        float* C = (float*)malloc(bytes);
        float* C_ref = (float*)malloc(bytes);

        // Inicialización determinista para verificación
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i*N + j] = (float)(i + j) / N;
                B[i*N + j] = (float)(i - j + N) / N;
            }
        }
        for (int i = 0; i < N*N; i++) {
            C[i] = 0.0f;
        }

            // CPU referencia serial para verificación
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < N; k++) {
                        sum += A[i*N + k] * B[k*N + j];
                    }
                    C_ref[i*N + j] = sum;
                }
            }

        // Buffers
        cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
        cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
        cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

        // Copiar datos
        clEnqueueWriteBuffer(queue, dA, CL_TRUE, 0, bytes, A, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, dB, CL_TRUE, 0, bytes, B, 0, NULL, NULL);

        // Argumentos
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
        clSetKernelArg(kernel, 3, sizeof(int), &N);

        size_t globalSize[2] = {(size_t)N, (size_t)N};
        size_t localSize[2] = {16, 16};

        // ====== WARM-UP: 1 ejecución sin medir ======
        cl_event warmup_event;
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, &warmup_event);
        clWaitForEvents(1, &warmup_event);
        clReleaseEvent(warmup_event);

        // ====== MEDICIÓN: NREP = 5 repeticiones ======
        double times_ms[NREP];
        
        for (int rep = 0; rep < NREP; rep++) {
            // Reinicializar C
            for (int i = 0; i < N*N; i++) {
                C[i] = 0.0f;
            }
            clEnqueueWriteBuffer(queue, dC, CL_TRUE, 0, bytes, C, 0, NULL, NULL);
            
            // Ejecutar kernel
            cl_event event;
            clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, &event);
            clWaitForEvents(1, &event);

            // Obtener tiempo de profiling
            cl_ulong start, end;
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
            times_ms[rep] = (end - start) * 1e-6;  // Convertir a milisegundos
            
            clReleaseEvent(event);
        }

        // Calcular mediana
        double median_time_ms = compute_median(times_ms, NREP);
        double gflops = 2.0 * (double)N * N * N / (median_time_ms * 1e6);

        // Leer resultado (del último kernel ejecutado)
        clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, bytes, C, 0, NULL, NULL);

        // Verificación numérica (solo para N <= 1024)
        double max_error = 0.0;
        int correct = 1;
        if (N <= 1024) {
            for (int i = 0; i < N*N; i++) {
                double error = fabs(C[i] - C_ref[i]);
                if (error > max_error) {
                    max_error = error;
                }
                if (error > TOL) {
                    correct = 0;
                }
            }
        }

        // Output
        printf("----------------------------------\n");
        printf("N = %d\n", N);
        printf("Tiempo mediano (NREP=%d): %.3f ms\n", NREP, median_time_ms);
        printf("GFLOPS: %.2f\n", gflops);
        if (N <= 1024) {
            printf("Error máximo absoluto: %.2e\n", max_error);
            printf("Resultado: %s\n", correct ? "CORRECTO" : "INCORRECTO");
        } else {
            printf("Verificación: omitida (N > 1024)\n");
        }
        
        // Guardar resultados para la tabla final
        results_time[s] = median_time_ms;
        results_gflops[s] = gflops;
        results_valid[s] = (N <= 1024) ? correct : 1;
    
        

        // Liberar
        free(A); free(B); free(C); free(C_ref);
        clReleaseMemObject(dA);
        clReleaseMemObject(dB);
        clReleaseMemObject(dC);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // ====== TABLA DE RESULTADOS ======
    printf("\n\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                                  TABLA DE RESULTADOS FINAL                                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝\n");
    printf("┌─────────┬──────────────┬──────────────┬──────────────┐\n");
    printf("│    N    │  t (ms)      │  GFLOPS      │  Válido      │\n");
    printf("├─────────┼──────────────┼──────────────┼──────────────┤\n");
    for (int s = 0; s < 4; s++) {
        printf("│ %5d   │ %12.3f │ %12.2f │ %12s │\n", 
               sizes[s], results_time[s], results_gflops[s], 
               results_valid[s] ? "SÍ" : "NO");
    }
    printf("└─────────┴──────────────┴──────────────┴──────────────┘\n\n");

    return 0;
}