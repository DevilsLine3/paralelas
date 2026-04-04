#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define TOL 1e-3

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

int main() {
    int sizes[] = {512, 1024, 2048, 4096};

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

        // Inicializar
        for (int i = 0; i < N*N; i++) {
            A[i] = (float)rand() / RAND_MAX;
            B[i] = (float)rand() / RAND_MAX;
            C[i] = 0.0f;
        }

        if (N <= 1024) {
            // CPU referencia
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < N; k++) {
                        sum += A[i*N + k] * B[k*N + j];
                    }
                    C_ref[i*N + j] = sum;
                }
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

        // Ejecutar kernel con medición
        cl_event event;
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, &event);
        clWaitForEvents(1, &event);

        // Tiempo
        cl_ulong start, end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

        double time_ms = (end - start) * 1e-6;
        double time_s = time_ms / 1000.0;
        double gflops = (2.0 * N * N * N) / (time_s * 1e9);

        // Leer resultado
        clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, bytes, C, 0, NULL, NULL);

        // Verificación
        int correct = 1;
        if (N <= 1024) {
            for (int i = 0; i < N*N; i++) {
                if (fabs(C[i] - C_ref[i]) > TOL) {
                    correct = 0;
                    break;
                }
            }
        }

        // Output
        printf("----------------------------------\n");
        printf("N = %d\n", N);
        printf("Tiempo kernel: %.3f ms\n", time_ms);
        printf("GFLOPS: %.2f\n", gflops);
        if (N <= 1024) {
            printf("Resultado: %s\n", correct ? "correct" : "incorrect");
        } else {
            printf("Verificacion omitida para N > 1024\n");
        }

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

    return 0;
}