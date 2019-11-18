#include <CL/cl.h>
#include <omp.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>


#define RET_CODE_CHECK(retCode, func, message)                                             \
    retCode = func;                                                                        \
    if (retCode) printf("Error: retCode = %d [%s]\n", static_cast<int>(retCode), message);

#define RET_CODE_FUNC_CHECK(retCode, func, message)                                        \
    func;                                                                                  \
    if (retCode) printf("Error: retCode = %d [%s]\n", static_cast<int>(retCode), message);

#define RET_CODE_RETURN_CHECK(retCode, func, result, message)                              \
    result = func;                                                                         \
    if (retCode) printf("Error: retCode = %d [%s]\n", static_cast<int>(retCode), message);


#define BLOCK_SIZE 16


auto omp_gemm(const cl_uint n, const float *a, const float *b, float *c) {
    int i, j, k;
    float c_ij;
    auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for shared (n, a, b, c) private(i, j, k, c_ij)
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            c_ij = 0.0f;
            for (k = 0; k < n; ++k)
                c_ij += a[i * n + k] * b[k * n + j];
            c[i * n + j] = c_ij;
        }
    }

    auto time = std::chrono::steady_clock::now() - t0;

    return time;
}


std::string readKernel(const char *filename) {
    std::ifstream ifs(filename);
    std::string content{ std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>() };

    return content;
}


void initializeKernel(cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
                      cl_device_id& device, cl_program& program, cl_int& retCode,
                      const char *filename, const char *kernelName, cl_device_type deviceType) {
    cl_uint platformsCount = 0;
    clGetPlatformIDs(0, nullptr, &platformsCount);

    cl_platform_id* platforms = new cl_platform_id[platformsCount];
    clGetPlatformIDs(platformsCount, platforms, nullptr);

    cl_platform_id platform = platforms[0];
    cl_context_properties properties[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    RET_CODE_RETURN_CHECK(retCode, clCreateContextFromType((platform == nullptr) ? nullptr : properties,
        deviceType, 0, 0, &retCode), context, "clCreateContextFromType")

    size_t deviceCount = 0;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, 0, &deviceCount);

    cl_device_id* devices = new cl_device_id[deviceCount];
    clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceCount, devices, 0);
    device = devices[0];

    char deviceName[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, deviceName, nullptr);

    RET_CODE_RETURN_CHECK(retCode, clCreateCommandQueueWithProperties(context, device, 0, &retCode), queue, "clCreateCommandQueueWithProperties")

    std::string content = readKernel(filename);
    size_t kernelLen = content.length();
    char* kernelSource = new char[kernelLen + 1];
    for (cl_uint i = 0; i < kernelLen; ++i)
        kernelSource[i] = content[i];
    kernelSource[kernelLen] = '\0';

    RET_CODE_RETURN_CHECK(retCode, clCreateProgramWithSource(context, 1, (const char**)&kernelSource,
        &kernelLen, &retCode), program, "clCreateProgramWithSource")
    clBuildProgram(program, 1, &device, 0, 0, 0);

    // size_t logSize = 1000, actualLogSize;
    // char *log = new char[logSize];
    // clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, &actualLogSize);
    // printf("log:\n%s\n", log);

    kernel = clCreateKernel(program, kernelName, 0);

    delete[] kernelSource;
}


void setKernelArguments(const cl_uint n, const float *a, const float *b, float *c,
                        cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
                        cl_device_id& device, cl_int& retCode, cl_mem& aBuffer, cl_mem& bBuffer,
                        cl_mem& cBuffer) {
    cl_uint biteSize = sizeof(float) * n * n;

    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 0, sizeof(cl_uint), &n), "clSetKernelArg")

    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_READ_ONLY, biteSize, 0, &retCode), aBuffer, "clCreateBuffer")
    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, aBuffer, CL_TRUE, 0, biteSize, a, 0, 0, 0), "clEnqueueWriteBuffer")
    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 1, sizeof(cl_mem), &aBuffer), "clSetKernelArg")

    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_READ_ONLY, biteSize, 0, &retCode), bBuffer, "clCreateBuffer")
    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, bBuffer, CL_TRUE, 0, biteSize, b, 0, 0, 0), "clEnqueueWriteBuffer")
    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 2, sizeof(cl_mem), &bBuffer), "clSetKernelArg")

    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_READ_WRITE, biteSize, 0, &retCode), cBuffer, "clCreateBuffer")
    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, cBuffer, CL_TRUE, 0, biteSize, c, 0, 0, 0), "clEnqueueWriteBuffer")
    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 3, sizeof(cl_mem), &cBuffer), "clSetKernelArg")
}


auto opencl_gemm_impl(const cl_uint n, const float *a, const float *b, float *c,
                      const char *filename, const char *kernelName, cl_device_type deviceType) {
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_device_id device;
    cl_program program;
    cl_mem aBuffer, bBuffer, cBuffer;
    cl_int retCode = 0;
    size_t groupSize = 0;

    initializeKernel(kernel, context, queue, device, program, retCode, filename, kernelName, deviceType);
    setKernelArguments(n, a, b, c, kernel, context, queue, device, retCode, aBuffer, bBuffer, cBuffer);

    cl_event event;
    const size_t nWorkItems[] = {n, n};
    const size_t groupSizes[] = {BLOCK_SIZE, BLOCK_SIZE};
    auto t0 = std::chrono::steady_clock::now();
    RET_CODE_CHECK(retCode, clEnqueueNDRangeKernel(queue, kernel, 2, 0, nWorkItems, groupSizes, 0, 0, &event), "clEnqueueNDRangeKernel")
    clWaitForEvents(1, &event);
    auto time = std::chrono::steady_clock::now() - t0;
    RET_CODE_CHECK(retCode, clEnqueueReadBuffer(queue, cBuffer, CL_TRUE, 0, sizeof(float) * n * n, c, 0, 0, 0), "clEnqueueReadBuffer")

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(cBuffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return time;
}


auto opencl_gemm_cpu(const cl_uint n, const float *a, const float *b, float *c) {
    return opencl_gemm_impl(n, a, b, c, "gemm_kernel.cl", "gemm", CL_DEVICE_TYPE_CPU);
}


auto opencl_gemm_gpu(const cl_uint n, const float *a, const float *b, float *c) {
    return opencl_gemm_impl(n, a, b, c, "gemm_kernel.cl", "gemm", CL_DEVICE_TYPE_GPU);
}


auto opencl_gemm_block_cpu(const cl_uint n, const float *a, const float *b, float *c) {
    return opencl_gemm_impl(n, a, b, c, "gemm_block_kernel.cl", "gemm_block", CL_DEVICE_TYPE_CPU);
}


auto opencl_gemm_block_gpu(const cl_uint n, const float *a, const float *b, float *c) {
    return opencl_gemm_impl(n, a, b, c, "gemm_block_kernel.cl", "gemm_block", CL_DEVICE_TYPE_GPU);
}