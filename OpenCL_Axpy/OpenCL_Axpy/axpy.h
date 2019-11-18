#include <CL/cl.h>
#include <omp.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>


#define RET_CODE_CHECK(retCode, func)                                      \
    retCode = func;                                                        \
    if (retCode) printf("Error: retCode = %d\n", static_cast<int>(retCode));

#define RET_CODE_FUNC_CHECK(retCode, func)                                 \
    func;                                                                  \
    if (retCode) printf("Error: retCode = %d\n", static_cast<int>(retCode));

#define RET_CODE_RETURN_CHECK(retCode, func, result)                       \
    result = func;                                                         \
    if (retCode) printf("Error: retCode = %d\n", static_cast<int>(retCode));


template <typename FPType>
auto cpu_axpy(const size_t n, const FPType a, const FPType* x, const size_t incx, FPType* y, const size_t incy) {
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i * incy < n && i * incx < n; ++i)
        y[i * incy] = y[i * incy] + a * x[i * incx];
    auto time = std::chrono::steady_clock::now() - t0;

    return time;
}

template <typename FPType>
auto omp_axpy(const size_t n, const FPType a, const FPType *x, const size_t incx, FPType *y, const size_t incy) {
    int i;
    auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for shared (y, x ,a, n, incx, incy) private(i)
    for (i = 0; i < n; ++i)
    if (i * incy < n && i * incx < n)
        y[i * incy] = y[i * incy] + a * x[i * incx];

    auto time = std::chrono::steady_clock::now() - t0;

    return time;
}

template <typename FPType>
std::string readKernel() {
    std::ifstream ifs(sizeof(FPType) == sizeof(double) ? "daxpy_kernel.cl" : "saxpy_kernel.cl");
    std::string content{ std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>() };

    return content;
}


template <typename FPType>
void initializeKernel(cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
    cl_device_id& gpu, cl_program& program, cl_int& retCode, cl_device_type deviceType = CL_DEVICE_TYPE_GPU) {
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
        deviceType, 0, 0, &retCode), context)

    size_t gpuCount = 0;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, 0, &gpuCount);

    cl_device_id* gpus = new cl_device_id[gpuCount];
    clGetContextInfo(context, CL_CONTEXT_DEVICES, gpuCount, gpus, 0);
    gpu = gpus[0];

    char gpuName[128];
    clGetDeviceInfo(gpu, CL_DEVICE_NAME, 128, gpuName, nullptr);

    RET_CODE_RETURN_CHECK(retCode, clCreateCommandQueueWithProperties(context, gpu, 0, &retCode), queue)

    std::string content = readKernel<FPType>();
    size_t kernelLen = content.length();
    char* kernelSource = new char[kernelLen + 1];
    for (size_t i = 0; i < kernelLen; ++i)
        kernelSource[i] = content[i];
    kernelSource[kernelLen] = '\0';

    RET_CODE_RETURN_CHECK(retCode, clCreateProgramWithSource(context, 1, (const char**)&kernelSource,
        &kernelLen, &retCode), program)
    clBuildProgram(program, 1, &gpu, 0, 0, 0);

    kernel = clCreateKernel(program, sizeof(FPType) == sizeof(double) ? "daxpy" : "saxpy", 0);

    delete[] kernelSource;
}


template <typename FPType>
void setKernelArguments(const size_t n, const FPType a, const FPType* x, const size_t incx, FPType* y,
    const size_t incy, cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
    cl_device_id& gpu, cl_int& retCode, cl_mem& xBuffer, cl_mem& yBuffer, size_t& groupSize) {
    RET_CODE_CHECK(retCode, clGetKernelWorkGroupInfo(kernel, gpu, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &groupSize, 0))
    // groupSize = 8;
    size_t biteSize = sizeof(FPType) * (n / groupSize + !!(n % groupSize)) * groupSize;

    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 0, sizeof(size_t), &n))

    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 1, sizeof(FPType), &a))

    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_READ_ONLY, biteSize, 0, &retCode), xBuffer)
    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, xBuffer, CL_TRUE, 0, biteSize, x, 0, 0, 0))
    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 2, sizeof(cl_mem), &xBuffer))

    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 3, sizeof(size_t), &incx))

    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_READ_WRITE, biteSize, 0, &retCode), yBuffer)
    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, yBuffer, CL_TRUE, 0, biteSize, y, 0, 0, 0))
    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 4, sizeof(cl_mem), &yBuffer))

    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 5, sizeof(size_t), &incy))
}


template <typename FPType>
auto opencl_axpy(const size_t n, const FPType a, const FPType* x, const size_t incx, FPType* y, const size_t incy,
              cl_device_type deviceType = CL_DEVICE_TYPE_GPU) {
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_device_id gpu;
    cl_program program;
    cl_mem xBuffer, yBuffer;
    cl_int retCode = 0;
    size_t groupSize = 0;

    initializeKernel<FPType>(kernel, context, queue, gpu, program, retCode);
    setKernelArguments(n, a, x, incx, y, incy, kernel, context, queue, gpu, retCode, xBuffer, yBuffer, groupSize);

    size_t nWorkItems = (n / groupSize + !!(n % groupSize)) * groupSize;
    cl_event event;
    auto t0 = std::chrono::steady_clock::now();
    RET_CODE_CHECK(retCode, clEnqueueNDRangeKernel(queue, kernel, 1, 0, &nWorkItems, &groupSize, 0, 0, &event))
    clWaitForEvents(1, &event);
    auto time = std::chrono::steady_clock::now() - t0;
    RET_CODE_CHECK(retCode, clEnqueueReadBuffer(queue, yBuffer, CL_TRUE, 0, sizeof(FPType) * n, y, 0, 0, 0))

    clReleaseMemObject(xBuffer);
    clReleaseMemObject(yBuffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return time;
}