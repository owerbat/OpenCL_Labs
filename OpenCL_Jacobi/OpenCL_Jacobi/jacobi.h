#include <CL/cl.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <utility>


#define RET_CODE_CHECK(retCode, func, message)                                             \
    retCode = func;                                                                        \
    if (retCode) printf("Error: retCode = %d [%s]\n", static_cast<int>(retCode), message);

#define RET_CODE_FUNC_CHECK(retCode, func, message)                                        \
    func;                                                                                  \
    if (retCode) printf("Error: retCode = %d [%s]\n", static_cast<int>(retCode), message);

#define RET_CODE_RETURN_CHECK(retCode, func, result, message)                              \
    result = func;                                                                         \
    if (retCode) printf("Error: retCode = %d [%s]\n", static_cast<int>(retCode), message);


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

    if (retCode != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        char *log = new char[logSize];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
        printf("\n-------------------------------------\n");
        printf("log:\n%s", log);
        printf("-------------------------------------\n\n");
    }

    kernel = clCreateKernel(program, kernelName, 0);

    delete[] kernelSource;
}


void setKernelArguments(const size_t size, const float *a, float *b, float *x0, float *x1,
                        float *norm, cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
                        cl_device_id& device, cl_int& retCode, cl_mem& aBuffer, cl_mem& bBuffer,
                        cl_mem& x0Buffer, cl_mem& x1Buffer, cl_mem& normBuffer) {
    cl_uint biteSizeA = sizeof(float) * size * size;
    cl_uint biteSize  = sizeof(float) * size;

    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_READ_ONLY,
                          biteSizeA, 0, &retCode), aBuffer, "clCreateBuffer a")
    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, aBuffer, CL_TRUE, 0, biteSizeA, a, 0, 0, 0), "clEnqueueWriteBuffer a")
    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer), "clSetKernelArg a")

    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_READ_ONLY,
                          biteSize, 0, &retCode), bBuffer, "clCreateBuffer b")
    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, bBuffer, CL_TRUE, 0, biteSize, b, 0, 0, 0), "clEnqueueWriteBuffer b")
    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer), "clSetKernelArg b")

    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_READ_WRITE,
                          biteSize, 0, &retCode), x0Buffer, "clCreateBuffer x0")
    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, x0Buffer, CL_TRUE, 0, biteSize, x0, 0, 0, 0), "clEnqueueWriteBuffer x0")

    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_READ_WRITE,
                          biteSize, 0, &retCode), x1Buffer, "clCreateBuffer x1")
    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, x1Buffer, CL_TRUE, 0, biteSize, x1, 0, 0, 0), "clEnqueueWriteBuffer x1")

    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                          biteSize, 0, &retCode), normBuffer, "clCreateBuffer norm")
    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, normBuffer, CL_TRUE, 0, biteSize, norm, 0, 0, 0), "clEnqueueWriteBuffer norm")
    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 4, sizeof(cl_mem), &normBuffer), "clSetKernelArg norm")
}


auto opencl_jacobi_impl(const size_t size, const float *a, float *b, float *x0, float *x1,
                      float *norm, const char *filename, const char *kernelName, cl_device_type deviceType) {
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_device_id device;
    cl_program program;
    cl_mem aBuffer, bBuffer, x0Buffer, x1Buffer, normBuffer;
    cl_int retCode = 0;
    size_t groupSize = 256;

    initializeKernel(kernel, context, queue, device, program, retCode, filename, kernelName, deviceType);
    setKernelArguments(size, a, b, x0, x1, norm, kernel, context, queue, device, retCode,
                       aBuffer, bBuffer, x0Buffer, x1Buffer, normBuffer);

    cl_event event;
        printf("1\n");
    size_t nWorkItems = (size / groupSize + !!(size % groupSize)) * groupSize;
    size_t iter = -1;
    const size_t nIter = 200;
    float sum = FLT_MAX;
    const float tol = 1e-7f;

    auto t0 = std::chrono::steady_clock::now();
    while (++iter <= nIter && sqrt(sum) > tol) {
        RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 2, sizeof(cl_mem), &x0Buffer), "clSetKernelArg x0")
        RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 3, sizeof(cl_mem), &x1Buffer), "clSetKernelArg x1")

        RET_CODE_CHECK(retCode, clEnqueueNDRangeKernel(queue, kernel, 2, 0, &nWorkItems, &groupSize, 0, 0, &event), "clEnqueueNDRangeKernel")
        // printf("2\n");
        clWaitForEvents(1, &event);

        sum = 0.0f;
        for (size_t i = 0; i < size; ++i)
            sum += norm[i] * norm[i];

        std::swap(x0, x1);
        std::swap(x0Buffer, x1Buffer);
    }
    auto time = std::chrono::steady_clock::now() - t0;

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(x0Buffer);
    clReleaseMemObject(x1Buffer);
    clReleaseMemObject(normBuffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return time;
}


auto opencl_jacobi_cpu(const size_t size, const float *a, float *b, float *x0, float *x1, float *norm) {
    return opencl_jacobi_impl(size, a, b, x0, x1, norm, "jacobi_kernel.cl", "jacobi", CL_DEVICE_TYPE_CPU);
}


auto opencl_jacobi_gpu(const size_t size, const float *a, float *b, float *x0, float *x1, float *norm) {
    return opencl_jacobi_impl(size, a, b, x0, x1, norm, "jacobi_kernel.cl", "jacobi", CL_DEVICE_TYPE_GPU);
}