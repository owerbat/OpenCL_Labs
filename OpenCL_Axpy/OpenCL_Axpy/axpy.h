#include <CL/cl.h>
#include <iostream>
#include <cstdio>
#include <fstream>


#define RET_CODE_CHECK(retCode, func)                                      \
    retCode = func;                                                        \
    if (retCode) printf("Error: retCode = %d\n", static_cast<int>(retCode));

#define RET_CODE_FUNC_CHECK(retCode, func)                                 \
    func;                                                                  \
    if (retCode) printf("Error: retCode = %d\n", static_cast<int>(retCode));

#define RET_CODE_RETURN_CHECK(retCode, func, result)                       \
    result = func;                                                         \
    if (retCode) printf("Error: retCode = %d\n", static_cast<int>(retCode));


// template <typename FPType>
// void cpu_axpy(const size_t n, const FPType a, const FPType *x, const size_t incx, FPType *y, const size_t incy);

// int readKernel(char *kernelSource, size_t& kernelLen);

// void initializeKernel(cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
//                       cl_device_id& gpu, cl_program& program, cl_int& retCode);

// template <typename FPType>
// void setKernelArguments(const size_t n, const FPType a, const FPType *x, const size_t incx, FPType *y,
//                         const size_t incy, cl_kernel& kernel, cl_context& context,
//                         cl_command_queue& queue, cl_int& retCode, cl_mem& xBuffer, cl_mem& yBuffer);

// template <typename FPType>
// void gpu_axpy(const size_t n, const FPType a, const FPType *x, const size_t incx, FPType *y, const size_t incy);


template <typename FPType>
void cpu_axpy(const size_t n, const FPType a, const FPType *x, const size_t incx, FPType *y, const size_t incy) {
    for (size_t i = 0; i < n; ++i)
        y[i * incy] = y[i * incy] + a * x[i * incx];
}


std::string readKernel(size_t& kernelLen) {
    // FILE *fp;
    // fopen_s(&fp, "axpy_kernel.cl", "rb");

    // if (!fp) {
    //     printf("Kernel loading failed\n");
    //     return 1;
    // }

    // fseek(fp, 0, SEEK_END);
    // kernelLen = ftell(fp);
    // rewind(fp);

    // kernelSource = (char*)malloc(kernelLen + 1);
    // kernelSource[kernelLen] = '\0';
    // fread(kernelSource, sizeof(char), kernelLen, fp);

    // fclose(fp);
    // printf("ptr = %p\n", kernelSource);

    std::ifstream ifs("axpy_kernel.cl");
    std::string content{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};

    kernelLen = content.length();
    // kernelSource = new char[kernelLen+1];
    // for (size_t i = 0; i < kernelLen; ++i)
    //     kernelSource[i] = content[i];
    // kernelSource[kernelLen] = '\0';

    return content;
}


void initializeKernel(cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
                      cl_device_id& gpu, cl_program& program, cl_int& retCode) {
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
        CL_DEVICE_TYPE_GPU, 0, 0, &retCode), context)

    size_t gpuCount = 0;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, 0, &gpuCount);

    cl_device_id* gpus = new cl_device_id[gpuCount];
    clGetContextInfo(context, CL_CONTEXT_DEVICES, gpuCount, gpus, 0);
    gpu = gpus[0];

    RET_CODE_RETURN_CHECK(retCode, clCreateCommandQueueWithProperties(context, gpu, 0, &retCode), queue)

    char *kernelSource = nullptr;
    size_t kernelLen;
    // RET_CODE_CHECK(retCode, readKernel(kernelSource, kernelLen))
    std::string content = readKernel(kernelLen);
    kernelSource = new char[kernelLen+1];
    for (size_t i = 0; i < kernelLen; ++i)
        kernelSource[i] = content[i];
    kernelSource[kernelLen] = '\0';

    printf("ptr = %p\n", kernelSource);
    for (size_t i = 0; i < kernelLen; ++i)
        std::cout << kernelSource[i];
    std::cout << "\nkS: " << (const char**)&kernelSource
              << "\nlen: " << kernelLen << std::endl;

    RET_CODE_RETURN_CHECK(retCode, clCreateProgramWithSource(context, 1, (const char**)&kernelSource,
        &kernelLen, &retCode), program)
    clBuildProgram(program, 1, &gpu, 0, 0, 0);
    printf("hi\n");

    kernel = clCreateKernel(program, "axpy", 0);

    delete[] kernelSource;
}


template <typename FPType>
void setKernelArguments(const size_t n, const FPType a, const FPType *x, const size_t incx, FPType *y,
                        const size_t incy, cl_kernel& kernel, cl_context& context,
                        cl_command_queue& queue, cl_int& retCode, cl_mem& xBuffer, cl_mem& yBuffer) {
    size_t biteSize = sizeof(FPType) * n;

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
void gpu_axpy(const size_t n, const FPType a, const FPType *x, const size_t incx, FPType *y, const size_t incy) {
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_device_id gpu;
    cl_program program;
    cl_mem xBuffer, yBuffer;
    cl_int retCode = 0;

    initializeKernel(kernel, context, queue, gpu, program, retCode);
    setKernelArguments(n, a, x, incx, y, incy, kernel, context, queue, retCode, xBuffer, yBuffer);

    // size_t group;
    // RET_CODE_CHECK(retCode, clGetKernelWorkGroupInfo(kernel, gpu, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, 0))

    // RET_CODE_CHECK(retCode, clEnqueueNDRangeKernel(queue, kernel, 1, 0, &n, &group, 0, 0, 0))
    // RET_CODE_CHECK(retCode, clEnqueueReadBuffer(queue, yBuffer, CL_TRUE, 0, sizeof(FPType) * n, y, 0, 0, 0))

    // clReleaseMemObject(xBuffer);
    // clReleaseMemObject(yBuffer);
    // clReleaseProgram(program);
    // clReleaseKernel(kernel);
    // clReleaseCommandQueue(queue);
    // clReleaseContext(context);
}
