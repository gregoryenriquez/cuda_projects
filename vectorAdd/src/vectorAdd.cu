#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#define MAX_TEXT_BLOCKS 100
#define MAX_CHAR_SIZE 10000
#define BUFFER_SIZE (MAX_TEXT_BLOCKS * MAX_CHAR_SIZE)

__device__ char TEXT_BUFFER[MAX_TEXT_BLOCKS][MAX_CHAR_SIZE];
__device__ bool MATCHES[MAX_TEXT_BLOCKS];

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void wordSearch(int itemsInBuffer, char* phrase, int phraseLen) {
    printf("wordSearch() is called, blockIdx is: %d ", blockIdx.x);
    printf("phrase is: %s ", phrase);
    printf("TEXT_BUFFER[%d] is: %s ", blockIdx.x, TEXT_BUFFER[blockIdx.x]);
    printf("phraseLen is: %d ", phraseLen);
    int containsMatch = 0;

    if (blockIdx.x < itemsInBuffer) {
        int strLen = 0;
        for (int i = 0; TEXT_BUFFER[blockIdx.x][i] != '\0'; i++) {
            strLen++;
        }
        printf("blockIdx < itemsInBuffer = true\n");
        for (int charIdx = 0; charIdx < strLen; charIdx++) {
            // printf("charIdx < strlen(TEXT_BUFFER[%d]) = true\n", blockIdx.x);
            if (TEXT_BUFFER[blockIdx.x][charIdx] == phrase[0]) {
                printf("TEXT_BUFFER[blockIdx.x][charIdx] == phrase[0] = true\n");
                int wordLen = 1;
                charIdx++;
                while (TEXT_BUFFER[blockIdx.x][charIdx] == phrase[wordLen] && wordLen < phraseLen) {
                    printf("TEXT_BUFFER[charIdx] == phrase[wordLen] && wordLen < phraseLen = true\n");
                    charIdx++;
                    wordLen++;
                    if (wordLen == phraseLen) {
                        containsMatch = 1;
                        MATCHES[blockIdx.x] = 1;
                        printf("MATCHES[blockIdx.x] = %d text is %s\n", MATCHES[blockIdx.x], TEXT_BUFFER[blockIdx.x]);
                        break;
                    }
                }
                if (containsMatch == 1) {
                    printf("Match found in block: %d, \n", blockIdx.x);
                    return;
                }
            }
        }
    }
    MATCHES[blockIdx.x] = 0;


}


/* HOST MAIN ROUTINE */
int
main(int argc, char **argv)
{
	if (argc < 3) {
		printf("Insufficient arguments\n");
		exit(0);
	}

	char* filePath = argv[1];
	char* phrase = argv[2];

	int phraseLen = 0;
	for (int i = 0; phrase[i] != '\0'; i++) {
		if (i == 1000) break;
		phraseLen++;
	}

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    const int h_bufferSize = MAX_CHAR_SIZE * MAX_TEXT_BLOCKS; // 4000000 bytes of chars = 4MB
    const char* reviewIdentifier = "review/text";
    char hBuffer[MAX_TEXT_BLOCKS][MAX_CHAR_SIZE];

    printf("Size of hbuffer: %d", (int)sizeof(hBuffer));

    bool *matches = (bool *)malloc(sizeof(bool) * MAX_TEXT_BLOCKS);
    if (hBuffer == NULL) {
    	printf("Failed to create buffer\n");
    }
    char* hBufferPtr = hBuffer[0];
    printf("[char text blocks of %d chars max]\n", MAX_CHAR_SIZE);
    printf("[host buffer of %d chars\n", h_bufferSize);

    /* DEVICE ALLOCATE MEMORY */
    /* Allocate memory for global variable MATCHES on device */
    err = cudaMalloc((void **)&MATCHES, sizeof(bool) * MAX_TEXT_BLOCKS);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate matches array matches (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Allocate memory on device for command line argument phrase */
    char *d_phrase = NULL;
    size_t d_phraseSize = phraseLen * sizeof(char);
    err = cudaMalloc((void **)&d_phrase, d_phraseSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate char array phrase (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Allocate memory on device for phrase length */
    int *d_phraseLen = NULL;
    size_t d_phraseLenSize = sizeof(int);
    err = cudaMalloc((void **)&d_phraseLen, d_phraseLenSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate phrase length phraseLen (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    FILE *file = fopen(filePath, "r");
    int lineId = 0;
    int lineCount = 0;

    if (file != NULL) {
    	char line[MAX_CHAR_SIZE];
    	if (line == NULL) {
        	printf("Failed to allocate text block!\n");
        	exit(EXIT_FAILURE);
    	}
    	while (fgets(line, MAX_CHAR_SIZE, file) != NULL && lineCount < MAX_TEXT_BLOCKS + 1) {
    		size_t tempLen = strlen(reviewIdentifier);
    		char firstWord[tempLen];
    		strncpy(firstWord, line, tempLen);

    		if (strncmp(reviewIdentifier, firstWord, tempLen) == 0) {
                if (lineCount % 10 == 0) {
                    printf("Line: %d\n", lineCount);
                }

        		// printf("%s\n", line);
                hBufferPtr = hBuffer[lineCount]; // shift ptr to next block
                strncpy(hBufferPtr, line, MAX_CHAR_SIZE);
                // printf("hBufferPtr = %p", hBufferPtr);
                lineCount++;

                /* Host text buffer hBuffer is full and ready to be copied to device TEXT_BUFFER */
        		if (lineCount == MAX_TEXT_BLOCKS) {
        			printf("[Host buffer is full, copying memory to device]\n");

        		    /* COPY MEMORY FROM HOST TO DEVICE*/
        			/* Copy text buffer to device*/
                    printf("[Copying text buffer to device...]\n");
                    for (int i = 0; i < MAX_TEXT_BLOCKS; i++) {
                        printf("%s\n", hBuffer[i]);                        
                    }

                    err = cudaMemcpyToSymbol(TEXT_BUFFER, hBuffer, BUFFER_SIZE, 0, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "Failed to copy text buffer hBuffer (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }                    

        		    /* Copy phrase to device */
                    printf("[Copying phrase to device...]\n");
                    err = cudaMemcpy(d_phrase, phrase, d_phraseSize, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess)
                    {
                        fprintf(stderr, "Failed to copy phrase d_phrase (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }

        		    /* Copy phrase length to device */
                    printf("[Copying phrase len to device...]\n");
                    err = cudaMemcpy(d_phraseLen, &phraseLen, d_phraseLenSize, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess)
                    {
                        fprintf(stderr, "Failed to copy phrase len d_phraseLen (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }

            		/* Initialize and launch CUDA kernel */
                    printf("[Initializing CUDA kernel and launching]\n");
                    wordSearch<<<MAX_TEXT_BLOCKS,1>>>(MAX_TEXT_BLOCKS, d_phrase, phraseLen);
                    err = cudaGetLastError();
        		    if (err != cudaSuccess) {
        		        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        		        exit(EXIT_FAILURE);
        		    }

                    // cudaFree(d_phrase);
                    // cudaFree(d_phraseLen);

                    /* Retrieve MATCHES result from device to host*/
                    printf("[Copying device memory matches to host memory]\n");
                    err = cudaMemcpyFromSymbol(matches, MATCHES, sizeof(bool) * MAX_TEXT_BLOCKS, 0, cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "Failed to copy device matches array to host (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }

                    for (int i = 0; i < sizeof(bool) * MAX_TEXT_BLOCKS; i++) {
                        if (matches[i] == 1) {
                            printf("1");
                        } else {
                            printf("0");
                        }
                    }
                    printf("\n");

                    // exit(0);

        			// end cuda
        			// lineCount = 0;
           //          printf("[Setting hbuffer to empty via memset\n");
        			// memset(&hBuffer[0], 0, sizeof(hBuffer)); // clear the buffer
                    // set the buffer ptr back to the beginning
           //          printf("[Settings hBufferPtr back to the beginning\n");
        			// hBufferPtr = hBuffer[0];

                    break;

        		}
    		}
    		lineId++;
    	}
    } else {
    	printf("Could not read file %s", filePath);
    	return 0;
    }

    printf("End CUDA search\n");

    fclose(file);
    free(matches);
    // cudaFree(d_text);
    // cudaFree(d_textLen);
    cudaFree(d_phrase);
    cudaFree(d_phraseLen);
    // cudaFree(d_textBlocks);
    // cudaFree(d_matches);

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

