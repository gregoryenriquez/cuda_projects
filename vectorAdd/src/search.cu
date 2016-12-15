#include <stdio.h>
#include <time.h>

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
 * Searches for a given phrase in the item reviews for a given file, lines must be prefixed with 'review/text'.
 */
__global__ void wordSearch(int itemsInBuffer, char* phrase, int phraseLen) {
    // printf("wordSearch() is called, blockIdx is: %d, phrase is %s, phraseLen is %d, TEXT_BUFFER[%d] is: %s, ", 
            // blockIdx.x, phrase, phraseLen, blockIdx.x, TEXT_BUFFER[blockIdx.x]);

    if (threadIdx.x < itemsInBuffer) {
        int strLen = 0;
        for (int i = 0; TEXT_BUFFER[threadIdx.x][i] != '\0'; i++) {
            strLen++;
        }
        for (int charIdx = 0; charIdx < strLen; charIdx++) {
            if (TEXT_BUFFER[threadIdx.x][charIdx] == phrase[0]) {
                int wordLen = 1;
                charIdx++;
                while (TEXT_BUFFER[threadIdx.x][charIdx] == phrase[wordLen] && wordLen < phraseLen) {
                    charIdx++;
                    wordLen++;
                    if (wordLen == phraseLen) {
                        MATCHES[threadIdx.x] = 1;
                        // printf("MATCHES[%d] = %d text is %s\n", blockIdx.x, MATCHES[blockIdx.x], TEXT_BUFFER[blockIdx.x]);
                        return;
                    }
                }
            }
        }
    }
    MATCHES[threadIdx.x] = 0;
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
    double timeSpentAllocatingDeviceMemory;
    double timeSpentCopyingToDeviceMemory;
    double timeSpentRunningKernel;
    double timeSpentCopyingFromDeviceMemory;

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

    printf("Size of hbuffer: %d\n", (int)sizeof(hBuffer));

    bool *matches = (bool *)malloc(sizeof(bool) * MAX_TEXT_BLOCKS);
    if (hBuffer == NULL) {
        printf("Failed to create buffer\n");
    }
    char* hBufferPtr = hBuffer[0];
    printf("[char text blocks of %d chars max]\n", MAX_CHAR_SIZE);
    printf("[host buffer of %d chars\n", h_bufferSize);

    clock_t begin, end;
    double time_spent;
    begin = clock();
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
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent to allocate device memory: %f\n", time_spent);
    timeSpentAllocatingDeviceMemory = time_spent;
    begin = 0;
    end = 0;

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
                hBufferPtr = hBuffer[lineCount]; // shift ptr to next block
                strncpy(hBufferPtr, line, MAX_CHAR_SIZE);
                lineCount++;

                /* Host text buffer hBuffer is full and ready to be copied to device TEXT_BUFFER */
                if (lineCount == MAX_TEXT_BLOCKS) {
                    printf("[Host buffer is full, copying memory to device]\n");

                    begin = clock();
                    /* COPY MEMORY FROM HOST TO DEVICE*/
                    /* Copy text buffer to device*/
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
                    end = clock();
                    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
                    timeSpentCopyingToDeviceMemory = time_spent;

                    begin = clock();
                    /* Initialize and launch CUDA kernel */
                    printf("[Initializing CUDA kernel and launching]\n");
                    wordSearch<<<1, MAX_TEXT_BLOCKS>>>(MAX_TEXT_BLOCKS, d_phrase, phraseLen);
                    err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }
                    end = clock();
                    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
                    timeSpentRunningKernel = time_spent;

                    begin = clock();
                    /* Retrieve MATCHES result from device to host*/
                    printf("[Copying device memory matches to host memory]\n");
                    err = cudaMemcpyFromSymbol(matches, MATCHES, sizeof(bool) * MAX_TEXT_BLOCKS, 0, cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "Failed to copy device matches array to host (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }
                    end = clock();
                    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
                    timeSpentCopyingFromDeviceMemory = time_spent;

                    printf("Line Matches Booealn Array, 1 = match, 0 = no match: \n[");
                    for (int i = 0; i < sizeof(bool) * MAX_TEXT_BLOCKS; i++) {
                        if (matches[i] == 1) {
                            printf("1 ");
                        } else {
                            printf("0 ");
                        }
                    }
                    printf("]\n");
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
    cudaFree(d_phrase);
    cudaFree(d_phraseLen);

    /* Performance Results */
    printf("|-------Performance-------|\n");
    printf("Lines processed: 100\n");
    printf("Time spent in allocating device memory: %fs\n", timeSpentAllocatingDeviceMemory);
    printf("Time spent in copying 100 lines to device memory: %fs\n", timeSpentCopyingToDeviceMemory);
    printf("Time spent in executing kernel: %fs\n", timeSpentRunningKernel);
    printf("Time spent in copying matches result to host: %fs\n", timeSpentCopyingFromDeviceMemory);
    printf("Total \"CUDA\" execution time: %fs\n", timeSpentCopyingToDeviceMemory + timeSpentRunningKernel +
        timeSpentCopyingFromDeviceMemory);
    printf("|-------Performance-------|\n");
    printf("Done\n");
    return 0;
}

