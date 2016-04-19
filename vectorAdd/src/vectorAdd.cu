/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#define MAX_TEXT_BLOCKS = 2;
#define MAX_CHAR_SIZE = 10000;
#define BUFFER_SIZE = MAX_TEXT_BLOCKS * MAX_CHAR_SIZE;

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

__global__ void wordSearch(const char *textBuffer, int textLen, int itemsInBuffer, char* phrase, int phraseLen, bool *matches) {
    printf("wordSearch() is called, blockIdx is: %d ", blockIdx.x);
    printf("phrase is: %s ", phrase);
    printf("TEXT_BUFFER[%d] is: %s ", blockIdx.x, TEXT_BUFFER[blockIdx.x]);
    printf("phraseLen is: %d ", phraseLen);
    int containsMatch = 0;
    int spaceAsciiVal = 32;
    const char space = (char)spaceAsciiVal;

    int charIdx = 0; // set starting char position
    if (blockIdx.x < itemsInBuffer) {
        printf("blockIdx < itemsInBuffer = true\n");
        while (charIdx < MAX_CHAR_SIZE)) { // check end position = start of next block
            // printf("charIdx < ((blockIdx.x + 1) * textLen) = true\n");
            if (TEXT_BUFFER[blockIdx.x][charIdx] == phrase[0]) {
                printf("TEXT_BUFFER[blockIdx.x][charIdx] == phrase[0] = true\n");
                int wordLen = 1;
                charIdx++;
                while (TEXT_BUFFER[blockIdx.x][charIdx] == phrase[wordLen] && wordLen < phraseLen) {
                    printf("textBuffer[charIdx] == phrase[wordLen] && wordLen < phraseLen = true\n");
                    charIdx++;
                    wordLen++;
                    if (wordLen == phraseLen) {
                        containsMatch = 1;
                        MATCHES[blockIdx.x] = 1;
                        printf("matches[blockIdx.x] = %d text is %s\n", blockIdx.x, textBuffer);
                        break;
                    }
                }
                if (containsMatch == 1) {
                    printf("Match found in block: %d, \n", blockIdx.x);
                    return;
                }
            }
            charIdx++;
        }
    }
    MATCHES[blockIdx.x] = 0;


}

// __global__ void wordSearch(const char *textBuffer, int textLen, int itemsInBuffer, char* phrase, int phraseLen, bool *matches) {
//     printf("wordSearch() is called, blockIdx is: %d ", blockIdx.x);
//     printf("phrase is: %s ", phrase);
//     printf("textBuffer[%d] is: %s ", blockIdx.x, textBuffer[blockIdx.x]);
//     printf("phraseLen is: %d ", phraseLen);
// 	int textIdx = blockIdx.x;
// 	int containsMatch = 0;
// 	int spaceAsciiVal = 32;
// 	const char space = (char)spaceAsciiVal;

// 	int charIdx = blockIdx.x * textLen; // set starting char position
// 	if (textIdx < itemsInBuffer) {
//         printf("textIdx < itemsInBuffer = true\n");
// 		while (charIdx < ((blockIdx.x + 1) * textLen)) { // check end position = start of next block
//             // printf("charIdx < ((blockIdx.x + 1) * textLen) = true\n");
// 			if (textBuffer[charIdx] == phrase[0]) {
//                 printf("textBuffer[charIdx] == phrase[0] = true\n");
// 				int wordLen = 1;
// 				charIdx++;
// 				while (textBuffer[charIdx] == phrase[wordLen] && wordLen < phraseLen) {
//                     printf("textBuffer[charIdx] == phrase[wordLen] && wordLen < phraseLen = true\n");
// 					charIdx++;
// 					wordLen++;
// 					if (wordLen == phraseLen && textBuffer[charIdx] == space) {
// 						containsMatch = 1;
// 						matches[textIdx] = 1;
//                         printf("matches[textIdx] = %d text is %s\n", textIdx, textBuffer);
// 						break;
// 					}
// 				}
// 				if (containsMatch == 1) {
//                     printf("Match found in block: %d, \n", textIdx);
// 					return;
// 				}
// 			}
//             charIdx++;
// 		}
// 	}
// 	matches[textIdx] = 0;


// }

/**
 * Host main routine
 */
int
main(int argc, char **argv)
{
	if (argc < 3) {
		printf("Insufficient arguments\n");
		exit(0);
	}

	char* filePath = argv[1];
	char* phrase = argv[2];

    //for debugging
    const int numOfEntriesToRead = 2;

	int phraseLen = 0;
	for (int i = 0; phrase[i] != '\0'; i++) {
		if (i == 1000) break;
		phraseLen++;
	}

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // TODO: remove start and end spaces from query

    const int maxChars = 10000;  // 40000 bytes = 40kb per line
    const int *maxCharsPtr = &maxChars;
    const int maxBlocksInBuffer = numOfEntriesToRead; // 100 blocks o text
    const int *maxBlocksInBufferPtr = &maxBlocksInBuffer;
    const int h_bufferSize = maxChars * maxBlocksInBuffer; // 4000000 bytes of chars
    const char* reviewIdentifier = "review/text";
    char hBuffer[maxBlocksInBuffer][maxChars] = ""; // 

    printf("Size of hbuffer: %d", (int)sizeof(hBuffer));

    const int numOfLinesToSearch = numOfEntriesToRead;
    bool *matches = (bool *)malloc(sizeof(bool) * maxBlocksInBuffer);
    int matchesSize = numOfLinesToSearch * sizeof(bool);

    if (hBuffer == NULL) {
    	printf("Failed to create buffer\n");
    }
    char* hBufferPtr = hBuffer[0][0];
    printf("[char text blocks of %d chars max]\n", maxChars);
    printf("[host buffer of %d chars\n", h_bufferSize);

    FILE *file = fopen(filePath, "r");
    int lineId = 0;
    int bufferCount = 0;
    int lineCount = 0;

    // initialize matches
    bool *d_matches = NULL;
    size_t d_matchesSize = sizeof(bool) * maxBlocksInBuffer; // HACK, will only work with 1 block!!!
    err = cudaMalloc((void **)&d_matches, d_matchesSize);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate matches array matches (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_matches, matches, d_matchesSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to mem copy matches array (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    /* DEVICE ALLOCATE MEMORY */
    char *d_text = NULL;
    size_t d_bufferSize = h_bufferSize * sizeof(char);
    err = cudaMalloc((void **)&d_text, d_bufferSize);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate text buffer hBuffer (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_textLen = NULL;
    size_t d_textLenSize = sizeof(int);
    err = cudaMalloc((void **)&d_textLen, d_textLenSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate text buffer char len textLen (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_textBlocks = NULL;
    size_t d_textBlockSize = sizeof(int);
    err = cudaMalloc((void **)&d_textBlocks, d_textBlockSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate num of text blocks textBlocks (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    char *d_phrase = NULL;
    size_t d_phraseSize = phraseLen * sizeof(char);
    err = cudaMalloc((void **)&d_phrase, d_phraseSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate char array phrase (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_phraseLen = NULL;
    size_t d_phraseLenSize = sizeof(int);
    err = cudaMalloc((void **)&d_phraseLen, d_phraseLenSize);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate phrase length phraseLen (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (file != NULL) {
    	char line[maxChars];
    	if (line == NULL) {
        	printf("Failed to allocate text block!\n");
        	exit(EXIT_FAILURE);
    	}
    	while (fgets(line, maxChars, file) != NULL && lineCount < 100) {
    		size_t tempLen = strlen(reviewIdentifier);
    		char firstWord[tempLen];
    		strncpy(firstWord, line, tempLen);

    		if (strncmp(reviewIdentifier, firstWord, tempLen) == 0) {
                if (lineCount % 11 == 0) {
                    printf("Line: %d\n", lineCount);
                }


        		printf("%s\n", line);

        		bufferCount++;
                hBufferPtr = &hBufferPtr[lineCount]; // shift ptr to next block
                // for (int i = 0; i < strlen(line); i++) {
                //     hBuffer[lineCount * maxChars + i] = line[i];
                // }


                strncpy(hBufferPtr, line, maxChars);
                printf("hBufferPtr = %p", hBufferPtr);
                printf(" end of hbuffer = %p\n", &hBuffer[maxChars - 1][maxBlocksInBuffer - 1]);                  
                lineCount++;

        		if (bufferCount == maxBlocksInBuffer || lineCount == numOfEntriesToRead) {
        			printf("Host buffer is full, copying memory to device\n");

                    // for (int i = 0; i < sizeof(hBuffer); i++) {
                    //     printf("%c", hBuffer[i]);
                    // }
                    // printf("\n");
                    // exit(0);

        		    /* COPY MEMORY */
        			// copy text buffer
                    printf("[Copying text buffer to device...]\n");
                    printf("%s\n", hBuffer);
        		    // err = cudaMemcpy(d_text, hBuffer, d_bufferSize, cudaMemcpyHostToDevice);
        		    // if (err != cudaSuccess)
        		    // {
        		    //     fprintf(stderr, "Failed to copy text buffer hBuffer (error code %s)!\n", cudaGetErrorString(err));
        		    //     exit(EXIT_FAILURE);
        		    // }

                    err = cudaMemcpyToSymbol(TEXT_BUFFER, hBuffer, BUFFER_SIZE);
                    if (err != cudaSuccess)
                    {
                        fprintf(stderr, "Failed to copy text buffer hBuffer (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }                    

        		    //copy text len
                    printf("[Copying text len to device...]\n");
        		    err = cudaMemcpy(d_textLen, maxCharsPtr, d_textLenSize, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess)
                    {
                        fprintf(stderr, "Failed to copy text length d_textLen (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }

        		    //copy itemsInBuffer
                    printf("[Copying num of blocks to device...]\n");
                    err = cudaMemcpy(d_textBlocks, maxBlocksInBufferPtr, d_textBlockSize, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess)
                    {
                        fprintf(stderr, "Failed to copy block size d_textBlocks (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }

        		    //copy phrase
                    printf("[Copying phrase to device...]\n");
                    err = cudaMemcpy(d_phrase, phrase, d_phraseSize, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess)
                    {
                        fprintf(stderr, "Failed to copy phrase d_phrase (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }

        		    // copy prhase len
                    printf("[Copying phrase len to device...]\n");
                    err = cudaMemcpy(d_phraseLen, &phraseLen, d_phraseLenSize, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess)
                    {
                        fprintf(stderr, "Failed to copy phrase len d_phraseLen (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }

            		// run device function
                    printf("[Initializing CUDA kernel and launching]\n");

                    // strcpy(d_phrase, phrase);
                    // d_textLen = maxChars;
                    // d_textBlocks = maxBlocksInBuffer;
                    // d_phraseLen = phraseLen;

                    // printf("d_textLen: %d\n", d_textLen);
                    // printf("d_textBlocks: %d\n", d_textBlocks);
                    // printf("d_phrase: %s\n", d_phrase);
                    // printf("d_phraseLen: %d\n", d_phraseLen);
                    // printf("d_matches: %d\n", d_matches);
                    // printf("d_text: %s\n", d_text);

                    /* DEBUG
                    cudaFree(d_text);
                    cudaFree(d_textLen);
                    cudaFree(d_phrase);
                    cudaFree(d_phraseLen);
                    cudaFree(d_textBlocks);
                    exit(0);
                    */
                    printf("size of matches: %zu\n", sizeof(matches));
                    printf("size of d_matches: %zu\n", sizeof(d_matches));

                    wordSearch<<<maxBlocksInBuffer,1>>>(d_text, maxChars, maxBlocksInBuffer, d_phrase, phraseLen, d_matches);
        			// wordSearch<<<numOfBlocks,1>>>(d_text, d_textLen, d_textBlocks, d_phrase, d_phraseLen, d_matches);
                    err = cudaGetLastError();
        		    if (err != cudaSuccess)
        		    {
        		        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        		        exit(EXIT_FAILURE);
        		    }
                    cudaFree(d_text);
                    cudaFree(d_textLen);
                    cudaFree(d_phrase);
                    cudaFree(d_phraseLen);
                    cudaFree(d_textBlocks);

                    printf("[copying device memory matches to host memory\n");
                    err = cudaMemcpy(matches, d_matches, matchesSize, cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "Failed to copy device matches array to host (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }

                    for (int i = 0; i < sizeof(matches); i++) {
                        if (matches[i] == 1) {
                            printf("1");
                        }
                    }
                    printf("\n");

                    cudaFree(d_matches);
                    exit(0);
//        		    int threadsPerBlock = 256;
//        		    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
//        		    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
//        		    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);



        			// end cuda
        			bufferCount = 0;
                    printf("[Setting hbuffer to empty via memset\n");
        			memset(&hBuffer[0], 0, sizeof(hBuffer)); // clear the buffer
                    // set the buffer ptr back to the beginning
                    printf("[Settings hBufferPtr back to the beginning\n");
        			hBufferPtr = hBuffer;

                    break;

        		} else {

        		}


    		}
    		lineId++;
    	}
    } else {
    	printf("Could not read file %s", filePath);
    	return 0;
    }

    printf("End search\n");

    fclose(file);
    free(matches);
    cudaFree(d_text);
    cudaFree(d_textLen);
    cudaFree(d_phrase);
    cudaFree(d_phraseLen);
    cudaFree(d_textBlocks);
    cudaFree(d_matches);

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

