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

    double timeSpentRunningKernel;


    int phraseLen = 0;
    for (int i = 0; phrase[i] != '\0'; i++) {
        if (i == 1000) break;
        phraseLen++;
    }


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
                // if (lineCount % 50 == 0) {
                //     printf("Line: %d\n", lineCount);
                // }

                hBufferPtr = hBuffer[lineCount]; // shift ptr to next block
                strncpy(hBufferPtr, line, MAX_CHAR_SIZE);
                lineCount++;

                /* Host text buffer hBuffer is full and ready to be copied to device TEXT_BUFFER */
                if (lineCount == MAX_TEXT_BLOCKS) {
                    printf("[Host buffer is full, running search]\n");

                    begin = clock();

                    for (int i = 0; i < MAX_TEXT_BLOCKS; i++) {
                        int strLen = 0;
                        for (int j = 0; hBuffer[i][j] != '\0'; j++) {
                            strLen++;
                        }
                        matches[i] = 0;                        
                        for (int charIdx = 0; charIdx < strLen; charIdx++) {
                            if (hBuffer[i][charIdx] == phrase[0]) {
                                int wordLen = 1;
                                charIdx++;
                                while (hBuffer[i][charIdx] == phrase[wordLen] && wordLen < phraseLen) {
                                    charIdx++;
                                    wordLen++;
                                    if (wordLen == phraseLen) {
                                        matches[i] = 1;
                                        break;
                                    }
                                }
                            }

                        }

                        

                    }

                    end = clock();

                    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
                    timeSpentRunningKernel = time_spent;


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

    printf("End sequential search\n");

    fclose(file);
    free(matches);

    /* Performance Results */
    printf("|-------Performance-------|\n");
    printf("Time spent in running search: %fs\n", timeSpentRunningKernel);
    printf("|-------/Performance-------|\n");


    printf("Done\n");
    return 0;
}

