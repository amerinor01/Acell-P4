// Suma de vectores usando la memoria global de la GPU

#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <helper_timer.h>

StopWatchInterface *hTimer = NULL;
StopWatchInterface *kTimer = NULL;

typedef unsigned int *vector;


// Funciones para generar los datos de los vectores

void genVectors(vector A, vector B,vector D, unsigned int size){
    unsigned int i;
    for(i=0; i<size;i++){
        A[i] = i;
        B[i] = i;
        D[i] = i;
    }
}


// Función para mostrar los datos de los vectores

void printData(vector A, unsigned int n){
    int i = 0;

    for(i = 0; i < n; i++)
        printf("%u ", A[i]);
}

// CUDA Kernels
__global__ void SumVectorSec(vector A, vector B, vector C, vector D, unsigned int n, int stream){
    int i = 0;
    unsigned int k = (threadIdx.x + blockIdx.x * blockDim.x)* stream;
    for(i = 0; i< stream; i++){
            C[D[i+k]] = A[D[i+k]] + B[D[i+k]];
    }
}

// ------------------------
// MAIN function
// ------------------------
int main(int argc, char **argv)
{

   unsigned int n = 0;
   int blockSize;
   int stream;
   int debug;
   if (argc == 5)
     {
        blockSize = atoi(argv[1]);
        stream = atoi(argv[2]);
        n = (unsigned) atoi(argv[3]);
        debug = atoi(argv[4]);
     }
   else
     {
       printf("Sintaxis: \n");

       exit(0);
     }
   float timerValue;
   double ops;

   // Definir vectores en el host
   vector A;
   vector B;
   vector C;
   vector D;

   // Definir vectores en el device
   vector d_A;

   vector d_B;

   vector d_C;

   vector d_D;
   // Timers: para medir el tiempo total
   sdkCreateTimer(&hTimer);
   sdkResetTimer(&hTimer);
   sdkStartTimer(&hTimer);

   // Reservar memoria en el host para los vectores A, B y C, y asignar valores a A y B

    A = (unsigned int*) malloc(n * sizeof(unsigned int));
    B = (unsigned int*) malloc(n * sizeof(unsigned int));
    C = (unsigned int*) malloc(n * sizeof(unsigned int));
    D = (unsigned int*) malloc(n * sizeof(unsigned int));
    genVectors(A,B,D,n);

    if(debug){
    printf("Vector A: ");
    printData(A,n);
    printf("\n");
    }

    if(debug){
    printf("Vector B: ");
    printData(B,n);
    printf("\n");
    }
    if(debug){
    printf("Vector D: ");
    printData(D,n);
    printf("\n");
    }
   // Reservar memoria en el device para los vectores
    cudaMalloc((void **)&d_A, n* sizeof(unsigned int));
    cudaMalloc((void **)&d_B, n* sizeof(unsigned int));
    cudaMalloc((void **)&d_C, n* sizeof(unsigned int));
    cudaMalloc((void **)&d_D, n* sizeof(unsigned int));
   // Copiar los vectores A y B desde host a device
    cudaMemcpy(d_A, A, n* sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n* sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, n* sizeof(unsigned int), cudaMemcpyHostToDevice);

   // Definir el grid y los bloques de hilos
    dim3 dimBlock(blockSize);
    dim3 dimGrid( ceil(n / (float)blockSize));

   // Timers: para medir el tiempo del kernel
   sdkCreateTimer(&kTimer);
   sdkResetTimer(&kTimer);
   sdkStartTimer(&kTimer);

   // Ejecutar el kernel
    SumVectorSec<<<dimGrid, dimBlock>>>(d_A,d_B,d_C,d_D,n,stream);


   cudaDeviceSynchronize();

   sdkStopTimer(&kTimer);

   // Copiar el vector C desde device a host

    cudaMemcpy(C, d_C, n* sizeof(unsigned int), cudaMemcpyDeviceToHost);
   // Liberar memoria en el device y en el host
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    if(debug){
        printf("\nVector C: ");
        printData(C, n);
        printf("\n");
    }
   sdkStopTimer(&hTimer);

   timerValue = sdkGetTimerValue(&kTimer);
   timerValue = timerValue / 1000;
   sdkDeleteTimer(&kTimer);
   printf("Tiempo kernel: %f s\n", timerValue);

   timerValue = sdkGetTimerValue(&hTimer);
   timerValue = timerValue / 1000;
   sdkDeleteTimer(&hTimer);
   printf("Tiempo total: %f s\n", timerValue);
   ops = n / timerValue;
   printf("GFLOPS %f \n",(ops*n)/1000000000);


   return 0;
}
