#include "student_tiled_naive.hpp"

/*
    La fonction reprend le meme principe que sur CPU
    1 pixel -> 1 thread
    On fais le maximum entre la valeur du pixel d'entree 
    et la valeur du pixel dans la sous image de sortie
*/
__global__ void kernel_optimized_tiled(const uint8_t *const dev_inPtr,  uint32_t * dev_outPtr,const uint32_t inWidth, const uint32_t inHeight,
                                   const uint32_t outWidth, const uint32_t outHeight)
{
    // Index du thread correspondant au pixel d'entree    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t indexIn = x + y * inWidth;
    
    int ratioX = ceilf(inWidth / outWidth); // Nombre de pixels dans le sous bloc en x
    int ratioY =  ceilf(inHeight / outHeight); // Nombre de pixels dans le sous bloc en y
    
    // Index du thread correspondant au pixel de sortie
    int32_t xi = x / ratioX;
    int32_t yj = y / ratioY;
    int32_t indexOut = xi + yj * outWidth;

    if (xi < outWidth && yj < outHeight)
    {
        uint32_t pixValue = dev_inPtr[indexIn];
        /* 
            On verifie si le pixel n'est pas deja plus petit que
            la valeur dans la sortie
        */
        if(pixValue > dev_outPtr[indexOut]){
            /*  On fait quand meme le max dans le cas ou le if d'un autre
                thread ne serait pas passer
            */
            atomicMax(&dev_outPtr[indexOut], pixValue);
        }
    }
}

float tiled_GPU_optimized(los::Heightmap in, los::Heightmap &out)
{

    const size_t inSizeImg = in.getSize();
    const size_t outSizeImg = out.getSize();

    // Table de pixel pour l'hote
    uint8_t *inPtr = in.getPtr();
    uint8_t *outPtr = out.getPtr();

    // Table de pixel temporaire utilise pour atomic max
    uint32_t *tmpPtr = new uint32_t[outSizeImg];
    for (int i = 0; i < outSizeImg; i++)
    {
        tmpPtr[i] = 0;
    }

    // Table de pixel sur GPU
    uint8_t *dev_inPtr;
    uint8_t *dev_outPtr;
    uint32_t *dev_tmpPtr;

    const uint32_t inWidth = in.getWidth();
    const uint32_t inHeight = in.getHeight();
    const uint32_t outWidth = out.getWidth();
    const uint32_t outHeight = out.getHeight();

    // Allocation memoire GPU
    HANDLE_ERROR(cudaMalloc((void **)&dev_inPtr, inSizeImg * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_outPtr, outSizeImg * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_tmpPtr, outSizeImg * sizeof(uint32_t)));

    // Copie de l'hote sur le GPU 
    HANDLE_ERROR(cudaMemcpy(dev_inPtr, inPtr, inSizeImg * sizeof(uint8_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_outPtr, outPtr, outSizeImg * sizeof(uint8_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_tmpPtr, tmpPtr, outSizeImg  * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Configuration kernel
    dim3 dimBlock(NB_THREAD_TILED_OPTIMIZED,NB_THREAD_TILED_OPTIMIZED); // Nombre de threads par block
    dim3 dimGrid(ceil((float)inWidth / (float)dimBlock.x), ceil((float)inHeight / (float)dimBlock.y));
    printf("%d,%d\n",dimBlock.x,dimBlock.y);
    ChronoGPU chr;
    chr.start();
    // Kernel
    kernel_optimized_tiled<<<dimGrid, dimBlock>>>(dev_inPtr,dev_tmpPtr, inWidth, inHeight, outWidth, outHeight);
    chr.stop();

    // Copy du GPU sur l'hote
    HANDLE_ERROR(cudaMemcpy(tmpPtr, dev_tmpPtr, outSizeImg * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // Copie du tableau temporaire sur le tableau de sortie
    for(int i = 0;i<outSizeImg;i++){
        outPtr[i] = tmpPtr[i] ;
    }

    // Liberation de la memoire sur le GPU
    HANDLE_ERROR(cudaFree(dev_inPtr));
    HANDLE_ERROR(cudaFree(dev_outPtr));
    HANDLE_ERROR(cudaFree(dev_tmpPtr));

    return chr.elapsedTime();
}