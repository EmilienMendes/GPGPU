#include "student_view_fully_optimized.hpp"

// Calcul des angles avec des floats pour l'optimisation
__device__ float angle_calc_fully_optimized(float xa,float ya,uint8_t za,float xb,float yb,uint8_t zb){
    float numerator = (float) zb - (float) za;
    float denominator = sqrt( ( xb - xa ) * ( xb - xa ) + ( yb - ya ) * ( yb - ya ) );
    return atan(numerator/denominator);
}


__global__ void  kernel_fully_optimized_view_test( const uint8_t *const dev_inPtr, uint8_t *const dev_outPtr, const uint32_t width, const uint32_t height,
                                         int32_t cx,int32_t cy,uint8_t cz )
{
    // Indexation de chaque thread pour correspondre a un pixel
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t index = x+y*width;
    
    // Coloration du centre en rouge 
    if(x == cx && y == cy){
        dev_outPtr[index] = 155;
    }
    else if(index < width*height && x < width && y < height) {
        
        float Fcx = (float) cx;
        float Fcy = (float)cy;
        float px =(float) x;
        float py =  (float)y;
        uint8_t z = dev_inPtr[index];
        // Calcul de l'angle entre le point et le centre
        float base_angle  = angle_calc_fully_optimized(Fcx,Fcy,cz,px,py,z);
        // Rasterisation de la droite
        uint32_t D  = max(abs(px - Fcx), abs(py - Fcy));
        float tmpD  = (float) D;
        float stepx = (px - Fcx) / tmpD;
        float stepy = (py - Fcy) / tmpD;

        // Construction des D cases entre le point p et c
        bool isVisible = true;
        uint32_t k = 1;
        while (k < D && isVisible ){
            // Construction du point intermediaire
            float xi = Fcx + stepx * (float)k;
            float yj = Fcy + stepy * (float)k;
            int current_index  = (int)xi+(int)yj*width;
            uint8_t zi = dev_inPtr[current_index];
            // Calcul de l'angle entre le point intermediarire et le centre
            float current_angle = angle_calc_fully_optimized(Fcx,Fcy,cz,xi,yj,zi);
            if (current_angle >= base_angle) {
                isVisible = false;
            }
            k++;
        }
        if(isVisible){
            dev_outPtr[index] = 255;
        }
        else{
            dev_outPtr[index] = 0;
        }
    }
}

float view_test_GPU_fully_optimized(los::Heightmap in, los::Heightmap &out, Point center)
{
    const size_t sizeImg = in.getSize();

    // Table de pixel sur l'hote
    uint8_t *inPtr	= in.getPtr();
    uint8_t *outPtr	= out.getPtr();

    // Table de pixel sur GPU
    uint8_t *dev_inPtr;
    uint8_t *dev_outPtr;

    const uint32_t width	= in.getWidth();
    const uint32_t height	= in.getHeight();

    // Allocation memoire sur le GPU
    HANDLE_ERROR(cudaMalloc((void **)&dev_inPtr,sizeImg*sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_outPtr,sizeImg*sizeof(uint8_t)));

    // Copie de la memoire de l'hote sur le GPU
    HANDLE_ERROR(cudaMemcpy( dev_inPtr,inPtr, sizeImg*sizeof(uint8_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy( dev_outPtr,outPtr, sizeImg*sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Configuation du kernel
    dim3 dimBlock(NB_THREAD_VIEW_FULLY_OPTIMIZED,NB_THREAD_VIEW_FULLY_OPTIMIZED);
    dim3 dimGrid(ceil((float) width/(float)dimBlock.x),ceil((float) height/(float)dimBlock.y));

    ChronoGPU chr;
    chr.start();
    // Appel du kernel
    kernel_fully_optimized_view_test<<<dimGrid,dimBlock>>>(dev_inPtr,dev_outPtr,width,height,center.getX(),center.getY(),center.getZ());
    chr.stop();

    // Copie du GPU vers l'hote
    HANDLE_ERROR(cudaMemcpy( outPtr,dev_outPtr, sizeImg*sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // Liberation de la memoire sur GPU
    HANDLE_ERROR(cudaFree(dev_inPtr));
    HANDLE_ERROR(cudaFree(dev_outPtr));

    return chr.elapsedTime();
}



