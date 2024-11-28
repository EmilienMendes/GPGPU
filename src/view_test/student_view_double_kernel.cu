#include "student_view_double_kernel.hpp"

// Calcul de l'angle entre le point et le centre
__device__ float angle_calc_double_kernel(float xa,float ya,uint8_t za,float xb,float yb,uint8_t zb){
    float numerator = (float) zb - (float) za;
    float denominator = sqrt( ( xb - xa ) * ( xb - xa ) + ( yb - ya ) * ( yb - ya ) );
    return atan(numerator/denominator);
}

// Calcul de tous les angles entre chaque point et le centre
__global__ void kernel_angle_calc(const uint8_t *in, uint32_t width, uint32_t height, int32_t cx, int32_t cy, uint8_t cz, float *angles_tab){
    // Indexation du thread pour correspondre a un pixel
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = x + y * width;
    if(tid< width*height) {
        // Calcul de l'angle du point courant
        angles_tab[tid] = angle_calc_double_kernel(cx, cy, cz, x, y, in[tid]);
    }
}

__global__ void kernel_view_test_tab(uint8_t *out, uint32_t width, uint32_t height, int32_t cx, int32_t cy, float *angles_tab){
    // Index du thread pour correspondre a un pixel
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = x + y * width;

    if(tid < width*height && x < width && y < height) {
        // On met les points et le centre dans des float pour avoir une meilleur precision
        float Fcx = (float) cx;
        float Fcy = (float)cy;
        float px =(float) x;
        float py =  (float)y;
        
        // Angle point courant
        float base_angle  = angles_tab[tid];

        // Rasterisation de la droite
        uint32_t D  = max(abs(px - Fcx), abs(py - Fcy));
        float tmpD  = (float) D;
        float stepx = (px - Fcx) / tmpD;
        float stepy = (py - Fcy) / tmpD;

        // Construction des D cases entre le point p et le centre c
        bool isVisible = true;
        uint32_t k = 1;
        while (k < D && isVisible ){
            float xi = Fcx + stepx * (float)k;
            float yj = Fcy + stepy * (float)k;
            int current_index  = (int) xi + (int) yj * width;
            float current_angle = angles_tab[current_index];
            if (current_angle >= base_angle) {
                isVisible = false;
            }
            k++;
        }
        if(isVisible){
            out[tid] = 255;
        }
        else{
            out[tid] = 0;
        }
    }
}


float view_test_double_kernel(los::Heightmap in, los::Heightmap &out , Point center){

    const size_t sizeImg = in.getSize();

    // Table de pixel sur l'hote
    uint8_t *inPtr	= in.getPtr();
    uint8_t *outPtr	= out.getPtr();

    // Table de pixel sur GPU
    uint8_t *dev_inPtr;
    uint8_t *dev_outPtr;

    // Tableau des angles entre les points et le centre sur l'hote
    float *angles_tab = new float[sizeImg];
    for(int i = 0; i<sizeImg;i++){
        angles_tab[i] = 0;
    }

    // Tableau des angles sur le GPU
    float *dev_angles_tab;

    const uint32_t width = in.getWidth();
    const uint32_t height = in.getHeight();

    // Allocation memoire sur le GPU
    HANDLE_ERROR(cudaMalloc((void **)&dev_inPtr,sizeImg*sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_outPtr,sizeImg*sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_angles_tab,sizeImg*sizeof(float)));

    // Copie de la memoire de l'hote sur le GPU
    HANDLE_ERROR(cudaMemcpy( dev_inPtr,inPtr, sizeImg*sizeof(uint8_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy( dev_outPtr,outPtr, sizeImg*sizeof(uint8_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy( dev_angles_tab,angles_tab, sizeImg*sizeof(float), cudaMemcpyHostToDevice));

    // Configuation du kernel
    dim3 dimBlock(NB_THREAD_VIEW_DOUBLE_KERNEL,NB_THREAD_VIEW_DOUBLE_KERNEL);
    dim3 dimGrid(ceil((float) width/(float)dimBlock.x),ceil((float) height/(float)dimBlock.y));

    ChronoGPU chr;
    chr.start();
    // Appel du kernel
    kernel_angle_calc<<<dimGrid,dimBlock>>>(dev_inPtr, width, height, center.getX(), center.getY(), center.getZ(), dev_angles_tab);
    kernel_view_test_tab<<<dimGrid,dimBlock>>>(dev_outPtr, width, height, center.getX(), center.getY(), dev_angles_tab);

    chr.stop();

    // Copy du GPU vers l'hote
    HANDLE_ERROR(cudaMemcpy( outPtr,dev_outPtr, sizeImg*sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // Liberation de la memoire sur GPU
    HANDLE_ERROR(cudaFree(dev_inPtr));
    HANDLE_ERROR(cudaFree(dev_outPtr));
    HANDLE_ERROR(cudaFree(dev_angles_tab));

    return chr.elapsedTime();
}