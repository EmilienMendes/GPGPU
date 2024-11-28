#include "student_view_optimized.hpp"

    
// Fonction avec des doubles pour l'optimisation
__device__ double angle_calc_optimized(float xa,float ya,uint8_t za,float xb,float yb,uint8_t zb){
    double numerator = (double) zb - (double) za;
    double denominator = sqrt( ( xb - xa ) * ( xb - xa ) + ( yb - ya ) * ( yb - ya ) );
    return atan(numerator/denominator);
}
/*
    On combinera deux optimisations en une :
    La premiere grid stride loop qui fonctionne sur une grande image
    La deuxieme qui partage les calculs d'angle entre les differents threads du bloc
*/
__global__ void  kernel_optimized_view_test( const uint8_t *const dev_inPtr, uint8_t *const dev_outPtr, const uint32_t width, const uint32_t height,
                                         int32_t cx,int32_t cy,uint8_t cz )
{
    for(int y = blockIdx.y * blockDim.y + threadIdx.y;y<width;y+=blockDim.y*gridDim.y){
        for(int x = blockIdx.x*blockDim.x+threadIdx.x;x<height;x+=blockDim.x*gridDim.x){
            // Creation d'un tableau d'angle partage entre tous les threads du blocs            
            __shared__ double cache[NB_THREAD_VIEW_OPTIMIZED*NB_THREAD_VIEW_OPTIMIZED];
            // Indexation du threads par rapport au tableau en cache
            int32_t my_index = threadIdx.x + threadIdx.y * NB_THREAD_VIEW_OPTIMIZED;
            cache[my_index] = 0;
            // Remplissage du tableau
            __syncthreads();
            atomicAdd(&cache[my_index],angle_calc_optimized(cx,cy,cz,x,y,dev_inPtr[x+y*width]));
            __syncthreads();

            // Valeurs limites a ne pas depasser pour rester dans le cache
            int32_t min_value_x = x - threadIdx.x + 0 ;
            int32_t max_value_x = min_value_x + NB_THREAD_VIEW_OPTIMIZED ;
            int32_t min_value_y = y - threadIdx.y + 0;
            int32_t max_value_y = min_value_y + NB_THREAD_VIEW_OPTIMIZED ;
            
            int32_t index = x+y*width;
            float Fcx = (float) cx;
            float Fcy = (float)cy;
            float px =(float) x;
            float py =  (float)y;
            double base_angle  = cache[my_index]  ;
            uint32_t D  = max(abs(px - Fcx), abs(py - Fcy));
            float tmpD  = (float) D;
            float stepx = (px - Fcx) / tmpD;
            float stepy = (py - Fcy) / tmpD;

            // Construcions des D cases entre le point et le centre c
            bool isVisible = true;
            double current_angle;
            uint32_t k = 1;
            while (k < D && isVisible ){
                // Construction du point
                int xi = cx + stepx * k;
                int yj = cy + stepy * k;
                // Indexation de la valeur du cache par rapport au tableau reel
                int32_t current_index_cache  = (xi-min_value_x) + (yj-min_value_y) * NB_THREAD_VIEW_OPTIMIZED ;
                // Si le point est dans les limites du cache, on utilise la valeur
                if(xi >= min_value_x && x <= max_value_x && yj >= min_value_y && yj <= max_value_y && current_index_cache < NB_THREAD_VIEW_OPTIMIZED * NB_THREAD_VIEW_OPTIMIZED  ){
                    current_angle = cache[current_index_cache] ;
                }
                // Calcul de l'angle qui est en dehors du cache
                else{ 
                   current_angle = angle_calc_optimized(cx,cy,cz,xi,yj,dev_inPtr[xi+yj*width]) ;
                } 
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
}

float view_test_GPU_optimized(los::Heightmap in, los::Heightmap &out, Point center)
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

    // Configuration du  kernel
    dim3 dimBlock(NB_THREAD_VIEW_NAIVE,NB_THREAD_VIEW_NAIVE);
    dim3 dimGrid(ceil((float) width/(float)dimBlock.x),ceil((float) height/(float)dimBlock.y));


    ChronoGPU chr;
    chr.start();
    // Appel du kernel
    kernel_optimized_view_test<<<dimGrid,dimBlock>>>(dev_inPtr,dev_outPtr,width,height,center.getX(),center.getY(),center.getZ());
    chr.stop();

    // Copie du GPU vers l'hote
    HANDLE_ERROR(cudaMemcpy( outPtr,dev_outPtr, sizeImg*sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // Liberation de la memoire sur GPU
    HANDLE_ERROR(cudaFree(dev_inPtr));
    HANDLE_ERROR(cudaFree(dev_outPtr));

    return chr.elapsedTime();
}



