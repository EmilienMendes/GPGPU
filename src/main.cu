#include <iostream>
#include <iomanip> 
#include <string>

#include "../utils/ppm.hpp"
#include "../utils/geometry.hpp"
#include "CPU/reference.hpp"
#include "view_test/student_view_naive.hpp"
#include "view_test/student_view_optimized.hpp"
#include "tiled/student_tiled_naive.hpp"
#include "tiled/student_tiled_optimized.hpp"
#include "view_test/student_view_double_kernel.hpp"
#include "view_test/student_view_fully_optimized.hpp"


// Sauvegarde de l'image avec un nouveau nom
void saveImage(std::string fileName,std::string suffix,los::Heightmap outImage){
    fileName.erase( fileName.end() - 4, fileName.end() ); // erase .ppm
    fileName+=suffix;
    outImage.saveTo( fileName.c_str());
} 


int main( int argc, char **argv )
{


    // ================================================================================================================
    // Image d'entree
    char fileName [1024] = R"(..\..\result\1.input.ppm)";
    //char fileName [1024] = R"(..\..\result\limousin-full.ppm)";
    std::cout << "Loading image: " << fileName << std::endl;
    const los::Heightmap input( fileName );

    // ================================================================================================================
    // Parametres pour view test

    const uint32_t width	= input.getWidth();
    const uint32_t height	= input.getHeight();
    const uint32_t center_x = 245;
    const uint32_t center_y = 497;
    // const uint32_t center_x = width/2;
    // const uint32_t center_y = height/2;
    Point center =  Point(center_x,center_y,input.getPixel(center_x,center_y));
    std::cout << "Image has " << width << " x " << height << " pixels" << std::endl;
    char resultFileView[1024] =  R"(..\..\result\2.result.ppm)";
    std::cout << "Loading image: " << resultFileView << std::endl;
    const los::Heightmap resultView( resultFileView );

    // ================================================================================================================
    // Parametre pour tiled 

    const uint32_t ratioX = 10;
    const uint32_t ratioY = 10;
    char resultFileTiled[1024] =  R"(..\..\result\3.tiled.ppm)";
    std::cout << "Loading image: " << resultFileTiled << std::endl;
    const los::Heightmap resultTiled( resultFileTiled );

    // ================================================================================================================
    // Image de sortie

    los::Heightmap outCPUView( width, height );
    los::Heightmap outCPUTiled( ratioX, ratioY );

    los::Heightmap outGPUViewNaive( width, height );
    los::Heightmap outGPUViewOptimized( width, height );
    los::Heightmap outGPUViewFullyOptimized( width, height );
    los::Heightmap outGPUViewDoubleKernel( width, height );

    
    los::Heightmap outGPUTiledNaive( ratioX, ratioY );
    los::Heightmap outGPUTiledOptimized( ratioX, ratioY );


    // ================================================================================================================
    // Version sequentielle

    std::cout << "============================================"	<< std::endl;
    std::cout << "  Version sequentielle pour CPU View Test   "	<< std::endl;
    std::cout << "============================================"	<< std::endl;

    const float timeCPU_view_test = view_test_CPU( input, outCPUView,center);
    saveImage(resultFileView,"CPU.ppm",outCPUView);
   
    std::cout << "-> Done : " << timeCPU_view_test << " ms" << std::endl << std::endl;

    std::cout << "============================================"	<< std::endl;
    std::cout << "      Sequential version on CPU Tiled       "	<< std::endl;
    std::cout << "============================================"	<< std::endl;
    
    const float timeCPU_tiled = tiled_CPU( input, outCPUTiled);
    saveImage(resultFileTiled,"CPU.ppm",outCPUTiled);
    std::cout << "-> Done : " << timeCPU_tiled << " ms" << std::endl << std::endl;

   
    
    // ================================================================================================================
    // GPU CUDA View Test

    std::cout << "============================================"	<< std::endl;
    std::cout << "       Parallel version on GPU View Test    "	<< std::endl;
    std::cout << "============================================"	<< std::endl;
    
    // Naive
    const float timeGPU_view_test_naive = view_test_GPU_naive( input, outGPUViewNaive, center);
    saveImage(resultFileView,"GPU_naive.ppm",outGPUViewNaive);
    std::cout << "Naive -> Done : " << timeGPU_view_test_naive << " ms" << std::endl << std::endl;
    
    // Optimized
    const float timeGPU_view_test_optimized = view_test_GPU_optimized( input, outGPUViewOptimized, center);
    saveImage(resultFileView,"GPU_optimized.ppm",outGPUViewOptimized);
    std::cout << "Optimized -> Done : " << timeGPU_view_test_optimized << " ms" << std::endl << std::endl;

    // Fully Optimized
    const float timeGPU_view_test_fully_optimized = view_test_GPU_fully_optimized( input, outGPUViewFullyOptimized, center);
    saveImage(resultFileView,"GPU_fully_optimized.ppm",outGPUViewFullyOptimized);
    std::cout << "Optimized -> Done : " << timeGPU_view_test_fully_optimized << " ms" << std::endl << std::endl;

    // Double Kernel
    const float timeGPU_view_test_double_kernel = view_test_double_kernel( input, outGPUViewDoubleKernel, center);
    saveImage(resultFileView,"GPU_double_kernel.ppm",outGPUViewDoubleKernel);
    std::cout << "Optimized -> Done : " << timeGPU_view_test_double_kernel << " ms" << std::endl << std::endl;

    // ================================================================================================================
    // GPU CUDA Tiled
    std::cout << "============================================"	<< std::endl;
    std::cout << "       Parallel version on GPU Tiled        "	<< std::endl;
    std::cout << "============================================"	<< std::endl;
    
    // Naive
    const float timeGPU_tiled_naive = tiled_GPU_naive( input, outGPUTiledNaive);
    saveImage(resultFileTiled,"GPU_naive.ppm",outGPUTiledNaive);
    std::cout << "Naive -> Done : " << timeGPU_tiled_naive << " ms" << std::endl << std::endl;

    // Optimized
    const float timeGPU_tiled_optimized = tiled_GPU_optimized( input, outGPUTiledOptimized);
    saveImage(resultFileTiled,"GPU_optimized.ppm",outGPUTiledOptimized);
    std::cout << "Optimized -> Done : " << timeGPU_tiled_optimized << " ms" << std::endl << std::endl;
    
    // ================================================================================================================
    // Verification des resultats pour tiled
    std::cout << "============================================"	<< std::endl;
    std::cout << "       Checking results for Tiled           "	<< std::endl;
    std::cout << "============================================"	<< std::endl;
    bool error =false;
    for(int i = 0;i<outGPUTiledNaive.getWidth() ;i++){
        for(int j = 0;j<outGPUTiledNaive.getHeight() && !error;j++){

            if(outCPUTiled.getPixel(i,j) !=  outGPUTiledNaive.getPixel(i,j)){
                std::cout <<"CPU/GPU Naive/GPU Optimized " << std::endl;
                std::cout<< "Erreur pour Naive : [" << i << "][" << j << "] : " << (uint32_t)outCPUTiled.getPixel(i,j) << "/" <<  (uint32_t) outGPUTiledNaive.getPixel(i,j) << "/" <<  (uint32_t) outGPUTiledOptimized.getPixel(i,j) << std::endl;
                error = true;
            }
            else if(outCPUTiled.getPixel(i,j) !=  outGPUTiledOptimized.getPixel(i,j)){
                std::cout <<"CPU/GPU Naive/GPU Optimized " << std::endl;
                std::cout<< " Erreur pour optimized [" << i << "][" << j << "] : " << (uint32_t)outCPUTiled.getPixel(i,j) << "/" <<  (uint32_t) outGPUTiledNaive.getPixel(i,j) << "/" <<  (uint32_t) outGPUTiledOptimized.getPixel(i,j) << std::endl;
                error = true;
            }
        }
    }
    if(!error) {
        std::cout<< "                Pas d'erreur                " <<std::endl;
    }
    

    // ================================================================================================================
    std::cout << "============================================" << std::endl;
    std::cout << "         Recapitulation des temps           " << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "-> CPU (View Test) : " << std::fixed << std::setprecision(2) << timeCPU_view_test << " ms" << std::endl;
    std::cout << "-> GPU (View Test Naive) : " << std::fixed << std::setprecision(2) << timeGPU_view_test_naive << " ms" << std::endl;
    std::cout << "-> GPU (View Test Optimized) : " << std::fixed << std::setprecision(2) << timeGPU_view_test_optimized << " ms" << std::endl;
    std::cout << "-> GPU (View Test Fully Optimized) : " << std::fixed << std::setprecision(2) << timeGPU_view_test_fully_optimized << " ms" << std::endl;
    std::cout << "-> GPU (Double Kernel ) : " << std::fixed << std::setprecision(2) << timeGPU_view_test_double_kernel << " ms" << std::endl;
    std::cout << "-> CPU (Tiled) : " << std::fixed << std::setprecision(2) << timeCPU_tiled << " ms" << std::endl;
    std::cout << "-> GPU (Tiled Naive ) : " << std::fixed << std::setprecision(2) << timeGPU_tiled_naive << " ms" << std::endl;
    std::cout << "-> GPU (Tiled Optimized ) : " << std::fixed << std::setprecision(2) << timeGPU_tiled_optimized << " ms" << std::endl;
    
    return EXIT_SUCCESS;
}