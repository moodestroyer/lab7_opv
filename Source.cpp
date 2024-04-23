#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace cv;


const int width = 800, height = 600;  
const double x_min = -2.0, x_max = 2.0;  
const double y_min = -1.5, y_max = 1.5;
const int max_iter = 1000;  
const double threshold = 2.0;  

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    const int local_height = height / size;
    const int local_y_start = rank * local_height;
    const int local_y_end = local_y_start + local_height;

    
    Mat mandelbrot_set(local_height, width, CV_8U);

    for (int j = local_y_start; j < local_y_end; ++j) {
        for (int i = 0; i < width; ++i) {
            double x0 = x_min + (x_max - x_min) * i / (width - 1);
            double y0 = y_min + (y_max - y_min) * j / (height - 1);
            double x = 0.0, y = 0.0;
            int iteration = 0;

            while (x * x + y * y < 4 && iteration < max_iter) {
                double x_temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = x_temp;
                ++iteration;
            }

            mandelbrot_set.at<uchar>(j - local_y_start, i) = 255 * iteration / max_iter;
        }
    }

    
    Mat full_mandelbrot_set(height, width, CV_8U);
    MPI_Gather(mandelbrot_set.data, local_height * width, MPI_UNSIGNED_CHAR, full_mandelbrot_set.data, local_height * width, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

   
    if (rank == 0) {
        namedWindow("Mandelbrot ", WINDOW_NORMAL);
        imshow("Mandelbrot ", full_mandelbrot_set);
        waitKey(0);
    }

    MPI_Finalize();
    return 0;
}

