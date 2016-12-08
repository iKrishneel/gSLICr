// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#include <time.h>
#include <stdio.h>

#include "gSLICr_Lib/gSLICr.h"
#include "NVTimer.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"

using namespace std;
using namespace cv;

void load_image(const Mat& inimg, gSLICr::UChar4Image* outimg) {
    gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

    for (int y = 0; y < outimg->noDims.y; y++)
       for (int x = 0; x < outimg->noDims.x; x++) {
          int idx = x + y * outimg->noDims.x;
          outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
          outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
          outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
       }
}

void load_image(const gSLICr::UChar4Image* inimg, Mat& outimg) {
    const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);
     for (int y = 0; y < inimg->noDims.y; y++)
        for (int x = 0; x < inimg->noDims.x; x++) {
           int idx = x + y * inimg->noDims.x;
           outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
           outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
           outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
        }
}

void edgeBasedClusterFilters(const cv::Mat, const int *);

int main(int argc, const char *argv[]) {

    // gSLICr settings
    gSLICr::objects::settings my_settings;
    my_settings.img_size.x = 448;
    my_settings.img_size.y = 448;
    my_settings.no_segs = 200;
    my_settings.spixel_size = 32;
    my_settings.coh_weight = 0.7f;
    my_settings.no_iters = 3;
    my_settings.color_space = gSLICr::XYZ;  // gSLICr::CIELAB for Lab,
                                            // or gSLICr::RGB for RGB
    my_settings.seg_method = gSLICr::GIVEN_SIZE;  // or
                                                  // gSLICr::GIVEN_NUM
                                                  // for given number
    my_settings.do_enforce_connectivity = true;  // whether or not run
                                                 // the enforce
                                                 // connectivity step
    
    // instantiate a core_engine
    gSLICr::engines::core_engine* gSLICr_engine =
       new gSLICr::engines::core_engine(my_settings);
    
    // gSLICr takes gSLICr::UChar4Image as input and out put
    gSLICr::UChar4Image* in_img =
       new gSLICr::UChar4Image(my_settings.img_size, true, true);
    gSLICr::UChar4Image* out_img =
       new gSLICr::UChar4Image(my_settings.img_size, true, true);
    
    Size s(my_settings.img_size.x, my_settings.img_size.y);
    Mat oldFrame, frame;
    Mat boundry_draw_frame; boundry_draw_frame.create(s, CV_8UC3);
    
    oldFrame = cv::imread(argv[1]);
    cv::resize(oldFrame, oldFrame,
               cv::Size(my_settings.img_size.x, my_settings.img_size.y));
    // cv::GaussianBlur(oldFrame, oldFrame, cv::Size(3, 3), 1, 0);
    
    StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);
    
    int key; int save_count = 0;
    // while (cap.read(oldFrame))
    {
       resize(oldFrame, frame, s);
       
       load_image(frame, in_img);
       
       sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
       gSLICr_engine->Process_Frame(in_img);
       sdkStopTimer(&my_timer);
       cout << "\rsegmentation in:[" << sdkGetTimerValue(&my_timer) << "]ms" << "\n";
        
       gSLICr_engine->Draw_Segmentation_Result(out_img);
		
       load_image(out_img, boundry_draw_frame);
       cv::namedWindow("segmentation", cv::WINDOW_NORMAL);
       imshow("segmentation", boundry_draw_frame);

       
       //! get mask
       const gSLICr::IntImage *gmask = gSLICr_engine->Get_Seg_Res();
       const int* inimg_ptr = gmask->GetData(MEMORYDEVICE_CPU);
              
       cv::RNG rng(0xFFFFFFFF);

       std::vector<cv::Scalar> colors(my_settings.no_segs);
       for (int i = 0; i < colors.size(); i++) {
          cv::Scalar c = cv::Scalar(
             std::abs(static_cast<float>(rng.uniform(0.0, 1.0))),
             std::abs(static_cast<float>(rng.uniform(0.0, 1.0))),
             std::abs(static_cast<float>(rng.uniform(0.0, 1.0))));
          colors[i] = c;
       }
       
       cv::Mat im_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
       for (int y = 0; y < im_mask.rows; y++) {
          for (int x = 0; x < im_mask.cols; x++) {
             int value = inimg_ptr[x + (y * im_mask.cols)];
             cv::Scalar color = colors[value];
             im_mask.at<cv::Vec3b>(y, x)[0] = color.val[0] * 255;
             im_mask.at<cv::Vec3b>(y, x)[1] = color.val[1] * 255;
             im_mask.at<cv::Vec3b>(y, x)[2] = color.val[2] * 255;
          }
       }

       edgeBasedClusterFilters(oldFrame, inimg_ptr);
       
       
       cv::namedWindow("mask", cv::WINDOW_NORMAL);
       cv::imshow("mask", im_mask);
       
       cv::waitKey(0);
       
       // key = waitKey(1);
       // if (key == 27) break;
       // else if (key == 's')
       // {
       // 	char out_name[100];
       // 	sprintf(out_name, "seg_%04i.pgm", save_count);
       // 	gSLICr_engine->Write_Seg_Res_To_PGM(out_name);
       // 	sprintf(out_name, "edge_%04i.png", save_count);
       // 	imwrite(out_name, boundry_draw_frame);
       // 	sprintf(out_name, "img_%04i.png", save_count);
       // 	imwrite(out_name, frame);
       // 	printf("\nsaved segmentation %04i\n", save_count);
       // 	save_count++;
       // }
    }

    destroyAllWindows();
    return 0;
}


void edgeBasedClusterFilters(const cv::Mat image,
                             const int *gmask) {
    if (image.empty()) {
       cout << "EMPTY INPUT IMAGE"  << "\n";
       return;
    }

    double low_thresh = 20.0;
    double high_thresh = 100.0;
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny =
       cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, 3, true);
    cv::cuda::GpuMat d_edges;
    cv::cuda::GpuMat d_gray;
    cv::cuda::cvtColor(cv::cuda::GpuMat(image), d_gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::cuda::Filter> gauss =
       cv::cuda::createGaussianFilter(d_gray.type(), -1, cv::Size(9, 9), 1, 1);
    gauss->apply(d_gray, d_gray);
    canny->detect(d_gray, d_edges);
    
    cv::Mat edges;
    d_edges.download(edges);


    //! traverse the edges
    std::vector<int> sp_indices;
    int index = -1;
    int prev_index = -1;
    for (int j = 0; j < edges.rows; j++) {
       for (int i = 0; i < edges.cols; i++) {
          index = gmask[i + (j * edges.cols)];
          if (edges.at<uchar>(j, i) > 0 && index != prev_index) {
             sp_indices.push_back(index);
          }
          prev_index = index;
       }
    }

    cout << "SIZE: " << sp_indices.size()  << "\n";
    
    cv::imshow("edges", edges);

}
