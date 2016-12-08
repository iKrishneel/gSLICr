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

#include <vector>
#include <algorithm>

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


void edgeBasedClusterFilters(std::vector<bool> &, const cv::Mat, const int *);
void getBoundingRects(const cv::Mat,
                      const std::vector<std::vector<cv::Point2f> > &);
   

int main(int argc, const char *argv[]) {

    // gSLICr settings
    gSLICr::objects::settings my_settings;
    my_settings.img_size.x = 448;
    my_settings.img_size.y = 448;
    my_settings.no_segs = 300;
    my_settings.spixel_size = 16;
    my_settings.coh_weight = 0.8f;
    my_settings.no_iters = 3;
    my_settings.color_space = gSLICr::XYZ;  // gSLICr::CIELAB for Lab,
                                            // or gSLICr::RGB for RGB
    my_settings.seg_method = gSLICr::GIVEN_NUM;  // or
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
       
       sdkResetTimer(&my_timer);
       sdkStartTimer(&my_timer);
       gSLICr_engine->Process_Frame(in_img);

        
       gSLICr_engine->Draw_Segmentation_Result(out_img);
		
       load_image(out_img, boundry_draw_frame);

       
       //! get mask
       const gSLICr::IntImage *gmask = gSLICr_engine->Get_Seg_Res();
       const int* inimg_ptr = gmask->GetData(MEMORYDEVICE_CPU);

       //! filter based on edges
       std::vector<bool> superpixels_flags;
       edgeBasedClusterFilters(superpixels_flags, oldFrame, inimg_ptr);

       
       sdkStopTimer(&my_timer);
       cout << "\rsegmentation in:[" << sdkGetTimerValue(&my_timer)
            << "]ms" << "\n";


       //! ploting and vizualization
       cv::RNG rng(0xFFFFFFFF);
       std::vector<cv::Scalar> colors(my_settings.no_segs);
       for (int i = 0; i < colors.size(); i++) {
          cv::Scalar c = cv::Scalar(
             std::abs(static_cast<float>(rng.uniform(0.0, 1.0))),
             std::abs(static_cast<float>(rng.uniform(0.0, 1.0))),
             std::abs(static_cast<float>(rng.uniform(0.0, 1.0))));
          colors[i] = c;
       }


       const std::vector<std::vector<cv::Point2f> > superpixel_points;
       
       cv::Mat im_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
       for (int y = 0; y < im_mask.rows; y++) {
          for (int x = 0; x < im_mask.cols; x++) {
             int value = inimg_ptr[x + (y * im_mask.cols)];
             cv::Scalar color = colors[value];
             if (superpixels_flags[x + (y * im_mask.cols)]) {
                im_mask.at<cv::Vec3b>(y, x)[0] = color.val[0] * 255;
                im_mask.at<cv::Vec3b>(y, x)[1] = color.val[1] * 255;
                im_mask.at<cv::Vec3b>(y, x)[2] = color.val[2] * 255;
             }
          }
       }

       cv::namedWindow("segmentation", cv::WINDOW_NORMAL);
       imshow("segmentation", boundry_draw_frame);
       
       cv::namedWindow("mask", cv::WINDOW_NORMAL);
       cv::imshow("mask", im_mask);
       cv::waitKey(0);
    }
    destroyAllWindows();
    return 0;
}


void edgeBasedClusterFilters(std::vector<bool> &superpixels_flag,
                             const cv::Mat image,
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

    
    //! sort the indices
    std::sort(sp_indices.begin(), sp_indices.end(), std::less<int>());

    //! remove duplicates
    sp_indices.erase(std::unique(sp_indices.begin(), sp_indices.end()),
                     sp_indices.end());
    
    //! flag superpixels
    superpixels_flag.clear();
    superpixels_flag.resize(static_cast<int>(image.rows * image.cols), false);

    //! memory for 2d image points
    std::vector<std::vector<cv::Point2f> > superpixel_points(
       static_cast<int>(sp_indices.size()));

    for (int i = 0; i < sp_indices.size(); i++) {
       int x = 0;
       int y = 0;
       for (int j = 0; j < image.rows * image.cols; j++) {
          if (gmask[j] == sp_indices[i]) {
             superpixels_flag[j] = true;
             superpixel_points[i].push_back(cv::Point2f(x, y));
          }
          
          if (x++ > image.cols - 1) {
             x = 0;
             if (y++ > image.rows - 1) {
                y = 0;
             }
          }
       }
    }

    getBoundingRects(image, superpixel_points);
    
}


void getBoundingRects(const cv::Mat image,
                      const std::vector<
                      std::vector<cv::Point2f> > &superpixel_points) {

    cv::Mat img = image.clone();
    for (int i = 0; i < superpixel_points.size(); i++) {
       cv::Rect rect = cv::boundingRect(superpixel_points[i]);
       cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
    }
    
    cv::namedWindow("rects", cv::WINDOW_NORMAL);
    cv::imshow("rects", img);
}
