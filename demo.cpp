// Copyright 2014-2016 Isis Innovation Limited and the authors of gSLICr

#include <time.h>
#include <stdio.h>

#include "gSLICr_Lib/gSLICr.h"
#include "NVTimer.h"

#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudawarping.hpp"

#include <boost/foreach.hpp>

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


int edgeBasedClusterFilters(std::vector<bool> &, const cv::Mat, const int *);
void getBoundingRects(std::vector<cv::Rect_<int> > &, const cv::Mat,
                      const std::map<int, std::vector<cv::Point2f> > &);
void rankBoxProposals(const cv::Mat, const cv::cuda::GpuMat,
                      const std::vector<cv::Rect_<int> >);
    
//! minimum size
const int MIN_CLUSTER_SIZE_ = 32;
   

int main(int argc, const char *argv[]) {

    //! setup device
    cout << "Device Info: " << cv::cuda::getDevice()  << "\n";
    cv::cuda::setDevice(cv::cuda::getDevice());
    
    // gSLICr settings
    gSLICr::objects::settings my_settings;
    my_settings.img_size.x = 448;
    my_settings.img_size.y = 448;
    my_settings.no_segs = 200;
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
    Mat boundry_draw_frame;
    boundry_draw_frame.create(s, CV_8UC3);
    
    oldFrame = cv::imread(argv[1]);
    cv::resize(oldFrame, oldFrame,
               cv::Size(my_settings.img_size.x, my_settings.img_size.y));

    //! image preprocessing
    cv::cuda::GpuMat d_image(oldFrame);
    cv::cuda::resize(d_image, d_image, cv::Size(my_settings.img_size.x,
                                                my_settings.img_size.y));
    
    
    double low_thresh = 20.0;
    double high_thresh = 100.0;
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny =
       cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, 3, true);
    cv::cuda::GpuMat d_edges;
    cv::cuda::GpuMat d_gray;
    // cv::cuda::cvtColor(cv::cuda::GpuMat(image), d_gray,
    // cv::COLOR_BGR2GRAY);
    cv::cuda::cvtColor(d_image, d_gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::cuda::Filter> gauss =
        cv::cuda::createGaussianFilter(d_gray.type(), -1, cv::Size(9, 9), 1, 1);
    gauss->apply(d_gray, d_gray);
    canny->detect(d_gray, d_edges);
    
    cv::Mat im_edges;
    d_edges.download(im_edges);
    
    
    StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);

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
    int ssize = edgeBasedClusterFilters(superpixels_flags,
                                        im_edges, inimg_ptr);
       
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

    // std::vector<std::vector<cv::Point2f> > superpixel_points(ssize);
    std::map<int, std::vector<cv::Point2f> > superpixel_points;
              
    cv::Mat im_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
    for (int y = 0; y < im_mask.rows; y++) {
        for (int x = 0; x < im_mask.cols; x++) {
            int value = inimg_ptr[x + (y * im_mask.cols)];
            cv::Scalar color = colors[value];
            if (superpixels_flags[x + (y * im_mask.cols)]) {
                im_mask.at<cv::Vec3b>(y, x)[0] = color.val[0] * 255;
                im_mask.at<cv::Vec3b>(y, x)[1] = color.val[1] * 255;
                im_mask.at<cv::Vec3b>(y, x)[2] = color.val[2] * 255;

                superpixel_points[value].push_back(cv::Point2f(x, y));
                
                // superpixel_points[i].push_back(cv::Point2f(x, y));
            }
        }
    }

    std::vector<cv::Rect_<int> > box_proposals;
    getBoundingRects(box_proposals, oldFrame, superpixel_points);


    //!
    rankBoxProposals(oldFrame, d_edges, box_proposals);
    
    
       
    std::cout << "# of proposals: " << superpixel_points.size()
              << " " << box_proposals.size() << "\n";
       
    cv::namedWindow("segmentation", cv::WINDOW_NORMAL);
    imshow("segmentation", boundry_draw_frame);
       
    cv::namedWindow("mask", cv::WINDOW_NORMAL);
    cv::imshow("mask", im_mask);
    cv::waitKey(0);

    destroyAllWindows();
    return 0;
}

int edgeBasedClusterFilters(std::vector<bool> &superpixels_flag,
                            const cv::Mat image, const int *gmask) {
    if (image.empty()) {
       cout << "EMPTY INPUT IMAGE"  << "\n";
       return -1;
    }
    
    //! traverse the edges
    std::vector<int> sp_indices;
    int index = -1;
    int prev_index = -1;

    for (int j = 0; j < image.rows; j++) {
       for (int i = 0; i < image.cols; i++) {
          index = gmask[i + (j * image.cols)];
          if (image.at<uchar>(j, i) > 0 && index != prev_index) {
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

    //! TODO(FIX): slow
    for (int i = 0; i < sp_indices.size(); i++) {
       for (int j = 0; j < image.rows * image.cols; j++) {
           superpixels_flag[j] = (gmask[j] == sp_indices[i]) ? true :
               superpixels_flag[j];
       }
    }

    return static_cast<int>(sp_indices.size());
}


void getBoundingRects(std::vector<cv::Rect_<int> > &box_proposals,
                      const cv::Mat image, const std::map<int,
                      std::vector<cv::Point2f> > &superpixel_points) {
    cv::Mat img = image.clone();
    int rejected_counter = 0;
    for (std::map<int, std::vector<cv::Point2f> >::const_iterator it =
            superpixel_points.begin(); it != superpixel_points.end(); it++) {
       cv::Rect rect = cv::boundingRect(it->second);
       if (rect.width > MIN_CLUSTER_SIZE_ && rect.height > MIN_CLUSTER_SIZE_) {
           box_proposals.push_back(rect);
           
           cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 1);
       } else {
          // cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1);
          rejected_counter++;
       }
    }

    cout << "TOTAL SIZE: " << superpixel_points.size() << " "
         << rejected_counter  << "\n";

    cv::namedWindow("rects", cv::WINDOW_NORMAL);
    cv::imshow("rects", img);
    // cv::waitKey(0);
}

/**
 * function to scale and rotate the box proposals
 */
void warpBoxProposals(std::vector<cv::Rect_<int> > &warped_rects,
                      const cv::Rect_<int> pbox, const cv::Size size_im,
                      const int levels, const float scale_step = 1.10f) {
    warped_rects.clear();
    if (levels == 0 || scale_step < 0.0) {
       warped_rects.push_back(pbox);
       return;
    }
    
    const int octave = std::floor(levels/2);
    float step = 0.0f;
    cv::Rect_<int> rect;

    for (int k = -octave; k < octave; k++) {
       step = std::pow(scale_step, k);
       cv::Point center(pbox.x + pbox.width/2, pbox.y + pbox.height/2);
       rect.width = pbox.width * step;
       rect.height = pbox.height * step;

       rect.x = center.x - (rect.width/2);
       rect.y = center.y - (rect.height/2);
       
       rect.x = (rect.x < 0) ? 0: rect.x;
       rect.y = (rect.y < 0) ? 0: rect.y;
       rect.width -= (rect.br().x > size_im.width) ?
          (rect.br().x - size_im.width) : 0;
       rect.height -= (rect.br().y > size_im.height) ?
          (rect.br().y - size_im.height) : 0;
       
       warped_rects.push_back(rect);
       if (k != 0) {
          warped_rects.push_back(
             cv::RotatedRect(
                cv::Point(rect.x + rect.width/2, rect.y + rect.height/2),
                rect.size(), 90).boundingRect());
       }
    }
}


void rankBoxProposals(
    const cv::Mat image, const cv::cuda::GpuMat d_edges,
    const std::vector<cv::Rect_<int> > box_proposals) {
    if (image.empty() || d_edges.empty()) {
        std::cout << "ERROR: [rankBoxProposals]Empty inputs!" << "\n";
        return;
    }

    //! temp on CPU
    cv::Mat im_edges;
    d_edges.download(im_edges);

    const int padding = 0;
    cv::Rect_<int> box;
    std::vector<cv::Rect_<int> > warped_rects;
    BOOST_FOREACH(cv::Rect_<int> rect, box_proposals) {
        box.x = rect.x - padding;
        box.y = rect.y - padding;
        box.width = rect.width + (2 * padding);
        box.height = rect.height + (2 * padding);

        // TODO(HERE): pass probability map directly and return the
        // top k ranks for the boxes in the octave
        warpBoxProposals(warped_rects, box, image.size(), 5, 1.2);
        
        
        cv::Mat roi = im_edges(box);
        cv::namedWindow("boxes", cv::WINDOW_NORMAL);
        cv::imshow("boxes", roi);
        cv::waitKey(0);
    }
}
