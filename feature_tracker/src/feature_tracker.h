#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
  FeatureTracker();

  /**
   * @brief
   *
   * @param[in] _img 输入图像
   * @param[in] _cur_time 图像的时间戳
   * 1、图像均衡化预处理
   * 2、光流追踪
   * 3、提取新的特征点（如果发布）
   * 4、所有特征点去畸变，计算速度
   */
  void readImage(const cv::Mat &_img, double _cur_time);

  void setMask();

  void addPoints();

  bool updateID(unsigned int i);

  void readIntrinsicParameter(const string &calib_file);

  void showUndistortion(const string &name);

  void rejectWithF();

  void undistortedPoints();

  cv::Mat mask;
  cv::Mat fisheye_mask;
  cv::Mat prev_img, cur_img, forw_img;
  // 记录flow image的强角点
  vector<cv::Point2f> n_pts;
  // 前一帧、当前帧、光流帧特征点的像素坐标
  vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
  // 记录去畸变的归一化相机坐标
  vector<cv::Point2f> prev_un_pts, cur_un_pts;
  // 为当前帧相对前一帧特征点沿x,y方向的像素移动速度
  vector<cv::Point2f> pts_velocity;

  // ?
  vector<int> ids;

  // 记录特征点被跟踪的次数
  vector<int> track_cnt;
  
  map<int, cv::Point2f> cur_un_pts_map;
  map<int, cv::Point2f> prev_un_pts_map;
  camodocal::CameraPtr m_camera;
  double cur_time;
  double prev_time;

  // ?
  static int n_id;
};
