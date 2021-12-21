#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;

// 构建Brief产生器，用于通过Brief模板文件对图像特征点计算Brief描述子
class BriefExtractor
{
public:
	virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
	BriefExtractor(const std::string &pattern_file);

	DVision::BRIEF m_brief;
};

// 构建关键帧类、描述子计算、匹配关键帧与回环帧
class KeyFrame
{
public:
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal,
			 vector<double> &_point_id, int _sequence);
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1> &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);

	/**
	 * @brief 寻找两帧之间联系，确定是否回环
	 *
	 * @param[in] old_kf
	 * @return true
	 * @return false
	 */
	bool findConnection(KeyFrame *old_kf);

	/**
	 * @brief 计算已有特征点的描述子
	 * 70个点的描述
	 *
	 */
	void computeWindowBRIEFPoint();

	/**
	 * @brief 额外提取fast特征点并计算描述子新的500个点
	 *
	 */
	void computeBRIEFPoint();

	/**
	 * @brief 計算兩個描述子的漢明距離
	 *
	 * @param a
	 * @param b
	 * @return int
	 */
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);

	/**
	 * @brief 暴力匹配法，通过遍历所有的候选描述子得到最佳匹配
	 *
	 * @param[in] window_descriptor 当前帧的一个描述子
	 * @param[in] descriptors_old 回环帧的描述子集合
	 * @param[in] keypoints_old 回环帧像素坐标集合
	 * @param[in] keypoints_old_norm 回环帧归一化坐标集合
	 * @param[out] best_match 最佳匹配的像素坐标
	 * @param[out] best_match_norm 最佳匹配的归一化相机坐标
	 * @return true
	 * @return false
	 */
	bool searchInAera(const BRIEF::bitset window_descriptor,
					  const std::vector<BRIEF::bitset> &descriptors_old,
					  const std::vector<cv::KeyPoint> &keypoints_old,
					  const std::vector<cv::KeyPoint> &keypoints_old_norm,
					  cv::Point2f &best_match,
					  cv::Point2f &best_match_norm);

	/**
	 * @brief 将当前帧的描述子依次和回环帧描述子进行匹配，得到匹配结果
	 *
	 * @param[out] matched_2d_old 匹配回环帧点的像素坐标集合
	 * @param[out] matched_2d_old_norm 匹配回环帧点的归一化相机坐标集合
	 * @param[out] status 状态位
	 * @param[in] descriptors_old 回环帧的描述子集合
	 * @param[in] keypoints_old 回环帧的像素坐标
	 * @param[in] keypoints_old_norm 回环帧的归一化坐标
	 */
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
						  std::vector<uchar> &status,
						  const std::vector<BRIEF::bitset> &descriptors_old,
						  const std::vector<cv::KeyPoint> &keypoints_old,
						  const std::vector<cv::KeyPoint> &keypoints_old_norm);
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
								const std::vector<cv::Point2f> &matched_2d_old_norm,
								vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
				   const std::vector<cv::Point3f> &matched_3d,
				   std::vector<uchar> &status,
				   Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info);

	Eigen::Vector3d getLoopRelativeT();

	/**
	 * @brief Get the Loop Relative Yaw object
	 * 
	 * @return double 
	 */
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();

	// 关键帧时间戳
	double time_stamp;
	int index;
	int local_index;
	Eigen::Vector3d vio_T_w_i;
	Eigen::Matrix3d vio_R_w_i;
	Eigen::Vector3d T_w_i;
	Eigen::Matrix3d R_w_i;
	Eigen::Vector3d origin_vio_T; // 原始VIO结果的位姿
	Eigen::Matrix3d origin_vio_R;
	cv::Mat image;
	cv::Mat thumbnail;
	vector<cv::Point3f> point_3d;
	vector<cv::Point2f> point_2d_uv;
	vector<cv::Point2f> point_2d_norm;
	vector<double> point_id;
	vector<cv::KeyPoint> keypoints;		 // fast角点的像素坐标
	vector<cv::KeyPoint> keypoints_norm; // fast角点对应的归一化相机系坐标
	vector<cv::KeyPoint> window_keypoints;
	vector<BRIEF::bitset> brief_descriptors;		// 额外提取的fast特征点的描述子
	vector<BRIEF::bitset> window_brief_descriptors; // 原来光流追踪的特征点的描述子
	bool has_fast_point;

	// ?
	int sequence;

	bool has_loop;
	int loop_index;
	Eigen::Matrix<double, 8, 1> loop_info;
};
