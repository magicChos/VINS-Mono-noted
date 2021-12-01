#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;

// 每个路标点由多个连续的图像观测到
// 路标点信息结构体
struct SFMFeature
{
	// 状态，记录是否被三角化
	bool state;
	// 记录特征点索引
	int id;
	// 记录所有观测到该特征点的图像帧ID和图像坐标
	vector<pair<int, Vector2d>> observation;
	// 3d坐标
	double position[3];
	// 深度
	double depth;
};

struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		: observed_u(observed_u), observed_v(observed_v)
	{
	}

	template <typename T>
	bool operator()(const T *const camera_R, const T *const camera_T, const T *point, T *residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p); // 旋转这个点
		p[0] += camera_T[0];
		p[1] += camera_T[1];
		p[2] += camera_T[2]; // 这其实就是Rcw * pw + tcw
		// 得到该相机坐标系下的3d坐标
		T xp = p[0] / p[2];
		T yp = p[1] / p[2]; // 归一化处理
							// 跟现有观测形成残差
		residuals[0] = xp - T(observed_u);
		residuals[1] = yp - T(observed_v);
		return true;
	}

	static ceres::CostFunction *Create(const double observed_x,
									   const double observed_y)
	{
		// 2, 4, 3, 3？
		return (new ceres::AutoDiffCostFunction<
				ReprojectionError3D, 2, 4, 3, 3>(
			new ReprojectionError3D(observed_x, observed_y)));
	}

	double observed_u;
	double observed_v;
};

class GlobalSFM
{
public:
	GlobalSFM();
	bool construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
				   const Matrix3d relative_R, const Vector3d relative_T,
				   vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						  Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);
	// 构造sfm时初始化赋值
	int feature_num;
};