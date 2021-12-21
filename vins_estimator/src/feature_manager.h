#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

// 每个路标点在一张图像中的信息
// 指的是每帧基本的数据
class FeaturePerFrame
{
public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        cur_td = td;
    }

    // imu和camera同步时间差
    double cur_td;
    // 3d特征点坐标
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;
    // 特征点的深度
    double z;
    // 是否被用了
    bool is_used;
    // 视差
    double parallax;
    //变换矩阵
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

// 每个路标点由多个连续的图像观测到
class FeaturePerId
{
public:
    // 特征点索引
    const int feature_id;
    // 首次被观测到时，该帧的索引
    int start_frame;
    // 能够观测到某个特征点的所有相关帧
    vector<FeaturePerFrame> feature_per_frame;

    // 该特征出现的次数(被多少帧观测到)
    int used_num;
    bool is_outlier;
    // 是否Marg边缘化
    bool is_margin;
    // 估计的逆深度
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    /**
     * @brief 返回最后一个观测到这个特征点的图像帧ID
     * @return int
     */
    int endFrame();
};

/** ----------------------------------------------------------------
 * 特征管理器类
 * ---------------------------------------------------------------*/
class FeatureManager
{
public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    /**
     * @brief 窗口中被跟踪的特征点数量
     * @return int
     */
    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    // void updateDepth(const VectorXd &x);

    /**
     * @brief 设置特征点逆深度
     * @param x
     */
    void setDepth(const VectorXd &x);

    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();

    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);

    /**
     * @brief 首次在原来最老帧出现的特征点转移到现在现在最老帧
     *
     * @param[in] marg_R  被移除的位姿
     * @param[in] marg_P
     * @param[in] new_R    转接地图点的位姿
     * @param[in] new_P
     */
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);

    /**
     * @brief 边缘化最老帧，直接将特征点保存的帧号前移
     *
     */
    void removeBack();

    /**
     * @brief 边缘化次新帧，对特征点在次新帧的信息移除
     *
     * @param frame_count
     */
    void removeFront(int frame_count);
    void removeOutlier();
    // 滑窗内所有路标点
    list<FeaturePerId> feature;

    // 被跟踪的个数
    int last_track_num;

private:
    /**
     * @brief 计算某个特征点it_per_id在次新帧和次次新帧的视差ans
     * @brief 判断观测到该特征点的frame中倒数第二帧和倒数第三帧的共视关系 实际是求取该特征点在两帧的归一化平面上的坐标点的距离ans
     * @param it_per_id
     * @param frame_count
     * @return double
     */
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);

    // 记录imu到世界坐标系的变换矩阵
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif