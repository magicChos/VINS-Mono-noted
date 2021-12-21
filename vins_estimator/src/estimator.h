#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>

class Estimator
{
public:
    Estimator();

    void setParameter();

    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);

    // 表示当前帧跟踪到上一帧中的特征点集合，也就是当前帧观测到的所有的路标点（不包括在当前帧新提取的点）
    // map int: feature Id
    // pair int: camera Id
    // pair value 7: x,y,z,u,v,ux,vx
    /**
     * @brief 实现了视觉与IMU的初始化以及非线性优化的紧耦合
     * @param [in] image:某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]构成的map,索引为feature_id
     * @param [in] header
     */
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                      const std_msgs::Header &header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();
    bool initialStructure();

    /**
     * @brief 实现了陀螺仪的偏置校准（加速度偏置没有处理），计算速度V、重力g和尺度s
     * @return true
     * @return false
     */
    bool visualInitialAlign();

    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);

    /**
     * @brief 负责维护滑动窗口
     * 如果次新帧是关键帧，则边缘化最老帧，将其看到的特征点和IMU数据转化为先验信息，如果次新帧不是关键帧，则舍弃视觉测量而保留IMU测量值，从而保证IMU预积分的连贯性。
     *
     */
    void slideWindow();

    /*********************************
     * @brief vio非线性优化求解里程计
     */
    void solveOdometry();

    void slideWindowNew();

    /**
     * @brief 首次在原来最老帧出现的特征点转移到现在现在最老帧
     *
     */
    void slideWindowOld();

    /**
     * @brief 负责利用边缘化残差构建优化模型，而且它负责整个系统所有的优化工作，边缘化残差的使用只是它功能的一部分
     *
     */
    void optimization();
    void vector2double();
    void double2vector();

    /**
     * @brief VIO是否正常检测
     * 1. 地图点被跟踪的数目；
     * 2. 零偏是否超过阈值
     * 3. 陀螺仪偏置是否正常
     * 4. 两帧之间运动是否过大
     * 5. 两帧之间姿态变化是否过大
     * @return true
     * @return false
     */
    bool failureDetection();

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    // 记录solver状态，默认为INITIAL
    SolverFlag solver_flag;
    // 记录边缘化状态
    MarginalizationFlag marginalization_flag;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];

    // imu时间间隔？
    double td;

    // 保存最老帧信息
    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;

    // 记录滑动窗口中所有关键帧的Header信息
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    // 统计一种有多少次marg第一帧的情况
    int sum_of_back;
    int sum_of_outlier, sum_of_front, sum_of_invalid;

    // 特征管理器对象
    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    // 滑窗中关键帧位置
    vector<Vector3d> key_poses;
    double initial_timestamp;

    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    // key:image的时间戳
    map<double, ImageFrame> all_image_frame;

    // 用来坐初始化用的
    IntegrationBase *tmp_pre_integration;

    // relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
