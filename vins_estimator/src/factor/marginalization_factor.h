#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    // 构造函数需要，cost function（约束），loss function：残差的计算方式，相关联的参数块，待边缘化的参数块的索引
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    // 优化变量内存地址
    std::vector<double *> parameter_blocks;
    //需要被边缘化的变量地址的id，也就是上面这个vector的id
    std::vector<int> drop_set;

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    // 残差，imu:15x1，视觉2x1
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    // 所有观测项
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    //保存了视觉观测项，imu观测项以及上一次的边缘化项，从中去分离出需要边缘化的状态量和需要保留的状态量
    std::vector<ResidualBlockInfo *> factors;

    // m为要边缘化的变量个数，n为要保留下来的变量个数
    int m, n;
    // <优化变量内存地址 , localSize>
    std::unordered_map<long, int> parameter_block_size; //global size   // 地址->global size
    int sum_block_size;
    // < 优化变量内存地址，在矩阵中的id>
    std::unordered_map<long, int> parameter_block_idx; //local size // 地址->参数排列的顺序idx
    // < 优化变量内存地址，数据>
    std::unordered_map<long, double *> parameter_block_data;    // 地址->参数块实际内容的地址

    // 上一次边缘化后留下的参数块大小
    std::vector<int> keep_block_size;
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    // 边缘化得到的雅可比矩阵
    Eigen::MatrixXd linearized_jacobians;
    // 边缘化得到的残差
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

// 由于边缘化的costfuntion不是固定大小的，因此只能继承最基本的类
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
