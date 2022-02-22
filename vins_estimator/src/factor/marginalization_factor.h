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

// 残差类，将不同的损失函数以及优化变量统一起来再一起添加到marginalization_info中
struct ResidualBlockInfo {
    // 构造函数需要，cost function（约束），loss function：残差的计算方式，相关联的参数块，待边缘化的参数块的索引
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set) :
        cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    // 记录优化变量数据
    std::vector<double *> parameter_blocks;
    // 待边缘化的优化变量id
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

struct ThreadsStruct {
    // 所有观测项
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx;  //local size
};

class MarginalizationInfo
{
public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;

    /**
     * @brief 添加残差块相关信息（优化变量，待边缘化变量）
     * 
     * @param residual_block_info 
     */
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);

    /**
     * @brief 并更新parameter_block_data
     * 
     */
    void preMarginalize();

    /**
     * @brief 执行边缘化，多线程构造先验项舒尔补AX=b的结构，计算Jacobian和残差
     * 
     */
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    //保存了视觉观测项，imu观测项以及上一次的边缘化项，从中去分离出需要边缘化的状态量和需要保留的状态量
    std::vector<ResidualBlockInfo *> factors;

    // m为要边缘化的变量个数
    int m;
    // n为要保留下来的变量个数
    int n;

    // <优化变量内存地址 , 优化变量长度>
    std::unordered_map<long, int> parameter_block_size; //global size   // 地址->global size
    int sum_block_size;
    // < 优化变量内存地址，在矩阵中的id>
    std::unordered_map<long, int> parameter_block_idx; //local size // 地址->参数排列的顺序idx
    // < 优化变量内存地址，优化变量对应的数据指针>
    std::unordered_map<long, double *> parameter_block_data; // 地址->参数块实际内容的地址

    // 进行边缘化之后保留下来的各个优化变量的长度
    std::vector<int> keep_block_size;
    std::vector<int> keep_block_idx; //local size
    std::vector<double *> keep_block_data;

    // 的是边缘化之后从信息矩阵H恢复出来雅克比矩阵
    Eigen::MatrixXd linearized_jacobians;
    // 边缘化得到的残差
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
};

// 由于边缘化的costfuntion不是固定大小的，因此只能继承最基本的类
class MarginalizationFactor : public ceres::CostFunction
{
public:
    MarginalizationFactor(MarginalizationInfo *_marginalization_info);

    /**
     * @brief 输出各项残差以及残差对应各优化变量的Jacobian
     * 
     * @param parameters 待优化变量 
     * @param residuals[out]  先验值（对于先验残差就是上一时刻的先验残差，last_marginalization_info，对于IMU就是预计分值pre_integrations[1]，对于视觉就是空间的的像素坐标pts_i, pts_j）
     * @param jacobians[out]  
     * @return true 
     * @return false 
     */
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo *marginalization_info;
};
