"""
@Author: 
@Time: 2021/11/24
功能说明: 
"""
import os
import wandb


def config_wandb(my_wandb: wandb, args):
    """
    配置wandb
    :param my_wandb: 当前wandb对象
    :param test_name: 当前测试名称，对当前测试进行记录
    :return:
    """
    # args参数存储数据原始路径
    args.raw_data_path = my_wandb.run.dir
    my_wandb.config.update(args)
    # 指定测试名称
    my_wandb.run.name = args.test_name
    # 创建model 目录, 默认在当前数据存储目录下的./files/models
    # my_wandb.run.dir 该目录到./files
    os.makedirs(os.path.join(my_wandb.run.dir, 'models'))