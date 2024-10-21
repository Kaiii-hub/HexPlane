import datetime
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf  # 配置文件
from torch.utils.tensorboard import SummaryWriter  # 日志记录

from config.config import Config  # 配置文件
from hexplane.dataloader import get_test_dataset, get_train_dataset
from hexplane.model import init_model 
from hexplane.render.render import evaluation, evaluation_path
from hexplane.render.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


def render_test(cfg):
    # 加载测试数据集并取 ndc_ray 和 white_bg 属性
    test_dataset = get_test_dataset(cfg, is_stack=True) 
    ndc_ray = test_dataset.ndc_ray # ndc_ray是归一化设备坐标系下的光线
    white_bg = test_dataset.white_bg  # 获取测试数据集的白色背景属性

    # 检查 cfg.systems.ckpt 指定的检查点路径是否存在。如果不存在，打印错误信息并返回
    if not os.path.exists(cfg.systems.ckpt):
        print("the ckpt path does not exists!!")
        return
    # 使用 torch.load 加载模型 HexPlane，并将其映射到指定的设备 device
    HexPlane = torch.load(cfg.systems.ckpt, map_location=device)

    # 获取检查点路径的目录路径  
    logfolder = os.path.dirname(cfg.systems.ckpt)

    # 如果 cfg.render_train 为 True，则创建训练图像的保存目录   
    if cfg.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)  # 创建训练图像的保存目录
        train_dataset = get_train_dataset(cfg, is_stack=True)  # 获取训练数据集 
        # 调用 evaluation 函数，渲染训练数据集的图像
        evaluation(
            train_dataset, # 训练数据集 
            HexPlane, # 模型
            cfg, # 配置文件
            f"{logfolder}/imgs_train_all/", # 训练图像的保存路径    
            prefix="train", # 图像的前缀
            N_vis=-1, # 可视化的图像数量
            N_samples=-1, # 采样的点数
            ndc_ray=ndc_ray,
            white_bg=white_bg, # 白色背景
            device=device, # 设备
        )
    # 如果 cfg.render_test 为 True，则创建测试图像的保存目录        
    if cfg.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset, # 测试数据集
            HexPlane, # 模型
            cfg, # 配置文件
            f"{logfolder}/imgs_test_all/",
            prefix="test", # 图像的前缀
                N_vis=-1, # 可视化的图像数量
            N_samples=-1, # 采样的点数
            ndc_ray=ndc_ray, # ndc_ray是归一化设备坐标系下的光线
            white_bg=white_bg, # 白色背景
            device=device, # 设备
        )

    if cfg.render_path:
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset, # 测试数据集      
            HexPlane, # 模型
            cfg, # 配置文件
            f"{logfolder}/imgs_path_all/", # 路径图像的保存路径 
            prefix="test", # 图像的前缀
            N_vis=-1, # 可视化的图像数量
            N_samples=-1, # 采样的点数
            ndc_ray=ndc_ray, # ndc_ray是归一化设备坐标系下的光线    
            white_bg=white_bg, # 白色背景
            device=device, # 设备
        )


def reconstruction(cfg):
    if cfg.data.datasampler_type == "rays":
        train_dataset = get_train_dataset(cfg, is_stack=False)
    else:
        train_dataset = get_train_dataset(cfg, is_stack=True)
    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg
    near_far = test_dataset.near_far

    if cfg.systems.add_timestamp:
        logfolder = f'{cfg.systems.basedir}/{cfg.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{cfg.systems.basedir}/{cfg.expname}"

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(logfolder, "logs"))
    cfg_file = os.path.join(f"{logfolder}", "cfg.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # init model.
    aabb = train_dataset.scene_bbox.to(device)
    HexPlane, reso_cur = init_model(cfg, aabb, near_far, device)

    # init trainer.
    trainer = Trainer(
        HexPlane,
        cfg,
        reso_cur,
        train_dataset,
        test_dataset,
        summary_writer,
        logfolder,
        device,
    )

    trainer.train()

    torch.save(HexPlane, f"{logfolder}/{cfg.expname}.th")
    # Render training viewpoints.
    if cfg.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = get_train_dataset(cfg, is_stack=True)
        evaluation(
            train_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_train_all/",
            prefix="train",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    # Render test viewpoints.
    if cfg.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/{cfg.expname}/imgs_test_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    # Render validation viewpoints.
    if cfg.render_path:
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/{cfg.expname}/imgs_path_all/",
            prefix="validation",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )


if __name__ == "__main__":
    # Load config file from base config, yaml and cli.
    base_cfg = OmegaConf.structured(Config())
    cli_cfg = OmegaConf.from_cli()
    base_yaml_path = base_cfg.get("config", None)
    yaml_path = cli_cfg.get("config", None)
    if yaml_path is not None:
        yaml_cfg = OmegaConf.load(yaml_path)
    elif base_yaml_path is not None:
        yaml_cfg = OmegaConf.load(base_yaml_path)
    else:
        yaml_cfg = OmegaConf.create()
    cfg = OmegaConf.merge(base_cfg, yaml_cfg, cli_cfg)  # merge configs

    # Fix Random Seed for Reproducibility.
    random.seed(cfg.systems.seed)
    np.random.seed(cfg.systems.seed)
    torch.manual_seed(cfg.systems.seed)
    torch.cuda.manual_seed(cfg.systems.seed)

    if cfg.render_only and (cfg.render_test or cfg.render_path):
        # Inference only.
        render_test(cfg)
    else:
        # Reconstruction and Inference.
        reconstruction(cfg)
