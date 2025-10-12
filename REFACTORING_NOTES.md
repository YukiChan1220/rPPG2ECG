# wash_data.py 重构说明

## 概述

`wash_data.py` 已经被重构为面向对象的架构，支持两种不同的 mirror_version：
- **mirror_version = '1'**: 处理 ECG + rPPG 信号
- **mirror_version = '2'**: 处理 ECG + rPPG + PPG (red, ir, green) 信号

## 主要类结构

### 1. SignalData
数据容器类，统一管理所有信号数据。
- 包含字段：time, rppg, ecg, ppg_red, ppg_ir, ppg_green
- 提供 `get_signal(name)` 方法动态获取信号
- 提供 `has_ppg()` 判断是否包含 PPG 信号

### 2. DataLoader (抽象基类)
数据加载器基类，使用策略模式支持不同的数据格式。

#### MirrorV1Loader
- 从 `merged_log.csv` 加载数据
- 处理格式: timestamp, rppg, ecg

#### MirrorV2Loader  
- 从 `merged_output.csv` 加载数据
- 处理格式: timestamp, rppg, ppg_red, ppg_ir, ppg_green, ecg

### 3. SignalCleaner
信号清洗类，实现三种滤波算法：
- **STD Filter**: 标准差阈值过滤
- **Diff Filter**: 幅度差异过滤
- **Welch Filter**: 频谱分析过滤（BPM容差）

### 4. DataProcessor
数据处理核心类：
- 根据 mirror_version 自动选择合适的加载器
- 使用 SignalCleaner 对信号进行清洗
- 管理多个信号的 mask（通过 masks 字典）
- 提供 `get_combined_mask()` 合并多个信号的 mask

### 5. PlotterBase (抽象基类)
可视化基类，使用模板方法模式。

#### RawDataPlotter
原始数据可视化：
- 动态创建多个子图（根据信号数量）
- 提供交互式滑块控制阈值参数
- mirror_version='2' 时自动添加 PPG 信号的滑块
- 支持实时更新和标记异常区域

#### CleanedDataPlotter
清洗后数据检查：
- 显示归一化后的信号
- 提供 Accept/Reject/Reverse 按钮
- 支持 ECG 信号反转功能

### 6. DataLogger
数据存储类：
- `log_cleaned_data()`: 将清洗后的数据分段保存
  - 自动检测并保存所有清洁的连续窗口
  - 支持保存多种信号（ECG, rPPG, PPG）
- `modify_cleaned_data()`: 修改已保存的数据
  - reject: 删除文件
  - reverse: 反转 ECG 信号
  - 自动归一化处理

### 7. Pipeline
主流程控制类：
- `start_cleaning()`: 原始数据清洗流程
  - 自动检测信号类型（是否包含 PPG）
  - 交互式调整过滤参数
  - 实时可视化和标记
  
- `start_checking_cleaning()`: 清洗结果检查流程
  - 加载已清洗的数据
  - 支持最终确认或调整

## 数据流

```
原始数据 -> DataLoader -> SignalData
                              ↓
                        DataProcessor (使用 SignalCleaner)
                              ↓
                        生成 masks (每个信号一个)
                              ↓
                        RawDataPlotter (可视化 + 用户交互)
                              ↓
                        DataLogger (保存清洗后的数据)
                              ↓
                        CleanedDataPlotter (最终检查)
```

## 版本差异

| 功能 | mirror_version='1' | mirror_version='2' |
|------|-------------------|-------------------|
| 信号类型 | ECG, rPPG | ECG, rPPG, PPG (red, ir, green) |
| 数据源 | merged_log.csv | merged_output.csv |
| 子图数量 | 2 | 5 |
| 滑块数量 | 3 | 6 |
| 清洗逻辑 | STD + Welch | STD + Welch (应用于所有信号) |

## 特性

1. **面向对象设计**: 高内聚、低耦合
2. **策略模式**: DataLoader 支持灵活扩展
3. **模板方法模式**: PlotterBase 定义可视化框架
4. **单一职责**: 每个类专注于特定功能
5. **代码复用**: SignalCleaner 被所有信号共享
6. **错误处理**: 仅输出必要的错误信息
7. **动态适配**: 根据数据自动调整界面和处理逻辑

## 使用方法

```python
# 设置 mirror_version
# global_vars.py 中修改: mirror_version = "1" 或 "2"

# 运行脚本
python wash_data.py

# 输入参数
# Data path: lab_mirror_data (version 1) 或 lab_ppg_mirror_data (version 2)
# Log path: 清洗后数据保存路径
# Starting point: 起始患者编号
# Ending point: 结束患者编号
```

## 输出格式

清洗后的数据保存为 CSV 格式：
- mirror_version='1': Time, rppg, ecg
- mirror_version='2': Time, rppg, ecg, ppg_red, ppg_ir, ppg_green

所有信号在保存时都会进行归一化处理（零均值，单位方差）。
