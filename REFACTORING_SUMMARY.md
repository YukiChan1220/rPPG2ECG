# wash_data.py 重构完成总结

## 完成的工作

### ✅ 1. 支持两种 Mirror Version
- **Version 1**: 处理 ECG + rPPG 信号（从 `merged_log.csv` 加载）
- **Version 2**: 处理 ECG + rPPG + PPG (red, ir, green) 信号（从 `merged_output.csv` 加载）

### ✅ 2. 面向对象重构
创建了以下类：
- **SignalData**: 数据容器，统一管理所有信号
- **DataLoader (抽象类)**: 定义加载接口
  - MirrorV1Loader: 版本1数据加载器
  - MirrorV2Loader: 版本2数据加载器
- **SignalCleaner**: 信号清洗器，实现三种滤波算法
- **DataProcessor**: 数据处理器，整合加载和清洗功能
- **PlotterBase (抽象类)**: 可视化基类
  - RawDataPlotter: 原始数据可视化
  - CleanedDataPlotter: 清洗后数据可视化
- **DataLogger**: 数据存储器
- **Pipeline**: 流程控制器

### ✅ 3. 实现的处理逻辑
对于 Version 2 的所有信号（ECG, rPPG, PPG Red, PPG IR, PPG Green）都实现了：
- **STD Threshold 筛选**: 标准差阈值过滤
- **Welch 频谱分析**: BPM 容差筛选
- **实时可视化**: 动态显示所有信号和标记区域
- **手动参数调整**: 每个信号都有独立的滑块控制

### ✅ 4. 数据存储
- 自动保存所有信号到 CSV 文件
- Version 1: Time, rppg, ecg
- Version 2: Time, rppg, ecg, ppg_red, ppg_ir, ppg_green
- 支持分段保存连续的清洁窗口

### ✅ 5. 代码优化
- 使用抽象基类和继承减少代码重复
- 采用策略模式支持不同的加载策略
- 采用模板方法模式统一可视化流程
- 最小化输出：仅显示错误和必要调试信息

## 设计模式应用

1. **策略模式**: DataLoader 根据 mirror_version 选择不同的加载策略
2. **模板方法模式**: PlotterBase 定义可视化框架，子类实现具体细节
3. **单一职责原则**: 每个类专注于单一功能
4. **开闭原则**: 通过继承扩展功能，无需修改现有代码

## 关键特性

### 动态适配
- 自动检测是否包含 PPG 信号
- 根据信号数量动态创建子图
- 根据 mirror_version 动态添加控制滑块

### 代码复用
- SignalCleaner 被所有信号类型共享
- PlotterBase 提供通用可视化框架
- DataLogger 统一处理不同格式的数据保存

### 错误处理
- 所有加载和处理操作都有异常捕获
- 仅输出必要的错误信息
- 不会因为单个文件错误而中断整个流程

## 使用示例

### 设置 Mirror Version
在 `global_vars.py` 中：
```python
mirror_version = "1"  # 或 "2"
```

### 运行脚本
```bash
python wash_data.py
```

### 交互式输入
```
Input data path: lab_ppg_mirror_data
Input log path: lab_ppg_washed
Input starting point (default 0): 3
Input ending point (default None): 3
```

### 处理流程
1. **原始数据清洗**:
   - 加载患者数据
   - 显示所有信号
   - 通过滑块调整每个信号的阈值
   - 红色区域表示被标记为异常的部分
   - 点击 Accept 保存，Reject 跳过

2. **清洗结果检查**:
   - 加载已清洗的数据
   - 显示归一化后的信号
   - Accept: 保持原样
   - Reject: 删除文件
   - Reverse: 反转 ECG 信号

## 文件输出

清洗后的数据会保存为多个文件，每个连续的清洁窗口保存为独立文件：
- patient_000003_1.csv
- patient_000003_2.csv
- patient_000003_3.csv
- ...

每个文件包含的列（Version 2）：
- Time
- rppg
- ecg
- ppg_red
- ppg_ir
- ppg_green

## 测试建议

1. **测试 Version 1**:
   ```python
   # global_vars.py
   mirror_version = "1"
   ```
   使用 `lab_mirror_data` 中的数据测试

2. **测试 Version 2**:
   ```python
   # global_vars.py
   mirror_version = "2"
   ```
   使用 `lab_ppg_mirror_data` 中的数据测试

## 注意事项

1. 确保 `global_vars.py` 中的 `mirror_version` 设置正确
2. 数据路径和格式需要与 mirror_version 匹配
3. Version 1 需要 `merged_log.csv`
4. Version 2 需要 `merged_output.csv`
5. 所有保存的数据都会自动归一化

## 未来扩展

如需支持新的数据格式，只需：
1. 继承 `DataLoader` 创建新的加载器
2. 在 `DataProcessor.__init__` 中添加选择逻辑
3. 无需修改其他代码

如需添加新的滤波算法：
1. 在 `SignalCleaner` 中添加新的过滤方法
2. 在 `SignalCleaner.clean()` 中调用
3. 在 `RawDataPlotter._init_plot()` 中添加相应控制滑块
