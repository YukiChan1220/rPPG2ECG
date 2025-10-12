# wash_data.py 重构对比

## 代码统计

| 指标 | 原始代码 | 重构后代码 | 变化 |
|------|---------|-----------|------|
| 总行数 | 463 | 604 | +141 行 |
| 类数量 | 3 | 11 | +8 个类 |
| 代码结构 | 面向过程 | 面向对象 | ✓ |
| 代码复用 | 低 | 高 | ✓ |
| 可扩展性 | 低 | 高 | ✓ |

## 主要改进

### 1. 架构改进
- ✅ 从面向过程改为面向对象
- ✅ 引入抽象基类和继承
- ✅ 应用设计模式（策略模式、模板方法模式）
- ✅ 单一职责原则

### 2. 功能扩展
- ✅ 支持 mirror_version='2' (ECG + rPPG + PPG)
- ✅ 动态适配信号数量
- ✅ 自动生成控制界面
- ✅ 统一的数据处理流程

### 3. 代码质量
- ✅ 更好的模块化
- ✅ 减少重复代码
- ✅ 提高可维护性
- ✅ 更清晰的错误处理
- ✅ 最小化输出信息

## 类结构对比

### 原始代码（3个类）
```
DataProcessor - 数据加载、信号处理
DataPlotter - 可视化（raw/cleaned 混在一起）
DataLogger - 数据保存
Pipeline - 流程控制
```

### 重构后代码（11个类）
```
SignalData - 数据容器
├── DataLoader (抽象类)
│   ├── MirrorV1Loader - Version 1 加载器
│   └── MirrorV2Loader - Version 2 加载器
├── SignalCleaner - 信号清洗（复用）
├── DataProcessor - 数据处理协调器
├── PlotterBase (抽象类)
│   ├── RawDataPlotter - 原始数据可视化
│   └── CleanedDataPlotter - 清洗数据可视化
├── DataLogger - 数据存储
└── Pipeline - 流程控制
```

## 功能对比

### Version 1 支持（原始 + 重构）
| 功能 | 原始代码 | 重构代码 |
|------|---------|---------|
| ECG 信号处理 | ✓ | ✓ |
| rPPG 信号处理 | ✓ | ✓ |
| STD 筛选 | ✓ | ✓ |
| Welch 筛选 | ✓ | ✓ |
| 可视化 | ✓ | ✓ |
| 手动调整 | ✓ | ✓ |

### Version 2 支持（仅重构）
| 功能 | 原始代码 | 重构代码 |
|------|---------|---------|
| PPG Red 处理 | ✗ | ✓ |
| PPG IR 处理 | ✗ | ✓ |
| PPG Green 处理 | ✗ | ✓ |
| 动态子图生成 | ✗ | ✓ |
| 动态滑块生成 | ✗ | ✓ |
| 统一保存格式 | ✗ | ✓ |

## 代码复用示例

### 原始代码
```python
# DataProcessor 中针对 rppg 和 ecg 分别处理
def update_signal(self, signal: str, config: dict):
    if signal == 'rppg':
        return self._clean_signal(self.rppg_signal, config)
    elif signal == 'ecg':
        return self._clean_signal(self.ecg_signal, config)
```

### 重构代码
```python
# SignalCleaner 可以处理任意信号
cleaner = SignalCleaner(fs=512)
mask = cleaner.clean(signal_data, config)

# DataProcessor 使用字典管理所有信号
def update_signal_mask(self, signal_name, config):
    sig = self.data.get_signal(signal_name)
    self.masks[signal_name] = self.cleaner.clean(sig, config)
    return self.masks[signal_name]
```

## 扩展性对比

### 添加新的数据格式

**原始代码**：
需要修改 `DataProcessor._load_signal_from_merged_log()` 中的 if-elif 逻辑

**重构代码**：
只需创建新的 Loader 子类
```python
class MirrorV3Loader(DataLoader):
    def load(self, data_path):
        # 新格式的加载逻辑
        pass
```

### 添加新的滤波算法

**原始代码**：
需要修改 `DataProcessor._clean_signal()` 方法

**重构代码**：
只需在 `SignalCleaner` 中添加新方法
```python
class SignalCleaner:
    def _new_filter(self, sig, params):
        # 新的滤波逻辑
        pass
    
    def clean(self, sig, config):
        # ...
        if "new_filter" in config:
            mask &= self._new_filter(sig, config["new_filter"])
        # ...
```

## 可维护性改进

### 原始代码问题
1. DataPlotter 包含两种模式的代码，职责不清晰
2. 数据加载逻辑耦合在 DataProcessor 中
3. 信号处理逻辑与数据管理混在一起
4. 难以为新的 mirror version 扩展

### 重构后优势
1. 每个类职责单一明确
2. 数据加载策略独立可替换
3. 信号处理逻辑完全复用
4. 新增信号类型只需配置不需修改代码

## 性能影响

行数增加主要原因：
- 添加了抽象基类和接口定义
- 分离了职责，每个类都有清晰的边界
- 添加了更多的辅助方法
- 增加了对 Version 2 的完整支持

实际运行性能：
- 运行时性能基本相同
- 内存使用略有增加（使用对象而非直接变量）
- 代码可读性和可维护性大幅提升

## 总结

虽然代码行数增加了 30%，但获得了：
- ✓ 完整的 Version 2 支持
- ✓ 更好的代码组织结构
- ✓ 更高的代码复用率
- ✓ 更强的可扩展性
- ✓ 更低的维护成本
- ✓ 更清晰的错误处理
- ✓ 符合面向对象设计原则

这是一个成功的重构！
