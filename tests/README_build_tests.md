# 构建测试程序使用说明

## 概述

这两个测试程序用于在内存不足的情况下，分阶段构建NNDescent图和NSG图。当`build_mmap`在NNDescent阶段因内存不足而失败时，可以使用这些程序继续构建过程。

## 程序说明

### 1. test_nndescent_build.cpp

**功能**: 读取已保存的cluster_data文件，为每个cluster构建NNDescent图

**使用方法**:
```bash
./test_nndescent_build <prefix> [k_nndescent] [l_nndescent] [iter] [s] [r]
```

**参数说明**:
- `prefix`: 索引目录路径（包含cluster_data子目录）
- `k_nndescent`: K参数（默认20）
- `l_nndescent`: L参数（默认100）
- `iter`: 迭代次数（默认10）
- `s`: S参数（默认10）
- `r`: R参数（默认100）

**示例**:
```bash
./test_nndescent_build /path/to/index 20 100 10 10 100
```

### 2. test_nsg_build.cpp

**功能**: 读取已保存的cluster_data和nndescent图，为每个cluster构建NSG图

**使用方法**:
```bash
./test_nsg_build <prefix> [L_nsg] [R_nsg] [C_nsg] [use_mmap]
```

**参数说明**:
- `prefix`: 索引目录路径（包含cluster_data和nndescent子目录）
- `L_nsg`: L参数（默认32）
- `R_nsg`: R参数（默认100）
- `C_nsg`: C参数（默认500）
- `use_mmap`: 是否使用mmap保存（默认1，即true）

**示例**:
```bash
./test_nsg_build /path/to/index 32 100 500 1
```

## 使用流程

### 场景1: build_mmap在NNDescent阶段失败

1. **确认cluster_data已保存**:
   ```bash
   ls /path/to/index/cluster_data/
   ```

2. **构建NNDescent图**:
   ```bash
   ./test_nndescent_build /path/to/index
   ```

3. **构建NSG图**:
   ```bash
   ./test_nsg_build /path/to/index
   ```

### 场景2: 只想重新构建NNDescent图

```bash
./test_nndescent_build /path/to/index 20 100 10 10 100
```

### 场景3: 只想重新构建NSG图

```bash
./test_nsg_build /path/to/index 32 100 500 1
```

## 程序特性

### 内存优化
- 使用并发控制，避免内存过载
- NNDescent阶段最多4个并发
- NSG阶段最多3个并发
- 自动检测和跳过缺失的文件

### 错误处理
- 自动跳过缺失的NNDescent图
- 详细的错误报告和进度显示
- 支持部分成功（至少完成一个cluster即认为成功）

### 性能监控
- 显示每个cluster的构建时间
- 显示总体构建统计
- 显示内存使用情况

## 目录结构要求

程序期望以下目录结构：
```
/path/to/index/
├── cluster_data/
│   ├── cluster_0.data
│   ├── cluster_1.data
│   └── ...
├── centroids.data
├── nndescent/          (由test_nndescent_build创建)
│   ├── nndescent_0.graph
│   ├── nndescent_1.graph
│   └── ...
└── nsg_graph/          (由test_nsg_build创建)
    ├── nsg_0.nsg
    ├── nsg_1.nsg
    └── ...
```

## 注意事项

1. **内存管理**: 程序会自动控制并发数量，避免内存过载
2. **文件依赖**: NSG构建需要先有NNDescent图
3. **参数调优**: 可以根据系统内存调整并发数量
4. **错误恢复**: 程序支持断点续传，可以重新运行失败的cluster

## 故障排除

### 常见问题

1. **"No cluster files found"**
   - 检查cluster_data目录是否存在
   - 检查cluster文件命名格式（cluster_X.data）

2. **"NNDescent graph not found"**
   - 确保先运行test_nndescent_build
   - 检查nndescent目录是否存在

3. **内存不足**
   - 减少并发数量（修改max_parallel_clusters）
   - 增加系统swap空间

4. **构建失败**
   - 检查参数设置是否合理
   - 查看详细错误信息
   - 尝试重新运行单个cluster 