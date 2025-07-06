#pragma once

#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "nsg/neighbor.h"
#include "nsg/index_nsg.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>

namespace CNNS {

// RAII 管理的 mmap 资源类
class MMapResource {
private:
    void* mapped_ptr_;
    size_t mapped_size_;
    int fd_;
    bool is_valid_;

public:
    MMapResource();
    MMapResource(const std::string& filename, size_t size, int file_descriptor);
    ~MMapResource();
    
    // 移动构造函数
    MMapResource(MMapResource&& other) noexcept;
    
    // 移动赋值运算符
    MMapResource& operator=(MMapResource&& other) noexcept;
    
    // 禁用拷贝
    MMapResource(const MMapResource&) = delete;
    MMapResource& operator=(const MMapResource&) = delete;
    
    void* get_ptr() const;
    size_t get_size() const;
    bool is_valid() const;
    
    // 手动释放资源
    void cleanup();
    
    // 重置为新的映射
    bool remap(const std::string& filename, size_t size, int file_descriptor);
};

class ClusterMMap {
public:
    ClusterMMap() = default;
    ~ClusterMMap() {
        if (mmap_base && mmap_base != MAP_FAILED) munmap(mmap_base, mmap_length);
        if (fd != -1) close(fd);
    }

    float* data_ptr = nullptr;
    size_t points_num = 0;
    size_t dim = 0;
    void* mmap_base = nullptr;
    size_t mmap_length = 0;
    int fd = -1;
};

class MappingMMap {
public:
    MappingMMap() = default;
    ~MappingMMap() {
        if (mmap_base && mmap_base != MAP_FAILED) munmap(mmap_base, mmap_length);
        if (fd != -1) close(fd);
    }

    faiss::idx_t* data = nullptr;
    size_t length = 0;
    void* mmap_base = nullptr;
    size_t mmap_length = 0;
    int fd = -1;
};

// 数据格式枚举
enum class DataFormat {
    FVECS,  // .fvecs format (float)
    BVECS,  // .bvecs format (unsigned char)
    IVECS   // .ivecs format (int)
};

// 改进的内存映射函数，返回 RAII 管理的资源
std::tuple<MMapResource, const unsigned char*, size_t> mmap_bvecs_managed(const std::string& filename);
std::tuple<MMapResource, const float*, size_t> mmap_fvecs_managed(const std::string& filename);
std::tuple<MMapResource, const int*, size_t> mmap_ivecs_managed(const std::string& filename);

// 原有的 mmap 函数（保持向后兼容）
std::tuple<const unsigned char*, size_t, size_t> mmap_bvecs(const std::string& filename);
std::tuple<const float*, size_t, size_t> mmap_fvecs(const std::string& filename);
std::tuple<const int*, size_t, size_t> mmap_ivecs(const std::string& filename);

bool load_cluster_data_mmap(
    int cluster_id,
    unsigned global_dim,
    ClusterMMap& cluster_info,
    const std::string& prefix);

bool load_id_mapping_mmap(
    int cluster_id,
    unsigned& points_num,
    MappingMMap& mapping_info,
    const std::string& prefix);

bool load_nsg_index_mmap(
    int cluster_id,
    unsigned global_dim,
    unsigned points_num,
    efanna2e::IndexNSG*& nsg_index,
    const std::string& prefix);

void load_cluster_specific_data_and_nsg_mmap(
    int cluster_id,
    unsigned global_dim,
    std::map<int, ClusterMMap>& cluster_data_map,
    std::map<int, MappingMMap>& id_mapping_map,
    std::map<int, efanna2e::IndexNSG*>& cluster_nsg_indices,
    const std::string& prefix);

void load_cluster_nsg_mmap(
    int cluster_id,
    unsigned global_dim,
    std::map<int, MappingMMap>& id_mapping_map,
    std::map<int, efanna2e::IndexNSG*>& cluster_nsg_indices,
    const std::string& prefix);

template <typename T>
std::vector<unsigned> load_cluster_point_data_batch(
    int cluster_id,
    unsigned start_point_id,
    unsigned dim,
    T* data_buffer,  // 需要预先分配足够大的空间
    size_t batch_size,  // 批量大小
    const std::string& prefix);

// 通用的数据格式处理函数
template<typename T>
void convert_format_to_float_batch(
    const void* data_ptr,
    DataFormat format,
    size_t start_idx,
    size_t batch_size,
    unsigned dim,
    std::vector<float>& float_data);

// 通用的数据格式转换函数（用于聚类数据构建）
template<typename T>
void convert_format_to_original_batch(
    const void* data_ptr,
    DataFormat format,
    const std::vector<faiss::idx_t>& ids_in_cluster,
    unsigned dim,
    std::vector<T>& output_data);

// 通用的训练数据转换函数
template<typename T>
void convert_format_to_float_training(
    const void* data_ptr,
    DataFormat format,
    size_t train_size,
    unsigned dim,
    std::vector<float>& train_data);

} // namespace CNNS