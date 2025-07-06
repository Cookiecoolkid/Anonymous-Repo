#include "data_load.h"
#include <iostream>
#include <sys/stat.h>

namespace CNNS {

// MMapResource 类实现
MMapResource::MMapResource() : mapped_ptr_(nullptr), mapped_size_(0), fd_(-1), is_valid_(false) {}

MMapResource::MMapResource(const std::string& filename, size_t size, int file_descriptor) 
    : mapped_ptr_(nullptr), mapped_size_(size), fd_(file_descriptor), is_valid_(false) {
    
    if (fd_ != -1 && mapped_size_ > 0) {
        mapped_ptr_ = mmap(NULL, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        // Does it work for saving memory?
        // madvise(mapped_ptr_, mapped_size_, MADV_SEQUENTIAL | MADV_WILLNEED);
        madvise(mapped_ptr_, mapped_size_, MADV_RANDOM | MADV_DONTNEED);
        is_valid_ = (mapped_ptr_ != MAP_FAILED);
    }

    std::cout << "MMapResource constructed with size: " << mapped_size_ << std::endl;
}

MMapResource::~MMapResource() {
    cleanup();
}

// 移动构造函数
MMapResource::MMapResource(MMapResource&& other) noexcept 
    : mapped_ptr_(other.mapped_ptr_), mapped_size_(other.mapped_size_), 
      fd_(other.fd_), is_valid_(other.is_valid_) {
    // 重置other对象，避免重复释放
    other.mapped_ptr_ = nullptr;
    other.mapped_size_ = 0;
    other.fd_ = -1;
    other.is_valid_ = false;
}

// 移动赋值运算符
MMapResource& MMapResource::operator=(MMapResource&& other) noexcept {
    if (this != &other) {
        cleanup();
        mapped_ptr_ = other.mapped_ptr_;
        mapped_size_ = other.mapped_size_;
        fd_ = other.fd_;
        is_valid_ = other.is_valid_;
        // 重置other对象，避免重复释放
        other.mapped_ptr_ = nullptr;
        other.mapped_size_ = 0;
        other.fd_ = -1;
        other.is_valid_ = false;
    }
    return *this;
}

void* MMapResource::get_ptr() const { return mapped_ptr_; }
size_t MMapResource::get_size() const { return mapped_size_; }
bool MMapResource::is_valid() const { return is_valid_; }

// 手动释放资源
void MMapResource::cleanup() {
    if (is_valid_ && mapped_ptr_ != nullptr) {
        munmap(mapped_ptr_, mapped_size_);
        mapped_ptr_ = nullptr;
        is_valid_ = false;
        std::cout << "MMapResource cleaned up" << std::endl;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
    mapped_size_ = 0;
}

// 重置为新的映射
bool MMapResource::remap(const std::string& filename, size_t size, int file_descriptor) {
    cleanup();
    mapped_size_ = size;
    fd_ = file_descriptor;
    if (fd_ != -1 && mapped_size_ > 0) {
        mapped_ptr_ = mmap(NULL, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        is_valid_ = (mapped_ptr_ != MAP_FAILED);
    }
    return is_valid_;
}

// 改进的 mmap 函数，返回 RAII 管理的资源
std::tuple<MMapResource, const unsigned char*, size_t> mmap_bvecs_managed(const std::string& filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Cannot open file for memory mapping: " << filename << std::endl;
        return {MMapResource(), nullptr, 0};
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "fstat failed on " << filename << std::endl;
        close(fd);
        return {MMapResource(), nullptr, 0};
    }
    
    size_t file_size = sb.st_size;
    MMapResource mmap_res(filename, file_size, fd);
    
    if (!mmap_res.is_valid()) {
        std::cerr << "mmap failed on " << filename << std::endl;
        return {MMapResource(), nullptr, 0};
    }
    
    // 读取全局维度信息
    unsigned global_dim = *reinterpret_cast<unsigned*>(mmap_res.get_ptr());
    
    // 计算点数：每个向量有 4字节维度 + dim字节数据
    size_t points_num = file_size / (4 + global_dim);
    
    // 数据指针(dim + data + dim + data...)
    const unsigned char* data_ptr = reinterpret_cast<const unsigned char*>(mmap_res.get_ptr());
    
    return {std::move(mmap_res), data_ptr, points_num};
}

std::tuple<MMapResource, const float*, size_t> mmap_fvecs_managed(const std::string& filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Cannot open file for memory mapping: " << filename << std::endl;
        return {MMapResource(), nullptr, 0};
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "fstat failed on " << filename << std::endl;
        close(fd);
        return {MMapResource(), nullptr, 0};
    }
    
    size_t file_size = sb.st_size;
    MMapResource mmap_res(filename, file_size, fd);
    
    if (!mmap_res.is_valid()) {
        std::cerr << "mmap failed on " << filename << std::endl;
        return {MMapResource(), nullptr, 0};
    }
    
    // 读取全局维度信息
    unsigned global_dim = *reinterpret_cast<unsigned*>(mmap_res.get_ptr());
    
    // 计算点数：每个向量有 4字节维度 + dim*4字节数据
    size_t points_num = file_size / ((global_dim + 1) * 4);
    
    // 数据指针(dim + data + dim + data...)
    const float* data_ptr = reinterpret_cast<const float*>(mmap_res.get_ptr());
    
    return {std::move(mmap_res), data_ptr, points_num};
}

std::tuple<MMapResource, const int*, size_t> mmap_ivecs_managed(const std::string& filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Cannot open file for memory mapping: " << filename << std::endl;
        return {MMapResource(), nullptr, 0};
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "fstat failed on " << filename << std::endl;
        close(fd);
        return {MMapResource(), nullptr, 0};
    }
    
    size_t file_size = sb.st_size;
    MMapResource mmap_res(filename, file_size, fd);
    
    if (!mmap_res.is_valid()) {
        std::cerr << "mmap failed on " << filename << std::endl;
        return {MMapResource(), nullptr, 0};
    }
    
    // 读取全局维度信息
    unsigned global_dim = *reinterpret_cast<unsigned*>(mmap_res.get_ptr());
    
    // 计算点数：每个向量有 4字节维度 + dim*4字节数据
    size_t points_num = file_size / ((global_dim + 1) * 4);
    
    // 数据指针(dim + data + dim + data...)
    const int* data_ptr = reinterpret_cast<const int*>(mmap_res.get_ptr());
    
    return {std::move(mmap_res), data_ptr, points_num};
}

// 原有的 mmap 函数实现（保持向后兼容）
std::tuple<const unsigned char*, size_t, size_t> mmap_bvecs(const std::string& filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Cannot open file for memory mapping: " << filename << std::endl;
        return {nullptr, 0, 0};
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "fstat failed on " << filename << std::endl;
        close(fd);
        return {nullptr, 0, 0};
    }
    
    size_t file_size = sb.st_size;
    void* mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    if (mapped == MAP_FAILED) {
        std::cerr << "mmap failed on " << filename << std::endl;
        close(fd);
        return {nullptr, 0, 0};
    }
    
    // 读取全局维度信息
    unsigned global_dim = *reinterpret_cast<unsigned*>(mapped);
    
    // 计算点数：每个向量有 4字节维度 + dim字节数据
    size_t points_num = file_size / (4 + global_dim);
    
    // 数据指针(dim + data + dim + data...)
    const unsigned char* data_ptr = reinterpret_cast<const unsigned char*>(mapped);
    
    return {data_ptr, file_size, points_num};
}

std::tuple<const float*, size_t, size_t> mmap_fvecs(const std::string& filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Cannot open file for memory mapping: " << filename << std::endl;
        return {nullptr, 0, 0};
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "fstat failed on " << filename << std::endl;
        close(fd);
        return {nullptr, 0, 0};
    }
    
    size_t file_size = sb.st_size;
    void* mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    if (mapped == MAP_FAILED) {
        std::cerr << "mmap failed on " << filename << std::endl;
        close(fd);
        return {nullptr, 0, 0};
    }
    
    // 读取全局维度信息
    unsigned global_dim = *reinterpret_cast<unsigned*>(mapped);
    
    // 计算点数：每个向量有 4字节维度 + dim*4字节数据
    size_t points_num = file_size / ((global_dim + 1) * 4);
    
    // 数据指针(dim + data + dim + data...)
    const float* data_ptr = reinterpret_cast<const float*>(mapped);
    
    return {data_ptr, file_size, points_num};
}

std::tuple<const int*, size_t, size_t> mmap_ivecs(const std::string& filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Cannot open file for memory mapping: " << filename << std::endl;
        return {nullptr, 0, 0};
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "fstat failed on " << filename << std::endl;
        close(fd);
        return {nullptr, 0, 0};
    }
    
    size_t file_size = sb.st_size;
    void* mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    if (mapped == MAP_FAILED) {
        std::cerr << "mmap failed on " << filename << std::endl;
        close(fd);
        return {nullptr, 0, 0};
    }
    
    // 读取全局维度信息
    unsigned global_dim = *reinterpret_cast<unsigned*>(mapped);
    
    // 计算点数：每个向量有 4字节维度 + dim*4字节数据
    size_t points_num = file_size / ((global_dim + 1) * 4);
    
    // 数据指针(dim + data + dim + data...)
    const int* data_ptr = reinterpret_cast<const int*>(mapped);
    
    return {data_ptr, file_size, points_num};
}

bool load_cluster_data_mmap(
    int cluster_id,
    unsigned global_dim,
    ClusterMMap& cluster_info,
    const std::string& prefix) {
    
    std::string filename = prefix + "/cluster_data/cluster_" + std::to_string(cluster_id) + ".data";
    
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Cannot open cluster file " << filename << std::endl;
        return false;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "fstat failed on " << filename << std::endl;
        close(fd);
        return false;
    }

    size_t file_size = sb.st_size;
    void* mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    madvise(mapped, file_size, MADV_RANDOM | MADV_DONTNEED);

    if (mapped == MAP_FAILED) {
        std::cerr << "mmap failed on " << filename << std::endl;
        close(fd);
        return false;
    }

    size_t vec_size_bytes = global_dim* sizeof(float);
    size_t points_num = file_size / vec_size_bytes;

    cluster_info.data_ptr = reinterpret_cast<float*>(mapped);
    cluster_info.points_num = points_num;
    cluster_info.dim = global_dim;
    cluster_info.mmap_base = mapped;
    cluster_info.mmap_length = file_size;
    cluster_info.fd = fd;
    return true;
}

bool load_id_mapping_mmap(
    int cluster_id,
    unsigned& points_num,
    MappingMMap& mapping_info,
    const std::string& prefix) {

    std::string filename = prefix + "/mapping/mapping_" + std::to_string(cluster_id);
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Cannot open mapping file " << filename << std::endl;
        return false;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "fstat failed on " << filename << std::endl;
        close(fd);
        return false;
    }

    size_t file_size = sb.st_size;
    
    // 首先读取points_num
    unsigned file_points_num;
    ssize_t points_num_read = pread(fd, &file_points_num, sizeof(unsigned), 0);
    if (points_num_read != sizeof(unsigned)) {
        std::cerr << "Failed to read points_num from mapping file" << std::endl;
        close(fd);
        return false;
    }
    
    points_num = file_points_num;
    // 计算实际的mapping数据大小（减去points_num的大小）
    size_t mapping_data_size = file_size - sizeof(unsigned);
    
    // 检查剩余数据大小是否是sizeof(faiss::idx_t)的整数倍
    if (mapping_data_size % sizeof(faiss::idx_t) != 0) {
        std::cerr << "Mapping data size is not a multiple of int64_t size" << std::endl;
        close(fd);
        return false;
    }
    
    // 检查数据大小是否与points_num匹配
    size_t expected_size = points_num * sizeof(faiss::idx_t);
    if (mapping_data_size != expected_size) {
        std::cerr << "Mapping data size mismatch. Expected: " << expected_size 
                  << " bytes, Got: " << mapping_data_size << " bytes" << std::endl;
        close(fd);
        return false;
    }

    void* mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    madvise(mapped, file_size, MADV_RANDOM | MADV_DONTNEED);

    if (mapped == MAP_FAILED) {
        std::cerr << "mmap failed on " << filename << std::endl;
        close(fd);
        return false;
    }

    // mapping数据从points_num之后开始
    mapping_info.data = reinterpret_cast<faiss::idx_t*>(reinterpret_cast<char*>(mapped) + sizeof(unsigned));
    mapping_info.length = points_num;
    mapping_info.mmap_base = mapped;
    mapping_info.mmap_length = file_size;
    mapping_info.fd = fd;
    return true;
}

bool load_nsg_index_mmap(
    int cluster_id,
    unsigned global_dim,
    unsigned points_num,
    efanna2e::IndexNSG*& nsg_index,
    const std::string& prefix) {
    
    std::string filename = prefix + "/nsg_graph/nsg_" + std::to_string(cluster_id) + ".nsg";
    struct stat buffer;
    if (stat(filename.c_str(), &buffer) != 0) {
        std::cerr << "NSG file does not exist: " << filename << std::endl;
        return false;
    }

    nsg_index = new efanna2e::IndexNSG(global_dim, points_num, efanna2e::L2, nullptr);
    try {
        nsg_index->Load_mmap_with_dist(filename.c_str());
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load NSG index: " << e.what() << std::endl;
        delete nsg_index;
        nsg_index = nullptr;
        return false;
    }
}

void load_cluster_specific_data_and_nsg_mmap(
    int cluster_id,
    unsigned global_dim,
    std::map<int, ClusterMMap>& cluster_data_map,
    std::map<int, MappingMMap>& id_mapping_map,
    std::map<int, efanna2e::IndexNSG*>& cluster_nsg_indices,
    const std::string& prefix) {

    try {
        // 创建并加载聚类数据
        cluster_data_map[cluster_id] = ClusterMMap();
        if (!load_cluster_data_mmap(cluster_id, global_dim, cluster_data_map[cluster_id], prefix)) {
            throw std::runtime_error("Failed to load cluster data");
        }

        // 创建并加载映射数据
        id_mapping_map[cluster_id] = MappingMMap();
        unsigned points_num;
        if (!load_id_mapping_mmap(cluster_id, points_num, id_mapping_map[cluster_id], prefix)) {
            throw std::runtime_error("Failed to load id mapping");
        }
        
        // 加载NSG索引
        efanna2e::IndexNSG* nsg_index = nullptr;
        if (!load_nsg_index_mmap(cluster_id, global_dim, points_num, nsg_index, prefix)) {
            throw std::runtime_error("Failed to load NSG index");
        }
        cluster_nsg_indices[cluster_id] = nsg_index;
    } catch (const std::exception& e) {
        std::cerr << "Error loading cluster " << cluster_id << ": " << e.what() << std::endl;
        // 清理已加载的资源
        cluster_data_map.erase(cluster_id);
        id_mapping_map.erase(cluster_id);
        if (cluster_nsg_indices[cluster_id]) {
            delete cluster_nsg_indices[cluster_id];
            cluster_nsg_indices.erase(cluster_id);
        }
    }
}

void load_cluster_nsg_mmap(
    int cluster_id,
    unsigned global_dim,
    std::map<int, MappingMMap>& id_mapping_map,
    std::map<int, efanna2e::IndexNSG*>& cluster_nsg_indices,
    const std::string& prefix) {
    try {
        // 创建并加载映射数据
        id_mapping_map[cluster_id] = MappingMMap();
        unsigned points_num;
        if (!load_id_mapping_mmap(cluster_id, points_num, id_mapping_map[cluster_id], prefix)) {
            throw std::runtime_error("Failed to load id mapping");
        }
        
        // 加载NSG索引
        efanna2e::IndexNSG* nsg_index = nullptr;
        if (!load_nsg_index_mmap(cluster_id, global_dim, points_num, nsg_index, prefix)) {
            throw std::runtime_error("Failed to load NSG index");
        }
        cluster_nsg_indices[cluster_id] = nsg_index;
    } catch (const std::exception& e) {
        std::cerr << "Error loading cluster " << cluster_id << ": " << e.what() << std::endl;
        // 清理已加载的资源
        id_mapping_map.erase(cluster_id);
        if (cluster_nsg_indices[cluster_id]) {
            delete cluster_nsg_indices[cluster_id];
            cluster_nsg_indices.erase(cluster_id);
        }
    }
}

template <typename T>
std::vector<unsigned> load_cluster_point_data_batch(
    int cluster_id,
    unsigned start_point_id,
    unsigned dim,
    T* data_buffer,
    size_t batch_size,
    const std::string& prefix) {
    // 加载数据文件
    std::string data_filename = prefix + "/cluster_data/cluster_" + std::to_string(cluster_id) + ".data";
    int data_fd = open(data_filename.c_str(), O_RDONLY);
    if (data_fd == -1) {
        throw std::runtime_error("Cannot open data file " + data_filename);
    }

    // 计算每个向量的字节大小
    size_t vec_size_bytes = dim * sizeof(T);
    
    // 计算起始偏移量
    size_t start_offset = start_point_id * vec_size_bytes;
    
    // 读取数据
    ssize_t actual_read_size = pread(data_fd, data_buffer, batch_size, start_offset);
    if (actual_read_size == -1) {
        close(data_fd);
        throw std::runtime_error("Failed to read data");
    }
    
    // 计算实际读取的点数
    unsigned actual_points = actual_read_size / vec_size_bytes;
    
    // 生成已加载点的ID列表
    std::vector<unsigned> loaded_point_ids;
    for (unsigned i = 0; i < actual_points; i++) {
        loaded_point_ids.push_back(start_point_id + i);
    }
    
    close(data_fd);
    return loaded_point_ids;
}

#ifdef CNNS_TEMPLATE_INSTANTIATION
// 显式模板实例化
template std::vector<unsigned> CNNS::load_cluster_point_data_batch<float>(int cluster_id, unsigned start_point_id, unsigned dim, float* data_buffer, size_t batch_size, const std::string& prefix);
#endif

// 通用数据格式处理函数实现
template<typename T>
void convert_format_to_float_batch(
    const void* data_ptr,
    DataFormat format,
    size_t start_idx,
    size_t batch_size,
    unsigned dim,
    std::vector<float>& float_data) {
    
    switch (format) {
        case DataFormat::BVECS: {
            const unsigned char* bvec_data = static_cast<const unsigned char*>(data_ptr);
            #pragma omp parallel for
            for (size_t i = 0; i < batch_size; ++i) {
                // 每个向量：4字节维度 + dim字节数据
                const unsigned char* vec_start = bvec_data + (start_idx + i) * (4 + dim);
                const unsigned char* vec_data = vec_start + 4; // 跳过维度字段
                for (size_t j = 0; j < dim; ++j) {
                    float_data[i * dim + j] = static_cast<float>(vec_data[j]);
                }
            }
            break;
        }
        case DataFormat::FVECS: {
            const float* fvec_data = static_cast<const float*>(data_ptr);
            #pragma omp parallel for
            for (size_t i = 0; i < batch_size; ++i) {
                // 每个向量：4字节维度 + dim*4字节数据
                const float* vec_start = fvec_data + (start_idx + i) * (1 + dim);
                const float* vec_data = vec_start + 1; // 跳过维度字段
                std::copy(vec_data, vec_data + dim, float_data.begin() + i * dim);
            }
            break;
        }
        case DataFormat::IVECS: {
            const int* ivec_data = static_cast<const int*>(data_ptr);
            #pragma omp parallel for
            for (size_t i = 0; i < batch_size; ++i) {
                // 每个向量：4字节维度 + dim*4字节数据
                const int* vec_start = ivec_data + (start_idx + i) * (1 + dim);
                const int* vec_data = vec_start + 1; // 跳过维度字段
                for (size_t j = 0; j < dim; ++j) {
                    float_data[i * dim + j] = static_cast<float>(vec_data[j]);
                }
            }
            break;
        }
    }
}

template<typename T>
void convert_format_to_original_batch(
    const void* data_ptr,
    DataFormat format,
    const std::vector<faiss::idx_t>& ids_in_cluster,
    unsigned dim,
    std::vector<T>& output_data) {
    
    switch (format) {
        case DataFormat::BVECS: {
            const unsigned char* bvec_data = static_cast<const unsigned char*>(data_ptr);
            #pragma omp parallel for
            for (size_t idx = 0; idx < ids_in_cluster.size(); ++idx) {
                faiss::idx_t id = ids_in_cluster[idx];
                // 每个向量：4字节维度 + dim字节数据
                const unsigned char* vec_start = bvec_data + id * (4 + dim);
                const unsigned char* vec_data = vec_start + 4; // 跳过维度字段
                std::copy(vec_data, vec_data + dim, output_data.begin() + idx * dim);
            }
            break;
        }
        case DataFormat::FVECS: {
            const float* fvec_data = static_cast<const float*>(data_ptr);
            #pragma omp parallel for
            for (size_t idx = 0; idx < ids_in_cluster.size(); ++idx) {
                faiss::idx_t id = ids_in_cluster[idx];
                // 每个向量：4字节维度 + dim*4字节数据
                const float* vec_start = fvec_data + id * (1 + dim);
                const float* vec_data = vec_start + 1; // 跳过维度字段
                std::copy(vec_data, vec_data + dim, output_data.begin() + idx * dim);
            }
            break;
        }
        case DataFormat::IVECS: {
            const int* ivec_data = static_cast<const int*>(data_ptr);
            #pragma omp parallel for
            for (size_t idx = 0; idx < ids_in_cluster.size(); ++idx) {
                faiss::idx_t id = ids_in_cluster[idx];
                // 每个向量：4字节维度 + dim*4字节数据
                const int* vec_start = ivec_data + id * (1 + dim);
                const int* vec_data = vec_start + 1; // 跳过维度字段
                std::copy(vec_data, vec_data + dim, output_data.begin() + idx * dim);
            }
            break;
        }
    }
}

template<typename T>
void convert_format_to_float_training(
    const void* data_ptr,
    DataFormat format,
    size_t train_size,
    unsigned dim,
    std::vector<float>& train_data) {
    
    switch (format) {
        case DataFormat::BVECS: {
            const unsigned char* bvec_data = static_cast<const unsigned char*>(data_ptr);
            #pragma omp parallel for
            for (size_t i = 0; i < train_size; ++i) {
                // 每个向量：4字节维度 + dim字节数据
                const unsigned char* vec_start = bvec_data + i * (4 + dim);
                const unsigned char* vec_data = vec_start + 4; // 跳过维度字段
                for (size_t j = 0; j < dim; ++j) {
                    train_data[i * dim + j] = static_cast<float>(vec_data[j]);
                }
            }
            break;
        }
        case DataFormat::FVECS: {
            const float* fvec_data = static_cast<const float*>(data_ptr);
            #pragma omp parallel for
            for (size_t i = 0; i < train_size; ++i) {
                // 每个向量：4字节维度 + dim*4字节数据
                const float* vec_start = fvec_data + i * (1 + dim);
                const float* vec_data = vec_start + 1; // 跳过维度字段
                std::copy(vec_data, vec_data + dim, train_data.begin() + i * dim);
            }
            break;
        }
        case DataFormat::IVECS: {
            const int* ivec_data = static_cast<const int*>(data_ptr);
            #pragma omp parallel for
            for (size_t i = 0; i < train_size; ++i) {
                // 每个向量：4字节维度 + dim*4字节数据
                const int* vec_start = ivec_data + i * (1 + dim);
                const int* vec_data = vec_start + 1; // 跳过维度字段
                for (size_t j = 0; j < dim; ++j) {
                    train_data[i * dim + j] = static_cast<float>(vec_data[j]);
                }
            }
            break;
        }
    }
}

// 显式模板实例化
template void convert_format_to_float_batch<unsigned char>(
    const void* data_ptr, DataFormat format, size_t start_idx, 
    size_t batch_size, unsigned dim, std::vector<float>& float_data);

template void convert_format_to_float_batch<float>(
    const void* data_ptr, DataFormat format, size_t start_idx, 
    size_t batch_size, unsigned dim, std::vector<float>& float_data);

template void convert_format_to_float_batch<int>(
    const void* data_ptr, DataFormat format, size_t start_idx, 
    size_t batch_size, unsigned dim, std::vector<float>& float_data);

template void convert_format_to_original_batch<unsigned char>(
    const void* data_ptr, DataFormat format, 
    const std::vector<faiss::idx_t>& ids_in_cluster, unsigned dim, 
    std::vector<unsigned char>& output_data);

template void convert_format_to_original_batch<float>(
    const void* data_ptr, DataFormat format, 
    const std::vector<faiss::idx_t>& ids_in_cluster, unsigned dim, 
    std::vector<float>& output_data);

template void convert_format_to_original_batch<int>(
    const void* data_ptr, DataFormat format, 
    const std::vector<faiss::idx_t>& ids_in_cluster, unsigned dim, 
    std::vector<int>& output_data);

template void convert_format_to_float_training<unsigned char>(
    const void* data_ptr, DataFormat format, size_t train_size, 
    unsigned dim, std::vector<float>& train_data);

template void convert_format_to_float_training<float>(
    const void* data_ptr, DataFormat format, size_t train_size, 
    unsigned dim, std::vector<float>& train_data);

template void convert_format_to_float_training<int>(
    const void* data_ptr, DataFormat format, size_t train_size, 
    unsigned dim, std::vector<float>& train_data);

} // namespace CNNS