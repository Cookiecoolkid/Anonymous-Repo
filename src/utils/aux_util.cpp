#include "aux_util.h"
#include "index_build.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cctype>
#include <string>

namespace CNNS {

std::vector<float> load_fvecs(const std::string& filename, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Open file error: " << filename << std::endl;
        exit(1);
    }
    
    in.read((char*)&dim, 4);
    
    in.seekg(0, std::ios::end);
    size_t fsize = in.tellg();
    num = fsize / ((dim + 1) * 4);
    
    std::vector<float> data(num * dim);
    
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; ++i) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data.data() + i * dim), dim * sizeof(float));
    }
    in.close();
    
    return data;
}

std::vector<unsigned char> load_bvecs(const std::string& filename, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Open file error: " << filename << std::endl;
        exit(1);
    }
    
    in.read((char*)&dim, 4);
    
    in.seekg(0, std::ios::end);
    size_t fsize = in.tellg();
    num = fsize / (4 + dim);  // 4 bytes for dim + dim bytes for components
    
    std::vector<unsigned char> data(num * dim);
    
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; ++i) {
        in.seekg(4, std::ios::cur);  // Skip dimension
        in.read((char*)(data.data() + i * dim), dim * sizeof(unsigned char));
    }
    in.close();
    
    return data;
}

std::vector<int> load_ivecs(const std::string& filename, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Open file error: " << filename << std::endl;
        exit(1);
    }
    
    in.read((char*)&dim, 4);
    
    in.seekg(0, std::ios::end);
    size_t fsize = in.tellg();
    num = fsize / ((dim + 1) * 4);
    
    std::vector<int> data(num * dim);
    
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; ++i) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data.data() + i * dim), dim * sizeof(int));
    }
    in.close();
    
    return data;
}

std::vector<std::vector<unsigned>> loadGT(const char* filename) {
    std::ifstream in(filename, std::ios::binary | std::ios::in);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + std::string(filename));
    }

    std::vector<std::vector<unsigned>> results;
    while (in) {
        unsigned GK;
        in.read((char*)&GK, sizeof(unsigned));
        if (!in) break;

        std::vector<unsigned> result(GK);
        in.read((char*)result.data(), GK * sizeof(unsigned));
        if (!in) break;

        results.push_back(result);
    }

    in.close();
    return results;
}

std::vector<float> load_centroids(const std::string& filename, int& n_clusters, int& m, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Open file error: " << filename << std::endl;
        exit(1);
    }
    
    in.read((char*)&n_clusters, sizeof(n_clusters));
    in.read((char*)&m, sizeof(m));
    in.read((char*)&dim, sizeof(dim));
    
    size_t total_points = n_clusters * (m + 1);
    std::vector<float> centroids(total_points * dim);
    
    for (size_t i = 0; i < total_points; ++i) {
        unsigned point_dim;
        in.read((char*)&point_dim, sizeof(point_dim));
        if (point_dim != dim) {
            std::cerr << "Dimension mismatch in centroids file" << std::endl;
            exit(1);
        }
        in.read((char*)(centroids.data() + i * dim), dim * sizeof(float));
    }
    
    in.close();
    return centroids;
}

bool load_cluster_data(
    int cluster_id,
    unsigned global_dim,
    float*& cluster_data,
    unsigned& points_num,
    const std::string& prefix) {
    
    std::string cluster_filename = prefix + "/cluster_data/cluster_" + std::to_string(cluster_id) + ".fvecs";
    unsigned dim_cluster_data;
    std::ifstream in_cluster_data(cluster_filename, std::ios::binary);
    if (!in_cluster_data.is_open()) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Cannot open cluster file " << cluster_filename << std::endl;
        return false;
    }

    in_cluster_data.read((char*)&dim_cluster_data, 4);
    if (dim_cluster_data != global_dim) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Dimension mismatch in cluster file " << cluster_filename 
                  << ". Expected " << global_dim << ", got " << dim_cluster_data << std::endl;
        in_cluster_data.close();
        return false;
    }

    in_cluster_data.seekg(0, std::ios::end);
    size_t fsize_cluster_data = in_cluster_data.tellg();
    points_num = fsize_cluster_data / ((dim_cluster_data + 1) * 4);

    if (points_num == 0) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Cluster " << cluster_id << " has 0 points in " << cluster_filename << std::endl;
        in_cluster_data.close();
        return false;
    }

    cluster_data = new (std::nothrow) float[points_num * dim_cluster_data];
    if (!cluster_data) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Failed to allocate memory for cluster_data for cluster " << cluster_id << std::endl;
        in_cluster_data.close();
        return false;
    }

    in_cluster_data.seekg(0, std::ios::beg);
    for (size_t i = 0; i < points_num; i++) {
        in_cluster_data.seekg(4, std::ios::cur); // Skip dimension for each vector
        in_cluster_data.read((char*)(cluster_data + i * dim_cluster_data), dim_cluster_data * sizeof(float));
    }
    in_cluster_data.close();
    return true;
}

// 加载ID映射
bool load_id_mapping(
    int cluster_id,
    unsigned points_num,
    std::vector<faiss::idx_t>& id_mapping,
    const std::string& prefix) {
    
    std::string mapping_filename = prefix + "/mapping/mapping_" + std::to_string(cluster_id);
    std::ifstream mapping_file(mapping_filename, std::ios::binary);
    if (!mapping_file.is_open()) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Cannot open mapping file " << mapping_filename << std::endl;
        return false;
    }

    id_mapping.resize(points_num);
    mapping_file.read((char*)id_mapping.data(), points_num * sizeof(faiss::idx_t));
    if ((unsigned)mapping_file.gcount() != points_num * sizeof(faiss::idx_t)) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error reading id_mapping file " << mapping_filename << std::endl;
        mapping_file.close();
        return false;
    }
    mapping_file.close();
    return true;
}

DataFormat detect_file_format(const std::string& filename) {
    // 从文件扩展名检测格式
    std::string extension = filename.substr(filename.find_last_of('.'));
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == ".fvecs") {
        return DataFormat::FVECS;
    } else if (extension == ".bvecs") {
        return DataFormat::BVECS;
    } else if (extension == ".ivecs") {
        return DataFormat::IVECS;
    } else {
        // 默认假设为fvecs格式
        std::cerr << "Warning: Unknown file extension '" << extension << "', assuming .fvecs format" << std::endl;
        return DataFormat::FVECS;
    }
}



} // namespace CNNS
