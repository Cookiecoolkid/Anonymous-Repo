#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>
#include <set>
#include <list>
#include <unordered_map>

namespace efanna2e {

class IndexNSG : public Index {
 public:
  explicit IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexNSG();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;
  void SearchWithOptGraph(
      const float *query,
      size_t K,
      const Parameters &parameters,
      unsigned *indices);
  void OptimizeGraph(float* data);

  void Save_with_data(const char *filename);
  void Save_mmap(const char *filename);
  void Load_mmap(const char *filename);
  void Save_mmap_with_dist(const char *filename);
  void Load_mmap_with_dist(const char *filename);

  void Search_mmap(const float *query, const float *x, size_t K,
                  const Parameters &parameters, unsigned *indices);
  void Search_mmap_with_dist(const float *query, const float *x, size_t K,
                  const Parameters &parameters, unsigned *indices, float *distances, unsigned nearest_point_id, unsigned* hops = nullptr);

  void Search_mmap_with_dist_pread(const float *query, const float *x, size_t K, int64_t cluster_id, 
                std::string prefix, const Parameters &parameters, unsigned *indices, float *distances, float cluster_min_distance);

  protected:
    typedef std::vector<std::vector<unsigned > > CompactGraph;
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;

    CompactGraph final_graph_;

    Index *initializer_ = nullptr;
    void init_graph(const Parameters &parameters);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_);
    void sync_prune(unsigned q, std::vector<Neighbor>& pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_);
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_);
    void Load_nn_graph(const char *filename);
    void tree_grow(const Parameters &parameter);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);

    // 辅助函数：确保点数据已加载
    void ensure_point_loaded(unsigned point_id, int64_t cluster_id, const std::string& prefix);
    
    // LRU缓存管理函数
    void evict_lru_batch();
    unsigned get_batch_id(unsigned point_id) const;
    float* get_point_data(unsigned point_id);

  private:
    unsigned width;
    unsigned ep_;
    std::vector<std::mutex> locks;
    char* opt_graph_ = nullptr;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    KNNGraph nnd_graph;

    // mmap相关成员
    char* mmap_start = nullptr;  // mmap映射的起始地址
    size_t mmap_size = 0;       // mmap映射的大小
    unsigned* prefix_k = nullptr; // 前缀和数组的起始位置
    unsigned* neighbors_start = nullptr; // 邻居数组的起始位置
    unsigned* dist_start = nullptr; // 距离数组的起始位置
    
    // 缓存相关成员
    std::unordered_map<unsigned, std::vector<float>> batch_cache_;  // batch_id -> data
    std::list<unsigned> lru_list_;  // LRU队列，存储batch_id
    std::unordered_map<unsigned, std::list<unsigned>::iterator> lru_map_;  // batch_id -> LRU迭代器
    size_t batch_size_ = 64 * 1024;     // 单个batch大小 64KB (128维的一个点需要512B，也就是每次load 128个点)
    size_t max_cache_size_ = 8 * 1024 * 1024;  // 最大缓存大小 8MB
    size_t current_cache_size_ = 0;     // 当前缓存大小
};

}

#endif //EFANNA2E_INDEX_NSG_H
