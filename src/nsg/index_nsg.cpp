#include "index_nsg.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "exceptions.h"
#include "parameters.h"
#include "data_load.h"

namespace efanna2e {
#define _CONTROL_NUM 100
IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m,
                   Index *initializer)
    : Index(dimension, n, m), initializer_{initializer} {}

IndexNSG::~IndexNSG() {
    if (distance_ != nullptr) {
        delete distance_;
        distance_ = nullptr;
    }
    if (initializer_ != nullptr) {
        delete initializer_;
        initializer_ = nullptr;
    }
    if (opt_graph_ != nullptr) {
        delete opt_graph_;
        opt_graph_ = nullptr;
    }
    if (mmap_start != nullptr) {
        munmap(mmap_start, mmap_size);
        mmap_start = nullptr;
        mmap_size = 0;
        prefix_k = nullptr;
        neighbors_start = nullptr;
    }
}

void IndexNSG::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *)&width, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    unsigned GK = (unsigned)final_graph_[i].size();
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void IndexNSG::Save_mmap(const char *filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    assert(final_graph_.size() == nd_);

    out.write((char *)&width, sizeof(unsigned));
    out.write((char *)&ep_, sizeof(unsigned));
    out.write((char *)&nd_, sizeof(unsigned));

    std::vector<unsigned> prefix_k(nd_ + 1, 0);
    for (unsigned i = 0; i < nd_; i++) {
        prefix_k[i + 1] = prefix_k[i] + final_graph_[i].size();
    }

    std::vector<unsigned> all_neighbors_flattened(prefix_k[nd_]);
    for (unsigned i = 0; i < nd_; i++) {
        std::copy(final_graph_[i].begin(), final_graph_[i].end(), 
                 all_neighbors_flattened.begin() + prefix_k[i]);
    }

    out.write((char *)prefix_k.data(), sizeof(unsigned) * (nd_ + 1));
    out.write((char *)all_neighbors_flattened.data(), sizeof(unsigned) * prefix_k[nd_]);

    out.close();
}

void IndexNSG::Save_mmap_with_dist(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *)&width, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  out.write((char *)&nd_, sizeof(unsigned));

  std::vector<unsigned> dist(nd_, 0);
  for (unsigned i = 0; i < nd_; i++) {
    // DFS ensure that there is no isolation node
    unsigned min_neighbor_dist = std::numeric_limits<unsigned>::max();
    for (unsigned j = 0; j < final_graph_[i].size(); j++) {
      unsigned id = final_graph_[i][j];
      float dist_ij = distance_->compare(data_ + dimension_ * i, data_ + dimension_ * id, dimension_);
      if (dist_ij < min_neighbor_dist) {
        min_neighbor_dist = dist_ij;
      }
    }
    dist[i] = min_neighbor_dist;
  }

  std::vector<unsigned> prefix_k(nd_ + 1, 0);
  for (unsigned i = 0; i < nd_; i++) {
    prefix_k[i + 1] = prefix_k[i] + final_graph_[i].size();
  }

  std::vector<unsigned> all_neighbors_flattened(prefix_k[nd_]);
  for (unsigned i = 0; i < nd_; i++) {
    std::copy(final_graph_[i].begin(), final_graph_[i].end(), 
             all_neighbors_flattened.begin() + prefix_k[i]);
  }

  out.write((char *)dist.data(), sizeof(unsigned) * nd_);
  out.write((char *)prefix_k.data(), sizeof(unsigned) * (nd_ + 1));
  out.write((char *)all_neighbors_flattened.data(), sizeof(unsigned) * prefix_k[nd_]);

  out.close();
}

void IndexNSG::Load_mmap_with_dist(const char *filename) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "Error: fstat failed on " << filename << std::endl;
        close(fd);
        exit(EXIT_FAILURE);
    }

    mmap_size = sb.st_size;
    mmap_start = (char*)mmap(NULL, mmap_size, PROT_READ, MAP_SHARED, fd, 0);

    madvise(mmap_start, mmap_size, MADV_RANDOM | MADV_DONTNEED);

    if (mmap_start == MAP_FAILED) {
        std::cerr << "Error: mmap failed on " << filename << std::endl;
        close(fd);
        exit(EXIT_FAILURE);
    }

    width = *reinterpret_cast<unsigned*>(mmap_start);
    ep_ = *reinterpret_cast<unsigned*>(mmap_start + 4);
    nd_ = *reinterpret_cast<unsigned*>(mmap_start + 8);

    dist_start = reinterpret_cast<unsigned*>(mmap_start + 12);
    prefix_k = dist_start + nd_;
    neighbors_start = prefix_k + (nd_ + 1);

    close(fd);
}
    

void IndexNSG::Load(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&width, sizeof(unsigned));
  in.read((char *)&ep_, sizeof(unsigned));
  // width=100;
  unsigned cc = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof()) break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *)tmp.data(), k * sizeof(unsigned));
    final_graph_.push_back(tmp);
  }
  cc /= nd_;
  // std::cout<<cc<<std::endl;
}

void IndexNSG::Load_mmap(const char *filename) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "Error: fstat failed on " << filename << std::endl;
        close(fd);
        exit(EXIT_FAILURE);
    }

    mmap_size = sb.st_size;
    mmap_start = (char*)mmap(NULL, mmap_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_start == MAP_FAILED) {
        std::cerr << "Error: mmap failed on " << filename << std::endl;
        close(fd);
        exit(EXIT_FAILURE);
    }

    // 读取基本参数
    width = *reinterpret_cast<unsigned*>(mmap_start);
    ep_ = *reinterpret_cast<unsigned*>(mmap_start + 4);
    nd_ = *reinterpret_cast<unsigned*>(mmap_start + 8);

    // 设置前缀和数组和邻居数组的起始位置
    prefix_k = reinterpret_cast<unsigned*>(mmap_start + 12);  // 跳过width, ep_, n
    neighbors_start = prefix_k + (nd_ + 1);   // 跳过前缀和数组

    close(fd);
}

void IndexNSG::Load_nn_graph(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  in.read((char *)&k, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t num = (unsigned)(fsize / (k + 1) / 4);
  in.seekg(0, std::ios::beg);

  final_graph_.resize(num);
  final_graph_.reserve(num);
  unsigned kk = (k + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(k);
    final_graph_[i].reserve(kk);
    in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
  }
  in.close();
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  boost::dynamic_bitset<> flags{nd_, 0};
  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             boost::dynamic_bitset<> &flags,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    fullset.push_back(retset[i]);
    // flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::init_graph(const Parameters &parameters) {
  float *center = new float[dimension_];
  for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
  for (unsigned i = 0; i < nd_; i++) {
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] += data_[i * dimension_ + j];
    }
  }
  for (unsigned j = 0; j < dimension_; j++) {
    center[j] /= nd_;
  }
  std::vector<Neighbor> tmp, pool;
  ep_ = rand() % nd_;  // random initialize navigating point
  get_neighbors(center, parameters, tmp, pool);
  ep_ = tmp[0].id;
  delete center;
}

void IndexNSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                          const Parameters &parameter,
                          boost::dynamic_bitset<> &flags,
                          SimpleNeighbor *cut_graph_) {
  unsigned range = parameter.Get<unsigned>("R");
  unsigned maxc = parameter.Get<unsigned>("C");
  width = range;
  unsigned start = 0;

  for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
    unsigned id = final_graph_[q][nn];
    if (flags[id]) continue;
    float dist =
        distance_->compare(data_ + dimension_ * (size_t)q,
                           data_ + dimension_ * (size_t)id, (unsigned)dimension_);
    pool.push_back(Neighbor(id, dist, true));
  }

  std::sort(pool.begin(), pool.end());
  std::vector<Neighbor> result;
  if (pool[start].id == q) start++;
  result.push_back(pool[start]);

  while (result.size() < range && (++start) < pool.size() && start < maxc) {
    auto &p = pool[start];
    bool occlude = false;
    for (unsigned t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                     data_ + dimension_ * (size_t)p.id,
                                     (unsigned)dimension_);
      if (djk < p.distance /* dik */) {
        occlude = true;
        break;
      }
    }
    if (!occlude) result.push_back(p);
  }

  SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
  for (size_t t = 0; t < result.size(); t++) {
    des_pool[t].id = result[t].id;
    des_pool[t].distance = result[t].distance;
  }
  if (result.size() < range) {
    des_pool[result.size()].distance = -1;
  }
}

void IndexNSG::InterInsert(unsigned n, unsigned range,
                           std::vector<std::mutex> &locks,
                           SimpleNeighbor *cut_graph_) {
  SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
  for (size_t i = 0; i < range; i++) {
    if (src_pool[i].distance == -1) break;

    SimpleNeighbor sn(n, src_pool[i].distance);
    size_t des = src_pool[i].id;
    SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

    std::vector<SimpleNeighbor> temp_pool;
    int dup = 0;
    {
      LockGuard guard(locks[des]);
      for (size_t j = 0; j < range; j++) {
        if (des_pool[j].distance == -1) break;
        if (n == des_pool[j].id) {
          dup = 1;
          break;
        }
        temp_pool.push_back(des_pool[j]);
      }
    }
    if (dup) continue;

    temp_pool.push_back(sn);
    if (temp_pool.size() > range) {
      std::vector<SimpleNeighbor> result;
      unsigned start = 0;
      std::sort(temp_pool.begin(), temp_pool.end());
      result.push_back(temp_pool[start]);
      while (result.size() < range && (++start) < temp_pool.size()) {
        auto &p = temp_pool[start];
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
          float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                         data_ + dimension_ * (size_t)p.id,
                                         (unsigned)dimension_);
          if (djk < p.distance /* dik */) {
            occlude = true;
            break;
          }
        }
        if (!occlude) result.push_back(p);
      }
      {
        LockGuard guard(locks[des]);
        for (unsigned t = 0; t < result.size(); t++) {
          des_pool[t] = result[t];
        }
      }
    } else {
      LockGuard guard(locks[des]);
      for (unsigned t = 0; t < range; t++) {
        if (des_pool[t].distance == -1) {
          des_pool[t] = sn;
          if (t + 1 < range) des_pool[t + 1].distance = -1;
          break;
        }
      }
    }
  }
}

void IndexNSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
  /*
  std::cout << " graph link" << std::endl;
  unsigned progress=0;
  unsigned percent = 100;
  unsigned step_size = nd_/percent;
  std::mutex progress_lock;
  */
  unsigned range = parameters.Get<unsigned>("R");
  std::vector<std::mutex> locks(nd_);

#pragma omp parallel
  {
    // unsigned cnt = 0;
    std::vector<Neighbor> pool, tmp;
    boost::dynamic_bitset<> flags{nd_, 0};
#pragma omp for schedule(dynamic, 100)
    for (unsigned n = 0; n < nd_; ++n) {
      pool.clear();
      tmp.clear();
      flags.reset();
      get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
      sync_prune(n, pool, parameters, flags, cut_graph_);
      /*
    cnt++;
    if(cnt % step_size == 0){
      LockGuard g(progress_lock);
      std::cout<<progress++ <<"/"<< percent << " completed" << std::endl;
      }
      */
    }
  }

#pragma omp for schedule(dynamic, 100)
  for (unsigned n = 0; n < nd_; ++n) {
    InterInsert(n, range, locks, cut_graph_);
  }
}

void IndexNSG::Build(size_t n, const float *data, const Parameters &parameters) {
  std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
  unsigned range = parameters.Get<unsigned>("R");
  Load_nn_graph(nn_graph_path.c_str());
  data_ = data;
  init_graph(parameters);
  SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
  Link(parameters, cut_graph_);
  final_graph_.resize(nd_);

  for (size_t i = 0; i < nd_; i++) {
    SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
    unsigned pool_size = 0;
    for (unsigned j = 0; j < range; j++) {
      if (pool[j].distance == -1) break;
      pool_size = j;
    }
    pool_size++;
    final_graph_[i].resize(pool_size);
    for (unsigned j = 0; j < pool_size; j++) {
      final_graph_[i][j] = pool[j].id;
    }
  }

  tree_grow(parameters);

  unsigned max = 0, min = 1e6, avg = 0;
  for (size_t i = 0; i < nd_; i++) {
    auto size = final_graph_[i].size();
    max = max < size ? size : max;
    min = min > size ? size : min;
    avg += size;
    // printf("i = %ld, Max = %d, Min = %d, Cur_size = %ld\n", i, max, min, size);
  }
  avg /= 1.0 * nd_;
  printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);

  has_built = true;
  delete cut_graph_;
}

void IndexNSG::Search(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  data_ = x;
  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  unsigned tmp_l = 0;
  for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
    init_ids[tmp_l] = final_graph_[ep_][tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    float dist =
        distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = true;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;
        float dist =
            distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexNSG::Search_mmap(const float *query, const float *x, size_t K,
                          const Parameters &parameters, unsigned *indices) {
    const unsigned L = parameters.Get<unsigned>("L_search");
    data_ = x;
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    boost::dynamic_bitset<> flags{nd_, 0};

    // 从mmap中读取ep_的邻居
    unsigned tmp_l = 0;
    unsigned ep_neighbors_count = prefix_k[ep_ + 1] - prefix_k[ep_];
    
    for (; tmp_l < L && tmp_l < ep_neighbors_count; tmp_l++) {
        init_ids[tmp_l] = neighbors_start[prefix_k[ep_] + tmp_l];
        flags[init_ids[tmp_l]] = true;
    }

    // 随机填充剩余的init_ids
    while (tmp_l < L) {
        unsigned id = rand() % nd_;
        if (flags[id]) continue;
        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
    }


    // 计算初始距离
    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        float dist = distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
        retset[i] = Neighbor(id, dist, true);
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;

    while (k < (int)L) {
        int nk = L;

        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;

            // std::cout << "n = " << n << std::endl;

            // 从mmap中读取节点n的邻居
            unsigned n_neighbors_start = prefix_k[n];
            unsigned n_neighbors_count = prefix_k[n + 1] - prefix_k[n];

            // std::cout << "n_neighbors_count = " << n_neighbors_count << std::endl;

            for (unsigned m = 0; m < n_neighbors_count; ++m) {
                unsigned id = neighbors_start[n_neighbors_start + m];

                // std::cout << "id = " << id << std::endl;
                
                if (flags[id]) continue;
                flags[id] = 1;
                float dist = distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
                if (dist >= retset[L - 1].distance) continue;
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), L, nn);

                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }

    for (size_t i = 0; i < K; i++) {
        indices[i] = retset[i].id;
    }
}

void IndexNSG::Search_mmap_with_dist(const float *query, const float *x, size_t K,
                                  const Parameters &parameters, unsigned *indices, float *distances, unsigned nearest_point_id, unsigned* hops) {
    const unsigned L = parameters.Get<unsigned>("L_search");
    data_ = x;
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    boost::dynamic_bitset<> flags{nd_, 0};
    
    // 初始化跳数计数器 - 统计路径扩展次数
    unsigned local_hops = 0;

    unsigned tmp_l = 0;

    // 如果提供了nearest_point_id，优先使用它作为起始点
    if (nearest_point_id < nd_) {
        unsigned nearest_point_count = prefix_k[nearest_point_id + 1] - prefix_k[nearest_point_id];
        for (; tmp_l < nearest_point_count && tmp_l < L; tmp_l++) {
            init_ids[tmp_l] = neighbors_start[prefix_k[nearest_point_id] + tmp_l];
            flags[init_ids[tmp_l]] = true;
        }
    } else {
        unsigned ep_neighbors_count = prefix_k[ep_ + 1] - prefix_k[ep_];

        for (; tmp_l < L && tmp_l < ep_neighbors_count; tmp_l++) {
            init_ids[tmp_l] = neighbors_start[prefix_k[ep_] + tmp_l];
            flags[init_ids[tmp_l]] = true;
        }
    }

    // 随机填充剩余的init_ids
    while (tmp_l < L) {
        unsigned id = rand() % nd_;
        if (flags[id]) continue;

        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
    }

    // 计算初始距离
    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        float dist = distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
        retset[i] = Neighbor(id, dist, true);
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;

    while (k < (int)L) {
        int nk = L;
        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;
            // 计算当前点到query的距离
            float dist_n_to_query = distance_->compare(data_ + dimension_ * n, query, (unsigned)dimension_);
            
            // 使用三角不等式进行剪枝
            float min_dist_to_neighbor = dist_start[n];  // 当前点到其邻居的最小距离
            float lower_bound = std::abs(dist_n_to_query - min_dist_to_neighbor);
            
            // 如果下界大于当前最大距离，则跳过
            bool pruned = false;
            if (lower_bound >= retset[L - 1].distance) {
              pruned = true;
            }
            // 从mmap中读取节点n的邻居
            unsigned n_neighbors_start = prefix_k[n];
            unsigned n_neighbors_count = prefix_k[n + 1] - prefix_k[n];

            // 统计从当前节点n扩展到其邻居的跳数
            unsigned expanded = 0;
            for (unsigned m = 0; m < n_neighbors_count; ++m) {
                unsigned id = neighbors_start[n_neighbors_start + m];
                
                if (flags[id]) continue;
                flags[id] = 1;
                // 设置 flag = 1 后直接跳过
                if (pruned) continue;

                // 计算实际距离
                float dist = distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
                
                if (dist >= retset[L - 1].distance) continue;
                
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), L, nn);
                if (r < (int)L) expanded++;
                if (r < nk) nk = r;
            }
            
            // 如果从当前节点扩展了邻居，则增加跳数
            if (expanded > 0) {
                local_hops++;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }

    for (size_t i = 0; i < K; i++) {
        indices[i] = retset[i].id;
        if (distances != nullptr) {
            distances[i] = retset[i].distance;
        }
    }
    
    // 更新跳数统计
    if (hops != nullptr) {
        *hops = local_hops;
    }
}

/*
void IndexNSG::Search_mmap_with_dist(const float *query, const float *x, size_t K,
                                    const Parameters &parameters, unsigned *indices, 
                                    float *distances, unsigned nearest_point_id, unsigned* hops) {
    const unsigned L = parameters.Get<unsigned>("L_search");
    const unsigned beam_width = 4;
    data_ = x;
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    boost::dynamic_bitset<> flags{nd_, 0};
    unsigned local_hops = 0;

    // 初始化候选集（与原逻辑一致）
    unsigned tmp_l = 0;
    if (nearest_point_id < nd_) {
        unsigned nearest_point_count = prefix_k[nearest_point_id + 1] - prefix_k[nearest_point_id];
        for (; tmp_l < nearest_point_count && tmp_l < L; tmp_l++) {
            init_ids[tmp_l] = neighbors_start[prefix_k[nearest_point_id] + tmp_l];
            flags[init_ids[tmp_l]] = true;
        }
    } else {
        unsigned ep_neighbors_count = prefix_k[ep_ + 1] - prefix_k[ep_];
        for (; tmp_l < L && tmp_l < ep_neighbors_count; tmp_l++) {
            init_ids[tmp_l] = neighbors_start[prefix_k[ep_] + tmp_l];
            flags[init_ids[tmp_l]] = true;
        }
    }
    while (tmp_l < L) {
        unsigned id = rand() % nd_;
        if (flags[id]) continue;
        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
    }

    // 计算初始距离
    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        float dist = distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
        retset[i] = Neighbor(id, dist, true);
    }
    std::sort(retset.begin(), retset.begin() + L);

    // Beam Search主循环
    int k = 0;
    while (k < (int)L) {
        int nk = L;
        // 每轮处理前beam_width个候选点
        for (int beam_idx = 0; beam_idx < beam_width && k + beam_idx < (int)L; beam_idx++) {
            int current_k = k + beam_idx;
            if (retset[current_k].flag) {
                retset[current_k].flag = false;
                unsigned n = retset[current_k].id;
                float dist_n_to_query = distance_->compare(data_ + dimension_ * n, query, (unsigned)dimension_);
                float min_dist_to_neighbor = dist_start[n];
                float lower_bound = std::abs(dist_n_to_query - min_dist_to_neighbor);
                bool pruned = (lower_bound >= retset[L - 1].distance);

                // 扩展邻居
                unsigned n_neighbors_start = prefix_k[n];
                unsigned n_neighbors_count = prefix_k[n + 1] - prefix_k[n];
                unsigned expanded = 0;
                for (unsigned m = 0; m < n_neighbors_count; ++m) {
                    unsigned id = neighbors_start[n_neighbors_start + m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    if (pruned) continue;

                    float dist = distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
                    if (dist >= retset[L - 1].distance) continue;

                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    // if (r < (int)L) expanded++;
                    if (r < nk) nk = r; // 更新回溯位置
                }
                // if (expanded > 0) local_hops++;
            }
        }
        local_hops++;

        // 更新k：若发现更优候选点则回溯，否则推进beam_width
        if (nk <= k) 
            k = nk;
        else 
            k += beam_width;
    }

    // 返回结果（与原逻辑一致）
    for (size_t i = 0; i < K; i++) {
        indices[i] = retset[i].id;
        if (distances != nullptr) distances[i] = retset[i].distance;
    }
    if (hops != nullptr) *hops = local_hops;
}

*/
unsigned IndexNSG::get_batch_id(unsigned point_id) const {
    unsigned points_per_batch = batch_size_ / (dimension_ * sizeof(float));
    return point_id / points_per_batch;
}

float* IndexNSG::get_point_data(unsigned point_id) {
    unsigned batch_id = get_batch_id(point_id);
    auto it = batch_cache_.find(batch_id);
    if (it == batch_cache_.end()) {
        return nullptr; // batch未加载
    }
    
    // 更新LRU
    auto lru_it = lru_map_.find(batch_id);
    if (lru_it != lru_map_.end()) {
        lru_list_.erase(lru_it->second);
    }
    lru_list_.push_front(batch_id);
    lru_map_[batch_id] = lru_list_.begin();
    
    // 计算点在batch内的偏移
    unsigned points_per_batch = batch_size_ / (dimension_ * sizeof(float));
    unsigned offset_in_batch = (point_id % points_per_batch) * dimension_;
    
    return it->second.data() + offset_in_batch;
}

void IndexNSG::evict_lru_batch() {
    if (lru_list_.empty()) {
        return;
    }
    
    // 移除最久未使用的batch
    unsigned batch_id_to_evict = lru_list_.back();
    lru_list_.pop_back();
    lru_map_.erase(batch_id_to_evict);
    
    // 计算被移除batch的大小
    size_t batch_data_size = batch_cache_[batch_id_to_evict].size() * sizeof(float);
    current_cache_size_ -= batch_data_size;
    
    // 从缓存中移除
    batch_cache_.erase(batch_id_to_evict);
}

void IndexNSG::ensure_point_loaded(unsigned point_id, int64_t cluster_id, const std::string& prefix) {
    unsigned batch_id = get_batch_id(point_id);
    
    // 检查batch是否已加载
    if (batch_cache_.find(batch_id) != batch_cache_.end()) {
        // batch已加载，更新LRU
        auto lru_it = lru_map_.find(batch_id);
        if (lru_it != lru_map_.end()) {
            lru_list_.erase(lru_it->second);
        }
        lru_list_.push_front(batch_id);
        lru_map_[batch_id] = lru_list_.begin();
        return;
    }
    
    // 计算batch起始点ID
    unsigned points_per_batch = batch_size_ / (dimension_ * sizeof(float));
    unsigned batch_start = batch_id * points_per_batch;
    
    // 检查缓存大小限制
    size_t new_batch_size = batch_size_;
    while (current_cache_size_ + new_batch_size > max_cache_size_) {
        evict_lru_batch();
    }
    
    // 创建新的batch缓存
    std::vector<float> batch_data(batch_size_ / sizeof(float));
    
    // 批量加载数据
    std::vector<unsigned> loaded_ids = CNNS::load_cluster_point_data_batch<float>(
        cluster_id, batch_start, dimension_, batch_data.data(), batch_size_, prefix);
    
    // 添加到缓存
    batch_cache_[batch_id] = std::move(batch_data);
    current_cache_size_ += new_batch_size;
    
    // 更新LRU
    lru_list_.push_front(batch_id);
    lru_map_[batch_id] = lru_list_.begin();
}

void IndexNSG::Search_mmap_with_dist_pread(const float *query, const float *x, size_t K, 
                                  int64_t cluster_id, std::string prefix,
                                  const Parameters &parameters, unsigned *indices, float *distances, float cluster_min_distance) {
    assert(x == nullptr);
    const unsigned L = parameters.Get<unsigned>("L_search");
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    boost::dynamic_bitset<> flags{nd_, 0};

    // 初始化跳数计数器 - 统计路径扩展次数
    unsigned local_hops = 0;

    // 从mmap中读取ep_的邻居
    unsigned tmp_l = 0;
    unsigned ep_neighbors_count = prefix_k[ep_ + 1] - prefix_k[ep_];
    
    for (; tmp_l < L && tmp_l < ep_neighbors_count; tmp_l++) {
        init_ids[tmp_l] = neighbors_start[prefix_k[ep_] + tmp_l];
        flags[init_ids[tmp_l]] = true;
    }

    // 随机填充剩余的init_ids
    while (tmp_l < L) {
        unsigned id = rand() % nd_;
        if (flags[id]) continue;
        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
    }

    // 计算初始距离
    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        
        // 确保点数据已加载
        ensure_point_loaded(id, cluster_id, prefix);
        
        // 获取点数据
        float* point_data = get_point_data(id);
        
        float dist = distance_->compare(point_data, query, (unsigned)dimension_);
        retset[i] = Neighbor(id, dist, true);
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;

    while (k < (int)L) {
        int nk = L;

        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;

            // 确保当前点数据已加载
            ensure_point_loaded(n, cluster_id, prefix);
            
            // 获取当前点数据
            float* n_data = get_point_data(n);
            
            // 计算当前点到query的距离
            float dist_n_to_query = distance_->compare(n_data, query, (unsigned)dimension_);
            
            // 使用三角不等式进行剪枝
            float min_dist_to_neighbor = dist_start[n];  // 当前点到其邻居的最小距离
            float lower_bound = std::abs(dist_n_to_query - min_dist_to_neighbor);
            
            // 如果下界大于当前最大距离，则跳过
            bool is_pruned = false;
            if (lower_bound >= retset[L - 1].distance) is_pruned = true;

            // 从mmap中读取节点n的邻居
            unsigned n_neighbors_start = prefix_k[n];
            unsigned n_neighbors_count = prefix_k[n + 1] - prefix_k[n];

            // 统计从当前节点n扩展到其邻居的跳数
            unsigned expanded_neighbors = 0;
            for (unsigned m = 0; m < n_neighbors_count; ++m) {
                unsigned id = neighbors_start[n_neighbors_start + m];
                
                if (flags[id]) continue;
                flags[id] = 1;
                // 设置 flag = 1 后直接跳过
                if (is_pruned) continue;

                // 确保邻居点数据已加载
                ensure_point_loaded(id, cluster_id, prefix);
                
                // 获取邻居点数据
                float* neighbor_data = get_point_data(id);

                // 计算实际距离
                float dist = distance_->compare(query, neighbor_data, (unsigned)dimension_);

                if (dist >= retset[L - 1].distance) continue;
                
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), L, nn);
                expanded_neighbors++;

                if (r < nk) nk = r;
            }
            
            // 如果从当前节点扩展了邻居，则增加跳数
            if (expanded_neighbors > 0) {
                local_hops++;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }

    for (size_t i = 0; i < K; i++) {
        indices[i] = retset[i].id;
        if (distances != nullptr) {
            distances[i] = retset[i].distance;
        }
    }
}

void IndexNSG::SearchWithOptGraph(const float *query, size_t K,
                                  const Parameters &parameters, unsigned *indices) {
  unsigned L = parameters.Get<unsigned>("L_search");
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;

  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned tmp_l = 0;
  unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * ep_ + data_len);
  unsigned MaxM_ep = *neighbors;
  neighbors++;

  for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
    init_ids[tmp_l] = neighbors[tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
  }
  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float *x = (float *)(opt_graph_ + node_size * id);
    float norm_x = *x;
    x++;
    float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    flags[id] = true;
    L++;
  }
  // std::cout<<L<<std::endl;

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);
      unsigned MaxM = *neighbors;
      neighbors++;
      for (unsigned m = 0; m < MaxM; ++m)
        _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
        if (flags[id]) continue;
        flags[id] = 1;
        float *data = (float *)(opt_graph_ + node_size * id);
        float norm = *data;
        data++;
        float dist = dist_fast->compare(query, data, norm, (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        // if(L+1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexNSG::OptimizeGraph(float *data) {  // use after build or load

  data_ = data;
  data_len = (dimension_ + 1) * sizeof(float);
  neighbor_len = (width + 1) * sizeof(unsigned);
  node_size = data_len + neighbor_len;
  opt_graph_ = (char *)malloc(node_size * nd_);
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
  for (unsigned i = 0; i < nd_; i++) {
    char *cur_node_offset = opt_graph_ + i * node_size;
    float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
    std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
    std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                data_len - sizeof(float));

    cur_node_offset += data_len;
    unsigned k = final_graph_[i].size();
    std::memcpy(cur_node_offset, &k, sizeof(unsigned));
    std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
                k * sizeof(unsigned));
    std::vector<unsigned>().swap(final_graph_[i]);
  }
  CompactGraph().swap(final_graph_);
}

void IndexNSG::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
  unsigned tmp = root;
  std::stack<unsigned> s;
  s.push(root);
  if (!flag[root]) cnt++;
  flag[root] = true;
  while (!s.empty()) {
    unsigned next = nd_ + 1;
    for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
      if (flag[final_graph_[tmp][i]] == false) {
        next = final_graph_[tmp][i];
        break;
      }
    }
    // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
    if (next == (nd_ + 1)) {
      s.pop();
      if (s.empty()) break;
      tmp = s.top();
      continue;
    }
    tmp = next;
    flag[tmp] = true;
    s.push(tmp);
    cnt++;
  }
}

void IndexNSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                        const Parameters &parameter) {
  unsigned id = nd_;
  for (unsigned i = 0; i < nd_; i++) {
    if (flag[i] == false) {
      id = i;
      break;
    }
  }

  if (id == nd_) return;  // No Unlinked Node

  std::vector<Neighbor> tmp, pool;
  get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
  std::sort(pool.begin(), pool.end());

  unsigned found = 0;
  for (unsigned i = 0; i < pool.size(); i++) {
    if (flag[pool[i].id]) {
      // std::cout << pool[i].id << '\n';
      root = pool[i].id;
      found = 1;
      break;
    }
  }
  if (found == 0) {
    while (true) {
      unsigned rid = rand() % nd_;
      if (flag[rid]) {
        root = rid;
        break;
      }
    }
  }
  final_graph_[root].push_back(id);
}
void IndexNSG::tree_grow(const Parameters &parameter) {
  unsigned root = ep_;
  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned unlinked_cnt = 0;
  while (unlinked_cnt < nd_) {
    DFS(flags, root, unlinked_cnt);
    // std::cout << unlinked_cnt << '\n';
    if (unlinked_cnt >= nd_) break;
    findroot(flags, root, parameter);
    // std::cout << "new root"<<":"<<root << '\n';
  }
  for (size_t i = 0; i < nd_; ++i) {
    if (final_graph_[i].size() > width) {
      width = final_graph_[i].size();
    }
  }
}

void IndexNSG::Save_with_data(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  // 按点顺序写入数据
  for (unsigned i = 0; i < nd_; i++) {
    // 写入当前点的邻居数量
    unsigned num_neighbors = final_graph_[i].size();
    out.write((char *)&num_neighbors, sizeof(unsigned));
    
    // 写入当前点的所有邻居数据
    for (unsigned j = 0; j < final_graph_[i].size(); j++) {
      unsigned neighbor_id = final_graph_[i][j];
      out.write((char *)(data_ + dimension_ * neighbor_id), dimension_ * sizeof(float));
    }
  }

  out.close();
}

}
