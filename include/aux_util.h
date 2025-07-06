#ifndef AUX_UTIL_H
#define AUX_UTIL_H

#ifdef _WINDOWS
#include <Windows.h>
typedef HANDLE FileHandle;
#else
#include <unistd.h>
typedef int FileHandle;
#endif

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <memory>
#include <thread>
#include <mutex>
#include <omp.h>
#include <faiss/Index.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <sstream>

// 前向声明
namespace CNNS {
    enum class DataFormat;
}

namespace CNNS {

template <typename T>
T clamp(T value, T min_value, T max_value) {
    if (value < min_value)
        return min_value;
    else if (value > max_value)
        return max_value;
    else
        return value;
}

typedef uint64_t _u64;
typedef int64_t  _s64;
typedef uint32_t _u32;
typedef int32_t  _s32;
typedef uint16_t _u16;
typedef int16_t  _s16;
typedef uint8_t  _u8;
typedef int8_t   _s8;

// 加载fvecs文件
std::vector<float> load_fvecs(const std::string& filename, unsigned& num, unsigned& dim);

// 加载bvecs文件
std::vector<unsigned char> load_bvecs(const std::string& filename, unsigned& num, unsigned& dim);

// 加载ivecs文件
std::vector<int> load_ivecs(const std::string& filename, unsigned& num, unsigned& dim);

// 自动检测文件格式
DataFormat detect_file_format(const std::string& filename);

// 加载ground truth文件
std::vector<std::vector<unsigned>> loadGT(const char* filename);

// 加载质心文件
std::vector<float> load_centroids(const std::string& filename, int& n_clusters, int& m, unsigned& dim);


bool load_cluster_data(
    int cluster_id,
    unsigned global_dim,
    float*& cluster_data,
    unsigned& points_num,
    const std::string& prefix);

bool load_id_mapping(
    int cluster_id,
    unsigned points_num,
    std::vector<faiss::idx_t>& id_mapping,
    const std::string& prefix);


// taken from
// https://github.com/Microsoft/BLAS-on-flash/blob/master/include/utils.h
// round up X to the nearest multiple of Y
#define ROUND_UP(X, Y) \
  ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define DIV_ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0))

// round down X to the nearest multiple of Y
#define ROUND_DOWN(X, Y) (((uint64_t)(X) / (Y)) * (Y))

// alignment tests
#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)
#define IS_4096_ALIGNED(X) IS_ALIGNED(X, 4096)
#define METADATA_SIZE \
  4096  // all metadata of individual sub-component files is written in first
        // 4KB for unified files

#define BUFFER_SIZE_FOR_CACHED_IO (_u64) 1024 * (_u64) 1048576

inline bool file_exists(const std::string& name, bool dirCheck = false) {
  int val;
#ifndef _WINDOWS
  struct stat buffer;
  val = stat(name.c_str(), &buffer);
#else
  // It is the 21st century but Windows API still thinks in 32-bit terms.
  // Turns out calling stat() on a file > 4GB results in errno = 132 (OVERFLOW).
  // How silly is this!? So calling _stat64()
  struct _stat64 buffer;
  val = _stat64(name.c_str(), &buffer);
#endif

  if (val != 0) {
    switch (errno) {
      case EINVAL:
        std::cout << "Invalid argument passed to stat()" << std::endl;
        break;
      case ENOENT:
        // file is not existing, not an issue, so we won't cout anything.
        break;
      default:
        std::cout << "Unexpected error in stat():" << errno << std::endl;
        break;
    }
    return false;
  } else {
    // the file entry exists. If reqd, check if this is a directory.
    return dirCheck ? buffer.st_mode & S_IFDIR : true;
  }
}

inline _u64 get_file_size(const std::string& fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    _u64 end_pos = reader.tellg();
    reader.close();
    return end_pos;
  } else {
    std::cerr << "Could not open file: " << fname << std::endl;
    return 0;
  }
}

inline int delete_file(const std::string& fileName) {
  if (file_exists(fileName)) {
    auto rc = ::remove(fileName.c_str());
    if (rc != 0) {
      std::cerr
          << "Could not delete file: " << fileName
          << " even though it exists. This might indicate a permissions issue. "
             "If you see this message, please contact the diskann team."
          << std::endl;
    }
    return rc;
  } else {
    return 0;
  }
}

#ifdef _WINDOWS
#include <intrin.h>
#include <Psapi.h>


#ifndef AvxSupportedCPU
extern bool AvxSupportedCPU;
#endif

#ifndef Avx2SupportedCPU
extern bool Avx2SupportedCPU;
#endif

inline size_t getMemoryUsage() {
  PROCESS_MEMORY_COUNTERS_EX pmc;
  GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*) &pmc,
                       sizeof(pmc));
  return pmc.PrivateUsage;
}

inline std::string getWindowsErrorMessage(DWORD lastError) {
  char* errorText;
  FormatMessageA(
      // use system message tables to retrieve error text
      FORMAT_MESSAGE_FROM_SYSTEM
          // allocate buffer on local heap for error text
          | FORMAT_MESSAGE_ALLOCATE_BUFFER
          // Important! will fail otherwise, since we're not
          // (and CANNOT) pass insertion parameters
          | FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL,  // unused with FORMAT_MESSAGE_FROM_SYSTEM
      lastError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR) &errorText,  // output
      0,                   // minimum size for output buffer
      NULL);               // arguments - see note

  return errorText != nullptr ? std::string(errorText) : std::string();
}

inline void printProcessMemory(const char* message) {
  PROCESS_MEMORY_COUNTERS counters;
  HANDLE                  h = GetCurrentProcess();
  GetProcessMemoryInfo(h, &counters, sizeof(counters));
  std::cout << message << " [Peaking Working Set size: "
                << counters.PeakWorkingSetSize * 1.0 / (1024.0 * 1024 * 1024)
                << "GB Working set size: "
                << counters.WorkingSetSize * 1.0 / (1024.0 * 1024 * 1024)
                << "GB Private bytes "
                << counters.PagefileUsage * 1.0 / (1024 * 1024 * 1024) << "GB]"
                << std::endl;
}
#else


#ifndef AvxSupportedCPU
extern bool AvxSupportedCPU;
#endif

#ifndef Avx2SupportedCPU
extern bool Avx2SupportedCPU;
#endif

// need to check and change this
inline bool avx2Supported() {
  return true;
}
inline void printProcessMemory(const char*) {
}

inline size_t
getMemoryUsage() {  // for non-windows, we have not implemented this function
  return 0;
}

#endif
// unit:MB
inline size_t getProcessPeakRSS() {
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  return (size_t) rusage.ru_maxrss /1024L;
}

// Return Memory Usage in MB
inline size_t getCurrentRSS() {
  long rss = 0L;
  FILE *fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
      return (size_t) 0L;      /* Can't open? */
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
      fclose(fp);
      return (size_t) 0L;      /* Can't read? */
  }
  fclose(fp);
  return (size_t) (rss * (size_t) sysconf(_SC_PAGESIZE))/1024/1024L;
}

inline size_t getPeakMemoryFromProc() {
    size_t peak = 0;
    FILE* f = fopen("/proc/self/status", "r");
    if (f) {
        char line[128];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "VmPeak:", 7) == 0) {
                sscanf(line + 7, "%lu", &peak); // KB单位
                break;
            }
        }
        fclose(f);
    }
    return peak; // 返回KB值
}

inline size_t getMappedMemory() {
    FILE* f = fopen("/proc/self/maps", "r");
    size_t total = 0;
    if (f) {
        char line[1024];
        while (fgets(line, sizeof(line), f)) {
            void *start, *end;
            if (sscanf(line, "%p-%p", &start, &end) == 2) {
                total += (char*)end - (char*)start;
            }
        }
        fclose(f);
    }
    return total / 1024; // 转换为KB
}

inline size_t getPeakPhysicalMemoryKB() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    size_t hwm = 0;

    while (std::getline(status_file, line)) {
        if (line.find("VmHWM:") == 0) {
            std::istringstream iss(line.substr(6));
            iss >> hwm; // 单位是KB
            break;
        }
    }
    return hwm;
}

struct PageFaultStats {
    size_t minor;
    size_t major;
};

inline PageFaultStats getPageFaultStats() {
    std::ifstream stat_file("/proc/self/stat");
    std::string token;
    PageFaultStats stats = {0, 0};
    
    // 跳过前9个字段
    for (int i = 0; i < 9; i++) 
        stat_file >> token;
    
    stat_file >> stats.minor >> stats.major;
    return stats;
}

} // namespace CNNS

#endif // AUX_UTIL_H
