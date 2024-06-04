#include <cstdint>
#include <string>

#define CUTLASS_MAJOR 2
#define CUTLASS_MINOR 10
#define CUTLASS_PATCH 0
#define CUTLASS_BUILD 
#define CUTLASS_VERSION ((CUTLASS_MAJOR)*100 + (CUTLASS_MINOR)*10 + CUTLASS_PATCH)

namespace cutlass {

  inline uint32_t getVersion() {
    return CUTLASS_VERSION;
  }
  inline uint32_t getVersionMajor() {
    return CUTLASS_MAJOR;
  }
  inline uint32_t getVersionMinor() {
    return CUTLASS_MINOR;
  }
  inline uint32_t getVersionPatch() {
    return CUTLASS_PATCH;
  }
  inline uint32_t getVersionBuild() {
    return CUTLASS_BUILD + 0;
  }
  inline std::string getVersionString() {
    std::string version = "2.10.0";
    if (getVersionBuild()) {
      version += "." + std::to_string(getVersionBuild());
    }
    return version;
  }
  inline std::string getGitRevision() {
    return "6328be78";
  }

} // namespace cutlass
