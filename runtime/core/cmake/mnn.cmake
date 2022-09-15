if(MNN)
  set(MNN_VERSION "2.1.0")
  set(MNN_URL "https://github.com/alibaba/MNN/archive/refs/tags/${MNN_VERSION}.tar.gz")
  set(URL_HASH "SHA256=2e4c337208c1ed4c60be23fd699dce5c23dade6ac6f509e35ef8ab5b4a81b2fa")

  FetchContent_Declare(MNN
    URL ${MNN_URL}
    URL_HASH ${URL_HASH}
  )
  FetchContent_MakeAvailable(MNN)
  include_directories(${MNN_SOURCE_DIR}/include)
  link_directories(${MNN_SOURCE_DIR}/lib)

  add_definitions(-DUSE_MNN)
endif()
