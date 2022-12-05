if(ONNX)
  set(ONNX_VERSION "1.12.1")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-win-x64-${ONNX_VERSION}.zip")
    set(URL_HASH "SHA256=484b08c55867963bd8f74cc39d7c9b6199260f1184839cc40f37e50304597364")
  elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz")
    set(URL_HASH "SHA256=8f6eb9e2da9cf74e7905bf3fc687ef52e34cc566af7af2f92dafe5a5d106aa3d")
  elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-x64-${ONNX_VERSION}.tgz")
    set(URL_HASH "SHA256=71517c8571186eddd31e78134ac441571494fc2f524153165f4a2fec22940d66")
  else()
    message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Windows', 'Linux' or 'Darwin')")
  endif()

  FetchContent_Declare(onnxruntime
    URL ${ONNX_URL}
    URL_HASH ${URL_HASH}
  )
  FetchContent_MakeAvailable(onnxruntime)
  include_directories(${onnxruntime_SOURCE_DIR}/include)
  link_directories(${onnxruntime_SOURCE_DIR}/lib)

  if(MSVC)
    file(GLOB ONNX_DLLS "${onnxruntime_SOURCE_DIR}/lib/*.dll")
    file(COPY ${ONNX_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
  endif()

  add_definitions(-DUSE_ONNX)
endif()
