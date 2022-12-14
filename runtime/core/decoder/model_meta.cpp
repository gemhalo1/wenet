#include "model_meta.h"
#include "utils/json.h"
#include <fstream>
#include <sstream>

using namespace json;

namespace wenet {

static bool parseIntField(JSON& field, int& value) {
  if(!field.IsNull()) {
    value = atoi(field.ToString().c_str());
    return true;
  }
  return false;
}

ModelMetadata ModelMetadata::readJsonFile(const char* path) {

  std::ifstream meta_stream;
  meta_stream.open(path, std::ios::in);
  if (meta_stream.is_open()) {
    std::stringstream buffer;
    buffer << meta_stream.rdbuf();
    std::string jsonStr(buffer.str());
    meta_stream.close();

    auto jsonObj = JSON::Load(jsonStr);

    ModelMetadata meta;
    parseIntField(jsonObj.at("output_size"), meta.encoder_output_size_);
    parseIntField(jsonObj.at("num_blocks"), meta.num_blocks_);
    parseIntField(jsonObj.at("head"), meta.head_);
    parseIntField(jsonObj.at("cnn_module_kernel"), meta.cnn_module_kernel_);
    parseIntField(jsonObj.at("subsampling_rate"), meta.subsampling_rate_);
    parseIntField(jsonObj.at("right_context"), meta.right_context_);
    parseIntField(jsonObj.at("sos_symbol"), meta.sos_);
    parseIntField(jsonObj.at("eos_symbol"), meta.eos_);
    parseIntField(jsonObj.at("is_bidirectional_decoder"), meta.is_bidirectional_decoder_);
    parseIntField(jsonObj.at("chunk_size"), meta.chunk_size_);
    parseIntField(jsonObj.at("left_chunks"), meta.num_left_chunks_);
    parseIntField(jsonObj.at("decoding_window"), meta.decoding_window_);

    return meta;
  }

  return {};
}

}