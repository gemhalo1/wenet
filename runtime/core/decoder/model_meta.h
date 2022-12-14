#ifndef WENET_MODEL_META_H
#define WENET_MODEL_META_H

namespace wenet {

struct ModelMetadata {
  int right_context_ = 1;
  int subsampling_rate_ = 1;
  int sos_ = 0;
  int eos_ = 0;
  int is_bidirectional_decoder_ = 0;
  int chunk_size_ = 16;
  int num_left_chunks_ = -1;  // -1 means all left chunks
  int offset_ = 0;

  int encoder_output_size_ = 0;
  int num_blocks_ = 0;
  int cnn_module_kernel_ = 0;
  int head_ = 0;
  int decoding_window_ = 67;

  static ModelMetadata readJsonFile(const char* path);
};

}

#endif  // WENET_MODEL_META_H
