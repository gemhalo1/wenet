// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 ZeXuan Li (lizexuan@huya.com)
//                    Xingchen Song(sxc19@mails.tsinghua.edu.cn)
//                    hamddct@gmail.com (Mddct)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "decoder/mnn_asr_model.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "utils/string.h"
#include "model_meta.h"

//extern void write_data(const char* filename, void* data, size_t length);

namespace wenet {


MNN::ScheduleConfig MnnAsrModel::scheduleConfig = {
    .type = MNN_FORWARD_CPU,
    .numThread = 1
};

MnnSession::MnnSession(std::shared_ptr<MNN::Interpreter> interpreter, const MNN::ScheduleConfig& scheduleConfig) {
  interpreter_ = std::move(interpreter);
  session_ = interpreter_->createSession(scheduleConfig);
}

MnnSession::~MnnSession() {
  interpreter_->releaseSession(session_);
}

MNN::Tensor* MnnSession::getSessionInput(const char* name) {
  return interpreter_->getSessionInput(session_, name);
}

MNN::Tensor* MnnSession::getSessionOutput(const char* name) {
  return interpreter_->getSessionOutput(session_, name);
}

const std::map<std::string, MNN::Tensor*>& MnnSession::getSessionInputAll() const {
  return interpreter_->getSessionInputAll(session_);
}

const std::map<std::string, MNN::Tensor*>& MnnSession::getSessionOutputAll() const {
  return interpreter_->getSessionOutputAll(session_);
}

void MnnAsrModel::GetInputOutputInfo(
    const std::shared_ptr<MnnSession>& session,
    std::vector<const char*>* in_names, std::vector<const char*>* out_names) {
  auto& inputs = session->getSessionInputAll();

  // Input info
  in_names->clear();

  for(auto& x : inputs) {
    auto& name = x.first;
    auto& tensor = x.second;

    auto node_dims = tensor->shape();
    halide_type_t type = tensor->getType();
    MNN::Tensor::DimensionType dimType = tensor->getDimensionType();

    std::vector<int> strides;
    for(int i = 0; i < node_dims.size(); i++)
      strides.push_back(tensor->stride(i));

    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }

    std::stringstream strides_s;
    for (auto j : strides) {
      strides_s << j;
      strides_s << " ";
    }

    LOG(INFO) << "\tInput " << in_names->size() << " : name=" << name << " type=" << type.code
              << " dimention-type=" << dimType
              << " dims=" << shape.str()
              << " stides=" << strides_s.str();
    ;
    in_names->push_back(name.c_str());
  }

  // Output info
  auto& outputs = session->getSessionOutputAll();
  out_names->clear();

  for(auto& x : outputs) {
    auto& name = x.first;
    auto& tensor = x.second;

    auto node_dims = tensor->shape();
    halide_type_t type = tensor->getType();
    MNN::Tensor::DimensionType dimType = tensor->getDimensionType();

    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tOutput " << out_names->size() << " : name=" << name << " type=" << type.code
              << " dimention-type=" << dimType
              << " dims=" << shape.str();
    out_names->push_back(name.c_str());
  }
}

void MnnAsrModel::Read(const std::string& model_dir) {
  std::string encoder_mnn_path = model_dir + "/encoder.mnn";
  std::string rescore_mnn_path = model_dir + "/decoder.mnn";
  std::string ctc_mnn_path = model_dir + "/ctc.mnn";
  std::string meta_mnn_path = model_dir + "/meta.json";
  std::string pos_mnn_path = model_dir + "/pos_emb.bin";

  // 1. Load sessions
  try {
    encoder_interpreter_.reset(MNN::Interpreter::createFromFile(encoder_mnn_path.c_str()));
    encoder_session_ = std::make_shared<MnnSession>(encoder_interpreter_, scheduleConfig);

    rescore_interpreter_.reset(MNN::Interpreter::createFromFile(rescore_mnn_path.c_str()));
    rescore_session_ = std::make_shared<MnnSession>(rescore_interpreter_, scheduleConfig);
    
    ctc_interpreter_.reset(MNN::Interpreter::createFromFile(ctc_mnn_path.c_str()));
    ctc_session_ = std::make_shared<MnnSession>(ctc_interpreter_, scheduleConfig);
  } catch (std::exception const& e) {
    LOG(ERROR) << "error when load onnx model: " << e.what();
    exit(0);
  }

  // 2. Read metadata
  ModelMetadata meta = ModelMetadata::readJsonFile(meta_mnn_path.c_str());
  encoder_output_size_ = meta.encoder_output_size_;
  num_blocks_ = meta.num_blocks_;
  head_ = meta.head_;
  cnn_module_kernel_ = meta.cnn_module_kernel_;
  subsampling_rate_ = meta.subsampling_rate_;
  right_context_ = meta.right_context_;
  sos_ = meta.sos_;
  eos_ = meta.eos_;
  is_bidirectional_decoder_ = meta.is_bidirectional_decoder_;
  chunk_size_ = meta.chunk_size_;
  num_left_chunks_ = meta.num_left_chunks_;
  deocding_window_ = meta.decoding_window_;

  // 3. Read positional embedding data
  std::ifstream posemb_stream(pos_mnn_path, std::ios::binary | std::ios::in);
  if(posemb_stream) {
    posemb_stream.seekg(0, std::ios::end);
    size_t length = posemb_stream.tellg();
    pos_emb_table_ = std::make_shared<std::vector<float>>(length / sizeof(float));
    posemb_stream.seekg(0, std::ios::beg);
    posemb_stream.read((char*)pos_emb_table_->data(), length);
  }

  LOG(INFO) << "Mnn Model Info:";
  LOG(INFO) << "\tencoder_output_size " << encoder_output_size_;
  LOG(INFO) << "\tnum_blocks " << num_blocks_;
  LOG(INFO) << "\thead " << head_;
  LOG(INFO) << "\tcnn_module_kernel " << cnn_module_kernel_;
  LOG(INFO) << "\tsubsampling_rate " << subsampling_rate_;
  LOG(INFO) << "\tright_context " << right_context_;
  LOG(INFO) << "\tsos " << sos_;
  LOG(INFO) << "\teos " << eos_;
  LOG(INFO) << "\tis bidirectional decoder " << is_bidirectional_decoder_;
  LOG(INFO) << "\tchunk_size " << chunk_size_;
  LOG(INFO) << "\tnum_left_chunks " << num_left_chunks_;

  // 3. Read model nodes
  LOG(INFO) << "Mnn Encoder:";
  GetInputOutputInfo(encoder_session_, &encoder_in_names_, &encoder_out_names_);
  LOG(INFO) << "Mnn CTC:";
  GetInputOutputInfo(ctc_session_, &ctc_in_names_, &ctc_out_names_);
  LOG(INFO) << "Mnn Rescore:";
  GetInputOutputInfo(rescore_session_, &rescore_in_names_, &rescore_out_names_);
}

MnnAsrModel::MnnAsrModel(const MnnAsrModel& other) {
  // metadatas
  encoder_output_size_ = other.encoder_output_size_;
  num_blocks_ = other.num_blocks_;
  head_ = other.head_;
  cnn_module_kernel_ = other.cnn_module_kernel_;
  right_context_ = other.right_context_;
  subsampling_rate_ = other.subsampling_rate_;
  sos_ = other.sos_;
  eos_ = other.eos_;
  is_bidirectional_decoder_ = other.is_bidirectional_decoder_;
  chunk_size_ = other.chunk_size_;
  num_left_chunks_ = other.num_left_chunks_;
  offset_ = other.offset_;
  deocding_window_ = other.deocding_window_;

  // interpreter
  encoder_interpreter_ = other.encoder_interpreter_;
  rescore_interpreter_ = other.rescore_interpreter_;
  ctc_interpreter_ = other.ctc_interpreter_;

  // sessions
  encoder_session_ = std::make_shared<MnnSession>(encoder_interpreter_, scheduleConfig);
  rescore_session_ = std::make_shared<MnnSession>(rescore_interpreter_, scheduleConfig);
  ctc_session_ = std::make_shared<MnnSession>(ctc_interpreter_, scheduleConfig);

  {
    auto t = encoder_session_->getSessionInput("chunk");

    local_feats_tensor_ =
        std::make_unique<MNN::Tensor>(t, t->getDimensionType());

    t = encoder_session_->getSessionInput("att_mask");
    local_att_mask_tensor_ =
        std::make_unique<MNN::Tensor>(t, t->getDimensionType());

    t = encoder_session_->getSessionInput("pos_emb");
    local_pos_emb_tensor_ =
        std::make_unique<MNN::Tensor>(t, t->getDimensionType());

    t = encoder_session_->getSessionInput("cnn_cache");
    local_cnn_cache_tensor_ =
        std::make_unique<MNN::Tensor>(t, t->getDimensionType());
    std::fill(local_cnn_cache_tensor_->host<float>(), local_cnn_cache_tensor_->host<float>() + local_cnn_cache_tensor_->elementSize(), 0.0f);

    t = encoder_session_->getSessionInput("att_cache");
    local_att_cache_tensor_ =
        std::make_unique<MNN::Tensor>(t, t->getDimensionType());
    std::fill(local_att_cache_tensor_->host<float>(), local_att_cache_tensor_->host<float>() + local_att_cache_tensor_->elementSize(), 0.0f);
  }

  pos_emb_table_ = other.pos_emb_table_;

  // node names
  encoder_in_names_ = other.encoder_in_names_;
  encoder_out_names_ = other.encoder_out_names_;
  ctc_in_names_ = other.ctc_in_names_;
  ctc_out_names_ = other.ctc_out_names_;
  rescore_in_names_ = other.rescore_in_names_;
  rescore_out_names_ = other.rescore_out_names_;
}

std::shared_ptr<AsrModel> MnnAsrModel::Copy() const {
  auto asr_model = std::make_shared<MnnAsrModel>(*this);
  // Reset the inner states for new decoding
  asr_model->Reset();
  return asr_model;
}

void MnnAsrModel::Reset() {
  offset_ = chunk_size_ * num_left_chunks_;
  //  encoder_outs_.clear();
  cached_feature_.clear();
  encoder_outs_.clear();

  // Reset att_cache
  if(num_left_chunks_ <= 0) {
    LOG(FATAL) << "invalid num_left_chunks_ value: " << num_left_chunks_;
  }
}

bool MnnSession::resizeTensor(MNN::Tensor* tensor, const std::vector<int>& shape) {
  std::vector<int> currentShape = tensor->shape();

  if(shape != currentShape) {
    interpreter_->resizeTensor(tensor, shape);
    return true;
  }
  return false;
}

void MnnSession::resizeSession() {
  interpreter_->resizeSession(session_);
}

void MnnSession::runSession() {
  interpreter_->runSession(session_);
}

void MnnAsrModel::ForwardEncoderFunc(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* out_prob) {

  // 1. Prepare onnx required data, splice cached_feature_ and chunk_feats
  // chunk
  int num_frames = cached_feature_.size() + chunk_feats.size();
  const int feature_dim = chunk_feats[0].size();
  feats_.resize(0);
  for (size_t i = 0; i < cached_feature_.size(); ++i) {
    feats_.insert(feats_.end(), cached_feature_[i].begin(),
                  cached_feature_[i].end());
  }
  for (size_t i = 0; i < chunk_feats.size(); ++i) {
    feats_.insert(feats_.end(), chunk_feats[i].begin(), chunk_feats[i].end());
  }
  feats_.resize(deocding_window_ * feature_dim);
  std::copy(feats_.begin(), feats_.end(), local_feats_tensor_->host<float>());

  MNN::Tensor* chunk_tensor = encoder_session_->getSessionInput("chunk");
  chunk_tensor->copyFromHostTensor(local_feats_tensor_.get());

  // calculate the size of input after downsampling
  int real_chunk_size = num_frames / subsampling_rate_;

  // required_cache_size
  int required_cache_size = chunk_size_ * num_left_chunks_;

  // att_mask
  std::vector<int32_t> att_mask(required_cache_size + chunk_size_, 1);
  //for MNN we always use num_left_chunks_ > 0
  if (num_left_chunks_ > 0) {
    int chunk_idx = offset_ / chunk_size_ - num_left_chunks_;
    if (chunk_idx < num_left_chunks_) {
      for (int i = 0; i < (num_left_chunks_ - chunk_idx) * chunk_size_; ++i) {
        att_mask[i] = 0;
      }
    }
  }

  int real_chunks = ((num_frames - 1) / 2 - 1) / 2; //size of the chunk after downsampling
  for (size_t i = required_cache_size + real_chunks; i < att_mask.size(); i++) {
    att_mask[i] = 0;
  }

  std::copy(att_mask.begin(), att_mask.end(), local_att_mask_tensor_->host<int32_t>());
  MNN::Tensor* att_mask_tensor = encoder_session_->getSessionInput("att_mask");
  att_mask_tensor->copyFromHostTensor(local_att_mask_tensor_.get());

  //pos_emb
  MNN::Tensor* pos_emb_tensor = encoder_session_->getSessionInput("pos_emb");
  {
    int emb_start = (offset_ - required_cache_size) * encoder_output_size_;
    int emb_end = (offset_ + chunk_size_) * encoder_output_size_;

    std::copy(pos_emb_table_->begin() + emb_start,  pos_emb_table_->begin() + emb_end, local_pos_emb_tensor_->host<float>());
    pos_emb_tensor->copyFromHostTensor(local_pos_emb_tensor_.get());
  }

  MNN::Tensor* cnn_cache_tensor = encoder_session_->getSessionInput("cnn_cache");
  cnn_cache_tensor->copyFromHostTensor(local_cnn_cache_tensor_.get());

  MNN::Tensor* att_cache_tensor = encoder_session_->getSessionInput("att_cache");
  att_cache_tensor->copyFromHostTensor(local_att_cache_tensor_.get());


  //  if(offset_ == required_cache_size + 8) {
  //    write_data("/tmp/mnn_chunk.bin", local_feats_tensor_->host<float>(), local_feats_tensor_->elementSize() * sizeof(float));
  //    write_data("/tmp/mnn_pos_emb.bin", local_pos_emb_tensor_->host<float>(), local_pos_emb_tensor_->elementSize() * sizeof(float));
  //    write_data("/tmp/mnn_att_mask.bin", local_att_mask_tensor_->host<int32_t>(), local_att_mask_tensor_->elementSize() * sizeof(int32_t));
  //    write_data("/tmp/mnn_cnn_cache.bin", local_cnn_cache_tensor_->host<float>(), local_cnn_cache_tensor_->elementSize() * sizeof(float));
  //    write_data("/tmp/mnn_att_cache.bin", local_att_cache_tensor_->host<float>(), local_att_cache_tensor_->elementSize() * sizeof(float));
  //  }


  // 2. Encoder chunk forward

  encoder_session_->runSession();

  MNN::Tensor* output_tensor = encoder_session_->getSessionOutput("output");
  MNN::Tensor* r_att_cache_tensor = encoder_session_->getSessionOutput("r_att_cache");
  MNN::Tensor* r_cnn_cache_tensor = encoder_session_->getSessionOutput("r_cnn_cache");

  r_att_cache_tensor->copyToHostTensor(local_att_cache_tensor_.get());
  r_cnn_cache_tensor->copyToHostTensor(local_cnn_cache_tensor_.get());

  encoder_outs_.emplace_back(
      output_tensor->host<float>(),
      output_tensor->host<float>() + real_chunks * encoder_output_size_);

  offset_ += real_chunks;

  /////////////////////////////////////////////
  MNN::Tensor* hidden_tensor = ctc_session_->getSessionInput("hidden");
  hidden_tensor->copyFromHostTensor(output_tensor);
  ctc_session_->runSession();

  MNN::Tensor* probs_tensor = ctc_session_->getSessionOutput("probs");

  auto* logp_data = probs_tensor->host<float>();
  std::vector<int> probs_shape = probs_tensor->shape();

  //  int num_outputs = probs_shape[1];
  int num_outputs = real_chunk_size;
  int output_dim = probs_shape[2];
  out_prob->resize(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    (*out_prob)[i].resize(output_dim);
    memcpy((*out_prob)[i].data(), logp_data + i * output_dim,
           sizeof(float) * output_dim);
  }
}

float MnnAsrModel::ComputeAttentionScore(const float* prob,
                                         const std::vector<int>& hyp, int eos,
                                         int decode_out_len) {
  float score = 0.0f;

  for (size_t j = 0; j < hyp.size(); ++j) {
    score += *(prob + j * decode_out_len + hyp[j]);
  }

  score += *(prob + hyp.size() * decode_out_len + eos);

  return score;
}

static void pad_hyps(int* hyps_pad, const std::vector<std::vector<int>>& hyps, int sos, int max_hyps_len, bool reverse=false) {
  size_t num_hyps = hyps.size();
  int* p = hyps_pad;

  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    *p++ = sos;
    size_t j = 0;

    for (; j < hyp.size(); ++j) {
      *p++ = reverse ? hyp[hyp.size() - j - 1] : hyp[j];
    }

    if (j == max_hyps_len - 1) {
      continue;
    }
    for (; j < max_hyps_len - 1; ++j) {
      *p++ = 0;
    }
  }
}

void MnnAsrModel::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                     float reverse_weight,
                                     std::vector<float>* rescoring_score) {
  CHECK(rescoring_score != nullptr);
  int num_hyps = hyps.size();
  rescoring_score->resize(num_hyps, 0.0f);

  if (num_hyps == 0) {
    return;
  }
  // No encoder output
  if (encoder_outs_.size() == 0) {
    return;
  }

  std::vector<int> hyps_lens;
  int max_hyps_len = 0;
  for (size_t i = 0; i < num_hyps; ++i) {
    int length = hyps[i].size() + 1;
    max_hyps_len = std::max(length, max_hyps_len);
    hyps_lens.emplace_back(length);
  }

  size_t encoder_data_len = 0;
  for (const auto & encoder_out : encoder_outs_) {
    encoder_data_len += encoder_out.size();
  }

//  printf("encoder_data_len = %ld\n", encoder_data_len / encoder_output_size_);

  std::vector<int> real_encoder_out_shape = {1, (int)(encoder_data_len / encoder_output_size_), encoder_output_size_};

  rescore_session_->runSession();
  MNN::Tensor* encoder_out_tensor = rescore_session_->getSessionInput("encoder_out");
  rescore_session_->resizeTensor(encoder_out_tensor, real_encoder_out_shape);

  std::vector<int> real_hyps_shape= {num_hyps, max_hyps_len};
  MNN::Tensor* hyps_tensor = rescore_session_->getSessionInput("hyps");
  rescore_session_->resizeTensor(hyps_tensor, real_hyps_shape);

  MNN::Tensor* r_hyps_tensor = rescore_session_->getSessionInput("r_hyps");
  rescore_session_->resizeTensor(r_hyps_tensor, real_hyps_shape);

  std::vector<int> real_hyps_length_shape= {num_hyps};
  MNN::Tensor* hyps_lens_tensor = rescore_session_->getSessionInput("hyps_lens");
  rescore_session_->resizeTensor(hyps_lens_tensor, real_hyps_length_shape);

  rescore_session_->resizeSession();

  //encoder_out
  auto local_encoder_out = std::make_unique<MNN::Tensor>(encoder_out_tensor, encoder_out_tensor->getDimensionType());
  auto* local_encoder_out_data = local_encoder_out->host<float>();
  size_t data_offset = 0;
  for (auto & e : encoder_outs_) {
    std::copy(e.begin(), e.end(), local_encoder_out_data + data_offset);
    data_offset += e.size();
  }
  encoder_out_tensor->copyFromHostTensor(local_encoder_out.get());

  //  write_data("/tmp/mnn_encoder_out.bin", local_encoder_out->host<float>(), local_encoder_out->elementSize() * sizeof(float));

  //hyps
  auto local_hyps = std::make_unique<MNN::Tensor>(hyps_tensor, hyps_tensor->getDimensionType());
  pad_hyps(local_hyps->host<int>(), hyps, sos_, max_hyps_len, false);
  hyps_tensor->copyFromHostTensor(local_hyps.get());
  //  write_data("/tmp/mnn_hyps.bin", local_hyps->host<int>(), local_hyps->elementSize() * sizeof(int));

  //r_hyps
  auto local_r_hyps = std::make_unique<MNN::Tensor>(r_hyps_tensor, r_hyps_tensor->getDimensionType());
  pad_hyps(local_r_hyps->host<int>(), hyps, sos_, max_hyps_len, true);
  r_hyps_tensor->copyFromHostTensor(local_r_hyps.get());
  //  write_data("/tmp/mnn_r_hyps.bin", local_r_hyps->host<int>(), local_r_hyps->elementSize() * sizeof(int));

  //hyps_lens
  auto local_hyps_lens = std::make_unique<MNN::Tensor>(hyps_lens_tensor, hyps_lens_tensor->getDimensionType());
  std::copy(hyps_lens.begin(), hyps_lens.end(), local_hyps_lens->host<int>());
  hyps_lens_tensor->copyFromHostTensor(local_hyps_lens.get());

  //  write_data("/tmp/mnn_hyps_lens.bin", hyps_lens.data(), hyps_lens.size() * sizeof(float));
  rescore_session_->runSession();

  MNN::Tensor* score_tensor = rescore_session_->getSessionOutput("score");
  auto decoder_outs_data = score_tensor->host<float>();
  auto score_shape = score_tensor->shape();

  int decode_out_len = score_shape[2];

  MNN::Tensor* r_score_tensor = rescore_session_->getSessionOutput("r_score");
  auto r_decoder_outs_data = r_score_tensor->host<float>();

  //  write_data("/tmp/mnn_score.bin", decoder_outs_data, score_tensor->elementSize() * sizeof(float));
  //  write_data("/tmp/mnn_r_score.bin", r_decoder_outs_data, r_score_tensor->elementSize() * sizeof(float));

  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    float score = 0.0f;
    // left to right decoder score
    score = ComputeAttentionScore(
        decoder_outs_data + max_hyps_len * decode_out_len * i, hyp, eos_,
        decode_out_len);
    // Optional: Used for right to left score
    float r_score = 0.0f;
    if (is_bidirectional_decoder_ && reverse_weight > 0) {
      std::vector<int> r_hyp(hyp.size());
      std::reverse_copy(hyp.begin(), hyp.end(), r_hyp.begin());
      // right to left decoder score
      r_score = ComputeAttentionScore(
          r_decoder_outs_data + max_hyps_len * decode_out_len * i, r_hyp, eos_,
          decode_out_len);
    }
    // combined left-to-right and right-to-left score
    (*rescoring_score)[i] =
        score * (1 - reverse_weight) + r_score * reverse_weight;
  }
}

}  // namespace wenet
