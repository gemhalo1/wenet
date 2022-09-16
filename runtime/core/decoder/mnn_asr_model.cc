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
#include "boost/json/src.hpp"

#include "utils/string.h"

namespace json = boost::json;

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

json::value read_json( std::istream& is, json::error_code& ec )
{
  json::stream_parser p;
  std::string line;
  while( std::getline( is, line ) )
  {
    p.write( line, ec );
    if( ec )
      return nullptr;
  }
  p.finish( ec );
  if( ec )
    return nullptr;
  return p.release();
}

void MnnAsrModel::Read(const std::string& model_dir) {
  std::string encoder_mnn_path = model_dir + "/encoder.mnn";
  std::string rescore_mnn_path = model_dir + "/decoder_s.mnn";
  std::string ctc_mnn_path = model_dir + "/ctc.mnn";
  std::string meta_mnn_path = model_dir + "/meta.json";
  std::string pos_mnn_path = model_dir + "/pos_emb.bin";

  // 1. Load sessions
  try {
#ifdef _MSC_VER
    encoder_session_ = std::make_shared<Ort::Session>(
        env_, ToWString(encoder_onnx_path).c_str(), session_options_);
    rescore_session_ = std::make_shared<Ort::Session>(
        env_, ToWString(rescore_onnx_path).c_str(), session_options_);
    ctc_session_ = std::make_shared<Ort::Session>(
        env_, ToWString(ctc_onnx_path).c_str(), session_options_);
#else
    encoder_interpreter_.reset(MNN::Interpreter::createFromFile(encoder_mnn_path.c_str()));
    encoder_session_ = std::make_shared<MnnSession>(encoder_interpreter_, scheduleConfig);

    rescore_interpreter_.reset(MNN::Interpreter::createFromFile(rescore_mnn_path.c_str()));
    rescore_session_ = std::make_shared<MnnSession>(rescore_interpreter_, scheduleConfig);
    
    ctc_interpreter_.reset(MNN::Interpreter::createFromFile(ctc_mnn_path.c_str()));
    ctc_session_ = std::make_shared<MnnSession>(ctc_interpreter_, scheduleConfig);
#endif
  } catch (std::exception const& e) {
    LOG(ERROR) << "error when load onnx model: " << e.what();
    exit(0);
  }

  // 2. Read metadata
  std::ifstream meta_stream;
  meta_stream.open(meta_mnn_path.c_str(), std::ios::in);
  if(meta_stream.is_open()) {
    json::error_code ec;
    json::value value = read_json(meta_stream, ec);

    if(value.is_object()) {
      json::object obj = value.as_object();

#define CONVERT_INT_FIELD(f) atoi(obj[f].as_string().c_str())
      encoder_output_size_ = CONVERT_INT_FIELD("output_size");
      num_blocks_ = CONVERT_INT_FIELD("num_blocks");
      head_ = CONVERT_INT_FIELD("head");
      cnn_module_kernel_ = CONVERT_INT_FIELD("cnn_module_kernel");
      subsampling_rate_ = CONVERT_INT_FIELD("subsampling_rate");
      right_context_ = CONVERT_INT_FIELD("right_context");
      sos_ = CONVERT_INT_FIELD("sos_symbol");
      eos_ = CONVERT_INT_FIELD("eos_symbol");
      is_bidirectional_decoder_ = CONVERT_INT_FIELD("is_bidirectional_decoder");
      chunk_size_ = CONVERT_INT_FIELD("chunk_size");
      num_left_chunks_ = CONVERT_INT_FIELD("left_chunks");
      deocding_window_ = CONVERT_INT_FIELD("decoding_window");
    }
  }

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
    auto x =  t->getDimensionType();
    local_feats_tensor_ =
        std::make_unique<MNN::Tensor>(t, t->getDimensionType());

    t = encoder_session_->getSessionInput("att_mask");
    local_att_mask_tensor_ =
        std::make_unique<MNN::Tensor>(t, t->getDimensionType());

    t = encoder_session_->getSessionInput("pos_emb");
    local_pos_emb_tensor_ =
        std::make_unique<MNN::Tensor>(t, t->getDimensionType());
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
  offset_ = 0;
  //  encoder_outs_.clear();
  cached_feature_.clear();

  // Reset att_cache
  if (num_left_chunks_ > 0) {
    int required_cache_size = chunk_size_ * num_left_chunks_;

    offset_ = required_cache_size;

    std::vector<int> att_cache_shape  = {num_blocks_, head_, required_cache_size,
                                        encoder_output_size_ / head_ * 2};

    MNN::Tensor* t = encoder_session_->getSessionInput("att_cache");
    encoder_interpreter_->resizeTensor(t, att_cache_shape);

    //fill with 0.0f
    auto* buffer = t->host<float>();
    std::fill(buffer, buffer + t->elementSize(), 0.0f);
  } else {
    std::vector<int> att_cache_shape = {num_blocks_, head_, 0,
                                        encoder_output_size_ / head_ * 2};

    MNN::Tensor* t = encoder_session_->getSessionInput("att_cache");
    encoder_interpreter_->resizeTensor(t, att_cache_shape);
    auto* buffer = t->host<float>();
    std::fill(buffer, buffer + t->elementSize(), 0.0f);
  }

  // Reset cnn_cache
  {
    std::vector<int> cnn_cache_shape = {num_blocks_, 1, encoder_output_size_,
                                        cnn_module_kernel_ - 1};

    MNN::Tensor* t = encoder_session_->getSessionInput("cnn_cache");
    encoder_interpreter_->resizeTensor(t, cnn_cache_shape);

    // fill with 0.0f
    auto* buffer = t->host<float>();
    std::fill(buffer, buffer + t->elementSize(), 0.0f);
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
  std::vector<int32_t> att_mask(required_cache_size + chunk_size_);
  if (num_left_chunks_ > 0) {
    int chunk_idx = offset_ / chunk_size_ - num_left_chunks_;
    if (chunk_idx < num_left_chunks_) {
      for (int i = 0; i < (num_left_chunks_ - chunk_idx) * chunk_size_; ++i) {
        att_mask[i] = 0;
      }
    }
  }
  for(int i = real_chunk_size; i < chunk_size_; i++) {
    att_mask[required_cache_size + i] = 0;
  }
  std::copy(att_mask.begin(), att_mask.end(), local_att_mask_tensor_->host<int32_t>());
  MNN::Tensor* att_mask_tensor = encoder_session_->getSessionInput("att_mask");
  att_mask_tensor->copyFromHostTensor(local_att_mask_tensor_.get());

  //pos_emb
  MNN::Tensor* pos_emb_tensor = encoder_session_->getSessionInput("pos_emb");
  {
    int emb_start = (offset_ - required_cache_size) * encoder_output_size_;
    int emb_end = emb_start + (required_cache_size + chunk_size_) * encoder_output_size_;

    std::copy(pos_emb_table_->begin() + emb_start,  pos_emb_table_->begin() + emb_end, local_pos_emb_tensor_->host<float>());
    pos_emb_tensor->copyFromHostTensor(local_pos_emb_tensor_.get());
  }

  // 2. Encoder chunk forward

  encoder_session_->runSession();

  MNN::Tensor* output_tensor = encoder_session_->getSessionOutput("output");
  MNN::Tensor* r_att_cache_tensor = encoder_session_->getSessionOutput("r_att_cache");
  MNN::Tensor* r_cnn_cache_tensor = encoder_session_->getSessionOutput("r_cnn_cache");

  r_att_cache_tensor->copyToHostTensor(encoder_session_->getSessionInput("att_cache"));
  r_cnn_cache_tensor->copyToHostTensor(encoder_session_->getSessionInput("cnn_cache"));

  std::vector<int> output_shape = output_tensor->shape();

  offset_ += (int)output_shape[1];

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
#if 0
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += *(prob + j * decode_out_len + hyp[j]);
  }
  score += *(prob + hyp.size() * decode_out_len + eos);
#endif
  return score;
}

void MnnAsrModel::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                     float reverse_weight,
                                     std::vector<float>* rescoring_score) {
#if 0
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
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

  std::vector<int64_t> hyps_lens;
  int max_hyps_len = 0;
  for (size_t i = 0; i < num_hyps; ++i) {
    int length = hyps[i].size() + 1;
    max_hyps_len = std::max(length, max_hyps_len);
    hyps_lens.emplace_back(static_cast<int64_t>(length));
  }

  std::vector<float> rescore_input;
  int encoder_len = 0;
  for (int i = 0; i < encoder_outs_.size(); i++) {
    float* encoder_outs_data = encoder_outs_[i].GetTensorMutableData<float>();
    auto type_info = encoder_outs_[i].GetTensorTypeAndShapeInfo();
    for (int j = 0; j < type_info.GetElementCount(); j++) {
      rescore_input.emplace_back(encoder_outs_data[j]);
    }
    encoder_len += type_info.GetShape()[1];
  }

  const int64_t decode_input_shape[] = {1, encoder_len, encoder_output_size_};

  std::vector<int64_t> hyps_pad;

  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    hyps_pad.emplace_back(sos_);
    size_t j = 0;
    for (; j < hyp.size(); ++j) {
      hyps_pad.emplace_back(hyp[j]);
    }
    if (j == max_hyps_len - 1) {
      continue;
    }
    for (; j < max_hyps_len - 1; ++j) {
      hyps_pad.emplace_back(0);
    }
  }

  const int64_t hyps_pad_shape[] = {num_hyps, max_hyps_len};

  const int64_t hyps_lens_shape[] = {num_hyps};

  Ort::Value decode_input_tensor_ = Ort::Value::CreateTensor<float>(
      memory_info, rescore_input.data(), rescore_input.size(),
      decode_input_shape, 3);
  Ort::Value hyps_pad_tensor_ = Ort::Value::CreateTensor<int64_t>(
      memory_info, hyps_pad.data(), hyps_pad.size(), hyps_pad_shape, 2);
  Ort::Value hyps_lens_tensor_ = Ort::Value::CreateTensor<int64_t>(
      memory_info, hyps_lens.data(), hyps_lens.size(), hyps_lens_shape, 1);

  std::vector<Ort::Value> rescore_inputs;

  rescore_inputs.emplace_back(std::move(hyps_pad_tensor_));
  rescore_inputs.emplace_back(std::move(hyps_lens_tensor_));
  rescore_inputs.emplace_back(std::move(decode_input_tensor_));

  std::vector<Ort::Value> rescore_outputs = rescore_session_->Run(
      Ort::RunOptions{nullptr}, rescore_in_names_.data(), rescore_inputs.data(),
      rescore_inputs.size(), rescore_out_names_.data(),
      rescore_out_names_.size());

  float* decoder_outs_data = rescore_outputs[0].GetTensorMutableData<float>();
  float* r_decoder_outs_data = rescore_outputs[1].GetTensorMutableData<float>();

  auto type_info = rescore_outputs[0].GetTensorTypeAndShapeInfo();
  int decode_out_len = type_info.GetShape()[2];

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
#endif
}

}  // namespace wenet
