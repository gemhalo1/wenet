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

    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tInput " << in_names->size() << " : name=" << name << " type=" << "?"
              << " dims=" << shape.str();
    in_names->push_back(name.c_str());
  }

  // Output info
  auto& outputs = session->getSessionOutputAll();
  out_names->clear();

  for(auto& x : outputs) {
    auto& name = x.first;
    auto& tensor = x.second;

    auto node_dims = tensor->shape();

    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tOutput " << out_names->size() << " : name=" << name << " type=" << "?"
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
  std::string rescore_mnn_path = model_dir + "/decoder.mnn";
  std::string ctc_mnn_path = model_dir + "/ctc.mnn";
  std::string meta_mnn_path = model_dir + "/meta.json";

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
      }
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

  // interpreter
  encoder_interpreter_ = other.encoder_interpreter_;
  rescore_interpreter_ = other.rescore_interpreter_;
  ctc_interpreter_ = other.ctc_interpreter_;

  // sessions
  encoder_session_ = std::make_shared<MnnSession>(encoder_interpreter_, scheduleConfig);
  rescore_session_ = std::make_shared<MnnSession>(rescore_interpreter_, scheduleConfig);
  ctc_session_ = std::make_shared<MnnSession>(ctc_interpreter_, scheduleConfig);

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

void MnnAsrModel::ForwardEncoderFunc(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* out_prob) {
#if 0
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  // 1. Prepare onnx required data, splice cached_feature_ and chunk_feats
  // chunk
  int num_frames = cached_feature_.size() + chunk_feats.size();
  const int feature_dim = chunk_feats[0].size();
  std::vector<float> feats;
  for (size_t i = 0; i < cached_feature_.size(); ++i) {
    feats.insert(feats.end(), cached_feature_[i].begin(),
                 cached_feature_[i].end());
  }
  for (size_t i = 0; i < chunk_feats.size(); ++i) {
    feats.insert(feats.end(), chunk_feats[i].begin(), chunk_feats[i].end());
  }
  const int64_t feats_shape[3] = {1, num_frames, feature_dim};
  Ort::Value feats_ort = Ort::Value::CreateTensor<float>(
      memory_info, feats.data(), feats.size(), feats_shape, 3);
  // offset
  int64_t offset_int64 = static_cast<int64_t>(offset_);
  Ort::Value offset_ort = Ort::Value::CreateTensor<int64_t>(
      memory_info, &offset_int64, 1, std::vector<int64_t>{}.data(), 0);
  // required_cache_size
  int64_t required_cache_size = chunk_size_ * num_left_chunks_;
  Ort::Value required_cache_size_ort = Ort::Value::CreateTensor<int64_t>(
      memory_info, &required_cache_size, 1, std::vector<int64_t>{}.data(), 0);
  // att_mask
  Ort::Value att_mask_ort{nullptr};
  std::vector<uint8_t> att_mask(required_cache_size + chunk_size_, 1);
  if (num_left_chunks_ > 0) {
    int chunk_idx = offset_ / chunk_size_ - num_left_chunks_;
    if (chunk_idx < num_left_chunks_) {
      for (int i = 0; i < (num_left_chunks_ - chunk_idx) * chunk_size_; ++i) {
        att_mask[i] = 0;
      }
    }
    const int64_t att_mask_shape[] = {1, 1, required_cache_size + chunk_size_};
    att_mask_ort = Ort::Value::CreateTensor<bool>(
        memory_info, reinterpret_cast<bool*>(att_mask.data()), att_mask.size(),
        att_mask_shape, 3);
  }

  // 2. Encoder chunk forward
  std::vector<Ort::Value> inputs;
  for (auto name : encoder_in_names_) {
    if (!strcmp(name, "chunk")) {
      inputs.emplace_back(std::move(feats_ort));
    } else if (!strcmp(name, "offset")) {
      inputs.emplace_back(std::move(offset_ort));
    } else if (!strcmp(name, "required_cache_size")) {
      inputs.emplace_back(std::move(required_cache_size_ort));
    } else if (!strcmp(name, "att_cache")) {
      inputs.emplace_back(std::move(att_cache_ort_));
    } else if (!strcmp(name, "cnn_cache")) {
      inputs.emplace_back(std::move(cnn_cache_ort_));
    } else if (!strcmp(name, "att_mask")) {
      inputs.emplace_back(std::move(att_mask_ort));
    }
  }

  std::vector<Ort::Value> ort_outputs = encoder_session_->Run(
      Ort::RunOptions{nullptr}, encoder_in_names_.data(), inputs.data(),
      inputs.size(), encoder_out_names_.data(), encoder_out_names_.size());

  offset_ += static_cast<int>(
      ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
  att_cache_ort_ = std::move(ort_outputs[1]);
  cnn_cache_ort_ = std::move(ort_outputs[2]);

  std::vector<Ort::Value> ctc_inputs;
  ctc_inputs.emplace_back(std::move(ort_outputs[0]));

  std::vector<Ort::Value> ctc_ort_outputs = ctc_session_->Run(
      Ort::RunOptions{nullptr}, ctc_in_names_.data(), ctc_inputs.data(),
      ctc_inputs.size(), ctc_out_names_.data(), ctc_out_names_.size());
  encoder_outs_.push_back(std::move(ctc_inputs[0]));

  float* logp_data = ctc_ort_outputs[0].GetTensorMutableData<float>();
  auto type_info = ctc_ort_outputs[0].GetTensorTypeAndShapeInfo();

  int num_outputs = type_info.GetShape()[1];
  int output_dim = type_info.GetShape()[2];
  out_prob->resize(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    (*out_prob)[i].resize(output_dim);
    memcpy((*out_prob)[i].data(), logp_data + i * output_dim,
           sizeof(float) * output_dim);
  }
#endif
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
