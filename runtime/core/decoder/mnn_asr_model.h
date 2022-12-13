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


#ifndef DECODER_MNN_ASR_MODEL_H_
#define DECODER_MNN_ASR_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "MNN/Interpreter.hpp"  // NOLINT

#include "decoder/asr_model.h"
#include "utils/log.h"
#include "utils/utils.h"

namespace wenet {

// a MNN session shall be released from the interpreter, so a wrapper is needed
// to do it
class MnnAsrModel;

class MnnSession {
  friend class MnnAsrModel;
 public:
  MnnSession(std::shared_ptr<MNN::Interpreter> interpreter, const MNN::ScheduleConfig& scheduleConfig);
  ~MnnSession();

  MNN::Tensor* getSessionInput(const char* name);
  MNN::Tensor* getSessionOutput(const char* name);

  const std::map<std::string, MNN::Tensor*>& getSessionInputAll() const;
  const std::map<std::string, MNN::Tensor*>& getSessionOutputAll() const;

  bool resizeTensor(MNN::Tensor* tensor, const std::vector<int>& shape);
  void resizeSession();
  void runSession();
 private:
  std::shared_ptr<MNN::Interpreter> interpreter_;
  MNN::Session* session_ = nullptr;
};

class MnnAsrModel : public AsrModel {
 public:
  MnnAsrModel() = default;
  MnnAsrModel(const MnnAsrModel& other);
  void Read(const std::string& model_dir);
  void Reset() override;
  void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                          float reverse_weight,
                          std::vector<float>* rescoring_score) override;
  std::shared_ptr<AsrModel> Copy() const override;
  static void GetInputOutputInfo(const std::shared_ptr<MnnSession>& session,
                                 std::vector<const char*>* in_names,
                                 std::vector<const char*>* out_names);

 protected:
  void ForwardEncoderFunc(const std::vector<std::vector<float>>& chunk_feats,
                          std::vector<std::vector<float>>* ctc_prob) override;

  float ComputeAttentionScore(const float* prob, const std::vector<int>& hyp,
                              int eos, int decode_out_len);

 private:
  int encoder_output_size_ = 0;
  int num_blocks_ = 0;
  int cnn_module_kernel_ = 0;
  int head_ = 0;
  int deocding_window_ = 67;

  static MNN::ScheduleConfig scheduleConfig;

  std::shared_ptr<MNN::Interpreter> encoder_interpreter_ = nullptr;
  std::shared_ptr<MNN::Interpreter> rescore_interpreter_ = nullptr;
  std::shared_ptr<MNN::Interpreter> ctc_interpreter_ = nullptr;

  std::shared_ptr<std::vector<float>> pos_emb_table_ = nullptr;

  // sessions
  std::shared_ptr<MnnSession> encoder_session_ = nullptr;
  std::shared_ptr<MnnSession> rescore_session_ = nullptr;
  std::shared_ptr<MnnSession> ctc_session_ = nullptr;

  std::vector<float> feats_;
  std::unique_ptr<MNN::Tensor> local_feats_tensor_;
  std::unique_ptr<MNN::Tensor> local_att_mask_tensor_;
  std::unique_ptr<MNN::Tensor> local_pos_emb_tensor_;
  std::unique_ptr<MNN::Tensor> local_att_cache_tensor_;
  std::unique_ptr<MNN::Tensor> local_cnn_cache_tensor_;

  std::vector<std::vector<float>> encoder_outs_;

  // node names
  std::vector<const char*> encoder_in_names_, encoder_out_names_;
  std::vector<const char*> ctc_in_names_, ctc_out_names_;
  std::vector<const char*> rescore_in_names_, rescore_out_names_;
};

}  // namespace wenet

#endif  // DECODER_MNN_ASR_MODEL_H_
