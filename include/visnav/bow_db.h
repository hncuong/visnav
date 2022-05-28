/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <fstream>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

namespace visnav {

class BowDatabase {
 public:
  BowDatabase() {}

  inline void insert(const FrameCamId& fcid, const BowVector& bow_vector) {
    // TODO SHEET 3: add a bow_vector that corresponds to frame fcid to the
    // inverted index. You can assume the image hasn't been added before.
    for (const auto& kv : bow_vector) {
      auto word_id = kv.first;
      auto weight = kv.second;
      inverted_index[word_id].push_back(std::make_pair(fcid, weight));
    }
  }

  inline void query(const BowVector& bow_vector, size_t num_results,
                    BowQueryResult& results) const {
    // TODO SHEET 3: find num_results closest matches to the bow_vector in the
    // inverted index. Hint: for good query performance use std::unordered_map
    // to accumulate scores and std::partial_sort for getting the closest
    // results. You should use L1 difference as the distance measure. You can
    // assume that BoW descripors are L1 normalized.

    // Diff
    std::unordered_map<FrameCamId, double> frame_diffs;

    for (const auto& kv : bow_vector) {
      auto word_id = kv.first;
      auto weight = kv.second;

      // Query Frame with
      //      const auto &invertedIndex_ = this.getInvertedIndex();
      for (const auto& frame_weight : inverted_index.at(word_id)) {
        FrameCamId fcid = frame_weight.first;
        double frame_word_weight = frame_weight.second;

        // Now calculate l1 diff
        // L1 diff of two L1 normalized vectors
        double l1_diff = abs(weight - frame_word_weight) - abs(weight) -
                         abs(frame_word_weight);
        if (frame_diffs.find(fcid) == frame_diffs.end()) {
          // First appear. Add 2
          frame_diffs[fcid] = 2 + l1_diff;
        } else {
          frame_diffs[fcid] += l1_diff;
        }
      }
    }

    // Take only num_results
    std::vector<std::pair<FrameCamId, double>> diff_list(frame_diffs.begin(),
                                                         frame_diffs.end());
    size_t num_returns = diff_list.size();
    if (num_returns > num_results) num_returns = num_results;
    std::partial_sort(
        diff_list.begin(), diff_list.begin() + num_returns, diff_list.end(),
        [](auto& left, auto& right) { return left.second < right.second; });
    for (int i = 0; i < num_returns; i++) results.push_back(diff_list[i]);
  }

  void clear() { inverted_index.clear(); }

  void save(const std::string& out_path) {
    BowDBInverseIndex state;
    for (const auto& kv : inverted_index) {
      for (const auto& a : kv.second) {
        state[kv.first].emplace_back(a);
      }
    }
    std::ofstream os;
    os.open(out_path, std::ios::binary);
    cereal::JSONOutputArchive archive(os);
    archive(state);
  }

  void load(const std::string& in_path) {
    BowDBInverseIndex inverseIndexLoaded;
    {
      std::ifstream os(in_path, std::ios::binary);
      cereal::JSONInputArchive archive(os);
      archive(inverseIndexLoaded);
    }
    for (const auto& kv : inverseIndexLoaded) {
      for (const auto& a : kv.second) {
        inverted_index[kv.first].emplace_back(a);
      }
    }
  }

  const BowDBInverseIndexConcurrent& getInvertedIndex() {
    return inverted_index;
  }

 protected:
  BowDBInverseIndexConcurrent inverted_index;
};

}  // namespace visnav
