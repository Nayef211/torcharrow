/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gpt2_bpe_tokenizer.h"
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"

namespace facebook::torcharrow::functions {


// Based on the following example:
// ArrayWriter UDF example: ArrayWriterBenchmark.cpp (https://fburl.com/code/s0cg1vgb)
// OpaqueType UDF example: OpaqueType.cpp (https://fburl.com/code/8t5l8tpi)
template <typename T>
struct bpe_tokenize {
    VELOX_DEFINE_FUNCTION_TYPES(T);

    FOLLY_ALWAYS_INLINE bool call(
    ArrayWriter<int64_t>& result, 
    const arg_type<std::shared_ptr<GPT2BPEEncoder>>& bpe_encoder, 
    const arg_type<velox::Varchar>& text
  ) {
        result.copy_from(bpe_encoder->Encode(text.str()));
        return true;
    }
  };

} // namespace facebook::torcharrow::functions
