/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_FLASH_API_H
#define TRT_FLASH_API_H

#include "common/bertCommon.h"
#include "common/plugin.h"
#include "commonDatatype.h"

namespace nvinfer1
{
namespace plugin
{
int32_t
mha_fwd(void const* q,         // batch_size x seqlen_q x num_heads x head_size
        void const* k,         // batch_size x seqlen_k x num_heads_k x head_size
        void const* v,         // batch_size x seqlen_k x num_heads_k x head_size
        void const* out_,      // batch_size x seqlen_q x num_heads x head_size
        void const* softmax_lse,
        int32_t batch_size,
        int32_t num_heads,
        int32_t num_heads_k,
        int32_t head_size_og, 
        int32_t seqlen_q,
        int32_t seqlen_k,
        cudaStream_t stream);
} // name space plugin
} // namespace nvinfer1

#endif // TRT_FLASH_API_H
