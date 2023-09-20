#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import OrderedDict
from copy import deepcopy
from diffusers.models import AutoencoderKL, UNet2DConditionModel
import numpy as np
from onnx import shape_inference
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from cuda import cudart
import onnx

class Optimizer():
    def __init__(
        self,
        onnx_graph,
        verbose=False
    ):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def remove_casts(self):
        nRemoveCastNode = 0
        for node in self.graph.nodes:
            # Remove Cast nodes before qkv gemm
            if node.op in ["LayerNormalization"] and \
                len(node.o().outputs[0].outputs) == 3 and \
                node.o().o().op == "MatMul" and node.o().o(1).op == "MatMul" and node.o().o(2).op == "MatMul":
                for i in range(len(node.o().outputs[0].outputs)):
                    matMulNode = node.o().o()
                    matMulNode.inputs[0] = node.outputs[0]
                    nRemoveCastNode += 1
            
            # Remove double cast nodes after Softmax Node
            if node.op == "Softmax" and node.o().op == "Cast" and node.o().o().op == "Cast":
                node.o().o().o().inputs[0] = node.outputs[0]
                nRemoveCastNode += 1

        self.cleanup()
        return nRemoveCastNode

    def fuse_kv(self, node_k, node_v, fused_kv_idx, heads, num_dynamic=0):
        # Get weights of K
        weights_k = node_k.inputs[1].values
        # Get weights of V
        weights_v = node_v.inputs[1].values
        # Input number of channels to K and V
        C = weights_k.shape[0]
        # Number of heads
        H = heads
        # Dimension per head
        D = weights_k.shape[1] // H

        # Concat and interleave weights such that the output of fused KV GEMM has [b, s_kv, h, 2, d] shape
        weights_kv = np.dstack([weights_k.reshape(C, H, D), weights_v.reshape(C, H, D)]).reshape(C, 2 * H * D)

        # K and V have the same input
        input_tensor = node_k.inputs[0]
        # K and V must have the same output which we feed into fmha plugin
        output_tensor_k = node_k.outputs[0]
        # Create tensor
        constant_weights_kv = gs.Constant("Weights_KV_{}".format(fused_kv_idx), np.ascontiguousarray(weights_kv))

        # Create fused KV node
        fused_kv_node = gs.Node(op="MatMul", name="MatMul_KV_{}".format(fused_kv_idx), inputs=[input_tensor, constant_weights_kv], outputs=[output_tensor_k])
        self.graph.nodes.append(fused_kv_node)

        # Connect the output of fused node to the inputs of the nodes after K and V
        node_v.o(num_dynamic).inputs[0] = output_tensor_k
        node_k.o(num_dynamic).inputs[0] = output_tensor_k
        for i in range(0,num_dynamic):
            node_v.o().inputs.clear()
            node_k.o().inputs.clear()

        # Clear inputs and outputs of K and V to ge these nodes cleared
        node_k.outputs.clear()
        node_v.outputs.clear()
        node_k.inputs.clear()
        node_v.inputs.clear()

        self.cleanup()
        return fused_kv_node

    def insert_fmhca(self, node_q, node_kv, final_tranpose, mhca_idx, heads, num_dynamic=0):
        # Get inputs and outputs for the fMHCA plugin
        # We take an output of reshape that follows the Q GEMM
        output_q = node_q.o(num_dynamic).o().inputs[0]
        output_kv = node_kv.o().inputs[0]
        output_final_tranpose = final_tranpose.outputs[0]

        # Clear the inputs of the nodes that follow the Q and KV GEMM
        # to delete these subgraphs (it will be substituted by fMHCA plugin)
        node_kv.outputs[0].outputs[0].inputs.clear()
        node_kv.outputs[0].outputs[0].inputs.clear()
        node_q.o(num_dynamic).o().inputs.clear()
        for i in range(0,num_dynamic):
            node_q.o(i).o().o(1).inputs.clear()

        weights_kv = node_kv.inputs[1].values
        dims_per_head = weights_kv.shape[1] // (heads * 2)

        # Reshape dims
        shape = gs.Constant("Shape_KV_{}".format(mhca_idx), np.ascontiguousarray(np.array([0, 0, heads, 2, dims_per_head], dtype=np.int64)))

        # Reshape output tensor
        output_reshape = gs.Variable("ReshapeKV_{}".format(mhca_idx), np.dtype(np.float16), None)
        # Create fMHA plugin
        reshape = gs.Node(op="Reshape", name="Reshape_{}".format(mhca_idx), inputs=[output_kv, shape], outputs=[output_reshape])
        # Insert node
        self.graph.nodes.append(reshape)

        # Create fMHCA plugin
        fmhca = gs.Node(op="fMHCA", name="fMHCA_{}".format(mhca_idx), inputs=[output_q, output_reshape], outputs=[output_final_tranpose])
        # Insert node
        self.graph.nodes.append(fmhca)

        # Connect input of fMHCA to output of Q GEMM
        node_q.o(num_dynamic).outputs[0] = output_q

        if num_dynamic > 0:
            reshape2_input1_out = gs.Variable("Reshape2_fmhca{}_out".format(mhca_idx), np.dtype(np.int64), None)
            reshape2_input1_shape = gs.Node("Shape", "Reshape2_fmhca{}_shape".format(mhca_idx), inputs=[node_q.inputs[0]], outputs=[reshape2_input1_out])
            self.graph.nodes.append(reshape2_input1_shape)
            final_tranpose.o().inputs[1] = reshape2_input1_out

        # Clear outputs of transpose to get this subgraph cleared
        final_tranpose.outputs.clear()

        self.cleanup()

    def mha_mhca_detected(self, node, mha):
        # Go from V GEMM down to the S*V MatMul and all way up to K GEMM
        # If we are looking for MHCA inputs of two matmuls (K and V) must be equal.
        # If we are looking for MHA inputs (K and V) must be not equal.
        if node.op == "MatMul" and len(node.outputs) == 1 and \
            ((mha and len(node.inputs[0].inputs) > 0  and node.i().op == 'LayerNormalization') or \
            (not mha and len(node.inputs[0].inputs) == 0)):

            if node.o().op == 'Shape':
                if node.o(1).op == 'Shape':
                    num_dynamic_kv = 3 if node.o(2).op == 'Shape' else 2
                else:
                    num_dynamic_kv = 1
                # For Cross-Attention, if batch axis is dynamic (in QKV), assume H*W (in Q) is dynamic as well
                num_dynamic_q = num_dynamic_kv if mha else num_dynamic_kv + 1
            else:
                num_dynamic_kv = 0
                num_dynamic_q = 0

            o = node.o(num_dynamic_kv)
            if o.op == "Reshape" and \
                o.o().op == "Transpose" and \
                o.o().o().op == "Reshape" and \
                o.o().o().o().op == "MatMul" and \
                o.o().o().o().i(0).op == "Softmax" and \
                o.o().o().o().i(1).op == "Reshape" and \
                o.o().o().o().i(0).i().op == "Mul" and \
                o.o().o().o().i(0).i().i().op == "MatMul" and \
                o.o().o().o().i(0).i().i().i(0).op == "Reshape" and \
                o.o().o().o().i(0).i().i().i(1).op == "Transpose" and \
                o.o().o().o().i(0).i().i().i(1).i().op == "Reshape" and \
                o.o().o().o().i(0).i().i().i(1).i().i().op == "Transpose" and \
                o.o().o().o().i(0).i().i().i(1).i().i().i().op == "Reshape" and \
                o.o().o().o().i(0).i().i().i(1).i().i().i().i().op == "MatMul" and \
                node.name != o.o().o().o().i(0).i().i().i(1).i().i().i().i().name:
                # "len(node.outputs) == 1" to make sure we are not in the already fused node
                node_q = o.o().o().o().i(0).i().i().i(0).i().i().i()
                node_k = o.o().o().o().i(0).i().i().i(1).i().i().i().i()
                node_v = node
                final_tranpose = o.o().o().o().o(num_dynamic_q).o()
                # Sanity check to make sure that the graph looks like expected
                if node_q.op == "MatMul" and final_tranpose.op == "Transpose":
                    return True, num_dynamic_q, num_dynamic_kv, node_q, node_k, node_v, final_tranpose
        return False, 0, 0, None, None, None, None
    
    def fuse_kv_insert_fmhca(self, heads, mhca_index, sm):
        nodes = self.graph.nodes
        # Iterate over graph and search for MHCA pattern
        for idx, _ in enumerate(nodes):
            # fMHCA can't be at the 2 last layers of the network. It is a guard from OOB
            if idx + 1 > len(nodes) or idx + 2 > len(nodes):
                continue

            # Get anchor nodes for fusion and fMHCA plugin insertion if the MHCA is detected
            detected, num_dynamic_q, num_dynamic_kv, node_q, node_k, node_v, final_tranpose = \
                self.mha_mhca_detected(nodes[idx], mha=False)
            if detected:
                assert num_dynamic_q == 0 or num_dynamic_q == num_dynamic_kv + 1
                # Skip the FMHCA plugin for SM75 except for when the dim per head is 40.
                if sm == 75 and node_q.inputs[1].shape[1] // heads == 160:
                    continue
                # Fuse K and V GEMMS
                node_kv = self.fuse_kv(node_k, node_v, mhca_index, heads, num_dynamic_kv)
                # Insert fMHCA plugin
                self.insert_fmhca(node_q, node_kv, final_tranpose, mhca_index, heads, num_dynamic_q)
                return True
        return False
    
    def insert_fmhca_plugin(self, num_heads, sm):
        mhca_index = 0
        while self.fuse_kv_insert_fmhca(num_heads, mhca_index, sm):
            mhca_index += 1
        return mhca_index
    
    def fuse_qkv(self, node_q, node_k, node_v, fused_qkv_idx, heads, num_dynamic=0):
        # Get weights of Q
        weights_q = node_q.inputs[1].values
        # Get weights of K
        weights_k = node_k.inputs[1].values
        # Get weights of V
        weights_v = node_v.inputs[1].values

        # Input number of channels to Q, K and V
        C = weights_k.shape[0]
        # Number of heads
        H = heads
        # Hidden dimension per head
        D = weights_k.shape[1] // H

        # Concat and interleave weights such that the output of fused QKV GEMM has [b, s, h, 3, d] shape
        weights_qkv = np.dstack([weights_q.reshape(C, H, D), weights_k.reshape(C, H, D), weights_v.reshape(C, H, D)]).reshape(C, 3 * H * D)

        input_tensor = node_k.inputs[0]  # K and V have the same input
        # Q, K and V must have the same output which we feed into fmha plugin
        output_tensor_k = node_k.outputs[0]
        # Concat and interleave weights such that the output of fused QKV GEMM has [b, s, h, 3, d] shape
        constant_weights_qkv = gs.Constant("Weights_QKV_{}".format(fused_qkv_idx), np.ascontiguousarray(weights_qkv))

        # Created a fused node
        fused_qkv_node = gs.Node(op="MatMul", name="MatMul_QKV_{}".format(fused_qkv_idx), inputs=[input_tensor, constant_weights_qkv], outputs=[output_tensor_k])
        self.graph.nodes.append(fused_qkv_node)

        # Connect the output of the fused node to the inputs of the nodes after Q, K and V
        node_q.o(num_dynamic).inputs[0] = output_tensor_k
        node_k.o(num_dynamic).inputs[0] = output_tensor_k
        node_v.o(num_dynamic).inputs[0] = output_tensor_k
        for i in range(0,num_dynamic):
            node_q.o().inputs.clear()
            node_k.o().inputs.clear()
            node_v.o().inputs.clear()

        # Clear inputs and outputs of Q, K and V to ge these nodes cleared
        node_q.outputs.clear()
        node_k.outputs.clear()
        node_v.outputs.clear()

        node_q.inputs.clear()
        node_k.inputs.clear()
        node_v.inputs.clear()

        self.cleanup()
        return fused_qkv_node
    
    def insert_fmha(self, node_qkv, final_tranpose, mha_idx, heads, num_dynamic=0):
        # Get inputs and outputs for the fMHA plugin
        output_qkv = node_qkv.o().inputs[0]
        output_final_tranpose = final_tranpose.outputs[0]

        # Clear the inputs of the nodes that follow the QKV GEMM
        # to delete these subgraphs (it will be substituted by fMHA plugin)
        node_qkv.outputs[0].outputs[2].inputs.clear()
        node_qkv.outputs[0].outputs[1].inputs.clear()
        node_qkv.outputs[0].outputs[0].inputs.clear()

        weights_qkv = node_qkv.inputs[1].values
        dims_per_head = weights_qkv.shape[1] // (heads * 3)

        # Reshape dims
        shape = gs.Constant("Shape_QKV_{}".format(mha_idx), np.ascontiguousarray(np.array([0, 0, heads, 3, dims_per_head], dtype=np.int64)))

        # Reshape output tensor
        output_shape = gs.Variable("ReshapeQKV_{}".format(mha_idx), np.dtype(np.float16), None)
        # Create fMHA plugin
        reshape = gs.Node(op="Reshape", name="Reshape_{}".format(mha_idx), inputs=[output_qkv, shape], outputs=[output_shape])
        # Insert node
        self.graph.nodes.append(reshape)

        # Create fMHA plugin
        fmha = gs.Node(op="fMHA_V2", name="fMHA_{}".format(mha_idx), inputs=[output_shape], outputs=[output_final_tranpose])
        # Insert node
        self.graph.nodes.append(fmha)

        if num_dynamic > 0:
            reshape2_input1_out = gs.Variable("Reshape2_{}_out".format(mha_idx), np.dtype(np.int64), None)
            reshape2_input1_shape = gs.Node("Shape", "Reshape2_{}_shape".format(mha_idx), inputs=[node_qkv.inputs[0]], outputs=[reshape2_input1_out])
            self.graph.nodes.append(reshape2_input1_shape)
            final_tranpose.o().inputs[1] = reshape2_input1_out

        # Clear outputs of transpose to get this subgraph cleared
        final_tranpose.outputs.clear()

        self.cleanup()

    def fuse_qkv_insert_fmha(self, heads, mha_index):
        nodes = self.graph.nodes
        # Iterate over graph and search for MHA pattern
        for idx, _ in enumerate(nodes):
            # fMHA can't be at the 2 last layers of the network. It is a guard from OOB
            if idx + 1 > len(nodes) or idx + 2 > len(nodes):
                continue

            # Get anchor nodes for fusion and fMHA plugin insertion if the MHA is detected
            detected, num_dynamic_q, num_dynamic_kv, node_q, node_k, node_v, final_tranpose = \
                self.mha_mhca_detected(nodes[idx], mha=True)
            if detected:
                assert num_dynamic_q == num_dynamic_kv
                # Fuse Q, K and V GEMMS
                node_qkv = self.fuse_qkv(node_q, node_k, node_v, mha_index, heads, num_dynamic_kv)
                # Insert fMHA plugin
                self.insert_fmha(node_qkv, final_tranpose, mha_index, heads, num_dynamic_kv)
                return True
        return False
    
    def insert_fmha_plugin(self, num_heads):
        mha_index = 0
        while self.fuse_qkv_insert_fmha(num_heads, mha_index):
            mha_index += 1
        return mha_index

def get_path(version, inpaint=False):
    if version == "1.4":
        if inpaint:
            return "runwayml/stable-diffusion-inpainting"
        else:
            return "CompVis/stable-diffusion-v1-4"
    elif version == "1.5":
        if inpaint:
            return "runwayml/stable-diffusion-inpainting"
        else:
            return "/root/yh7/runwayml/stable-diffusion-v1-5"
    elif version == "2.0-base":
        if inpaint:
            return "stabilityai/stable-diffusion-2-inpainting"
        else:
            return "stabilityai/stable-diffusion-2-base"
    elif version == "2.0":
        if inpaint:
            return "stabilityai/stable-diffusion-2-inpainting"
        else:
            return "stabilityai/stable-diffusion-2"
    elif version == "2.1":
        return "stabilityai/stable-diffusion-2-1"
    elif version == "2.1-base":
        return "stabilityai/stable-diffusion-2-1-base"
    else:
        raise ValueError(f"Incorrect version {version}")

def get_embedding_dim(version):
    if version in ("1.4", "1.5"):
        return 768
    elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
        return 1024
    else:
        raise ValueError(f"Incorrect version {version}")

class BaseModel():
    def __init__(
        self,
        hf_token,
        fp16=False,
        device='cuda',
        verbose=True,
        path="",
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
    ):
        self.name = "SD Model"
        self.hf_token = hf_token
        self.fp16 = fp16
        self.device = device
        self.verbose = verbose
        self.path = path

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256   # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.cleanup()
        opt.info(self.name + ': cleanup')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ': finished')
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width)

class CLIP(BaseModel):
    def __init__(self,
        hf_token,
        device,
        verbose,
        path,
        max_batch_size,
        embedding_dim
    ):
        super(CLIP, self).__init__(hf_token, device=device, verbose=verbose, path=path, max_batch_size=max_batch_size, embedding_dim=embedding_dim)
        self.name = "CLIP"

    def get_model(self):
        return CLIPTextModel.from_pretrained(self.path,
            subfolder="text_encoder",
            use_auth_token=self.hf_token).to(self.device)

    def get_input_names(self):
        return ['input_ids']

    def get_output_names(self):
       return ['text_embeddings', 'pooler_output']

    def get_dynamic_axes(self):
        return {
            'input_ids': {0: 'B'},
            'text_embeddings': {0: 'B'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.select_outputs([0]) # delete graph output#1
        opt.cleanup()
        opt.info(self.name + ': remove output[1]')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        opt.select_outputs([0], names=['text_embeddings']) # rename network output
        opt.info(self.name + ': remove output[0]')
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ': finished')
        return opt_onnx_graph

def make_CLIP(version, hf_token, device, verbose, max_batch_size, inpaint=False):
    return CLIP(hf_token=hf_token, device=device, verbose=verbose, path=get_path(version, inpaint=inpaint),
                max_batch_size=max_batch_size, embedding_dim=get_embedding_dim(version))

class UNet(BaseModel):
    def __init__(self,
        hf_token,
        fp16=False,
        device='cuda',
        verbose=True,
        path="",
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4
    ):
        super(UNet, self).__init__(hf_token, fp16=fp16, device=device, verbose=verbose, path=path, max_batch_size=max_batch_size, embedding_dim=embedding_dim, text_maxlen=text_maxlen)
        self.unet_dim = unet_dim
        self.name = "UNet"

    def get_model(self):
        model_opts = {'revision': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        return UNet2DConditionModel.from_pretrained(self.path,
            subfolder="unet",
            use_auth_token=self.hf_token,
            **model_opts).to(self.device)

    def get_input_names(self):
        return ['sample', 'timestep', 'encoder_hidden_states']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            'sample': {0: '2B', 2: 'H', 3: 'W'},
            'encoder_hidden_states': {0: '2B'},
            'latent': {0: '2B', 2: 'H', 3: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'sample': [(2*min_batch, self.unet_dim, min_latent_height, min_latent_width), (2*batch_size, self.unet_dim, latent_height, latent_width), (2*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
            'encoder_hidden_states': [(2*min_batch, self.text_maxlen, self.embedding_dim), (2*batch_size, self.text_maxlen, self.embedding_dim), (2*max_batch, self.text_maxlen, self.embedding_dim)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (2*batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (2*batch_size, self.text_maxlen, self.embedding_dim),
            'latent': (2*batch_size, 4, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(2*batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
            torch.tensor([1.], dtype=torch.float32, device=self.device),
            torch.randn(2*batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device)
        )
    
    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.cleanup()
        opt.info(self.name + ': cleanup')
        
        num_heads = 8
        opt.remove_casts()
        opt.info(self.name + ': remove cast')
        
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        
        num_fmha_inserted = opt.insert_fmha_plugin(num_heads=num_heads)
        opt.info('UNet: inserted '+str(num_fmha_inserted)+' fMHA plugins')
        
        props = cudart.cudaGetDeviceProperties(0)[1]
        sm = props.major * 10 + props.minor
        num_fmhca_inserted = opt.insert_fmhca_plugin(num_heads=num_heads, sm=sm)
        opt.info('UNet: inserted '+str(num_fmhca_inserted)+' fMHCA plugins')
        
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ': finished')
        return onnx_opt_graph

def make_UNet(version, hf_token, device, verbose, max_batch_size, inpaint=False):
    return UNet(hf_token=hf_token, fp16=True, device=device, verbose=verbose, path=get_path(version, inpaint=inpaint),
            max_batch_size=max_batch_size, embedding_dim=get_embedding_dim(version), unet_dim=(9 if inpaint else 4))

class VAE(BaseModel):
    def __init__(self,
        hf_token,
        device,
        verbose,
        path,
        max_batch_size,
        embedding_dim
    ):
        super(VAE, self).__init__(hf_token, device=device, verbose=verbose, path=path, max_batch_size=max_batch_size, embedding_dim=embedding_dim)
        self.name = "VAE decoder"

    def get_model(self):
        vae = AutoencoderKL.from_pretrained(self.path,
            subfolder="vae",
            use_auth_token=self.hf_token).to(self.device)
        vae.forward = vae.decode
        return vae

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
       return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [(min_batch, 4, min_latent_height, min_latent_width), (batch_size, 4, latent_height, latent_width), (max_batch, 4, max_latent_height, max_latent_width)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'latent': (batch_size, 4, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)

def make_VAE(version, hf_token, device, verbose, max_batch_size, inpaint=False):
    return VAE(hf_token=hf_token, device=device, verbose=verbose, path=get_path(version, inpaint=inpaint),
            max_batch_size=max_batch_size, embedding_dim=get_embedding_dim(version))

class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, token, device, path):
        super().__init__()
        self.path = path
        self.vae_encoder = AutoencoderKL.from_pretrained(self.path, subfolder="vae", use_auth_token=token).to(device)
        
    def forward(self, x):
        return self.vae_encoder.encode(x).latent_dist.sample()

class VAEEncoder(BaseModel):
    def __init__(self,
        hf_token,
        device,
        verbose,
        path,
        max_batch_size,
        embedding_dim
    ):
        super(VAEEncoder, self).__init__(hf_token, device=device, verbose=verbose, path=path, max_batch_size=max_batch_size, embedding_dim=embedding_dim)
        self.name = "VAE encoder"

    def get_model(self):
        vae_encoder = TorchVAEEncoder(self.hf_token, self.device, self.path)
        return vae_encoder

    def get_input_names(self):
        return ['images']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            'images': {0: 'B', 2: '8H', 3: '8W'},
            'latent': {0: 'B', 2: 'H', 3: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, _, _, _, _ = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            'images': [(min_batch, 3, min_image_height, min_image_width), (batch_size, 3, image_height, image_width), (max_batch, 3, max_image_height, max_image_width)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'images': (batch_size, 3, image_height, image_width),
            'latent': (batch_size, 4, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device)

def make_VAEEncoder(version, hf_token, device, verbose, max_batch_size, inpaint=False):
    return VAEEncoder(hf_token=hf_token, device=device, verbose=verbose, path=get_path(version, inpaint=inpaint),
            max_batch_size=max_batch_size, embedding_dim=get_embedding_dim(version))

def make_tokenizer(version, hf_token):
    return CLIPTokenizer.from_pretrained(get_path(version),
            subfolder="tokenizer",
            use_auth_token=hf_token)
