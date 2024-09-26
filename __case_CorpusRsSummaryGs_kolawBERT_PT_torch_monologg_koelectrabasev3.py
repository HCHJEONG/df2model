from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from transformers import ElectraForMaskedLM # generator class
from transformers import ElectraForPreTraining # discriminator class
from transformers import ElectraTokenizer #  gen disc 공통
from transformers.optimization import get_cosine_schedule_with_warmup

from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch import nn

import torch
import torch.nn.functional as F
import gc

import pickle
import datetime
import numpy as np
import pandas as pd

samplingRate = 0.00001 # This should be 1.0 in case of real training
test_size = 0.2
batch_size = 8
epochs = 12

acc_train_history=[]
acc_test_history=[]
loss_train_history=[]
loss_test_history=[]

class MyConfig(dict):
  def __getattr__(self, name): return self[name]
  def __setattr__(self, name, value): self[name] = value
  
c = MyConfig({       
    'log_interval' : 1000,
    
    # how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
    'num_workers': 0, 
    'masking_rate': 0.15,
    
    # Electra Model 사용시 필요 / fp floating point
    'sampling': 'fp32_gumbel', 
    
    # Electra Loss 사용시 필요
    'gen_smooth_label': False,
    'disc_smooth_label': False,
    'loss_weights': (1.0, 50.0), # (gen loss weight, disc loss weight)
    
    # Scheduler 사용시 필요 
    # 'schedule': 'original_linear',
    'warmup_steps' : None,
    'warmup_ratio' : 0.1,
    
    # AdamW 사용시 필요
    'weight_decay' : 0.01,
    'adam_bias_correction': False,
    
    # 그래디언트 정규화 하나 더
    # 'max_grad_norm' : 1,
    # 'dr_rate' : None,   
    'learning_rate' : 5e-7, # 0.0000005
    # 'learning_rate' : 5e-5, # 0.00005
    })

dataFileName = [
    '..//web2df//dataset//listForCaseSentenceForgists.pickle', 
    '..//web2df//dataset//listForCaseSentenceForreasoning.pickle'
    ]
x_dataFieldName = 'unit_str'

seq_length = 512 
padToken = 0
unkToken = 1
clsToken = 2
sepToken = 3
maskToken = 4

class MLMDataset(Dataset):
    
    def __init__(self, inputs, device): # iterator of dict of (tokenizer.__call__() return) 
        self.inputs = inputs
        self.device = device
        
    def __getitem__(self, idx):
        # val[idx] 은 (1, 512) shape tensor with int64
        return {key: val[idx].detach().clone().to(self.device) for key, val in self.inputs.items()}
    
    def __len__(self):
        return len(self.inputs['input_ids'])

def dataloader_factory(inputs, device, batch_size, num_workers):       
           
    # dataloader setting
    # print("Torch Dataset / Torch DataLoader instantiating...")
    dataset = MLMDataset(inputs, device)
    dataLoader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle = True, 
                            num_workers = num_workers
                            # collate_fn=lambda x:x # 배치 리스트 요소를 데이터 개별 인스턴스로 세팅
                            )
    # print("done!")
    # print()
    return dataLoader 

class ELECTRAModel(nn.Module):
  
  def __init__(self, generator, discriminator, hf_tokenizer):
    super().__init__()
    self.generator, self.discriminator = generator, discriminator
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(0., 1.) # 위치 규모(tau or temperature) 두 하이퍼파라미터
    self.hf_tokenizer = hf_tokenizer

  def to(self, *args, **kwargs):
    "Also set dtype and device of contained gumbel distribution if needed"
    super().to(*args, **kwargs)
    
    a_tensor = next(self.parameters())
    device, dtype = a_tensor.device, a_tensor.dtype
    
    if c.sampling=='fp32_gumbel': 
        dtype = torch.float32
        
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

  def forward(self, data_batch):
    """
    masked_inputs (Tensor[int]): (Batch, Length)
    is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability 
    """
    masked_inputs = data_batch['input_ids'].squeeze(1)
    attention_mask = data_batch['attention_mask'].squeeze(1)
    token_type_ids = data_batch['token_type_ids'].squeeze(1)
    labels = data_batch['labels'].squeeze(1)
    is_mlm_applied = (masked_inputs == maskToken).squeeze(1)

    # print('\nmasked inputs')
    # print(masked_inputs)
    # print(masked_inputs.shape) # torch.Size([4, 512])
    # print(attention_mask)
    # print(token_type_ids)
    # print(labels)
    # print('\nis mlm applied')
    # print(is_mlm_applied)
    # print(is_mlm_applied.shape) # torch.Size([4, 512])
 
    # gen_outputs = self.generator(**data_batch)   
    gen_outputs = self.generator(
                                masked_inputs, 
                                attention_mask, 
                                token_type_ids
                                )  
    # print('\ngen_outpus')
    # print(type(gen_outputs)) # <class 'transformers.modeling_outputs.MaskedLMOutput'>
    # print(dir(gen_outputs))
    # print(len(gen_outputs)) # 1
    # for i in gen_outputs:
    #     print('what is in gen outputs?: ')
    #     print(i) # 'logits'
    #     print(type(i)) # str
    # print(gen_outputs.loss) # None
    # print(gen_outputs.logits)
    # print(gen_outputs.hidden_states) # None
    # print(gen_outputs.attentions) # None
    
    gen_logits = gen_outputs[0]
    # print('\ngen_outputs[0] a.k.a. gen_logits\n')
    # print(gen_logits)
    # print(type(gen_logits))
    # print(gen_logits.shape) # torch.Size([4, 512, 35000]) <= (B, L, vocab size)
    # print('now right after generator output...\n')

    # reduce size to save space and speed
    mlm_gen_logits = gen_logits[is_mlm_applied, :] # (B, L * 0.15, 35000)
    # ( #mlm_positions, vocab_size) torch.Size([18, 35000])
    # is mlm applied - torch.Size([4, 512])
    # print(mlm_gen_logits)
    # print(mlm_gen_logits.shape) # torch.Size([18, 35000])
    # print('\nnow right after filtering logits only for masked...\n')
          
    with torch.no_grad():
      # sampling
      pred_toks = self.sample(mlm_gen_logits) # ( #mlm_positions, )
      # produce inputs for discriminator
      generated = masked_inputs.clone() # (B,L)
      generated[is_mlm_applied] = pred_toks # (B,L)
    #   print(generated)
    #   for i, x in enumerate(generated):
        # print("\nlabel: ")
        # print(tokenizer.decode(labels[i]))
        # print("\ngenerated: ")
        # print(tokenizer.decode(x))
    #   input('...')

      # produce labels for discriminator
      is_replaced = is_mlm_applied.clone() # (B,L)
      # (B,L) is replaced True 부분이 generator가 못 맞춘 부분으로 disc에 대해서는 label로 기능함
      is_replaced[is_mlm_applied] = (pred_toks != labels[is_mlm_applied]) 
    #   for g in is_replaced:
    #     print(g)
    #     print()
    #   input('...')
      
    # print(generated)
    # print(generated.shape)
    # print(is_replaced)
    # print(is_replaced.shape)
    # input('...')
    
    # discriminator output은 tuple ( (loss: optional), logits, ...(optionals))
    disc_logits = self.discriminator(generated, attention_mask, token_type_ids)[0] 
    # (B, L, vocab size)
    # print(disc_logits)
    # print(disc_logits.shape)
    # print(disc_logits.dtype)
    # print(dir(disc_logits)) 
    # ['T', '__abs__', '__add__', '__and__', '__array__', '__array_priority__', '__array_wrap__', '__bool__', '__class__', '__complex__', '__contains__', '__deepcopy__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__div__', '__dlpack__', '__dlpack_device__', '__doc__', '__eq__', '__float__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__iand__', '__idiv__', '__ifloordiv__', '__ilshift__', '__imod__', '__imul__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__long__', '__lshift__', '__lt__', '__matmul__', '__mod__', '__module__', '__mul__', '__ne__', '__neg__', '__new__', '__nonzero__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdiv__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rfloordiv__', '__rlshift__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__torch_function__', '__truediv__', '__weakref__', '__xor__', '_backward_hooks', '_base', '_cdata', '_coalesced_', '_conj', '_conj_physical', '_dimI', '_dimV', '_fix_weakref', '_grad', '_grad_fn', '_indices', '_is_view', '_make_subclass', '_neg_view', '_nnz', '_python_dispatch', '_reduce_ex_internal', '_update_names', '_values', '_version', 'abs', 'abs_', 'absolute', 'absolute_', 'acos', 'acos_', 'acosh', 'acosh_', 'add', 'add_', 'addbmm', 'addbmm_', 'addcdiv', 'addcdiv_', 'addcmul', 'addcmul_', 'addmm', 'addmm_', 'addmv', 'addmv_', 'addr', 'addr_', 'align_as', 'align_to', 'all', 'allclose', 'amax', 'amin', 'aminmax', 'angle', 'any', 'apply_', 'arccos', 'arccos_', 'arccosh', 'arccosh_', 'arcsin', 'arcsin_', 'arcsinh', 'arcsinh_', 'arctan', 'arctan_', 'arctanh', 'arctanh_', 'argmax', 'argmin', 'argsort', 'as_strided', 'as_strided_', 'as_subclass', 'asin', 'asin_', 'asinh', 'asinh_', 'atan', 'atan2', 'atan2_', 'atan_', 'atanh', 'atanh_', 'backward', 'baddbmm', 'baddbmm_', 'bernoulli', 'bernoulli_', 'bfloat16', 'bincount', 'bitwise_and', 'bitwise_and_', 'bitwise_left_shift', 'bitwise_left_shift_', 'bitwise_not', 'bitwise_not_', 'bitwise_or', 'bitwise_or_', 'bitwise_right_shift', 'bitwise_right_shift_', 'bitwise_xor', 'bitwise_xor_', 'bmm', 'bool', 'broadcast_to', 'byte', 'cauchy_', 'cdouble', 'ceil', 'ceil_', 'cfloat', 'char', 'cholesky', 'cholesky_inverse', 'cholesky_solve', 'chunk', 'clamp', 'clamp_', 'clamp_max', 'clamp_max_', 'clamp_min', 'clamp_min_', 'clip', 'clip_', 'clone', 'coalesce', 'col_indices', 'conj', 'conj_physical', 'conj_physical_', 'contiguous', 'copy_', 'copysign', 'copysign_', 'corrcoef', 'cos', 'cos_', 'cosh', 'cosh_', 'count_nonzero', 'cov', 'cpu', 'cross', 'crow_indices', 'cuda', 'cummax', 'cummin', 'cumprod', 'cumprod_', 'cumsum', 'cumsum_', 'data', 'data_ptr', 'deg2rad', 'deg2rad_', 'dense_dim', 'dequantize', 'det', 'detach', 'detach_', 'device', 'diag', 'diag_embed', 'diagflat', 'diagonal', 'diff', 'digamma', 'digamma_', 'dim', 'dist', 'div', 'div_', 'divide', 'divide_', 'dot', 'double', 'dsplit', 'dtype', 'eig', 'element_size', 'eq', 'eq_', 'equal', 'erf', 'erf_', 'erfc', 'erfc_', 'erfinv', 'erfinv_', 'exp', 'exp2', 'exp2_', 'exp_', 'expand', 'expand_as', 'expm1', 'expm1_', 'exponential_', 'fill_', 'fill_diagonal_', 'fix', 'fix_', 'flatten', 'flip', 'fliplr', 'flipud', 'float', 'float_power', 'float_power_', 'floor', 'floor_', 'floor_divide', 'floor_divide_', 'fmax', 'fmin', 'fmod', 'fmod_', 'frac', 'frac_', 'frexp', 'gather', 'gcd', 'gcd_', 'ge', 'ge_', 'geometric_', 'geqrf', 'ger', 'get_device', 'grad', 'grad_fn', 'greater', 'greater_', 'greater_equal', 'greater_equal_', 'gt', 'gt_', 'half', 'hardshrink', 'has_names', 'heaviside', 'heaviside_', 'histc', 'histogram', 'hsplit', 'hypot', 'hypot_', 'i0', 'i0_', 'igamma', 'igamma_', 'igammac', 'igammac_', 'imag', 'index_add', 'index_add_', 'index_copy', 'index_copy_', 'index_fill', 'index_fill_', 'index_put', 'index_put_', 'index_select', 'indices', 'inner', 'int', 'int_repr', 'inverse', 'is_coalesced', 'is_complex', 'is_conj', 'is_contiguous', 'is_cuda', 'is_distributed', 'is_floating_point', 'is_inference', 'is_leaf', 'is_meta', 'is_mkldnn', 'is_mlc', 'is_neg', 'is_nonzero', 'is_ort', 'is_pinned', 'is_quantized', 'is_same_size', 'is_set_to', 'is_shared', 'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan', 'is_xpu', 'isclose', 'isfinite', 'isinf', 'isnan', 'isneginf', 'isposinf', 'isreal', 'istft', 'item', 'kron', 'kthvalue', 'layout', 'lcm', 'lcm_', 'ldexp', 'ldexp_', 'le', 'le_', 'lerp', 'lerp_', 'less', 'less_', 'less_equal', 'less_equal_', 'lgamma', 'lgamma_', 'log', 'log10', 'log10_', 'log1p', 'log1p_', 'log2', 'log2_', 'log_', 'log_normal_', 'log_softmax', 'logaddexp', 'logaddexp2', 'logcumsumexp', 'logdet', 'logical_and', 'logical_and_', 'logical_not', 'logical_not_', 'logical_or', 'logical_or_', 'logical_xor', 'logical_xor_', 'logit', 'logit_', 'logsumexp', 'long', 'lstsq', 'lt', 'lt_', 'lu', 'lu_solve', 'map2_', 'map_', 'masked_fill', 'masked_fill_', 'masked_scatter', 'masked_scatter_', 'masked_select', 'matmul', 'matrix_exp', 'matrix_power', 'max', 'maximum', 'mean', 'median', 'min', 'minimum', 'mm', 'mode', 'moveaxis', 'movedim', 'msort', 'mul', 'mul_', 'multinomial', 'multiply', 'multiply_', 'mv', 'mvlgamma', 'mvlgamma_', 'name', 'names', 'nan_to_num', 'nan_to_num_', 'nanmean', 'nanmedian', 'nanquantile', 'nansum', 'narrow', 'narrow_copy', 'ndim', 'ndimension', 'ne', 'ne_', 'neg', 'neg_', 'negative', 'negative_', 'nelement', 'new', 'new_empty', 'new_empty_strided', 'new_full', 'new_ones', 'new_tensor', 'new_zeros', 'nextafter', 'nextafter_', 'nonzero', 'norm', 'normal_', 'not_equal', 'not_equal_', 'numel', 'numpy', 'orgqr', 'ormqr', 'outer', 'output_nr', 'permute', 'pin_memory', 'pinverse', 'polygamma', 'polygamma_', 'positive', 'pow', 'pow_', 'prelu', 'prod', 'put', 'put_', 'q_per_channel_axis', 'q_per_channel_scales', 'q_per_channel_zero_points', 'q_scale', 'q_zero_point', 'qr', 'qscheme', 'quantile', 'rad2deg', 'rad2deg_', 'random_', 'ravel', 'real', 'reciprocal', 'reciprocal_', 'record_stream', 'refine_names', 'register_hook', 'reinforce', 'relu', 'relu_', 'remainder', 'remainder_', 'rename', 'rename_', 'renorm', 'renorm_', 'repeat', 'repeat_interleave', 'requires_grad', 'requires_grad_', 'reshape', 'reshape_as', 'resize', 'resize_', 'resize_as', 'resize_as_', 'resolve_conj', 'resolve_neg', 'retain_grad', 'retains_grad', 'roll', 'rot90', 'round', 'round_', 'rsqrt', 'rsqrt_', 'scatter', 'scatter_', 'scatter_add', 'scatter_add_', 'select', 'set_', 'sgn', 'sgn_', 'shape', 'share_memory_', 'short', 'sigmoid', 'sigmoid_', 'sign', 'sign_', 'signbit', 'sin', 'sin_', 'sinc', 'sinc_', 'sinh', 'sinh_', 'size', 'slogdet', 'smm', 'softmax', 'solve', 'sort', 'sparse_dim', 'sparse_mask', 'sparse_resize_', 'sparse_resize_and_clear_', 'split', 'split_with_sizes', 'sqrt', 'sqrt_', 'square', 'square_', 'squeeze', 'squeeze_', 'sspaddmm', 'std', 'stft', 'storage', 'storage_offset', 'storage_type', 'stride', 'sub', 'sub_', 'subtract', 'subtract_', 'sum', 'sum_to_size', 'svd', 'swapaxes', 'swapaxes_', 'swapdims', 'swapdims_', 'symeig', 't', 't_', 'take', 'take_along_dim', 'tan', 'tan_', 'tanh', 'tanh_', 'tensor_split', 'tile', 'to', 'to_dense', 'to_mkldnn', 'to_sparse', 'to_sparse_csr', 'tolist', 'topk', 'trace', 'transpose', 'transpose_', 'triangular_solve', 'tril', 'tril_', 'triu', 'triu_', 'true_divide', 'true_divide_', 'trunc', 'trunc_', 'type', 'type_as', 'unbind', 'unflatten', 'unfold', 'uniform_', 'unique', 'unique_consecutive', 'unsafe_chunk', 'unsafe_split', 'unsafe_split_with_sizes', 'unsqueeze', 'unsqueeze_', 'values', 'var', 'vdot', 'view', 'view_as', 'vsplit', 'where', 'xlogy', 'xlogy_', 'xpu', 'zero_']
    # input('...')
    
    return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

#   def _get_pad_mask_and_token_type(self, input_ids, sentA_lenths):
#     """
#     Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask and token_type_ids and won't be unnecessarily large, thus, prevent cpu processes loading batches from consuming lots of cpu memory and slow down the machine. 
#     """
#     attention_mask = input_ids != self.hf_tokenizer.pad_token_id
#     seq_len = input_ids.shape[1]
#     token_type_ids = torch.tensor([ ([0]*len + [1]*(seq_len-len)) for len in sentA_lenths.tolist()],  
#                                   device=input_ids.device)
#     return attention_mask, token_type_ids

  def sample(self, logits):
    "Reimplement gumbel softmax cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16 (https://github.com/pytorch/pytorch/issues/41663). Gumbel softmax is equal to what official ELECTRA code do, standard gumbel dist. = -ln(-ln(standard uniform dist.))"
    if c.sampling == 'fp32_gumbel':
      gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
      return (logits.float() + gumbel).argmax(dim=-1)
    elif c.sampling == 'fp16_gumbel':  # 5.06 ms
      gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
      return (logits + gumbel).argmax(dim=-1)
    elif c.sampling == 'multinomial':  # 2.X ms
      return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()

class ELECTRALoss():
    
  def __init__(self, loss_weights= c.loss_weights, gen_label_smooth=False, disc_label_smooth=False):
    
    self.loss_weights = loss_weights
    
    self.gen_loss_fc = CrossEntropyLoss() 
    # 2d tensor(softmax logits) and 1d tensor(idx) as args
    # self.gen_loss_fc = LabelSmoothingCrossEntropy(eps=gen_label_smooth) if gen_label_smooth else CrossEntropyLoss()
    # self.gen_loss_fc = LabelSmoothingCrossEntropyFlat(eps=gen_label_smooth) if gen_label_smooth else CrossEntropyLossFlat()
    
    self.disc_loss_fc = nn.BCEWithLogitsLoss() 
    # 1d tensor float(sigmoid logits) and 1d tensor float as args
    # self.disc_label_smooth = disc_label_smooth
    
  def __call__(self, pred, targ_ids):
    mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied = pred
    gen_loss = self.gen_loss_fc(mlm_gen_logits.float(), targ_ids[is_mlm_applied])
    
    non_pad = (attention_mask == 1) # padToken = 0
    # print(type(non_pad)) # torch Tensor, not tuple
    # print('...')
    
    # print(disc_logits.shape)
    disc_logits_ = disc_logits.masked_select(non_pad) # 2d ( B, L ) tensor -> 1d tensor
    # print(disc_logits.shape)
    # input('...')
    
    is_replaced = is_replaced.masked_select(non_pad) # 2d ( B, L ) tensor -> 1d tensor
    # if self.disc_label_smooth:
    #   is_replaced = is_replaced.float().masked_fill(~is_replaced, self.disc_label_smooth)
    
    disc_loss = self.disc_loss_fc(disc_logits_.float(), is_replaced.float())
    
    return (
            gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1], # 감마가 50
            gen_loss,
            disc_loss
            )

#정확도 측정을 위한 함수 정의
def calc_accuracy_for_minibatch(disc_logits, is_replaced, is_mlm_applied):
    # max_vals, max_indices = torch.max(prediclogitsTensorList, 1) #    
    # print(disc_logits)
    # print(F.softmax(disc_logits))
    # print(torch.round(F.softmax(disc_logits)).bool())
    # print(is_replaced)
    # ###
    
    result = []
    for idx in range(len(is_replaced)):
        pred = torch.round((torch.sign(disc_logits[idx].cpu().detach()) + 1) / 2).masked_select(is_mlm_applied[idx].cpu().detach()).numpy()
        # pred = torch.round(torch.sigmoid(disc_logits[idx].cpu().detach())).bool().masked_select(is_mlm_applied[idx].cpu().detach()).numpy()
        label = is_replaced[idx].cpu().detach().masked_select(is_mlm_applied[idx].cpu().detach()).numpy()
        # print('\npred: ')
        # print(pred)
        # print('\nlabel: ')
        # print(label)
        acc = accuracy_score(
           label,
           pred, 
           )
        result.append(acc)    
    # acc = 10000 * (1/(np.sqrt(torch.mul((prediclogitsTensorList - labellogitsTensorList), (prediclogitsTensorList - labellogitsTensorList)).sum().data.cpu().numpy()/labellogitsTensorList.size()[0])))
    # 두 리스트의 같은 위치의 요소를 비교해서 조건식을 충족하는 경우에는 그 충족 횟수의 합계를 내고
    # 그 합계를 리스트의 요소 갯수로 나누어 점수를 구함
    # 그 점수의 역수에 10000을 곱함 (0.001 ~ 0.01 => 10 ~ 1000)
    return result
'''
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
'''

if __name__ == '__main__':
    # print(torch.Tensor([True, False]))
    # print(type(torch.Tensor([True, False])) )
    # print(torch.Tensor([True, False]).dtype)
    # x=torch.Tensor([True, False])==True
    # print(x)
    # print(type(x))
    # print(x.dtype)
    # tensor([1., 0.])
    # <class 'torch.Tensor'>
    # torch.float32
    # tensor([ True, False])
    # <class 'torch.Tensor'>
    # torch.bool

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA 사용여부: {}\n".format(torch.cuda.is_available()))
    
    torch.manual_seed(42) # np.random.seed()와 동일 기능
    
    print('\ngenerator downloading: \n')
    generator =\
        ElectraForMaskedLM.from_pretrained("monologg/koelectra-base-v3-generator")
    print('\ngenerator config: \n', generator.config)
    print('\ndiscriminator downloading: \n')
    discriminator =\
        ElectraForPreTraining.from_pretrained("monologg/koelectra-base-v3-discriminator")
    print('\ndiscriminator config: \n', discriminator.config)
    
    print('\ntokenizer for discriminator downloading: \n')
    tokenizer =\
        ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    # print(tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
    # print("\n bos token: \n")
    # print(tokenizer.bos_token_id)
    # print("\n eos token: \n")
    # print(tokenizer.eos_token_id)
    # print("\n unk token: \n")
    # print(tokenizer.unk_token_id)
    # print("\n sep token: \n")
    # print(tokenizer.sep_token_id)
    # print("\n pad token: \n")
    # print(tokenizer.pad_token_id)
    # print("\n cls token: \n")
    # print(tokenizer.cls_token_id)
    # print("\n mask token: \n")
    # print(tokenizer.mask_token_id)
    # print("\n additional special tokens: \n")
    # print(tokenizer.additional_special_tokens_ids)
    # print()
       
    # senSample = '''
    # 위 경매절차에서 집행관이 이 사건 주택에 관한 현황을 조사할 당시 피고의 처인 소외 2는 위 임차기간이 '1995. 7. 31.부터 1996년 8월 현재까지'라고 진술하여 그에 따라 집행관 명의의 1996. 8. 24.자 부동산현황조사보고서가 작성되었으며, 한편 피고는 1996. 8. 31. 위 경매법원에 이 사건 주택에 대한 확정일자부 임차인으로서 권리신고 및 임차보증금 40,000,000원에 대한 배당요구를 함에 있어서, 임대차기간을 1996. 7. 30.까지로 하여 작성된 1995. 7. 29.자 임대차계약서만을 제출하였을 뿐 임대차기간의 연장에 관한 아무런 자료를 제출하지 않았고, 경매법원은 피고의 위와 같은 배당요구사실을 소유자인 소외 1에게 통지하지는 않았고 다만 원고가 낙찰받기 직전의 입찰명령에 첨부된 부동산목록에 피고가 임차인으로 된 [MASK]의 기간을 1996. 7. 30.까지로 표시하였다.''' # 임대차 masked
    # senSampleLabel = '''
    # 위 경매절차에서 집행관이 이 사건 주택에 관한 현황을 조사할 당시 피고의 처인 소외 2는 위 임차기간이 '1995. 7. 31.부터 1996년 8월 현재까지'라고 진술하여 그에 따라 집행관 명의의 1996. 8. 24.자 부동산현황조사보고서가 작성되었으며, 한편 피고는 1996. 8. 31. 위 경매법원에 이 사건 주택에 대한 확정일자부 임차인으로서 권리신고 및 임차보증금 40,000,000원에 대한 배당요구를 함에 있어서, 임대차기간을 1996. 7. 30.까지로 하여 작성된 1995. 7. 29.자 임대차계약서만을 제출하였을 뿐 임대차기간의 연장에 관한 아무런 자료를 제출하지 않았고, 경매법원은 피고의 위와 같은 배당요구사실을 소유자인 소외 1에게 통지하지는 않았고 다만 원고가 낙찰받기 직전의 입찰명령에 첨부된 부동산목록에 피고가 임차인으로 된 임대차의 기간을 1996. 7. 30.까지로 표시하였다.'''

    # print('\ntokenized result with tokenizer.tokenize with no option for ' + senSample + ": \n")
    # print(tokenizer.tokenize(senSample))
    # print('\ninput_ids for ' + senSample + " with tensored result of tokenizer.encode: \n")
    # input_ids = torch.tensor(tokenizer.encode(senSample, add_special_tokens=True)).unsqueeze(
    #     0
    # )  # Batch size 1
    # print(input_ids)
    # print('input_ids type and shape: ')
    # print(type(input_ids))
    # print(input_ids.shape)

    # inputs = tokenizer(senSample, return_tensors="pt")
    # print('\ntokenized masked with tokenizer for pt option: \n')
    # print(inputs["input_ids"])
    # print(type(inputs))
    # print(torch.tensor(inputs['input_ids'], dtype=torch.int32).shape)

    # labels = tokenizer(senSampleLabel, return_tensors="pt")["input_ids"]
    # print('\ntokenized label with tokenizer for pt option: \n')
    # print(labels)
    # print(type(labels))
    # print(torch.tensor(labels, dtype=torch.int32).shape)

    # print('\ntokenized label with tokenizer.tokenize without pt option(no cls sep tokens): \n')
    # tokenized = tokenizer.tokenize(senSampleLabel)
    # print(tokenized)
    # print(type(tokenized))
    # print(len(tokenized))
    
    # logits = generator(input_ids).logits
    # print('\ngenerator logits for sample sentence: \n', logits)
    # print('\ntype and shape of logits: \n')
    # print(type(logits))
    # print(logits.shape)

    # print('\ngenerator outputs with **inputs agrs for sample sentence: \n')
    # outputs = generator(**inputs)
    # print(type(outputs))

    # logits = discriminator(input_ids).logits
    # print('\ndiscriminator logits for sample sentence: \n', logits)
    # print('\ntype and shape of logits: \n')
    # print(type(logits))
    # print(logits.shape)

    # print('\ndiscriminator loss for sample sentence: \n')
    # outputs = discriminator(**inputs, labels=labels)
    # loss = outputs.loss
    # print('\nloss: ', loss)
    # print(type(loss))
    # print(loss.shape)

    # sentence = ["나는 내일 밥을 먹었다.", "너는 내일 밥을 먹어라."]
    # print('\ntesting for:')
    # print(sentence)
    # tokens = tokenizer.tokenize(sentence[0])
    # print('\ntokenized with tokenizer.tokenize for first sentence: \n')
    # print(tokens)
    # inputs = tokenizer.encode(sentence[0], return_tensors="pt")
    # print('\ntokenized with tokenizer.encode with pt option for first sentence: \n')
    # print(inputs)

    # inputss = tokenizer(
    #     sentence,
    #     return_tensors='pt',
    #     max_length=seq_length, 
    #     truncation=True, 
    #     padding='max_length'
    #     )
    
    # print('\ntokenized with tokenizer with pt option for 2 sens: \n')
    # print(inputss)
    # print(type(inputss))
    # print(inputss.input_ids.dtype)
    # print(dir(inputss))

    # print('\ndiscriminator output for the above 2 sens inputss: \n')
    # print(dir(discriminator(inputs))) 
    # print(discriminator(inputs).hidden_states)  # None
    # print(discriminator(inputs).items) # built-in method items of ElectraForPreTrainingOutput object
    # print(discriminator(inputs).keys) # built-in method keys of ElectraForPreTrainingOutput object
    # print(discriminator(inputs).logits) # rank 2 torch tensor object
    # print(discriminator(inputs).loss) # None
    # print(discriminator(inputs).values) # built-in method of ElectraForPreTrainingOutput object

    # print('\ntokenized with tokenizer only by two args of 2 sens: \n')
    # inputs = tokenizer(sentence[0], sentence[1])
    # print(inputs) # {'input_ids': [2, 2236, 4034, 8258, 2739, 4292, 2654, 4480, 4176, 18, 3, 2267, 4034, 8258, 2739, 4292, 2654, 4025, 4118, 18, 3], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    print("\ndata loading...")
    with open(dataFileName[0], 'rb') as f:
        listOfTuple = pickle.load(f)
    df1 = pd.DataFrame.from_records(listOfTuple, columns = ['cname', 'gistno', 'idx', 'unit_str'])
    # df1 = pd.DataFrame.from_records(listOfTuple, columns = ['case_full_no', 'numberingInItem', 'phrase'])
    print(df1.info())
    print()
    if len(dataFileName) > 1:
        with open(dataFileName[1], 'rb') as f:
            listOfTuple = pickle.load(f)
        df2 = pd.DataFrame.from_records(listOfTuple, columns = ['cname', 'gistno', 'idx', 'unit_str'])
        # df2 = pd.DataFrame.from_records(listOfTuple, columns = ['case_full_no', 'numberingInItem', 'phrase'])
        print(df2.info())
        print()
        df = pd.concat([df1, df2])
    else:
       df = df1
    print('\nconcatenated dataframe information: \n')
    print(df.info())
    print()
    df = df.sample(frac=samplingRate).reset_index(drop=True)  # shuffling하고 index reset
    print(f'\nsampled dataframe information with rate {samplingRate}: \n') 
    print(df.info())
    print()
    
    print("\ndata listing...\n")    
    dfGroupedByCaseFullNo = df.groupby('cname')
    bag = []

    for _, item in tqdm(dfGroupedByCaseFullNo):

        for i in item.itertuples():
            sentence = eval("i."+ x_dataFieldName).replace('판결요지', ' ').strip()
            # print()
            # print(i.cname)
            # print(i.gistno)
            # print(sentence)
            bag.append(sentence)
            # input('...')

    print("\ntokenizing")
    inputs= defaultdict(list)
    for i in tqdm(bag):
        input_ = tokenizer(
                        i, 
                        return_tensors='pt',
                        max_length=seq_length, 
                        truncation=True, 
                        padding='max_length'
                        )

        for key, value in input_.items(): # input_ids / token_ids / attention_mask 3회 반복

            inputs[key].append(value)
    
    print('\nlabel preparation and torch type check: \n')
    for key in list(inputs.keys()):
        
        for i, t in enumerate(tqdm(inputs[key])):
            inputs[key][i] = torch.tensor(t).type(torch.int64)
        
            if key == 'input_ids':
                inputs['labels'].append(inputs['input_ids'][i].detach().clone()) # MLM label
    
    print("\nmasking for inputs: \n")
    # print('\ninputs type: \n')
    # print(type(inputs))
    for idx, x in enumerate(tqdm(inputs['input_ids'], total = len(inputs['input_ids']))):
        for jdx, y in enumerate(x[0]): # x는 (1, 512) shape tensor with int64
            # print(type(y)) # x[0] (512) shape tensor with int64
            # print(y) # y는 x[0][idx] 원소(int64)
            # print(y.shape) # torch.Size([])
            if y != padToken and y != clsToken and y != sepToken and y != unkToken:
                if torch.rand(1) < c.masking_rate:
                    # print(f'\nitem at {idx}, {jdx}: ')
                    # print(y)
                    # print(y.type)
                    inputs['input_ids'][idx][0][jdx] = maskToken # 4
                    # print(y)
        # print(inputs['input_ids'][idx][0])
        # input('...')
    
    inputs = dict(inputs)
    print('\ntype of value in inputs dict: \n')
    print(type(inputs['input_ids']))
    print(inputs['input_ids'][0])
    print(inputs['labels'][0])
    # for i in inputs['input_ids']:
    #     print(i)
    #     print(type(i)) # list
    #     print(type(i[0])) # torch tensor
    #     input('...')
    
    # dataset sampling
    print("train / test dataset sampling...")
    # print(type(inputs))
    # input('...')
    df_ = pd.DataFrame(inputs)
    dfTrain, dfTest = \
        train_test_split(
        df_,
        test_size= test_size, 
        random_state= 42, 
        shuffle=True, # case full no 끼리 모이는 상태는 사라짐
        # stratify=df['label']
        )
    
    print("\nInfo of DataFrame for Training: \n")
    print(dfTrain.info())
    print("\nInfo of DataFrame for Training: \n")
    print(dfTest.info())
    inputsTrain = dfTrain.to_dict('list')
    inputsTest = dfTest.to_dict('list')

    print('\ntrain dataset(dict) key check, type check and others: \n')
    print(inputsTrain.keys())
    print(type(inputsTrain['input_ids'][0]))
    print(inputsTrain['input_ids'][0].shape) # dataloader 출력값
    print(inputsTrain['input_ids'][0][0].shape)
    print(inputsTrain['input_ids'][0][0].dtype)
    print(inputsTrain['input_ids'][0][0][0].dtype) 
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    # <class 'torch.Tensor'>
    # torch.Size([1, 512])
    # torch.Size([512])
    # torch.int64
    # torch.int64
    # print()    

    train_loader = dataloader_factory(inputsTrain, device, batch_size, c.num_workers)
    test_loader = dataloader_factory(inputsTest, device, batch_size, c.num_workers)
    
    print("\ntraining...\n")
    model = ELECTRAModel(generator, discriminator, tokenizer)
    model.to(device)    
    
    electra_loss_func = ELECTRALoss(loss_weights=c.loss_weights, gen_label_smooth=c.gen_smooth_label, disc_label_smooth=c.disc_smooth_label)
    
    # optimizer 설정
    print('\n named parameters in model: \n')
    print([(n, p.shape) for n, p in model.named_parameters()])
    print('\n parameters in model: \n')
    print([ x for x in model.parameters()])
    '''
    [('generator.electra.embeddings.word_embeddings.weight', torch.Size([35000, 768])), ('generator.electra.embeddings.position_embeddings.weight', torch.Size([512, 768])), ('generator.electra.embeddings.token_type_embeddings.weight', torch.Size([2, 768])), ('generator.electra.embeddings.LayerNorm.weight', torch.Size([768])), ('generator.electra.embeddings.LayerNorm.bias', torch.Size([768])), ('generator.electra.embeddings_project.weight', torch.Size([256, 768])), ('generator.electra.embeddings_project.bias', torch.Size([256])), ('generator.electra.encoder.layer.0.attention.self.query.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.0.attention.self.query.bias', torch.Size([256])), ('generator.electra.encoder.layer.0.attention.self.key.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.0.attention.self.key.bias', torch.Size([256])), ('generator.electra.encoder.layer.0.attention.self.value.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.0.attention.self.value.bias', torch.Size([256])), ('generator.electra.encoder.layer.0.attention.output.dense.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.0.attention.output.dense.bias', torch.Size([256])), ('generator.electra.encoder.layer.0.attention.output.LayerNorm.weight', torch.Size([256])), ('generator.electra.encoder.layer.0.attention.output.LayerNorm.bias', torch.Size([256])), ('generator.electra.encoder.layer.0.intermediate.dense.weight', torch.Size([1024, 256])), ('generator.electra.encoder.layer.0.intermediate.dense.bias', torch.Size([1024])), ('generator.electra.encoder.layer.0.output.dense.weight', torch.Size([256, 1024])), ('generator.electra.encoder.layer.0.output.dense.bias', torch.Size([256])), ('generator.electra.encoder.layer.0.output.LayerNorm.weight', torch.Size([256])), ('generator.electra.encoder.layer.0.output.LayerNorm.bias', torch.Size([256])), ('generator.electra.encoder.layer.1.attention.self.query.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.1.attention.self.query.bias', torch.Size([256])), ('generator.electra.encoder.layer.1.attention.self.key.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.1.attention.self.key.bias', torch.Size([256])), ('generator.electra.encoder.layer.1.attention.self.value.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.1.attention.self.value.bias', torch.Size([256])), ('generator.electra.encoder.layer.1.attention.output.dense.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.1.attention.output.dense.bias', torch.Size([256])), ('generator.electra.encoder.layer.1.attention.output.LayerNorm.weight', torch.Size([256])), ('generator.electra.encoder.layer.1.attention.output.LayerNorm.bias', torch.Size([256])), ('generator.electra.encoder.layer.1.intermediate.dense.weight', torch.Size([1024, 256])), ('generator.electra.encoder.layer.1.intermediate.dense.bias', torch.Size([1024])), ('generator.electra.encoder.layer.1.output.dense.weight', torch.Size([256, 1024])), ('generator.electra.encoder.layer.1.output.dense.bias', torch.Size([256])), ('generator.electra.encoder.layer.1.output.LayerNorm.weight', torch.Size([256])), ('generator.electra.encoder.layer.1.output.LayerNorm.bias', torch.Size([256])), ('generator.electra.encoder.layer.2.attention.self.query.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.2.attention.self.query.bias', torch.Size([256])), ('generator.electra.encoder.layer.2.attention.self.key.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.2.attention.self.key.bias', torch.Size([256])), ('generator.electra.encoder.layer.2.attention.self.value.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.2.attention.self.value.bias', torch.Size([256])), ('generator.electra.encoder.layer.2.attention.output.dense.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.2.attention.output.dense.bias', torch.Size([256])), ('generator.electra.encoder.layer.2.attention.output.LayerNorm.weight', torch.Size([256])), ('generator.electra.encoder.layer.2.attention.output.LayerNorm.bias', torch.Size([256])), ('generator.electra.encoder.layer.2.intermediate.dense.weight', torch.Size([1024, 256])), ('generator.electra.encoder.layer.2.intermediate.dense.bias', torch.Size([1024])), ('generator.electra.encoder.layer.2.output.dense.weight', torch.Size([256, 1024])), ('generator.electra.encoder.layer.2.output.dense.bias', torch.Size([256])), ('generator.electra.encoder.layer.2.output.LayerNorm.weight', torch.Size([256])), ('generator.electra.encoder.layer.2.output.LayerNorm.bias', torch.Size([256])), ('generator.electra.encoder.layer.3.attention.self.query.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.3.attention.self.query.bias', torch.Size([256])), ('generator.electra.encoder.layer.3.attention.self.key.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.3.attention.self.key.bias', torch.Size([256])), ('generator.electra.encoder.layer.3.attention.self.value.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.3.attention.self.value.bias', torch.Size([256])), ('generator.electra.encoder.layer.3.attention.output.dense.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.3.attention.output.dense.bias', torch.Size([256])), ('generator.electra.encoder.layer.3.attention.output.LayerNorm.weight', torch.Size([256])), ('generator.electra.encoder.layer.3.attention.output.LayerNorm.bias', torch.Size([256])), ('generator.electra.encoder.layer.3.intermediate.dense.weight', torch.Size([1024, 256])), ('generator.electra.encoder.layer.3.intermediate.dense.bias', torch.Size([1024])), ('generator.electra.encoder.layer.3.output.dense.weight', torch.Size([256, 1024])), ('generator.electra.encoder.layer.3.output.dense.bias', torch.Size([256])), ('generator.electra.encoder.layer.3.output.LayerNorm.weight', torch.Size([256])), ('generator.electra.encoder.layer.3.output.LayerNorm.bias', torch.Size([256])), ('generator.electra.encoder.layer.4.attention.self.query.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.4.attention.self.query.bias', torch.Size([256])), ('generator.electra.encoder.layer.4.attention.self.key.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.4.attention.self.key.bias', torch.Size([256])), ('generator.electra.encoder.layer.4.attention.self.value.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.4.attention.self.value.bias', torch.Size([256])), ('generator.electra.encoder.layer.4.attention.output.dense.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.4.attention.output.dense.bias', torch.Size([256])), ('generator.electra.encoder.layer.4.attention.output.LayerNorm.weight', torch.Size([256])), ('generator.electra.encoder.layer.4.attention.output.LayerNorm.bias', torch.Size([256])), ('generator.electra.encoder.layer.4.intermediate.dense.weight', torch.Size([1024, 256])), ('generator.electra.encoder.layer.4.intermediate.dense.bias', torch.Size([1024])), ('generator.electra.encoder.layer.4.output.dense.weight', torch.Size([256, 1024])), ('generator.electra.encoder.layer.4.output.dense.bias', torch.Size([256])), ('generator.electra.encoder.layer.4.output.LayerNorm.weight', torch.Size([256])), ('generator.electra.encoder.layer.4.output.LayerNorm.bias', torch.Size([256])), ('generator.electra.encoder.layer.5.attention.self.query.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.5.attention.self.query.bias', torch.Size([256])), ('generator.electra.encoder.layer.5.attention.self.key.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.5.attention.self.key.bias', torch.Size([256])), ('generator.electra.encoder.layer.5.attention.self.value.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.5.attention.self.value.bias', torch.Size([256])), ('generator.electra.encoder.layer.5.attention.output.dense.weight', torch.Size([256, 256])), ('generator.electra.encoder.layer.5.attention.output.dense.bias', torch.Size([256])), ('generator.electra.encoder.layer.5.attention.output.LayerNorm.weight', torch.Size([256])), ('generator.electra.encoder.layer.5.attention.output.LayerNorm.bias', torch.Size([256])), ('generator.electra.encoder.layer.5.intermediate.dense.weight', torch.Size([1024, 256])), ('generator.electra.encoder.layer.5.intermediate.dense.bias', torch.Size([1024])), ('generator.electra.encoder.layer.5.output.dense.weight', torch.Size([256, 1024])), ('generator.electra.encoder.layer.5.output.dense.bias', torch.Size([256])), ('generator.electra.encoder.layer.5.output.LayerNorm.weight', torch.Size([256])), ('generator.electra.encoder.layer.5.output.LayerNorm.bias', torch.Size([256])), ('generator.electra.encoder.layer.6.attention.self.query.weight', torch.Size([256
    '''
    no_decay = ['bias', 'LayerNorm.weight'] # weight decay란 weight가 지나치게 커지는 것을 막는 것을 의미
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': c.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=c.learning_rate)

    # scheduler for learning rate tuning 설정
    t_total = len(train_loader) * epochs
    t_step = t_total / batch_size
    c.warmup_steps = int(t_step * c.warmup_ratio)
    scheduler = \
        get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=c.warmup_steps, 
            num_training_steps=int(t_step))
    
    # for epoch in tqdm(range(epochs)):
        
    #     for data_batch in train_loader:

    #         print(data_batch) # tensor.int64 rank 2 in dic
    #         print(type(data_batch)) # dict
            
    #         input('...')

    for epoch in range(epochs):
                        
        train_acc = 0.0
        test_acc = 0.
        gc.collect()
        
        #TRAINING
        model.train()
        torch.cuda.empty_cache()
        print()
        print()
        print("######################## NEW_EPOCH ########################")
        # setup loop with TQDM and dataloader
        loop = tqdm(train_loader, leave=True)
        mean_loss = 0
        mean_score = 0
        for batch_id, batch in enumerate(loop):
            targ_ids = batch['labels'].squeeze(1)
            
            # create random array of floats with equal dimensions to input_ids tensor
            # print(batch['input_ids'].shape) # torch.Size([4, 512])
            # rand = torch.rand(batch['input_ids'].shape)
            # create mask array
            # print(batch['input_ids'].unique(return_counts=True))
            # print(batch['input_ids'].unique(return_counts=True)[1][0])
            # print( (4 * 512)/batch['input_ids'].unique(return_counts=True)[1][0])
            # ratio = ( (batch_size * 512)\
            #     /((batch_size * 512) - batch['input_ids'].unique(return_counts=True)[1][0]))
            # print(ratio)
            # # input('...')
            # mask_arr = \
            #     (rand < (c.masking_rate)) * \
            #     (batch['input_ids'] != clsToken) * \
            #     (batch['input_ids'] != sepToken) * \
            #     (batch['input_ids'] != padToken)            
            # selection = []
            # for i in range(batch['input_ids'].shape[0]):
            #     selection.append(
            #         torch.flatten(mask_arr[i].nonzero()).tolist()
            #     )           
            # # masking 
            # for i in range(batch['input_ids'].shape[0]):
            #     batch['input_ids'][i, selection[i]] = maskToken
            
            optimizer.zero_grad()
            
            # process            
            outputs = model(batch) 
            # outpus = ( mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied )

            # print(dir(outputs))            
            # print(type(outputs)) # tuple
            # for o in outputs:
            #     print(type(o))
            #     print(o.shape) 
                
            # mlm_gen_logits # torch.Size([21, 35000])
            # generated # torch.Size([4, 512])            
            # disc_logits # torch.Size([4, 512])            
            # is_replaced # torch.Size([4, 512]) # gen이 못 맞춘 부분 True            
            # attention_mask # torch.Size([4, 512])            
            # is_mlm_applied # torch.Size([4, 512]) # masked 부분 True          
            # input('...')
            
            # extract loss
            # print('\nnow extracting loss...\n')
            # print(targ_ids[outputs[5]])
            # print(targ_ids[outputs[5]].shape)
            # print(outputs[0].shape)
            # print(targ_ids.shape)
            # for n, i in enumerate(targ_ids):
            #     print()
            #     print(tokenizer.decode(batch['labels'].squeeze(1)[n]))
            #     print(tokenizer.decode(outputs[1][n])) # generated ids
            # input('...')
            loss, gen_loss, disc_loss = electra_loss_func(outputs, targ_ids)
            loss.backward()

            # self.disc_loss_fc = nn.BCEWithLogitsLoss()
            # def __call__(self, pred, targ_ids):
            #     mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied = pred
            #     gen_loss = self.gen_loss_fc(mlm_gen_logits.float(), targ_ids[is_mlm_applied])
            
                # non_pad = (non_pad == 1)
                # # print(type(non_pad)) # torch Tensor, not tuple
                # # print('...')

                # disc_logits = disc_logits.masked_select(non_pad) # -> 1d tensor
                # is_replaced = is_replaced.masked_select(non_pad) # -> 1d tensor
                # if self.disc_label_smooth:
                #     is_replaced = is_replaced.float().masked_fill(~is_replaced, self.disc_label_smooth)
                # disc_loss = self.disc_loss_fc(disc_logits.float(), is_replaced.float())
                # return gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1]

            # loss = outputs.loss

            # update parameters
            optimizer.step()  

            # Update learning rate schedule
            scheduler.step()      
                                 
            # fl score for mini-batch               
            _, _, disc_logits, is_replaced, _, is_mlm_applied = outputs
            # mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied 

            # print('\n', non_pad, '\n', is_mlm_applied)
            # print('\n', non_pad.shape, '\n', is_mlm_applied.shape)
            # print('\n', non_pad.dtype, '\n', is_mlm_applied.dtype)            

            # print('\n', disc_logits, '\n', is_replaced)
            # print('\n', disc_logits.shape, '\n', is_replaced.shape)
            # print('\n', disc_logits.dtype, '\n', is_replaced.dtype)

            # print('\n', disc_logits, '\n', is_replaced)
            # print('\n', disc_logits.shape, '\n', is_replaced.shape)
            # print('\n', disc_logits.dtype, '\n', is_replaced.dtype)
            # print('\n', disc_logits[is_mlm_applied], '\n', is_replaced[is_mlm_applied])
            # print('\n', disc_logits[is_mlm_applied].shape, '\n', is_replaced[is_mlm_applied].shape)
            # print('\n', disc_logits[is_mlm_applied].dtype, '\n', is_replaced[is_mlm_applied].dtype)
            train_acc = calc_accuracy_for_minibatch(disc_logits, is_replaced, is_mlm_applied)
                      
            # print relevant info to progress bar  
            acc_train_history.append(train_acc)
            loss_train_history.append((loss.data.cpu().detach().numpy(), gen_loss.cpu().detach().numpy(), disc_loss.cpu().detach().numpy()))
            mean_score = sum([sum(x) for x in acc_train_history])/len(acc_train_history)
            mean_disc_loss = sum([x[2] for x in loss_train_history])/len(loss_train_history)

            print()
            loop.set_description(f'Epoch {epoch} Batch {batch_id}')
            # loss.item() -> loss 스칼라 값 반환 python float class / loss.data 는 tensor class
            loop.set_postfix(trn_loss=loss.item(), m_trn_loss = mean_disc_loss, m_trn_acc = mean_score) 

            if batch_id % c.log_interval == 0 and batch_id > 10:
                
                print()
                print("epoch {} batch id {} loss {} gen_loss {} disc_loss {} train acc {}".format(
                   epoch, 
                   batch_id, 
                   loss.data.cpu().detach().numpy(), 
                   gen_loss.cpu().detach().numpy(), 
                   disc_loss.cpu().detach().numpy(), 
                   train_acc
                   ))
                print()

        with open(f"model/acc_train_epch_{epoch}_btch_{batch_id}_"+str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))+".history", 'wb') as f:
            pickle.dump(acc_train_history, f)
        with open(f"model/loss_train_epch_{epoch}_btch_{batch_id}_"+str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))+".history", 'wb') as f:
            pickle.dump(loss_train_history, f)

        torch.save(model.state_dict(), f"model/koelectra_model_epch_{epoch}_btch_{batch_id}_"+str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))+".pth")    
        
        # EVALUATING
        model.eval()
        with torch.no_grad():  
            
            mean_loss = 0
            mean_score = 0  
             
            loop = tqdm(test_loader, leave=True)
            for batch_id, batch in enumerate(loop):

                input_ids, attention_mask, token_type_ids, label = batch
                targ_ids = batch['labels'].squeeze(1)
            
                outputs = model(batch)

                # fl score for mini-batch
                _, _, disc_logits, is_replaced, _, is_mlm_applied = outputs
                test_acc = calc_accuracy_for_minibatch(disc_logits, is_replaced, is_mlm_applied)                              

                # if batch_id % c.log_interval == 0 and batch_id > 10: 
                loss, gen_loss, disc_loss = electra_loss_func(outputs, targ_ids)
                acc_test_history.append(test_acc)
                loss_test_history.append((loss.data.cpu().detach().numpy(), gen_loss.data.cpu().detach().numpy(), disc_loss.data.cpu().detach().numpy()))        
                mean_score = sum([sum(x) for x in acc_test_history])/len(acc_test_history)
                mean_disc_loss = sum([x[2] for x in loss_test_history])/len(loss_test_history)

                # print relevant info to progress bar
                print()
                loop.set_description(f'Epoch {epoch} Batch {batch_id}')
                loop.set_postfix(tst_loss=loss.item(), m_tst_loss=mean_disc_loss, m_tst_acc=mean_score)        

                if batch_id % c.log_interval == 0 and batch_id > 10:
                    
                    print()
                    print("epoch {} batch id {} loss {} gen_loss {} disc_loss {} test acc {}".format(
                    epoch, 
                    batch_id, 
                    loss.data.cpu().detach().numpy(), 
                    gen_loss.cpu().detach().numpy(), 
                    disc_loss.cpu().detach().numpy(), 
                    test_acc
                    ))
                    print()

        with open(f"model/acc_test_epch_{epoch}_btch_{batch_id}_"+str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))+".history", 'wb') as f:
            pickle.dump(acc_test_history, f)
        with open(f"model/loss_test_epch_{epoch}_btch_{batch_id}_"+str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))+".history", 'wb') as f:
            pickle.dump(loss_test_history, f)
          
                        
    torch.save(model.state_dict(), f"model/koelectra_model_epch_{epoch}_btch_{batch_id}_"+str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))+"_final.pth")  

      