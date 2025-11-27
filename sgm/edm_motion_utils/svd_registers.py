from sgm.edm_motion_utils.modified_svd import *
from sgm.edm_motion_utils.normal_svd import *
from sgm.edm_motion_utils.featureExtract_svd import *


# motion aligment module

def register_Constrain(model, epipolar_constrains):
    for _name, _module in model.diffusion_model.named_modules():

        # prevent alignment in early blocks
        if any([_name.__contains__('input_blocks.%d.' % i) for i in range(0, 30)]):
            continue
        if any([_name.__contains__('middle_block.%d.'%i) for i in range(0, 30)]):
            continue
        if any([_name.__contains__('output_blocks.%d.' % i) for i in range(0, 11)]):
            continue
        if any([_name.__contains__('output_blocks.%d.' % i) for i in range(15, 30)]):
            continue

        if _module.__class__.__name__ == 'VideoTransformerBlock' or _module.__class__.__name__ == 'BasicTransformerBlock':
            setattr(_module, "constrains", epipolar_constrains)

def register_attnmap_overriding(model):
    for _name, _module in model.diffusion_model.named_modules():

        # prevent alignment in early blocks
        if any([_name.__contains__('input_blocks.%d'%i) for i in range(0, 30)]):
            continue
        if any([_name.__contains__('output_blocks.%d'%i) for i in range(10, 30)]):
            continue

        if _module.__class__.__name__ == 'VideoTransformerBlock' or _module.__class__.__name__ == 'BasicTransformerBlock':
            setattr(_module, "attn_map_overriding", True)

def cancle_attnmap_overriding(model):
    for _name, _module in model.diffusion_model.named_modules():

        # prevent alignment in early blocks
        if any([_name.__contains__('input_blocks.%d' % i) for i in range(0, 30)]):
            continue
        if any([_name.__contains__('output_blocks.%d' % i) for i in range(10, 30)]):
            continue

        if _module.__class__.__name__ == 'VideoTransformerBlock' or _module.__class__.__name__ == 'BasicTransformerBlock':
            if hasattr(_module, "attn_map_overriding"):
                delattr(_module, "attn_map_overriding")

def register_SVD_MAM(model):
    '''
    replace the temporal attn block with Motion Alignment Module
    args:
        model: instance of OpenAIWrapper.
    '''
    # bound_method = forward_VideoUnet.__get__(
    #     model.diffusion_model,
    #     model.diffusion_model.__class__)
    # setattr(model.diffusion_model, 'forward', bound_method)

    # ['input_blocks', 'middle_block', 'output_blocks']
    for _name, _module in model.diffusion_model.named_modules():

        # prevent alignment in early blocks
        # if any([_name.__contains__('input_blocks.%d'%i) for i in range(0, 30)]):
        #     continue
        # # if any([_name.__contains__('middle_block.%d'%i) for i in range(0, 30)]):
        # #     continue
        # if any([_name.__contains__('output_blocks.%d'%i) for i in range(10, 100)]):
        #     continue

        if _module.__class__.__name__ == 'VideoTransformerBlock':

            # VideoTransformerBlock -> ReplacementVideoTransformerBlock
            bound_method = ReplacementVideoTransformerBlock_forward.__get__(
                _module, _module.__class__)
            setattr(_module, 'forward', bound_method)
            bound_method = ReplacementVideoTransformerBlock__forward.__get__(
                _module, _module.__class__)
            setattr(_module, '_forward', bound_method)

            # MemoryEfficientCrossAttention -> extend_MemoryEfficientCrossAttention
            bound_method = extend_MemoryEfficientCrossAttention.__get__(
                _module.attn1, _module.attn1.__class__)
            setattr(_module.attn1, 'forward', bound_method)
            bound_method = extend_MemoryEfficientCrossAttention.__get__(
                _module.attn2, _module.attn2.__class__)
            setattr(_module.attn2, 'forward', bound_method)


        if _module.__class__.__name__ == 'BasicTransformerBlock':

            # BasicTransformerBlock -> ReplacementBasicTransformerBlock
            bound_method = ReplacementBasicTransformerBlock__forward.__get__(
                _module, _module.__class__)
            setattr(_module, '_forward', bound_method)

            # MemoryEfficientCrossAttention -> extend_MemoryEfficientCrossAttention
            bound_method = extend_MemoryEfficientCrossAttention.__get__(
                _module.attn1, _module.attn1.__class__)
            setattr(_module.attn1, 'forward', bound_method)
            bound_method = extend_MemoryEfficientCrossAttention.__get__(
                _module.attn2, _module.attn2.__class__)
            setattr(_module.attn2, 'forward', bound_method)

        pass


def register_SVD_FeatureExtract(model, feautreSaveMatch):
    '''
        replace the temporal attn block with Motion Alignment Module
        args:
            model: instance of OpenAIWrapper.
        '''
    for _name, _module in model.diffusion_model.named_modules():

        if _module.__class__.__name__ == 'BasicTransformerBlock':
            bound_method = FeatureExtractBasicTransformerBlock__forward.__get__(
                _module, _module.__class__)
            setattr(_module, 'feautreSaveMatch', feautreSaveMatch)
            setattr(_module, '_forward', bound_method)


def register_SVD(model):
    '''
    replace the temporal attn block with Motion Alignment Module
    args:
        model: instance of OpenAIWrapper.
    '''
    for _name, _module in model.diffusion_model.named_modules():

        if _module.__class__.__name__ == 'VideoTransformerBlock':

            bound_method = NormalVideoTransformerBlock_forward.__get__(
                _module, _module.__class__)
            setattr(_module, 'forward', bound_method)
            bound_method = NormalVideoTransformerBlock__forward.__get__(
                _module, _module.__class__)
            setattr(_module, '_forward', bound_method)


        if _module.__class__.__name__ == 'BasicTransformerBlock':

            bound_method = NormalBasicTransformerBlock__forward.__get__(
                _module, _module.__class__)
            setattr(_module, '_forward', bound_method)
