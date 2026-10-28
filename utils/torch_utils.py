import torch, torchvision
import torch.nn as nn
from collections import OrderedDict    

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
   
def check_cuda():
    print(GREEN + f"PyTorch 版本: {torch.__version__}")
    print(f"Torchvision 版本: {torchvision.__version__}")     
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count() 
        print(f"GPU 数量: {device_count}") 
 
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")    
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  计算能力: {torch.cuda.get_device_capability(i)}")  
        
        print(f"当前设备索引: {torch.cuda.current_device()}")
        print(f"当前设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}" + RESET)  
   
def fuse_conv_and_bn(conv, bn):
    
    fusedconv = (
        nn.Conv2d(     
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size, 
            stride=conv.stride,  
            padding=conv.padding,
            dilation=conv.dilation,   
            groups=conv.groups, 
            bias=True,
        )     
        .requires_grad_(False)   
        .to(conv.weight.device)
    )  

    
    w_conv = conv.weight.view(conv.out_channels, -1)  
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))     
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
  
    
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv
     
def model_fuse_test(model):
    model.eval()    
    for name, m in model.named_modules():
        if hasattr(m, 'convert_to_deploy'):     
            print(BLUE + f"Converting module: {m.__class__}" + RESET)  
            m.convert_to_deploy()
    return model
   
def get_param_by_string(model, param_str): 
    
    keys = param_str.split('.') 
 
    
    param = model
    for key in keys:  
        if key.isdigit():  
            key = int(key)  
            param = param[key]  
        else: 
            param = getattr(param, key)  
     
    return param     

def is_parallel(model):
      
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))
 
def de_parallel(model):    
    
    return model.module if is_parallel(model) else model  

class FeatureExtractor:    
    def __init__(self, is_Ultralytics): 
        self.features = OrderedDict()  
        self.hooks = [] 
        self.is_Ultralytics = is_Ultralytics
    
    def get_activation(self, name):     
        def hook(model, input, output):     
            self.features[name] = output.detach()
        return hook    
 
    def register_hooks(self, model, layer_names):   
        


    
        self.layer_names = layer_names
        model = de_parallel(model)
        for name in layer_names:     
            if self.is_Ultralytics:     
                layer = eval(name)  
                hook = layer.register_forward_hook(self.get_activation(name)) 
            else:
                layer = get_param_by_string(model, name)   
                hook = layer.register_forward_hook(self.get_activation(name))    
            self.hooks.append(hook)     
    
    def get_features_in_order(self):    
          
        return [self.features[name] for name in self.layer_names if name in self.features]    
     
    def clear_features(self):     
        self.features.clear()  
  
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()  
        self.hooks.clear()
        self.clear_features()