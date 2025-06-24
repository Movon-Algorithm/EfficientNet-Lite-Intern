import torch
from collections import OrderedDict
from efficientnet_lite import build_efficientnet_lite

# 모델 초기화
model = build_efficientnet_lite('efficientnet_lite0', num_classes=2)

# checkpoint 로드
checkpoint = torch.load('./models/checkpoint-005000.pth.tar', map_location='cpu')

# "module." prefix 제거
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    new_state_dict[k.replace('module.', '')] = v
model.load_state_dict(new_state_dict)
model.eval()

# 더미 입력
dummy_input = torch.randn(1, 3, 224, 224)

# ONNX export
torch.onnx.export(model, dummy_input, "efficientnet_lite0.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                  opset_version=11)

print("Convert Done")