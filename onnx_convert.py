import argparse
import torch
from collections import OrderedDict
from efficientnet_lite import efficientnet_lite_params, build_efficientnet_lite

def parse_args():
    parser = argparse.ArgumentParser(description='Export EfficientNet Lite model to ONNX')
    parser.add_argument('--model_name', type=str, default='efficientnet_lite0', help='Model name')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint (.pt) file')
    parser.add_argument('--save_path', type=str, default=None, help='Output ONNX file path (default: model_name.onnx)')
    return parser.parse_args()

def main():
    args = parse_args()

    model_name = args.model_name
    checkpoint_path = args.checkpoint_path
    save_path = args.save_path or f"{model_name}.onnx"

    num_classes = 2  # 필요에 따라 변경하거나 argparse로 받도록 추가 가능

    # 모델 초기화
    model = build_efficientnet_lite(model_name, num_classes=num_classes)

    # checkpoint 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # checkpoint에서 state_dict만 추출
    state_dict = checkpoint['state_dict']

    # "module." prefix 제거 (멀티 GPU 학습 시)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    # 입력 크기 (efficientnet_lite_params에서 가져오기)
    input_size = efficientnet_lite_params[model_name][2]

    # 더미 입력 생성
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # ONNX export
    torch.onnx.export(
        model, dummy_input, save_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11
    )

    print(f"Convert Done. ONNX model saved at: {save_path}")

if __name__ == "__main__":
    main()
