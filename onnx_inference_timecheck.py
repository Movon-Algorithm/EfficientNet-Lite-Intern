import time
import numpy as np
import onnxruntime as ort
import argparse

def main():
    parser = argparse.ArgumentParser(description='ONNX 모델 추론 및 시간 측정')
    parser.add_argument('--model_path', type=str, required=True, help='ONNX 모델 파일 경로')
    parser.add_argument('--input_size', type=int, default=224, help='모델 입력 이미지 크기')
    args = parser.parse_args()

    # 랜덤 입력 생성 (float32, shape = [1, 3, input_size, input_size])
    input_numpy = np.random.rand(1, 3, args.input_size, args.input_size).astype(np.float32)

    # ONNX Runtime 세션 생성
    ort_session = ort.InferenceSession(args.model_path)

    # 추론 및 시간 측정
    start_time = time.time()
    outputs = ort_session.run(None, {'input': input_numpy})
    end_time = time.time()

    logits = outputs[0]

    print(f"Inference Time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output logits sample: {logits[0][:5]}")  # 첫 5개 값 출력 예시

if __name__ == '__main__':
    main()
