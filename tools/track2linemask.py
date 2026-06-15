import cv2
import numpy as np

def create_line_mask(input_path: str, output_path: str):
    # BGR 이미지 로드
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {input_path}")

    # 정확히 흰색(B=255,G=255,R=255)인 픽셀 찾기
    #white_mask = np.all(img == [255, 255, 255], axis=2)
    white_mask = np.all(img >= 200, axis=2)

    # (H,W) 단일 채널 생성
    gray_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    gray_mask[white_mask] = 255

    # 저장
    cv2.imwrite(output_path, gray_mask)

    return gray_mask


if __name__ == "__main__":
    create_line_mask(
        "track.png",
        "line_mask.png"
    )