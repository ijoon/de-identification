# de-identification

## 프로그램 실행 방법
패키지 설치
```
pip install -r requirements.txt
```
프로그램 실행 
```
python ./reference_tool.py --file ./videos/sample1.mp4 --scale 1.0
```
--file 옵션은 동영상 파일 경로 입니다. (현재는 실시간 검출을 하지 않고 검출 결과를 미리 동영상 별로 json 파일에 저장해두었습니다.)

--scale 옵션은 영상이 FullHD 인 경우에는 절반으로 줄여야합니다. (메모리 부족 방지)

## 얼굴(True Positive) 등록하는 방법
아래 과정을 여러번 반복하세요.
얼굴 등록이 한 번에 되지 않습니다.

1. 스페이스바를 누른다. (영상 정지)
2. 등록하고 싶은 얼굴을 `왼쪽 `클릭한다.
3. S 키를 눌러서 얼굴을 저장한다.
4. T 키를 눌러서 얼굴을 등록한다. (SVM Training)
5. 다시 스페이스바를 눌러서 영상을 재생한다.


## 오인식(False Positive) 제거하는 방법

1. 스페이스바를 누른다. (영상 정지)
2. 제거하고 싶은 얼굴을 `오른쪽` 클릭한다.
3. S 키를 눌러서 얼굴을 저장한다.
4. T 키를 눌러서 얼굴을 등록한다. (SVM Training)
5. 다시 스페이스바를 눌러서 영상을 재생한다.


## 검출기 구현 함수
- 검출기는 frame_processor.py에서 FrameProcessor Class안에있는 detect_objects 함수에서 구현해주시면 됩니다.
```Python
def detect_objects(self, img):
        """
        얼굴과 번호판을 검출하여 좌표를 반환합니다.
        입력: RGB 이미지
        출력: dictionary
        출력 dictionary 객체에 0번 키에는 얼굴 검출 박스.
        출력 dictionary 객체에 1번 키에는 번호판 검출 박스.
        """
        detected = {}

        """
        여기에서 얼굴 검출을 해주세요.
        """
        face_bboxes = self.face_detector.detect(img)
        detected[0] =  face_bboxes

        """
        여기에서 번호판 검출을 해주세요.
        """
        # license_bboxes = license_detector(img)
        # detected[1] = license_bboxes
        detected[1] = []

        return detected
```

- 위 코드에서 self.face_detector는 실시간으로 동작하는 것이 아닌 검출 결과를 사전에 저장해놓고 출력만 하는 것입니다.
- 빠른 데모를 위해 검출 결과를 미리 저장해놓았습니다.
