import sumolib
import os

# 검사할 네트워크 파일 이름
# 이 스크립트와 같은 폴더에 Town05.with_tls.net.xml 파일이 있어야 합니다.
NET_FILE = "Town05.with_tls.net.xml"

if not os.path.exists(NET_FILE):
    print(f"오류: {NET_FILE}을 찾을 수 없습니다. 스크립트와 같은 폴더에 파일을 위치시켜 주세요.")
else:
    net = sumolib.net.readNet(NET_FILE)
    tls_nodes = net.getTrafficLights()
    tls_ids = sorted([tl.getID() for tl in tls_nodes])

    # 1. 콤마로 구분된 문자열로 출력 (netconvert, 스크립트 등에 사용)
    tls_ids_str = ",".join(tls_ids)
    print("--- 1. 복사해서 사용할 ID 문자열 ---")
    print(tls_ids_str)

    # 2. 파이썬 리스트 형태로 출력
    print("\n--- 2. 파이썬 코드용 ID 리스트 ---")
    print(tls_ids)

    print(f"\n총 {len(tls_ids)}개의 신호등 ID를 찾았습니다.")
