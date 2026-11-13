# 선택적 독립 엔진 검증

이 폴더는 Quantum Lab의 PySCF 결과를 **별도로 설치한 Psi4**와 비교하기 위한 환경
기록이다. Psi4는 앱의 필수 의존성이나 화면에서 선택하는 계산 엔진이 아니며, 앱의
wheel에 포함하지 않는다.

## 설치와 실행

앱 의존성을 `uv sync --frozen`으로 설치한 Linux x86-64 환경에서 실행한다.
Conda 또는 micromamba가 별도로 필요하다. 잠금 파일은 실제 검증에 사용한
Psi4 1.11·Python 3.12 환경의 패키지 URL·버전·빌드·체크섬을 고정한다.
추가로 `psi4-package-manifest.json`에 설치한 패키지의 SHA-256과 라이선스 메타데이터를
기록했다. 전체 배포 조건 검토를 대신하는 자료는 아니다.

```bash
micromamba create --yes --prefix "$PWD/.cache/psi4-validation" \
  --file validation/psi4-linux-64.lock
uv run python scripts/cross_engine.py \
  --psi4-python "$PWD/.cache/psi4-validation/bin/python"
```

이미 Psi4가 설치된 별도 환경이 있으면 해당 환경의 Python 경로를 지정할 수 있다.
그 경우 엔진 버전은 결과에 기록되지만 이 잠금 환경과 같다고 가정하지 않는다.
Windows WSL2에서도 Linux용 설치 절차를 사용할 수 있으나 **실제 WSL2 실행 검증은
아직 하지 않았다.** 네이티브 Windows·ARM·macOS용 잠금 파일은 아니다.

## 무엇을 비교하는가

| 입력 | 조건 | 사전에 정한 전체 에너지 허용오차 |
| --- | --- | --- |
| H₂ RHF | STO-3G | 1e-8 Eh |
| H₂O·포름알데히드 RHF | def2-SVP | 1e-8 Eh |
| H₂O MP2 | def2-SVP, 동결 점유 궤도 0개 / 1개 | 1e-7 Eh |
| H₂O CCSD | STO-3G, 동결 0개 | 1e-7 Eh |
| H₂O PBE | def2-SVP, 각 엔진의 두 적분 격자 | 5e-6 Eh |

MP2·CCSD는 RHF 기준 에너지 1e-8 Eh, 상관 보정 1e-7 Eh도 별도로 검사한다.
모든 입력에서 BSE의 같은 기저 정의를 사용하고 spherical AO, all-electron,
conventional 적분, 전하·스핀·동결 점유 오비탈 수를 고정한다. 앱이 사용하는 Å→Bohr
변환을 명시해 Psi4에 같은 Bohr 좌표를 전달한다. 핵 반발 에너지 허용오차는 1e-10 Eh다.

PBE의 PySCF level 5/6과 Psi4 99×590 / 175×974 격자는 **서로 다른 구적법**이다.
각 엔진에서 격자를 세분했을 때 에너지 변화가 2e-6 Eh 이내인지 별도로 검사한다.
PBE 사례 하나만 선택하면 격자 세분 검사는 `not_evaluated`, 전체 결과는
`incomplete`로 남고 성공 종료하지 않는다. 이를 DFT 격자 수렴까지 통과한 것으로
해석하지 않는다.

## 결과 읽기

- 기본 위치는 `.quantum-lab/cross-engine/`이다. `runs/` 아래에 매 실행을 새로 저장하고
  `report.json`은 최신 실행을 가리킨다. 예전 기록을 덮어쓰지 않는다.
- 각 사례에 전체 입력, 공통 BSE JSON·Psi4 기저 파일·참고문헌, 입력 체크섬,
  엔진 버전·설정·결과·로그·실행 시간·최대 RSS가 남는다.
- 양쪽 수렴, 유한한 수치, 같은 입력 체크섬·기저 함수 수·전자 수·좌표·동결 정의를
  확인한 뒤만 에너지 비교를 통과시킨다. 엔진 부재·시간 초과·계산 실패는 성공으로
  건너뛰지 않으며 프로세스 종료 코드가 0이 아니다.
- `--case h2_rhf_sto3g`로 하나만 선택하거나 `--case`를 여러 번 지정할 수 있다.
  `--timeout`은 엔진별·사례별 최대 실행 초다.

이 검사는 작은 분자에서 **독립 구현 간의 수치 일치**를 확인한다. 실험값의 정확도,
다른 구조·원소·기저·방법, 주기계·들뜬 상태·진동 물성까지 보증하지 않는다.

앱 측 실행 시간에는 기저 해석·물성 분석·결과 파일 저장이 포함되며, Psi4 측은
`energy()` 호출 구간이다. 프로세스 시간은 모듈 불러오기도 포함한다. 측정 구간이
다르므로 이 기록을 엔진 속도 순위로 사용하지 않는다.
