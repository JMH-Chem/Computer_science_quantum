# 외부 코드·데이터 출처와 배포 조건

이 문서는 기술적인 출처 관리 안내다. 프로젝트 자체 소스의 배포 라이선스는 아직 정하지 않았으며, 의존성의 라이선스가 자동으로 프로젝트 전체의 라이선스가 되지는 않는다. 배포 전에 고정된 버전의 실제 LICENSE와 하위 의존성을 확인해야 한다.

## 실행·개발 의존성

| 구성요소 | 역할 | 공식 출처·대표 라이선스 |
| --- | --- | --- |
| PySCF | 분자·주기계 전자구조, 적분 | [소스와 LICENSE](https://github.com/pyscf/pyscf), Apache-2.0 |
| NumPy / SciPy | 수치 배열·선형대수 | [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy), BSD 계열 |
| ASE | 구조 입력·결정·밴드 경로 | [ASE 공식 저장소](https://gitlab.com/ase/ase), LGPL 계열; 설치 버전 확인 |
| geomeTRIC | 구조 최적화 | [공식 저장소](https://github.com/leeping/geomeTRIC), BSD-3-Clause |
| Basis Set Exchange | 기저 정의·메타데이터·참고문헌 | [공식 저장소](https://github.com/MolSSI-BSE/basis_set_exchange), BSD-3-Clause 코드; 개별 자료의 출처도 보존 |
| FastAPI / Pydantic / Uvicorn | API·검증·서버 | [FastAPI](https://github.com/fastapi/fastapi), [Pydantic](https://github.com/pydantic/pydantic), [Uvicorn](https://github.com/encode/uvicorn), 각 저장소 LICENSE 확인 |
| React / TypeScript / Vite | 브라우저 화면·빌드 | [React](https://github.com/facebook/react), [TypeScript](https://github.com/microsoft/TypeScript), [Vite](https://github.com/vitejs/vite), 각 저장소 LICENSE 확인 |
| 3Dmol.js | 분자·결정·CUBE 등가면 | [공식 저장소](https://github.com/3dmol/3Dmol.js), BSD-3-Clause와 포함 구성요소 고지 |
| Plotly.js | 과학 차트 | [공식 저장소](https://github.com/plotly/plotly.js), MIT |

PySCF의 Apache 라이선스만 확인하고 Libxc, libcint, BLAS 등 실제 바이너리에 들어 있는 구성요소의 조건을 생략하지 않는다. Python·npm 잠금 파일은 버전 재현성 자료이며 모든 저작권 고지를 대신하지 않는다.

## 조사 대상의 배포 경계

| 프로그램 | 공개 저장소에서 확인할 조건 | 현재 처리 |
| --- | --- | --- |
| Psi4 | LGPL-3.0 계열 및 문서 조건 | 조사만 수행 |
| Quantum ESPRESSO | GPL-2.0-or-later 및 하위 구성요소 | 조사만 수행 |
| GPAW | GPL-3.0-or-later | 조사만 수행 |
| CP2K | GPL 계열과 파일·의존성별 조건 | 조사만 수행 |
| NWChem | Educational Community License 2.0 | 조사만 수행 |
| xTB | LGPL-3.0-or-later 및 포함 자료 | 조사만 수행 |
| ORCA·Gaussian | 공급자의 라이선스·사용·배포 조건 | 자동 설치·재배포하지 않음 |

프로그램을 별도 프로세스로 실행한다는 사실만으로 재배포 의무나 사용 조건이 사라지지는 않는다. 사용자 설치본을 연결하는 기능을 추가할 때도 해당 엔진과 입력·출력 자료의 조건을 확인한다.

## 과학 자료의 출처

기저·ECP·가상퍼텐셜은 이름과 실제 수치 정의, 원문 참고문헌, 제공 엔진·데이터베이스 버전을 함께 기록한다. 계산 결과를 논문에 사용하면 사용한 프로그램뿐 아니라 방법론, 기저, 함수형, 가상퍼텐셜, 보정에 필요한 인용도 확인한다.

예제 구조는 구조·단위·생성 방법을 명시한다. 다이아몬드 예제의 격자상수와 기본 셀은 [PySCF 공식 밴드 예제](https://github.com/pyscf/pyscf/blob/master/examples/pbc/09-band_ase.py)에 근거한다. 예제는 수치 재현·학습용 입력이며 실험 구조의 불확실성이나 모든 연구 조건을 대표하지 않는다.
