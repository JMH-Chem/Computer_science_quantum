# 문헌 비교 입력과 출처

이 디렉터리는 선택한 출판값을 실제 앱 계산과 비교하기 위한 입력·판정 규칙·출처
기록이다. 계산 종료, 같은 방법의 값 재현, 서로 다른 근사 방법의 성능 평가를 구분한다.
[해석 보고서](../../docs/literature-validation.md)를 먼저 읽고 숫자를 사용한다.

## 재현

저장소 루트에서 잠금 파일 기반 환경을 설치한 뒤 다음을 실행한다.

```bash
uv sync --frozen
uv run python scripts/literature_validation.py --run --case water-eomccsd
```

전체 9개 계산은 다음 명령이다. CPU 시간이 필요하며 중간 파일과 결과를
`.quantum-lab/literature-validation/`에 저장한다. `--timeout`의 기본값은 계산당
3600초이며, 시간 초과는 통과가 아니다.

```bash
uv run python scripts/literature_validation.py --run
```

이미 실행한 공개 CLI 작업을 모으려면 입력 파일의 확장자 없는 이름과 해당 작업의
`result.json` 경로를 연결한 JSON 파일을 만든다. 경로는 수집 JSON 기준 상대 경로도
가능하다. 에너지 숫자만 적은 임의 파일은 받지 않는다.

```json
{
  "water-eomccsd": "/absolute/path/to/data/jobs/JOB_ID/result.json"
}
```

```bash
uv run python scripts/literature_validation.py --collect collected-results.json --case water-eomccsd
```

실행기는 입력·DB·저장 결과·환경 기록의 일치와 수렴 상태를 확인한다. 작업별
로그·이벤트·텍스트 결과와 해시를 별도 실행 디렉터리에 복사하고 `report.json`을
만든다. 원래 작업의 버전·해시를 보존하며 과거 실패를 새 성공으로 바꾸지 않는다.
부분 수집으로 전체 사례를 통과한 것처럼 표시하지 않는다.

## 입력과 비교의 의미

| 입력 파일 | 앱 조건 | 비교 대상 | 해석 |
| --- | --- | --- | --- |
| `water-eomccsd.json` | EOM-EE-CCSD/aVTZ, 동결 점유 오비탈 1개, singlet 3상태 | CCSD/aVTZ 7.60·9.36·9.96 eV | 구조·기저·동결 조건을 맞춘 수치 재현, 허용오차 0.01 eV |
| `formaldehyde-tddft.json` | PBE0-TDDFT/aVTZ, 격자 level 4, singlet 1상태 | exFCI/aVTZ TBE 3.98 eV | 방법이 달라 차이를 보고하며 합격 오차를 두지 않음 |
| `water-dimer.json` | CCSD(T)/aVDZ, 동결 2개 | S22B −4.989 kcal/mol | 아래 조각과 함께 전자적 상호작용 계산, BSSE·기저 오차가 남음 |
| `water-fragment-a.json` | CCSD(T)/aVDZ, 동결 1개 | 이합체의 첫 물 조각 | 이합체에서 추출한 좌표 유지 |
| `water-fragment-b.json` | CCSD(T)/aVDZ, 동결 1개 | 이합체의 두 번째 물 조각 | 이합체에서 추출한 좌표 유지 |
| `diamond-k2.json` | PBE/GTH-PBE/GTH-DZVP, k2³·FFT27³ | 저자 QE/SG15 PBE 간접 갭 4.1724807238 eV | 기저·가상퍼텐셜이 달라 진단 비교 |
| `diamond-k3.json` | 위와 같고 k3³·FFT27³ | 위와 같음 | k점 변화 검사 |
| `diamond-mesh33.json` | 위와 같고 k3³·FFT33³ | 위와 같음 | FFT 격자 변화 검사 |
| `diamond-k4.json` | 위와 같고 k4³·FFT33³ | 위와 같음 | 다음 k점 세분 검사 |

aVTZ는 aug-cc-pVTZ, aVDZ는 aug-cc-pVDZ다. 입력의 Å 좌표와 전체 정밀도
설정을 사용한다. 포름알데히드 XYZ의 첫 바닥상태 프레임만 사용했으며, 뒤의 들뜬
상태 구조를 섞지 않았다. `criteria.json`의 원래 분자 비교 정책은 해당 수치 결과를
보기 전에 정했다. 다이아몬드 k4³ 추가 검사는 k2³·k3³ 결과를 본 뒤 계획했으며,
그 결과를 보기 전, k4³ 실행 전에 0.025 eV 진단 기준을 기록했다. 이 순서를 전체
검사가 처음부터 사전 등록된 것처럼 표현하지 않는다. 포름알데히드의 최초 미수렴
시도(3상태·10⁻¹⁰)와 최종 입력(1상태·10⁻⁸)도 구분한다.

## 문헌·데이터 귀속

**QUEST.** Pierre-François Loos, Anthony Scemama, Aymeric Blondel, Yann Garniron,
Michel Caffarel, Denis Jacquemin, “A Mountaineering Strategy to Excited States:
Highly Accurate Reference Energies and Benchmarks,” *JCTC* **14**, 4360–4379 (2018),
[DOI 10.1021/acs.jctc.8b00406](https://doi.org/10.1021/acs.jctc.8b00406).
구조와 선택된 수치는 [LCPQ/QUESTDB_website](https://github.com/LCPQ/QUESTDB_website/tree/9cdb57208fafef696a3a6f049fab09fdc0abb587)의
커밋 `9cdb57208fafef696a3a6f049fab09fdc0abb587`에서 추출했다. 데이터베이스는
[Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/), 개별
내용은 [Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1-0/)로
제공된다. 이 디렉터리의 해당 추출 좌표·참조 수치는 그 출처와 조건을 유지한다.
앱 자체의 라이선스가 원 데이터의 조건을 대체하지 않는다. 논문 전체 PDF를
재배포하지 않으며 저자 공개본 URL·해시와 확인 범위를 `quest-reference.json`에 기록했다.

**S22B.** Marshall 등, *J. Chem. Phys.* **135**, 194102 (2011)의 기준값과
Jurečka 등, *Phys. Chem. Chem. Phys.* **8**, 1985 (2006)의 구조를 인용한다.
실제로 읽은 출처는 [Psi4의 S22.py](https://github.com/psi4/psi4/blob/64fc54046ff5a91733b1f386417286a05621f44e/psi4/share/psi4/databases/S22.py),
커밋 `64fc54046ff5a91733b1f386417286a05621f44e`이다. Psi4 소스 헤더는 LGPL version 3,
Copyright (c) 2007–2026 The Psi4 Developers를 명시한다. 이 디렉터리에는 전체
Python 소스를 복제하지 않고 선택된 좌표·기준값·서지 기록을 보존했다. 정확한
위치·SHA-256·원 논문 전문을 읽지 않았다는 범위는 `s22-water-reference.json`에 있다.

**다이아몬드.** “Many-body perturbation theory vs. density functional theory:
a systematic benchmark for band gaps of solids,” *npj Computational Materials*
(2025), [DOI 10.1038/s41524-025-01855-4](https://doi.org/10.1038/s41524-025-01855-4).
실제로 읽은 데이터·코드는 [MaxGrossmann/mbpt_benchmark](https://github.com/MaxGrossmann/mbpt_benchmark/tree/1f1af6474ecb7b8c3d3bd8318d067022bb5b3019),
커밋 `1f1af6474ecb7b8c3d3bd8318d067022bb5b3019`이다. README는 MIT 조건과
Copyright (c) 2025 Max Großmann, Marc Thieme, and Malte Grunert를 명시한다.
여기에는 선택된 수치와 출처를 기록하며 원 프로그램·가상퍼텐셜 파일을 배포하지 않는다.
원 ICSD 구조는 라이선스로 저자 저장소에서 제외되어 있다. 앱 입력은 공개 격자
상수에서 표준 다이아몬드 셀을 구성한 것이며 해당 ICSD 파일의 사본이 아니다.
확인한 파일의 URL·SHA-256과 단위 확인은 `materials-reference.json`에 있다.

원본 URL은 고정 커밋을 가리킨다. `.cache/` 경로는 이번 검토 환경의 원본 보관
위치이고 저장소를 새로 받으면 존재하지 않을 수 있다. 각 manifest의 URL과
SHA-256으로 같은 자료를 다시 확인할 수 있다. 수치 비교에 논문 전체 파일이나
독점 ICSD 파일이 필요하지는 않다.
