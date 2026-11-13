# 기존 전자구조 프로그램 조사와 구조 선택

이 문서는 2026-11-14 첫 구현을 위한 대표 프로그램 조사입니다. 공개 소스·공식 문서로 확인한 구조를 정리하며, 모든 프로그램을 설치하거나 동일 조건 벤치마크를 마쳤다는 보고서는 아닙니다. 비공개 코드의 내부 구현은 추정하지 않습니다.

## 비교 기준

프로그램마다 지원하는 물리 문제와 표현이 다릅니다. 기능 이름만 비교하지 않고 다음 항목을 함께 기록합니다.

1. 분자 또는 주기계, 경계 조건과 전하·스핀 처리.
2. 전자 상태 표현: Gaussian 궤도, 평면파, 실공간 격자, PAW와 가상퍼텐셜.
3. 적분·격자·선형대수·SCF·응답 계산을 나누는 방식.
4. Python API, 독립 실행 입력, 중간값·체크포인트 접근 가능성.
5. 방법론과 에너지·기울기·응답·동역학의 실제 지원 조합.
6. 재현성, 수렴 검사, 병렬화, 메모리, 설치와 배포 조건.

## 대표 프로그램

| 프로그램 | 계산 구조와 강점 | 이 프로젝트의 선택 |
| --- | --- | --- |
| [PySCF](https://github.com/pyscf/pyscf) | Python 계산 흐름과 C 수치 핵심. 분자와 주기계 모듈, 적분·SCF·DFT·상관·응답 모듈을 조합한다. callback에서 실제 반복 상태를 얻을 수 있다. | 첫 엔진. 분자/주기계 어댑터는 분리하고 작업 시스템은 공유한다. |
| [Psi4](https://github.com/psi4/psi4) | C++ 계산 핵심과 Python driver. `energy`, `gradient`, `optimize`, `frequency` 같은 작업 인터페이스와 분자 상관 계산에 강점이 있다. | 향후 분자 방법 확장과 독립 수치 검증 후보. 현재 연결하지 않는다. |
| [Quantum ESPRESSO](https://github.com/QEF/q-e) | PWscf, PHonon, TD-DFPT 등 패키지와 FFT·선형대수·KS solver·가상퍼텐셜 라이브러리로 구성된다. 평면파 기반 주기계와 응답·재료 계산을 다룬다. | 평면파 재료 어댑터의 비교 기준. PySCF 주기계와 같은 기저라고 표현하지 않는다. |
| [GPAW](https://gitlab.com/gpaw/gpaw) | ASE와 연결되는 Python 중심 PAW 코드. 실공간 격자, 평면파, 원자 중심 궤도 표현을 제공한다. | PAW·평면파·실공간 확장 후보. 별도 설치·수렴 검증이 필요하다. |
| [CP2K](https://github.com/cp2k/cp2k) | Fortran 중심이며 Quickstep의 GPW/GAPW 접근, 주기계와 분자 동역학, MPI·스레드·GPU 경로를 제공한다. Gaussian 궤도와 보조 격자/평면파의 역할을 구분해야 한다. | 큰 시스템·동역학 확장 후보. 첫 버전에 함께 번들하지 않는다. |
| [NWChem](https://github.com/nwchemgit/nwchem) | 대규모 병렬 전자구조 계산을 지향하는 범용 코드. 분자 계산과 평면파 기반 계산을 포함하는 모듈형 패키지다. | HPC·다양한 방법론 확장 비교 대상. |
| [xTB](https://github.com/grimme-lab/xtb) | 파라미터화된 확장 tight-binding 기반 빠른 근사 계산. 구조 최적화와 큰 분자의 탐색에 활용된다. | 빠른 구조 준비 후보. DFT와 같은 정확도의 결과로 표시하지 않는다. |
| [ORCA](https://www.faccts.de/docs/orca/6.1/manual/) | 공개 매뉴얼에서 분자 전자구조·상관·분광 계산의 입력과 출력, 워크플로를 분석할 수 있다. | 공개 문서 비교 대상. 자동 다운로드·재배포하지 않는다. 사용자 설치본 연결은 후속 설계다. |
| [Gaussian](https://gaussian.com/man/) | 공개 사용자 자료로 키워드, 계산 경로, 입력·출력과 작업 흐름을 비교할 수 있다. | 공개 문서 범위만 분석한다. 소스와 배포 권한을 가정하지 않는다. |

공식 웹 문서의 일부 도메인은 개발 환경의 네트워크 정책으로 접근이 제한될 수 있다. 접근 가능한 공식 GitHub/GitLab 문서·소스를 우선 사용했고, 직접 검토하지 못한 부분을 실행 검증으로 취급하지 않는다.

## 공통 계산 흐름과 수식

### 구조와 표현

구조에서 원자 번호·좌표·전하·스핀을 읽고 원자 궤도 기저, 적분 또는 수치 격자를 구성한다. 분자 Gaussian 기저에서는 보통

\[
\psi_i(\mathbf r)=\sum_\mu C_{\mu i}\chi_\mu(\mathbf r)
\]

로 궤도를 나타낸다. 겹침 행렬 \(S_{\mu\nu}\), 한 전자 적분, 두 전자 적분이나 density fitting 자료가 계산의 기반이 된다. 주기계에서는 격자와 Bloch 위상, k점 가중치가 추가된다.

PySCF PBC의 FFTDF에서 사용하는 평면파는 전자밀도 fitting의 보조 표현이다. 주 궤도 기저는 Gaussian이므로 순수 평면파 궤도 코드와 cutoff를 직접 대응시키지 않는다.

### SCF

초기 밀도에서 Fock 또는 Kohn–Sham 행렬을 만들고 일반화 고유값 문제를 반복해서 푼다.

\[
FC=SC\epsilon.
\]

닫힌껍질 RHF의 spin-summed 밀도는 \(P_{\mu\nu}=2\sum_i^{occ}C_{\mu i}C_{\nu i}\)이고, 에너지는

\[
E=\frac12\sum_{\mu\nu}P_{\mu\nu}(H_{\mu\nu}+F_{\mu\nu})+E_{nuc}
\]

로 얻는다. DFT에서는 교환·상관 함수형과 수치 적분이 더해지며 같은 RHF 에너지 식으로 대체할 수 없다.

DIIS·감쇠·level shift는 수렴을 돕는 알고리즘이다. SCF 에너지가 항상 감소한다는 보장은 없고, 에너지 변화만 작아도 밀도와 궤도 잔차가 수렴하지 않을 수 있다. 따라서 에너지·ΔE·잔차·밀도 변화를 함께 기록한다.

### 구조 최적화

핵 좌표를 갱신할 때마다 전자구조를 다시 계산해 에너지와 기울기를 얻는다. 바깥 루프는 최적화 단계, 안쪽 루프는 SCF다. 둘 다 실제 물리 시간에 해당하지 않는다. 최적화 종료와 최종 SCF 수렴을 각각 확인한다.

### 여기 상태와 응답

주파수 영역 선형응답 TDDFT는 바닥상태 주변의 응답에서 여기 에너지와 전이 특성을 얻는다. TDA는 응답 방정식의 일부 결합을 생략하는 근사다. 얻은 에너지와 oscillator strength로 흡수 stick spectrum을 만들 수 있지만, 이것만으로 형광 수명·양자수율·내부전환 궤적을 얻지는 못한다.

NTO는 전이 성격을 간결한 hole/particle 궤도 쌍으로 표현한다. 구현에 따라 정의가 다를 수 있다. PySCF의 해당 NTO 도구가 사용하는 X 성분 기반 관례를 결과에 남기고, 이를 완전한 실시간 전자 이동으로 표현하지 않는다.

### 저장과 검증

입력, 계산에 사용한 기저 정의, 엔진·라이브러리 버전, 수렴 조건, 최종 상태와 수치 결과를 함께 저장한다. 체크포인트는 표시 재계산의 편의를 위한 자료이며, 버전 독립적인 영구 교환 형식으로 보장하지 않는다.

## 이 프로젝트에서 선택한 구조

Python API와 반복 callback이 있는 PySCF를 첫 엔진으로 사용하면 로그 문자열 해석에 의존하지 않고 실제 계산값을 화면으로 전달할 수 있다. 분자와 주기계를 별도 어댑터로 두면 향후 다른 엔진을 연결해도 공통 작업 기록과 화면을 재사용할 수 있다.

첫 교육용 RHF는 **적분은 PySCF, SCF 선형대수는 자체 NumPy 구현**으로 나눈다. 적분까지 전부 직접 만든 계산 엔진이라는 표현은 쓰지 않는다. 교차 검증은 같은 적분과 조건에서 에너지·밀도·정규직교성을 비교한다.

독립 프로그램 비교에서는 구조·전하·스핀·기저 정의·ECP·함수형 구현·수치 격자·수렴 조건을 가능한 범위에서 일치시킨다. PySCF와 같은 PySCF 직접 호출의 일치는 어댑터 검증이며, 독립 엔진에 의한 물리적 검증과는 다르다.

## 추가 공식 근거

- [PySCF SCF callback 구현](https://github.com/pyscf/pyscf/blob/master/pyscf/scf/hf.py)
- [PySCF TDHF/TDDFT 구현과 NTO](https://github.com/pyscf/pyscf/blob/master/pyscf/tdscf/rhf.py)
- [PySCF 주기계 밴드 예제](https://github.com/pyscf/pyscf/blob/master/examples/pbc/09-band_ase.py)
- [Basis Set Exchange API와 데이터](https://github.com/MolSSI-BSE/basis_set_exchange)
- [geomeTRIC 구조 최적화](https://github.com/leeping/geomeTRIC)
- [3Dmol.js 구조·단위격자·등가면 API](https://github.com/3dmol/3Dmol.js)

구체적인 실행 결과와 시간·메모리 수치는 [검증 기록](validation.md)에 기록한다. 실행하지 않은 비교는 이 문서의 기능 조사만으로 통과 처리하지 않는다.
