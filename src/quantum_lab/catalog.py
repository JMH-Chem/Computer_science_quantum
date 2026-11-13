"""Local, source-linked capability catalog. Listed is not equivalent to validated."""
from functools import lru_cache
from numbers import Integral

from fastapi import APIRouter, HTTPException, Query

from .examples import examples

router = APIRouter()
SOURCE = "https://pyscf.org/user/dft.html"

LESSONS = [
    {"id": "scf", "title": "SCF 반복은 시간이 아닙니다", "body": "SCF는 전자밀도와 유효 퍼텐셜이 서로 일치할 때까지 푸는 수치 반복입니다. DIIS 가속으로 에너지가 잠시 오를 수 있습니다. 에너지 변화와 밀도 변화·궤도 기울기를 함께 확인하세요.", "source": "https://pyscf.org/user/scf.html"},
    {"id": "basis", "title": "기저 함수 · Basis set", "body": "전자 오비탈을 표현하는 함수의 집합입니다. 큰 기저가 표현력을 높이지만 비용과 선형종속 위험도 높입니다. STO-3G는 최소 기저 학습용입니다. 기저와 수치 격자를 따로 수렴시켜야 합니다.", "source": "https://www.basissetexchange.org/"},
    {"id": "functional", "title": "교환·상관 함수형 · XC functional", "body": "DFT에서 정확히 알 수 없는 교환·상관 에너지를 근사합니다. LDA, GGA, meta-GGA, hybrid, range-separated 등 계열이 있으며 모든 계에 가장 좋은 단일 함수형은 없습니다. 지원 목록에 있는 이름도 특정 응답·기울기 계산에서는 지원되지 않을 수 있습니다.", "source": SOURCE},
    {"id": "orbital", "title": "오비탈의 두 색상", "body": "오비탈 등가면의 두 색상은 파동함수의 양·음 위상입니다. 양전하·음전하 표시가 아닙니다. 전자밀도는 점유된 상태들의 기여를 합친 공간 분포이며 음수가 아닙니다.", "source": "https://pyscf.org/user/tools.html"},
    {"id": "excitation", "title": "여기 에너지와 S₁→S₀", "body": "선형응답 TDDFT는 고정 핵 구조에서 수직 여기 에너지와 전이 특성을 계산합니다. 여기 에너지는 HOMO–LUMO 간격과 같지 않습니다. 아래쪽 화살표 애니메이션은 개념 설명이며, 실제 시간·수명·양자수율·이완 궤적은 계산하지 않습니다.", "source": "https://pyscf.org/user/tddft.html"},
    {"id": "nto", "title": "자연 전이 오비탈 · NTO", "body": "전이를 소수의 hole–particle 오비탈 쌍으로 표현합니다. 이 앱의 TDDFT NTO는 PySCF의 X-only 관례를 사용하며 Y 성분은 포함하지 않습니다. 화면에는 선택한 상태의 가장 큰 가중치 쌍을 표시합니다.", "source": "https://github.com/pyscf/pyscf/blob/master/pyscf/tdscf/rhf.py"},
    {"id": "bands", "title": "밴드갭과 광학 갭", "body": "밴드는 k점에 따른 Kohn–Sham 궤도 에너지입니다. 표시된 갭은 계산한 k점·경로에서 찾은 KS 갭이며 전체 Brillouin zone의 엄밀한 갭이나 광학 여기 에너지로 보장되지 않습니다. Γ점만으로 얻은 결과에는 k점 수렴 검사가 필요합니다.", "source": "https://pyscf.org/user/pbc/scf.html"},
    {"id": "periodic", "title": "결정의 기저와 가상퍼텐셜", "body": "첫 버전의 주기계는 Gaussian 궤도 기저와 GTH 가상퍼텐셜을 사용합니다. FFT 격자는 보조 밀도 표현이며 평면파 궤도 기저와 같은 설정이 아닙니다. 밀도 적분은 가상퍼텐셜이 남긴 원자가 전자 수와 비교합니다.", "source": "https://pyscf.org/user/pbc/df.html"},
    {"id": "optimization", "title": "구조 최적화의 단계", "body": "힘을 이용해 원자 좌표를 바꾸고 각 구조에서 다시 SCF를 풉니다. 최적화 단계와 그 안의 SCF 반복은 별도 축입니다. 구조 최적화 과정도 실제 분자 운동의 시간 궤적은 아닙니다.", "source": "https://pyscf.org/user/geomopt.html"},
    {"id": "rhf", "title": "직접 만든 RHF 풀이기", "body": "PySCF가 공급한 원자 궤도 적분으로 S의 직교화, Fock 행렬, 점유 오비탈과 밀도 갱신을 NumPy에서 계산합니다. 자체 구현 범위와 적분 제공자를 구분합니다. 결합 해리처럼 다중참조 성격이 커지는 상황에서는 RHF 자체의 한계가 있습니다.", "source": "https://pyscf.org/user/scf.html"},
]

METHODS = [
    ("HF", "Hartree–Fock", "available", "분자 RHF/UHF. 평균장·단일 Slater determinant."),
    ("DFT", "Kohn–Sham DFT", "available", "분자 RKS/UKS, 결정 PBE. 함수형과 수치 격자에 의존."),
    ("tda", "Tamm–Dancoff approximation", "available", "닫힌껍질 singlet, PBE/PBE0/B3LYP."),
    ("tddft", "Linear-response TDDFT", "available", "수직 여기 에너지·전이 세기. 실제 시간 동역학 아님."),
    ("educational_rhf", "교육용 NumPy RHF", "available", "자체 SCF + PySCF 적분. 작은 닫힌껍질 분자."),
    ("tdhf", "TDHF / HF-TDA (CIS)", "available", "닫힌껍질 RHF singlet 응답. 수직 여기·전이 세기, 실제 시간 동역학 아님."),
    ("MP2", "MP2", "available", "RHF 기준 2차 섭동 상관. 동결 궤도 지정 가능, 준축퇴·작은 갭 주의."),
    ("CCSD", "CCSD", "available", "닫힌껍질 RHF 기준 단일·이중 들뜸 coupled cluster. 수렴 상태 별도 확인."),
    ("CCSD(T)", "CCSD(T)", "available", "수렴한 CCSD 위 비반복 삼중 들뜸 보정. 단일참조 한계, 높은 비용."),
    ("CISD", "CISD", "available", "RHF 기준 단일·이중 들뜸 CI. 크기 일관성 한계가 있음."),
    ("FCI", "Full CI", "available", "작은 유한 기저의 전체 CI. 기저 불완전성은 남으며 행렬식 수 제한 적용."),
    ("CASCI", "CASCI", "available", "고정 RHF 궤도의 활성 공간 CI. 사용자가 활성 전자·궤도 수를 선택."),
    ("CASSCF", "CASSCF", "available", "활성 공간 CI와 궤도 최적화. 활성 공간 밖의 동적 상관은 완전하지 않음."),
    ("frequencies", "Harmonic frequencies / IR / RRHO", "available", "닫힌껍질 HF·일부 DFT Hessian, 선택적 유한차분 IR. 정지점·양의 진동 모드에서 RRHO 열화학."),
    ("eomccsd", "EOM-EE-CCSD", "available", "닫힌껍질 CCSD 기준 singlet 여기 에너지. 전이 세기·NTO는 제공하지 않음."),
    ("NEVPT2", "Strongly-contracted NEVPT2", "available", "정준 RHF 궤도의 singlet CASCI 기준 위 2차 동적 상관. 활성 공간 선택과 비용 제한 적용."),
    ("GW_BSE", "GW / BSE", "planned", "준입자·전자–정공 응답. KS 갭과 다른 물리량."),
    ("semiempirical", "xTB / semiempirical", "planned", "빠른 구조 탐색을 위한 추가 엔진."),
    ("dynamics", "Nonadiabatic dynamics", "planned", "상태 간 결합·핵 운동·앙상블을 포함한 별도 설계 필요."),
]

LESSONS += [
    {"id": "correlation", "title": "상관 에너지 · Correlation energy", "body": "RHF가 놓치는 전자 간 상관을 MP2, CI, CC 등으로 근사합니다. 전체 에너지와 RHF 기준의 차이를 상관 에너지로 표시합니다. CCSD(T)와 MP2는 변분 상한이 아니며 더 낮은 수치가 항상 더 정확한 것은 아닙니다. 이 앱의 상관 계산 오비탈·밀도와 원자 전하는 명시적으로 RHF 기준입니다.", "source": "https://pyscf.org/user/cc.html"},
    {"id": "active_space", "title": "활성 공간 · Active space", "body": "CASCI는 선택한 궤도 안에서 전자 배치를 모두 다루고 CASSCF는 궤도도 최적화합니다. 현재는 에너지 순서의 연속 활성 궤도를 선택하므로 전자 수·궤도 수만으로 화학적으로 적절한 공간을 보장할 수 없습니다. 저장된 점유수와 궤도 성격을 확인해야 합니다.", "source": "https://pyscf.org/user/mcscf.html"},
    {"id": "vibrations", "title": "조화 진동 · Harmonic vibrations", "body": "핵 좌표에 대한 에너지의 두 번째 미분인 Hessian으로 정상 모드를 구합니다. 음수 표시는 허수 진동수이며 불안정 방향 또는 수치 오차를 뜻할 수 있습니다. 애니메이션은 선택한 모드의 변위를 확대해 보여 주는 모식도로 실제 시간 궤적이 아닙니다. IR 세기는 쌍극자 미분에서 얻으며 선폭은 표시 설정입니다.", "source": "https://github.com/pyscf/pyscf/blob/master/pyscf/hessian/thermo.py"},
    {"id": "thermochemistry", "title": "열화학과 기준 상태 · RRHO", "body": "강체 회전자·조화 진동자·이상 기체 근사입니다. 정지점과 양의 진동 모드를 확인한 경우에 온도·압력에 따른 ZPE, 엔탈피·엔트로피·Gibbs 에너지를 제공합니다. 용매·비조화성·내부 회전 효과는 포함하지 않습니다.", "source": "https://github.com/pyscf/pyscf/blob/master/pyscf/hessian/thermo.py"},
    {"id": "atomic_charges", "title": "원자 전하 · Population analysis", "body": "Mulliken과 Löwdin 전하는 전자밀도를 원자별로 나누는 서로 다른 정의입니다. 기저 의존성이 있고 직접 측정한 관측량이나 유일한 원자 전하가 아닙니다. 총 전하 보존과 분석에 사용한 밀도의 기준을 확인합니다.", "source": "https://pyscf.org/user/scf.html"},
    {"id": "dos", "title": "상태 밀도 · Density of states", "body": "계산한 균일 k점 격자의 Kohn–Sham 궤도 에너지에 Gaussian 폭을 주어 나타냅니다. 단위는 states/(eV cell)이고 두 스핀과 k점 가중치를 포함합니다. 거친 k점의 봉우리를 실제 재료의 날카로운 피크나 광학 스펙트럼으로 해석하지 않습니다.", "source": "https://pyscf.org/user/pbc/scf.html"},
]


@lru_cache(maxsize=1)
def functional_catalog():
    from pyscf.dft import libxc
    names = ["PBE", "PBE0", "B3LYP"] + sorted(n for n, code in libxc.XC_CODES.items()
                                                     if isinstance(code, Integral) and code > 0)
    result = []
    for name in dict.fromkeys(names):
        try:
            base_family = libxc.xc_type(name)
            hybrid = bool(libxc.is_hybrid_xc(name))
            omega, alpha, beta = libxc.rsh_coeff(name)
            ranged = abs(omega) > 1e-12
            family = ("range-separated hybrid " if ranged and hybrid else "hybrid " if hybrid else "") + base_family
            kinetic = "_K_" in name
            component = "kinetic" if kinetic else "exchange" if "_X_" in name else "correlation" if "_C_" in name else "exchange-correlation"
            needs_laplacian = bool(libxc.needs_laplacian(name))
            # Match the derivatives currently required by the molecular
            # adapter; catalog availability is not a numerical validation claim.
            derivatives = bool(libxc.test_deriv_order(name, 2, raise_error=False))
            supported = not kinetic and not needs_laplacian and derivatives
            preset = name in {"PBE", "PBE0", "B3LYP"}
            tasks = ["single_point"] if supported else []
            if preset and supported and not libxc.is_nlc(name):
                tasks += ["optimize", "tda", "tddft"]
            description = "검증 대상 프리셋 · 조합별 기록 확인" if preset else "단일점 실행 후보 · 조합별 검증 필요"
            if component in {"exchange", "correlation"}:
                description += " · XC의 단일 성분이며 완전한 교환·상관 조합이 아님"
            if kinetic:
                description = "운동에너지 함수형: 이 앱의 KS 교환·상관 함수형으로 지원하지 않음"
            elif needs_laplacian:
                description = "밀도 Laplacian 의존: 현재 어댑터 미지원"
            elif not derivatives:
                description = "현재 어댑터에 필요한 도함수 미지원"
            result.append({"name": name, "family": family, "status": "available" if supported else "unsupported",
                           "description": description, "base_family": base_family,
                           "component": component, "hybrid": hybrid, "range_separated": ranged,
                           "verification": "combination_not_asserted", "nonlocal_correlation": bool(libxc.is_nlc(name)),
                           "references": list(libxc.xc_reference(name)),
                           "reference": SOURCE, "supported_tasks": tasks})
        except Exception:
            continue
    return result


@lru_cache(maxsize=1)
def basis_metadata():
    import basis_set_exchange as bse
    return bse.get_metadata()


@router.get("/examples")
def get_examples():
    return examples()


@router.get("/catalog")
def catalog():
    return {"functionals": functional_catalog(),
            "bases": [{"name": n, "status": "available", "description": d} for n, d in [
                ("sto-3g", "학습용 최소 기저"), ("def2-svp", "분자 이중 제타"), ("def2-tzvp", "분자 삼중 제타"),
                ("gth-szv", "주기계 빠른 학습용"), ("gth-dzvp", "주기계 이중 제타")]],
            "methods": [dict(zip(("id", "name", "status", "description"), row)) for row in METHODS],
            "lessons": LESSONS,
            "status_note": "available은 실행 가능한 후보입니다. 조합별 수치 검증 기록은 docs/validation.md를 확인하세요."}


@router.get("/catalog/bases")
def search_bases(q: str = Query(default="", max_length=160), elements: str = Query(default="", max_length=500)):
    from ase.data import atomic_numbers
    symbols = {x.strip().capitalize() for x in elements.split(",") if x.strip()}
    if any(symbol not in atomic_numbers or symbol == "X" for symbol in symbols):
        raise HTTPException(422, "기저 검색에는 유효한 원소 기호를 쉼표로 구분해 입력하세요.")
    requested = {str(atomic_numbers[symbol]) for symbol in symbols}
    matches = []
    for key, metadata in basis_metadata().items():
        if metadata.get("role") != "orbital":
            continue
        if q.lower() not in key.lower() and q.lower() not in metadata["display_name"].lower():
            continue
        version = metadata["versions"][metadata["latest_version"]]
        supported = version.get("elements", [])
        if requested - set(supported):
            continue
        matches.append({"name": metadata["display_name"], "elements": supported,
                        "description": metadata.get("description", ""), "status": "available",
                        "role": metadata.get("role"), "verification": "combination_not_asserted",
                        "may_require_ecp": "scalar_ecp" in metadata.get("function_types", []),
                        "version": metadata["latest_version"], "reference": "https://www.basissetexchange.org/"})
        if len(matches) == 100:
            break
    return matches
