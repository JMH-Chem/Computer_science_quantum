"""Local, source-linked capability catalog. Listed is not equivalent to validated."""
from functools import lru_cache
from numbers import Integral

from fastapi import APIRouter, HTTPException, Query

from .examples import examples

router = APIRouter()
SOURCE = "https://pyscf.org/user/dft.html"

LESSONS = [
    {"id": "scf", "title": "SCF 반복은 시간이 아닙니다", "body": "SCF는 전자 분포를 조금씩 고쳐 계산값이 서로 맞아가는 과정입니다. 그래프의 가로축은 실제 시간이 아니라 계산 순서입니다. 수렴은 에너지·밀도 등의 변화가 정한 기준 안으로 작아졌다는 뜻이며 실험값과의 정확도는 별도 확인해야 합니다. DIIS 같은 수렴 가속을 사용하면 에너지가 잠시 오를 수 있으므로 에너지 변화와 밀도 변화·궤도 기울기를 함께 확인하세요.", "source": "https://pyscf.org/user/scf.html"},
    {"id": "basis", "title": "기저 함수 · Basis set", "body": "기저는 전자 상태의 공간 모양을 표현하는 데 쓰는 수학 함수들의 집합입니다. 더 큰 기저는 표현을 풍부하게 하지만 계산량도 늘리며 정확도가 자동으로 보장되지는 않습니다. STO-3G는 원리를 살펴보는 작은 학습용 기저입니다. 큰 기저의 선형종속 위험과 수치 격자 오차도 확인하고, 기저와 격자를 각각 바꾸어 결과가 안정되는지 비교하세요.", "source": "https://www.basissetexchange.org/"},
    {"id": "functional", "title": "교환·상관 함수형 · XC functional", "body": "함수형은 DFT에서 전자 사이 상호작용의 일부를 근사하는 계산 규칙입니다. 모든 분자와 물성에 가장 정확한 한 가지 함수형은 없으며 전체 에너지가 더 낮다고 더 정확한 것도 아닙니다. LDA, GGA, meta-GGA, hybrid, range-separated 등 계열이 있습니다. 목록에 있는 이름도 특정 응답·기울기 계산에서는 지원되지 않을 수 있으므로 작업별 지원 상태를 확인하세요.", "source": SOURCE},
    {"id": "orbital", "title": "오비탈의 두 색상", "body": "오비탈의 두 색은 전하의 양과 음이 아니라 파동함수의 부호를 구분합니다. 이를 양·음의 위상이라고 합니다. 전자밀도는 전자가 어디에 얼마나 분포하는지를 나타내며 음수가 아닙니다. 표시하는 등가면 값을 바꾸면 보이는 경계가 달라지지만 계산한 전자 수가 바뀌지는 않습니다.", "source": "https://pyscf.org/user/tools.html"},
    {"id": "excitation", "title": "여기 에너지와 S₁→S₀", "body": "여기 에너지는 전자의 상태를 바꾸는 데 필요한 에너지 차이입니다. 여기서는 원자 위치를 고정한 수직 여기 에너지를 계산하므로 실험 흡수 띠의 최대값이나 형광 에너지와 바로 같지는 않습니다. 선형응답 TDDFT의 여기 에너지는 HOMO–LUMO 궤도 간격과도 다릅니다. 아래쪽 화살표는 개념 설명이며 실제 시간·수명·양자수율·이완 궤적을 계산한 것이 아닙니다.", "source": "https://pyscf.org/user/tddft.html"},
    {"id": "nto", "title": "자연 전이 오비탈 · NTO", "body": "NTO는 전자 상태의 변화를 소수의 오비탈 쌍으로 요약해 보여 주는 표현입니다. 전이에 관여하는 hole–particle 쌍의 모양을 비교할 수 있지만, 이는 전자 한 개의 실제 이동 경로나 전체 전자밀도 변화 자체는 아닙니다. 이 앱은 PySCF의 X-only 관례로 Y 성분을 제외하고, 선택한 상태에서 가장 큰 가중치를 갖는 쌍을 표시합니다.", "source": "https://github.com/pyscf/pyscf/blob/master/pyscf/tdscf/rhf.py"},
    {"id": "bands", "title": "밴드갭과 광학 갭", "body": "이 화면의 밴드갭은 전자 상태 사이의 계산된 에너지 간격이며 빛 흡수 실험의 값과 바로 같지는 않습니다. 밴드는 k점에 따른 Kohn–Sham 궤도 에너지입니다. 표시된 KS 갭은 계산한 k점·경로 안에서 찾은 값이며 전체 Brillouin zone의 엄밀한 갭으로 보장되지 않습니다. 기저·k점·수치 격자 수렴을 따로 확인하세요.", "source": "https://pyscf.org/user/pbc/scf.html"},
    {"id": "periodic", "title": "결정의 기저와 가상퍼텐셜", "body": "결정 계산은 원자 배치가 공간에서 반복된다고 가정합니다. 이 앱은 Gaussian 궤도 기저와 GTH 가상퍼텐셜을 사용하며, 핵 가까이의 일부 전자를 유효 퍼텐셜로 대신합니다. 따라서 표시된 밀도는 남겨 둔 원자가 전자의 밀도입니다. FFT 격자는 보조 밀도 표현이며 평면파 궤도 기저와 같은 설정은 아닙니다.", "source": "https://pyscf.org/user/pbc/df.html"},
    {"id": "optimization", "title": "구조 최적화의 단계", "body": "구조 최적화는 원자 위치를 바꾸며 가까운 에너지 최소점을 찾는 계산입니다. 각 구조에서 전자 분포를 다시 계산하므로 바깥쪽 최적화 단계와 안쪽 SCF 반복을 구분합니다. 원자가 실제로 그 경로를 따라 움직인다는 뜻은 아닙니다. 최적화 후 진동 분석을 통해 최소점인지 추가로 확인할 수 있습니다.", "source": "https://pyscf.org/user/geomopt.html"},
    {"id": "rhf", "title": "직접 만든 RHF 풀이기", "body": "교육용 RHF는 전자 분포를 갱신하는 계산 절차를 직접 살펴보는 기능입니다. 원자 궤도 적분은 PySCF가 공급하고, S의 직교화·Fock 행렬·점유 오비탈·밀도 갱신은 자체 NumPy 코드로 계산합니다. 결합이 끊어지는 상황처럼 여러 전자 배치가 중요해지면 RHF 근사 자체의 한계가 커집니다.", "source": "https://pyscf.org/user/scf.html"},
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
    {"id": "correlation", "title": "상관 에너지 · Correlation energy", "body": "상관 계산은 평균적인 전자 분포를 넘어 전자들이 서로 영향을 주는 효과를 더 다룹니다. MP2, CI, CC 등으로 RHF가 놓친 상관을 근사하고 RHF 기준과의 에너지 차이를 표시합니다. 전체 에너지의 음수 부호는 기준점에 따른 것이며 다른 분자의 안정성을 그 숫자만으로 비교하지 않습니다. CCSD(T)와 MP2는 변분 상한이 아니므로 더 낮은 수치가 항상 더 정확하지는 않습니다. 저장한 오비탈·밀도·원자 전하는 RHF 기준입니다.", "source": "https://pyscf.org/user/cc.html"},
    {"id": "active_space", "title": "활성 공간 · Active space", "body": "활성 공간은 자세히 다룰 전자와 오비탈을 골라 계산량을 줄이는 선택입니다. CASCI는 그 공간의 전자 배치를 모두 다루고 CASSCF는 궤도도 최적화합니다. 현재는 에너지 순서의 연속된 궤도를 초기 선택하므로 전자 수·궤도 수만으로 화학적으로 적절한 공간을 보장할 수 없습니다. 저장된 점유수와 궤도 성격을 확인해야 합니다.", "source": "https://pyscf.org/user/mcscf.html"},
    {"id": "vibrations", "title": "조화 진동 · Harmonic vibrations", "body": "조화 진동은 원자들이 평형 구조 근처에서 작게 흔들린다고 가정한 계산입니다. 실제 분자의 진동은 완전한 조화 운동이 아니므로 실험의 기본 진동수와 차이가 날 수 있습니다. Hessian이라는 에너지의 두 번째 미분으로 모드를 구하며, 음수 표시는 허수 진동수로 불안정 방향 또는 수치 오차를 뜻할 수 있습니다. 움직임은 확대된 모식도이며 실제 시간 궤적이 아닙니다. IR 세기는 쌍극자 미분에서 구하고 선폭은 그림의 표시 설정입니다.", "source": "https://github.com/pyscf/pyscf/blob/master/pyscf/hessian/thermo.py"},
    {"id": "thermochemistry", "title": "열화학과 기준 상태 · RRHO", "body": "열화학은 전자 에너지에 분자의 운동과 온도·압력의 효과를 더해 해석합니다. 이 앱은 강체 회전자·조화 진동자·이상 기체 근사(RRHO)를 사용합니다. 정지점과 양의 진동 모드를 확인한 경우에 ZPE, 엔탈피·엔트로피·Gibbs 에너지를 제공합니다. 용매·비조화성·내부 회전 효과는 포함하지 않습니다.", "source": "https://github.com/pyscf/pyscf/blob/master/pyscf/hessian/thermo.py"},
    {"id": "atomic_charges", "title": "원자 전하 · Population analysis", "body": "원자 전하는 분자의 전자 분포를 원자별로 나누어 보는 분석값입니다. Mulliken과 Löwdin은 나누는 규칙이 달라 같은 밀도에서도 결과가 다를 수 있습니다. 기저 의존성이 있고 직접 측정한 유일한 원자 전하는 아닙니다. 총 전하 보존과 분석에 사용한 밀도의 기준을 확인하세요.", "source": "https://pyscf.org/user/scf.html"},
    {"id": "dos", "title": "상태 밀도 · Density of states", "body": "상태 밀도는 에너지 구간마다 가능한 전자 상태가 얼마나 있는지 보여 줍니다. 계산한 균일 k점 격자의 Kohn–Sham 에너지를 Gaussian 폭으로 펼친 값이며, 단위는 states/(eV cell)입니다. 두 스핀과 k점 가중치를 포함합니다. 거친 k점의 봉우리나 표시 폭이 만든 모양을 실제 재료의 날카로운 피크 또는 광학 스펙트럼으로 해석하지 않습니다.", "source": "https://pyscf.org/user/pbc/scf.html"},
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
